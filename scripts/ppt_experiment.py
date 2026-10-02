from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import e8_decay_experiment as e8
from scripts import experiment as campaign
from scripts.train import BASE_EXPERIMENT
from src.data import PackedTokens
from src.synthetic_ppt import ARMS, PPT_STEPS, SPEC, verify
from src.training import EXPERIMENT, atomic_json, sha256, weight_hash

SEED = 42
VARIANT = 'retrieval'
VERIFICATION = False
DATA = ROOT / 'data/synthetic-ppt/e9-v1'
BASE_CONTRACT = copy.deepcopy(e8._BASE_CONTRACT)


def lengths():
    return (2, 3) if VERIFICATION else (PPT_STEPS, 1500)


def contract(variant=None, iterations=None):
    variant = variant or VARIANT
    ppt, pt = lengths()
    total = ppt + pt
    if SEED not in (42, 43, 44) or variant not in ARMS or iterations not in (None, total):
        raise ValueError('E9 requires seeds 42/43/44, retrieval/grammatical, and its fixed two-phase budget')
    milestones = list(range(1, total + 1)) if VERIFICATION else sorted(set(
        [50, 100, 150, 200, ppt] + [ppt + n for n in BASE_CONTRACT['milestones']]))
    return {**BASE_CONTRACT, 'seed': SEED, 'variant': variant, 'iterations': total,
            'milestones': milestones, 'e9_format': 1, 'verification': VERIFICATION,
            'ppt': dict(steps=ppt, tokens=ppt * 4096, spec=SPEC, dataset=DATA.relative_to(ROOT).as_posix()),
            'pt': dict(steps=pt, tokens=pt * 4096, cooldown_fraction=0.15),
            'transfer': dict(weights_only=True, reset_optimizer=True, reset_scaler=True,
                             reset_cursor=True, reset_rng_to_seed=True)}


def schedule(step):
    ppt, pt = lengths()
    local = step if step <= ppt else step - ppt
    if step <= ppt:
        return min(local / EXPERIMENT['warmup'], 1.0)
    start = int(pt * 0.85)
    if local <= start:
        return min(local / EXPERIMENT['warmup'], 1.0)
    return 0.5 * (1 + math.cos(math.pi * (local - start) / (pt - start)))


def optimizer(model):
    groups = [dict(params=[p for p in model.parameters() if p.ndim == 2], weight_decay=0.1),
              dict(params=[p for p in model.parameters() if p.ndim != 2], weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=BASE_CONTRACT['lr'], betas=tuple(BASE_CONTRACT['betas']), foreach=False)


class PPTTrainer(e8.DecayTrainer):
    def step(self):
        ppt, _ = lengths()
        if self.counters['iterations'] == ppt:
            self.optimizer = optimizer(self.model)
            self.scaler = torch.amp.GradScaler('cuda', init_scale=EXPERIMENT['init_scale'])
            self.data = PackedTokens(ROOT / 'data/processed/train.bin', 256)
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
        previous = e8.decay_factor
        try:
            e8.decay_factor = schedule
            row = super().step()
        finally:
            e8.decay_factor = previous
        if row['skipped']:
            raise RuntimeError('E9 requires zero skipped updates; stopping')
        row.update(phase='ppt' if row['iterations'] <= ppt else 'pt',
                   phase_iteration=row['iterations'] if row['iterations'] <= ppt else row['iterations'] - ppt)
        return row


def fresh(variant=None, iterations=None):
    global VARIANT
    if variant is not None:
        VARIANT = variant
    value = contract(variant, iterations)
    campaign.configure(BASE_EXPERIMENT)
    trainer = PPTTrainer()
    e8.seed_baseline(trainer, SEED)
    if weight_hash(trainer.model) != e8.BASELINES[SEED]['baseline_weights_sha256']:
        raise ValueError('E9 seed initialization mismatch')
    trainer.initial_hash = weight_hash(trainer.model)
    trainer.optimizer = optimizer(trainer.model)
    trainer.data = PackedTokens(DATA / f'{VARIANT}-seed{SEED}.bin', 256)
    trainer.identity.update(verify(DATA))
    for name in ['scripts/ppt_experiment.py', 'src/synthetic_ppt.py', 'scripts/e8_decay_experiment.py',
                 'scripts/experiment.py', 'scripts/train.py']:
        trainer.identity[name] = sha256(ROOT / name)
    if trainer.optimizer.state or trainer.data.position or any(trainer.counters.values()):
        raise RuntimeError('Fresh phase requires empty state')
    campaign.configure(value)
    return trainer


@contextmanager
def factories():
    replacements = dict(contract=contract, fresh=fresh, restore=restore, VARIANTS=list(ARMS))
    previous = {name: getattr(campaign, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(campaign, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(campaign, name, value)


def restore(checkpoint):
    checksums = json.loads((checkpoint / 'checksums.json').read_text())
    if set(checksums) != {'model.safetensors', 'state.pt', 'metadata.json'}:
        raise ValueError('Incomplete checkpoint manifest')
    for name, expected in checksums.items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    value = metadata['experiment']
    if value['variant'] != VARIANT or value['seed'] != SEED:
        raise ValueError('Checkpoint seed/arm mismatch')
    if value != contract(value['variant'], value['iterations']):
        raise ValueError('Checkpoint experiment mismatch')
    trainer = fresh(value['variant'], value['iterations'])
    if metadata['counters']['iterations'] > lengths()[0]:
        trainer.data = PackedTokens(ROOT / 'data/processed/train.bin', 256)
    trainer.restore(checkpoint)
    return trainer, metadata


def main():
    global SEED, VARIANT, VERIFICATION
    if '--seed' in sys.argv:
        index = sys.argv.index('--seed')
        SEED = int(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    if '--verification' in sys.argv:
        VERIFICATION = True
        sys.argv.remove('--verification')
    if '--variant' in sys.argv:
        VARIANT = sys.argv[sys.argv.index('--variant') + 1]
    if not VERIFICATION:
        if '--recovery-check' not in sys.argv:
            raise ValueError('A passing seed/arm-specific --recovery-check is required before principal training')
        index = sys.argv.index('--recovery-check')
        evidence = json.loads(Path(sys.argv[index + 1]).read_text())
        del sys.argv[index:index + 2]
        if (evidence['status'] != 'passed' or evidence['seed'] != SEED or evidence['variant'] != VARIANT
                or evidence['entrypoint_sha256'] != sha256(Path(__file__))
                or evidence['generator_sha256'] != sha256(ROOT / 'src/synthetic_ppt.py')):
            raise ValueError('Recovery evidence does not match principal seed/arm/source')
    if '--continue-baseline' in sys.argv or '--plateau-stop' in sys.argv:
        raise ValueError('E9 requires fixed budgets and no baseline continuation')
    if '--iterations' not in sys.argv:
        sys.argv += ['--iterations', str(sum(lengths()))]
    with factories():
        campaign.main()
    run = ROOT / 'runs' / sys.argv[sys.argv.index('--run-id') + 1]
    metrics = json.loads((run / 'metrics.json').read_text())
    boundary = lengths()[0]
    history = metrics['history']
    phases = {}
    for phase, rows in [('ppt', [r for r in history if r['iterations'] <= boundary]),
                        ('pt', [r for r in history if r['iterations'] > boundary])]:
        seconds = sum(r['seconds'] for r in rows)
        phases[phase] = dict(iterations=len(rows), tokens=len(rows) * 4096,
                             train_seconds=seconds, tokens_s=len(rows) * 4096 / seconds if seconds else None)
    atomic_json(run / 'phase-accounting.json', dict(phases=phases, transfer=EXPERIMENT['transfer'],
                initial_seed_weights_sha256=metrics['initial_weights_sha256']))


if __name__ == '__main__':
    main()
