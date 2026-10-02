from contextlib import contextmanager
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from scripts import experiment as campaign
from scripts.train import BASE_EXPERIMENT
from src.model import Transformer
from src.training import EXPERIMENT, Trainer, isolated_eval, sha256, weight_hash

ITERATIONS = 1500
SEED = 42
VARIANT = 'adamw-decay'
VARIANTS = [VARIANT]
COOLDOWN_FRACTION = 0.15
COOLDOWN_START = int(ITERATIONS * (1.0 - COOLDOWN_FRACTION))
BASELINES = {
    42: dict(baseline_weights_sha256='194d4675760675037cfaef56e78f3e8aad74ca07d244068e782ac60d8ac1ef64',
             directory=None),
    43: dict(baseline_weights_sha256=None, directory=ROOT / 'output/initializations/3m-seed43-0f4e40f4b755'),
    44: dict(baseline_weights_sha256=None, directory=ROOT / 'output/initializations/3m-ffn-seed44-3f197002/gelu'),
}
_CONTRACT = campaign.contract
_BASE_CONTRACT = _CONTRACT('adamw', ITERATIONS)


def decay_factor(step):
    if step <= COOLDOWN_START:
        return min(step / EXPERIMENT['warmup'], 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * (step - COOLDOWN_START) / (ITERATIONS - COOLDOWN_START)))


def contract(variant=None, iterations=ITERATIONS):
    variant = variant or VARIANT
    if SEED not in BASELINES or variant != VARIANT or iterations != ITERATIONS:
        raise ValueError('E8 authorizes only AdamW GELU 3M with cooldown, 6,144,000 tokens, seeds 42/43/44')
    return {**_BASE_CONTRACT, 'seed': SEED, 'variant': variant, 'e8_format': 1,
            'decay': dict(kind='cosine_to_zero', start=COOLDOWN_START,
                          end=ITERATIONS, fraction=COOLDOWN_FRACTION, base_lr=_BASE_CONTRACT['lr'])}


def seed_baseline(trainer, seed):
    if seed == 42:
        return
    directory = BASELINES[seed]['directory']
    checksums = json.loads((directory / 'checksums.json').read_text())
    if set(checksums) != {'model.safetensors', 'initialization.json'}:
        raise ValueError('Incomplete baseline initialization manifest')
    for name, expected in checksums.items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Baseline checksum mismatch: {name}')
    metadata = json.loads((directory / 'initialization.json').read_text())
    if metadata['seed'] != seed or metadata['parameters'] != 3000384 or metadata['training_updates'] != 0:
        raise ValueError('Incompatible baseline initialization')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    expected_model = Transformer(trainer.config)
    if weight_hash(expected_model) != metadata['weights_sha256']:
        raise ValueError('Seed baseline weights identity mismatch')
    trainer.model.load_state_dict(expected_model.state_dict(), strict=True)
    BASELINES[seed]['baseline_weights_sha256'] = metadata['weights_sha256']
    for name in checksums:
        path = directory / name
        trainer.identity[path.relative_to(ROOT).as_posix()] = sha256(path)


def architecture():
    contract()
    campaign.configure(BASE_EXPERIMENT)
    trainer = DecayTrainer()
    seed_baseline(trainer, SEED)
    if BASELINES[SEED]['baseline_weights_sha256'] != weight_hash(trainer.model):
        raise ValueError('Baseline weights do not match the seed artifact')
    if sum(p.numel() for p in trainer.model.parameters()) != 3000384:
        raise ValueError('Parameter budget mismatch')
    return trainer


def sources(trainer):
    names = ['scripts/e8_decay_experiment.py', 'scripts/experiment.py']
    return {**trainer.identity, **{name: sha256(ROOT / name) for name in names}}


def fresh(variant=None, iterations=ITERATIONS):
    global VARIANT
    if variant is not None:
        VARIANT = variant
    value = contract(variant, iterations)
    trainer = architecture()
    trainer.identity = sources(trainer)
    groups = [dict(params=[p for p in trainer.model.parameters() if p.ndim == 2], weight_decay=0.1),
              dict(params=[p for p in trainer.model.parameters() if p.ndim != 2], weight_decay=0.0)]
    trainer.optimizer = torch.optim.AdamW(groups, lr=value['lr'], betas=tuple(value['betas']), foreach=False)
    if trainer.optimizer.state or trainer.data.position or any(trainer.counters.values()):
        raise RuntimeError('Fresh optimizer, cursor and counters required')
    campaign.configure(value)
    return trainer


class DecayTrainer(Trainer):
    def step(self):
        if self.counters['iterations'] >= EXPERIMENT['iterations']:
            raise RuntimeError('Authorized budget exhausted')
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr = EXPERIMENT['lr'] * decay_factor(self.counters['effective'] + 1)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        losses, batches = [], hashlib.sha256()
        for _ in range(4):
            x, y = self.data.batch(4, 'cuda')
            batches.update(x.cpu().numpy().tobytes())
            batches.update(y.cpu().numpy().tobytes())
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(self.model(x).float().flatten(0, 1), y.flatten())
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite training loss; stopping')
            self.scaler.scale(loss / 4).backward()
            losses.append(loss.detach().item())
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale
        self.counters['iterations'] += 1
        self.counters['tokens'] += 4096
        self.counters['skipped' if skipped else 'effective'] += 1
        self.counters['consecutive_skips'] = self.counters['consecutive_skips'] + 1 if skipped else 0
        if self.counters['consecutive_skips'] >= 3:
            raise RuntimeError('Persistent instability: three consecutive skipped updates')
        if not all(torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise RuntimeError('Nonfinite weights; stopping')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.train_seconds += elapsed
        row = dict(**self.counters, loss=sum(losses) / 4, lr=lr,
                   grad_norm=norm if math.isfinite(norm) else None,
                   scale=self.scaler.get_scale(), batch_sha256=batches.hexdigest(),
                   cursor=self.data.position, seconds=elapsed)
        self.history.append(row)
        return row


@contextmanager
def factories():
    # ponytail: process-scoped factories preserve legacy checkpoint hashes; concurrent variants need separate processes.
    replacements = dict(contract=contract, fresh=fresh, VARIANTS=VARIANTS)
    previous = {name: getattr(campaign, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(campaign, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(campaign, name, value)


def restore(checkpoint):
    with factories():
        return campaign.restore(checkpoint)


def main():
    global SEED
    if '--seed' in sys.argv:
        index = sys.argv.index('--seed')
        SEED = int(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    if '--variant' in sys.argv:
        VARIANT = sys.argv[sys.argv.index('--variant') + 1]
    if '--continue-baseline' in sys.argv or '--plateau-stop' in sys.argv:
        raise ValueError('E8 requires fresh weights and the full fixed budget')
    with factories():
        campaign.main()


if __name__ == '__main__':
    main()
