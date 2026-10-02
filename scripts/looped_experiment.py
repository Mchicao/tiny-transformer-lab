from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F
from scripts import experiment as campaign
from scripts.train import BASE_EXPERIMENT
from src.looped import LoopedConfig, install
from src.model import ModelConfig, Transformer
from src.training import EXPERIMENT, Trainer, atomic_json, isolated_eval, sha256, weight_hash

ITERATIONS = 1500
SEED = 42
VARIANT = 'looped2-final'
VARIANTS = ['looped2-final', 'looped2-deep', 'looped4-final', 'looped4-deep', 'dense10']
DENSE10_PARAMETERS = 5214144
BASELINES = {
    42: dict(baseline_weights_sha256='194d4675760675037cfaef56e78f3e8aad74ca07d244068e782ac60d8ac1ef64',
             directory=None),
    43: dict(baseline_weights_sha256=None, directory=ROOT / 'output/initializations/3m-seed43-0f4e40f4b755'),
    44: dict(baseline_weights_sha256=None, directory=ROOT / 'output/initializations/3m-ffn-seed44-3f197002/gelu'),
}
INDEX = ROOT / 'output/initializations/3m-looped-index.json'
_CONTRACT = campaign.contract
_BASE_CONTRACT = _CONTRACT('adamw', ITERATIONS)


def parse(variant=None):
    variant = variant or VARIANT
    head, _, recipe = variant.partition('-')
    if head == 'dense10':
        return 10, 'dense'
    if head[:6] != 'looped' or recipe not in {'final', 'deep'}:
        raise ValueError('Authorized variants are looped{2,4}-{final,deep} and dense10')
    return int(head[6:]), recipe


def index_key():
    loops, _ = parse()
    return f'k{loops}-seed{SEED}' if VARIANT != 'dense10' else f'dense10-seed{SEED}'


def contract(variant=None, iterations=ITERATIONS):
    variant = variant or VARIANT
    loops, recipe = parse(variant)
    if SEED not in BASELINES or variant not in VARIANTS or iterations != ITERATIONS:
        raise ValueError('Only looped 3M / dense 10-layer AdamW, 6,144,000 tokens, seeds 42/43/44 are authorized')
    if variant == 'dense10':
        return {**_BASE_CONTRACT, 'seed': SEED, 'variant': variant, 'dense10_format': 1}
    return {**_BASE_CONTRACT, 'seed': SEED, 'variant': variant,
            'looped_format': 1, 'loop_k': loops, 'loss_recipe': recipe}


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
    trainer = LoopTrainer()
    seed_baseline(trainer, SEED)
    baseline_hash = weight_hash(trainer.model)
    if BASELINES[SEED]['baseline_weights_sha256'] != baseline_hash:
        raise ValueError('Baseline weights do not match the seed artifact')
    loops, _ = parse()
    if VARIANT == 'dense10':
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        config = ModelConfig(**{**vars(trainer.config), 'layers': 10})
        trainer.model = Transformer(config).cuda()
        trainer.config = config
        raw = json.loads((ROOT / 'configs/5m-dense10.json').read_text(encoding='utf-8'))
        baseline_hash = weight_hash(trainer.model)
    else:
        trainer.model = install(trainer.model, loops, SEED)
        trainer.config = trainer.model.config
        raw = json.loads((ROOT / f'configs/3m-looped-k{loops}.json').read_text(encoding='utf-8'))
    if raw != {**asdict(trainer.config), 'precision': 'fp16',
               'microbatch': 4, 'gradient_accumulation': 4, 'optimizer': 'adamw'}:
        raise ValueError('Incompatible model config')
    parameters = sum(p.numel() for p in trainer.model.parameters())
    if parameters != (DENSE10_PARAMETERS if VARIANT == 'dense10' else 3000384 + loops * 192):
        raise ValueError('Parameter budget mismatch')
    return trainer, baseline_hash


def sources(trainer):
    names = ['configs/3m-looped-k2.json', 'configs/3m-looped-k4.json',
             'src/looped.py', 'scripts/looped_experiment.py', 'scripts/experiment.py']
    if VARIANT == 'dense10':
        names = ['configs/5m-dense10.json', 'scripts/looped_experiment.py', 'scripts/experiment.py']
    identity = {**trainer.identity, **{name: sha256(ROOT / name) for name in names}}
    return identity


def initialization_directory():
    if not INDEX.exists():
        raise FileNotFoundError(INDEX)
    index = json.loads(INDEX.read_text())
    key = index_key()
    if key not in index:
        raise KeyError(f'Missing initialization: {key}')
    return ROOT / 'output/initializations' / index[key]


def prepare():
    trainer, baseline_hash = architecture()
    loops, _ = parse()
    if not all(torch.isfinite(p).all().item() for p in trainer.model.parameters()):
        raise RuntimeError('Nonfinite initialization')
    parameters = DENSE10_PARAMETERS if VARIANT == 'dense10' else 3000384 + loops * 192
    metadata = dict(seed=SEED, variant=VARIANT, loops=None if VARIANT == 'dense10' else loops,
                    model=asdict(trainer.config), parameters=parameters,
                    weights_sha256=weight_hash(trainer.model),
                    baseline_weights_sha256=baseline_hash, sources=sources(trainer), training_updates=0,
                    loop_initialization=None if VARIANT == 'dense10' else
                    'PCG64 SeedSequence(seed, spawn_key=(loops,)); uniform variance 0.02^2')
    destination = ROOT / 'output/initializations' / f'{"5m-dense10" if VARIANT == "dense10" else f"3m-looped-k{loops}"}-seed{SEED}-{uuid.uuid4().hex[:8]}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name('.pending-' + destination.name + '-' + uuid.uuid4().hex[:8])
    staging.mkdir()
    save_file({name: value.cpu().contiguous() for name, value in trainer.model.state_dict().items()}, str(staging / 'model.safetensors'))
    with (staging / 'model.safetensors').open('rb+') as handle:
        os.fsync(handle.fileno())
    atomic_json(staging / 'initialization.json', metadata)
    atomic_json(staging / 'checksums.json', {name: sha256(staging / name) for name in ['model.safetensors', 'initialization.json']})
    os.replace(staging, destination)
    index = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    index[index_key()] = destination.name
    atomic_json(INDEX, index)
    fresh()
    logging.info(json.dumps(dict(status='prepared', backend=trainer.backend, initialization=str(destination), **metadata)))


def fresh(variant=None, iterations=ITERATIONS):
    global VARIANT
    if variant is not None:
        VARIANT = variant
    value = contract(variant, iterations)
    trainer, baseline_hash = architecture()
    destination = initialization_directory()
    checksums = json.loads((destination / 'checksums.json').read_text())
    if set(checksums) != {'model.safetensors', 'initialization.json'}:
        raise ValueError('Incomplete initialization manifest')
    for name, expected in checksums.items():
        if sha256(destination / name) != expected:
            raise ValueError(f'Initialization checksum mismatch: {name}')
    metadata = json.loads((destination / 'initialization.json').read_text())
    loops, _ = parse()
    parameters = DENSE10_PARAMETERS if VARIANT == 'dense10' else 3000384 + loops * 192
    expected_loops = None if VARIANT == 'dense10' else loops
    if (metadata['seed'], metadata['loops'], metadata['parameters'], metadata['training_updates']) != (SEED, expected_loops, parameters, 0):
        raise ValueError('Incompatible initialization counters')
    if metadata['model'] != asdict(trainer.config) or metadata['sources'] != sources(trainer):
        raise ValueError('Incompatible initialization identity')
    if metadata['baseline_weights_sha256'] != baseline_hash:
        raise ValueError('Baseline linkage mismatch')
    initial = load_file(str(destination / 'model.safetensors'))
    for name, parameter in trainer.model.state_dict().items():
        torch.testing.assert_close(parameter.cpu(), initial[name], rtol=0, atol=0)
    trainer.model.load_state_dict(initial, strict=True)
    trainer.initial_hash = weight_hash(trainer.model)
    if trainer.initial_hash != metadata['weights_sha256']:
        raise ValueError('Looped random weights mismatch')
    trainer.identity = sources(trainer)
    for name in checksums:
        path = destination / name
        trainer.identity[path.relative_to(ROOT).as_posix()] = sha256(path)
    groups = [dict(params=[p for p in trainer.model.parameters() if p.ndim == 2], weight_decay=0.1),
              dict(params=[p for p in trainer.model.parameters() if p.ndim != 2], weight_decay=0.0)]
    trainer.optimizer = torch.optim.AdamW(groups, lr=value['lr'], betas=tuple(value['betas']), foreach=False)
    if trainer.optimizer.state or trainer.data.position or any(trainer.counters.values()):
        raise RuntimeError('Fresh optimizer, cursor and counters required')
    campaign.configure(value)
    return trainer


class LoopTrainer(Trainer):
    def _loss(self, x, y):
        with torch.autocast('cuda', dtype=torch.float16):
            if EXPERIMENT.get('loss_recipe') == 'deep':
                logits = self.model.forward_loops(x)
                losses = [F.cross_entropy(l.float().flatten(0, 1), y.flatten()) for l in logits]
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Nonfinite training loss; stopping')
                return loss
            return F.cross_entropy(self.model(x).float().flatten(0, 1), y.flatten())

    def step(self):
        if self.counters['iterations'] >= EXPERIMENT['iterations']:
            raise RuntimeError('Authorized budget exhausted')
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr = EXPERIMENT['lr'] * min((self.counters['effective'] + 1) / EXPERIMENT['warmup'], 1.0)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        losses, batches = [], hashlib.sha256()
        for _ in range(4):
            x, y = self.data.batch(4, 'cuda')
            batches.update(x.cpu().numpy().tobytes())
            batches.update(y.cpu().numpy().tobytes())
            loss = self._loss(x, y)
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

    def evaluate(self):
        if not hasattr(self.model, 'forward_loops'):
            return Trainer.evaluate(self)
        start = time.perf_counter()
        tokens = np.memmap(ROOT / 'data/processed/validation.bin', dtype='<u2', mode='r')
        totals = [0.0] * self.model.loops
        count = 0
        with isolated_eval(self.model):
            for position in range(0, len(tokens) - 1, 256):
                chunk = torch.tensor(np.array(tokens[position:position + 257], dtype=np.int64), device='cuda')[None]
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = self.model.forward_loops(chunk[:, :-1])
                    targets = chunk[:, 1:].flatten()
                    for index, l in enumerate(logits):
                        totals[index] += F.cross_entropy(l.float().flatten(0, 1), targets, reduction='sum').item()
                count += chunk.shape[1] - 1
        loops_nll = [total / count for total in totals]
        if count != len(tokens) - 1 or not all(math.isfinite(value) for value in loops_nll):
            raise RuntimeError('Validation NLL/token count invalid')
        row = dict(iteration=self.counters['iterations'], tokens=self.counters['tokens'],
                   nll=loops_nll[-1], loops_nll=loops_nll, evaluated_tokens=count,
                   seconds=time.perf_counter() - start)
        self.evaluations.append(row)
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
    global SEED, VARIANT
    if '--prepare-initialization' in sys.argv:
        for flag in ['--seed', '--variant', '--prepare-initialization']:
            if flag in sys.argv:
                index = sys.argv.index(flag)
                value = sys.argv[index + 1] if flag != '--prepare-initialization' else None
                if flag == '--seed':
                    SEED = int(value)
                elif flag == '--variant':
                    VARIANT = value
                del sys.argv[index:index + (1 if flag == '--prepare-initialization' else 2)]
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        prepare()
        return
    if '--seed' in sys.argv:
        index = sys.argv.index('--seed')
        SEED = int(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    if '--variant' in sys.argv:
        VARIANT = sys.argv[sys.argv.index('--variant') + 1]
    if '--continue-baseline' in sys.argv or '--plateau-stop' in sys.argv:
        raise ValueError('Looped runs require fresh weights and the full fixed budget')
    with factories():
        campaign.main()


if __name__ == '__main__':
    main()
