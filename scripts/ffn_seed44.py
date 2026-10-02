from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import random
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from safetensors.torch import load_file, save_file
import torch
from scripts import experiment as campaign
from scripts.train import BASE_EXPERIMENT
from src.model import Transformer
from src.swiglu import install
from src.training import Trainer, atomic_json, sha256, weight_hash

ITERATIONS, SEED = 1500, 44
FFN = None
INITIALIZATIONS = ROOT / 'output/initializations/3m-ffn-seed44-3f197002'
_CONTRACT = campaign.contract


def contract(variant='adamw', iterations=ITERATIONS):
    if FFN not in {'gelu', 'swiglu'} or variant != 'adamw' or iterations != ITERATIONS:
        raise ValueError('Only seed44 GELU/SwiGLU AdamW, 6,144,000 tokens, is authorized')
    return {**_CONTRACT(variant, iterations), 'seed': SEED, 'ffn_replication_format': 1,
            'ffn': FFN, 'ffn_hidden': 768 if FFN == 'gelu' else 512}


def architecture():
    contract()
    campaign.configure(BASE_EXPERIMENT)
    trainer = Trainer()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    expected = Transformer(trainer.config)
    trainer.model.load_state_dict(expected.state_dict(), strict=True)
    baseline_hash = weight_hash(trainer.model)
    if baseline_hash == trainer.initial_hash:
        raise ValueError('Seed44 must differ from the original seed42')
    shared = {name: value.clone() for name, value in trainer.model.state_dict().items() if '.ffn.' not in name}
    if FFN == 'swiglu':
        trainer.config = install(trainer.model, seed=SEED)
    for name, value in shared.items():
        torch.testing.assert_close(trainer.model.state_dict()[name], value, rtol=0, atol=0)
    raw = json.loads((ROOT / f'configs/3m-{FFN}-seed44.json').read_text(encoding='utf-8'))
    if raw != {**asdict(trainer.config), 'seed': SEED, 'precision': 'fp16',
               'microbatch': 4, 'gradient_accumulation': 4, 'optimizer': 'adamw'}:
        raise ValueError('Incompatible seed44 config')
    if sum(p.numel() for p in trainer.model.parameters()) != 3000384:
        raise ValueError('Parameter budget mismatch')
    return trainer, baseline_hash


def sources(trainer):
    names = ['configs/3m-gelu-seed44.json', 'configs/3m-swiglu-seed44.json',
             'src/swiglu.py', 'scripts/ffn_seed44.py', 'scripts/experiment.py']
    return {**trainer.identity, **{name: sha256(ROOT / name) for name in names}}


def prepare():
    trainer, baseline_hash = architecture()
    if not all(torch.isfinite(p).all().item() for p in trainer.model.parameters()):
        raise RuntimeError('Nonfinite initialization')
    metadata = dict(seed=SEED, ffn=FFN, model=asdict(trainer.config), parameters=3000384,
                    weights_sha256=weight_hash(trainer.model), original_weights_sha256=trainer.initial_hash,
                    baseline_seed44_weights_sha256=baseline_hash, sources=sources(trainer), training_updates=0,
                    shared_tensors_exact=22, ffn_initialization='GELU: original PCG64 seed44 policy; '
                    'SwiGLU: separate PCG64 seed44 stream, same variance and residual scaling')
    destination = INITIALIZATIONS / FFN
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name('.pending-' + FFN + '-' + uuid.uuid4().hex[:8])
    staging.mkdir()
    save_file({name: value.cpu().contiguous() for name, value in trainer.model.state_dict().items()}, str(staging / 'model.safetensors'))
    with (staging / 'model.safetensors').open('rb+') as handle:
        os.fsync(handle.fileno())
    atomic_json(staging / 'initialization.json', metadata)
    atomic_json(staging / 'checksums.json', {name: sha256(staging / name) for name in ['model.safetensors', 'initialization.json']})
    os.replace(staging, destination)
    fresh()
    logging.info(json.dumps(dict(status='prepared', backend=trainer.backend, initialization=str(destination), **metadata)))


def fresh(variant='adamw', iterations=ITERATIONS):
    value = contract(variant, iterations)
    trainer, baseline_hash = architecture()
    destination = INITIALIZATIONS / FFN
    checksums = json.loads((destination / 'checksums.json').read_text())
    if set(checksums) != {'model.safetensors', 'initialization.json'}:
        raise ValueError('Incomplete initialization')
    for name, expected in checksums.items():
        if sha256(destination / name) != expected:
            raise ValueError(f'Initialization checksum mismatch: {name}')
    metadata = json.loads((destination / 'initialization.json').read_text())
    if (metadata['seed'], metadata['ffn'], metadata['parameters'], metadata['training_updates']) != (SEED, FFN, 3000384, 0):
        raise ValueError('Incompatible initialization counters')
    if metadata['model'] != asdict(trainer.config) or metadata['sources'] != sources(trainer):
        raise ValueError('Incompatible initialization identity')
    if metadata['original_weights_sha256'] != trainer.initial_hash or metadata['baseline_seed44_weights_sha256'] != baseline_hash:
        raise ValueError('Original random checkpoint identity mismatch')
    initial = load_file(str(destination / 'model.safetensors'))
    for name, parameter in trainer.model.state_dict().items():
        torch.testing.assert_close(parameter.cpu(), initial[name], rtol=0, atol=0)
    trainer.model.load_state_dict(initial, strict=True)
    trainer.initial_hash = weight_hash(trainer.model)
    if trainer.initial_hash != metadata['weights_sha256']:
        raise ValueError('Random weights mismatch')
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


@contextmanager
def factories():
    # ponytail: process-scoped factories preserve old checkpoint hashes; concurrent runs require separate processes.
    replacements = dict(contract=contract, fresh=fresh, VARIANTS=['adamw'])
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
    global FFN
    if '--ffn' not in sys.argv:
        raise ValueError('Explicit --ffn gelu/swiglu required')
    index = sys.argv.index('--ffn')
    FFN = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    contract()
    if sys.argv[1:] == ['--prepare-initialization']:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        prepare()
        return
    if '--continue-baseline' in sys.argv or '--plateau-stop' in sys.argv:
        raise ValueError('Seed44 requires fresh weights and a fixed budget')
    with factories():
        campaign.main()


if __name__ == '__main__':
    main()
