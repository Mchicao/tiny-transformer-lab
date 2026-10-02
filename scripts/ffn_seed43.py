from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safetensors.torch import load_file, save_file
import torch
from scripts import experiment as campaign
from scripts.train import SEED43_EXPERIMENT, make_trainer
from src.swiglu import install
from src.training import Trainer, atomic_json, sha256, weight_hash

ITERATIONS = 1500
INITIALIZATION = ROOT / 'output/initializations/3m-swiglu-seed43-b6cfe902'
_CONTRACT = campaign.contract


def contract(variant='adamw', iterations=ITERATIONS):
    if variant != 'adamw' or iterations != ITERATIONS:
        raise ValueError('SwiGLU comparison authorizes only AdamW and 6,144,000 tokens')
    return {**_CONTRACT(variant, iterations), 'seed': 43,
            'ffn_format': 1, 'ffn': 'swiglu', 'ffn_hidden': 512}


def architecture():
    campaign.configure(SEED43_EXPERIMENT)
    trainer = make_trainer(43, campaign.INITIALIZATION)
    trainer.config = install(trainer.model, seed=43)
    raw = json.loads((ROOT / 'configs/3m-swiglu-seed43.json').read_text())
    if raw != {**asdict(trainer.config), 'seed': 43, 'precision': 'fp16',
               'microbatch': 4, 'gradient_accumulation': 4, 'optimizer': 'adamw'}:
        raise ValueError('Incompatible SwiGLU config')
    if sum(p.numel() for p in trainer.model.parameters()) != 3000384:
        raise ValueError('SwiGLU parameter budget mismatch')
    return trainer


def sources(trainer):
    return {**trainer.identity, **{name: sha256(ROOT / name) for name in
            ['configs/3m-swiglu-seed43.json', 'src/swiglu.py', 'scripts/ffn_seed43.py', 'scripts/experiment.py']}}


def prepare():
    trainer = architecture()
    original = load_file(str(campaign.INITIALIZATION / 'model.safetensors'))
    unchanged = [name for name in original if '.ffn.' not in name]
    for name in unchanged:
        torch.testing.assert_close(trainer.model.state_dict()[name].cpu(), original[name], rtol=0, atol=0)
    if not all(torch.isfinite(p).all().item() for p in trainer.model.parameters()):
        raise RuntimeError('Nonfinite SwiGLU initialization')
    metadata = dict(seed=43, model=asdict(trainer.config), parameters=3000384,
                    weights_sha256=weight_hash(trainer.model), original_weights_sha256=trainer.initial_hash,
                    sources=sources(trainer), training_updates=0, shared_tensors_exact=len(unchanged),
                    ffn_initialization='Separate PCG64 seed43 stream; uniform variance 0.02^2; down scaled by 1/sqrt(2*layers)')
    INITIALIZATION.parent.mkdir(parents=True, exist_ok=True)
    if INITIALIZATION.exists():
        raise FileExistsError(INITIALIZATION)
    staging = INITIALIZATION.with_name('.pending-' + INITIALIZATION.name + '-' + uuid.uuid4().hex[:8])
    staging.mkdir()
    save_file({name: value.cpu().contiguous() for name, value in trainer.model.state_dict().items()}, str(staging / 'model.safetensors'))
    with (staging / 'model.safetensors').open('rb+') as handle:
        os.fsync(handle.fileno())
    atomic_json(staging / 'initialization.json', metadata)
    atomic_json(staging / 'checksums.json', {name: sha256(staging / name) for name in ['model.safetensors', 'initialization.json']})
    os.replace(staging, INITIALIZATION)
    fresh()
    logging.info(json.dumps(dict(status='prepared', initialization=str(INITIALIZATION),
                                backend=trainer.backend, **metadata)))


def fresh(variant='adamw', iterations=ITERATIONS):
    value = contract(variant, iterations)
    trainer = architecture()
    checksums = json.loads((INITIALIZATION / 'checksums.json').read_text())
    if set(checksums) != {'model.safetensors', 'initialization.json'}:
        raise ValueError('Incomplete SwiGLU initialization')
    for name, expected in checksums.items():
        if sha256(INITIALIZATION / name) != expected:
            raise ValueError(f'Initialization checksum mismatch: {name}')
    metadata = json.loads((INITIALIZATION / 'initialization.json').read_text())
    if (metadata['seed'], metadata['parameters'], metadata['training_updates']) != (43, 3000384, 0):
        raise ValueError('Incompatible SwiGLU initialization counters')
    if metadata['model'] != asdict(trainer.config) or metadata['sources'] != sources(trainer):
        raise ValueError('Incompatible SwiGLU initialization identity')
    if metadata['original_weights_sha256'] != trainer.initial_hash:
        raise ValueError('Original random checkpoint identity mismatch')
    initial = load_file(str(INITIALIZATION / 'model.safetensors'))
    for name, parameter in trainer.model.state_dict().items():
        torch.testing.assert_close(parameter.cpu(), initial[name], rtol=0, atol=0)
    trainer.model.load_state_dict(initial, strict=True)
    trainer.initial_hash = weight_hash(trainer.model)
    if trainer.initial_hash != metadata['weights_sha256']:
        raise ValueError('SwiGLU random weights mismatch')
    trainer.identity = sources(trainer)
    for name in checksums:
        path = INITIALIZATION / name
        trainer.identity[path.relative_to(ROOT).as_posix()] = sha256(path)
    groups = [dict(params=[p for p in trainer.model.parameters() if p.ndim == 2], weight_decay=0.1),
              dict(params=[p for p in trainer.model.parameters() if p.ndim != 2], weight_decay=0.0)]
    trainer.optimizer = torch.optim.AdamW(groups, lr=value['lr'], betas=tuple(value['betas']), foreach=False)
    if trainer.optimizer.state or trainer.data.position or any(trainer.counters.values()):
        raise RuntimeError('SwiGLU requires fresh optimizer, cursor and counters')
    campaign.configure(value)
    return trainer


@contextmanager
def factories():
    # ponytail: process-scoped factories preserve legacy checkpoint hashes; concurrent variants need separate processes.
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
    if sys.argv[1:] == ['--prepare-initialization']:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        prepare()
        return
    if '--continue-baseline' in sys.argv:
        raise ValueError('GELU trained weights cannot initialize the SwiGLU comparison')
    with factories():
        campaign.main()


if __name__ == '__main__':
    main()
