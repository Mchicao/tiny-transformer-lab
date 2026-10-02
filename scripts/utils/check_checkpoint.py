import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import PackedTokens
from src.model import ModelConfig, Transformer


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_hash(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def atomic_write(path, writer):
    temporary = path.with_suffix(path.suffix + '.tmp')
    writer(temporary)
    temporary.replace(path)


def identity():
    return {name: sha256(ROOT / name) for name in [
        'configs/3m.json', 'data/processed/tokenizer.json',
        'data/processed/train.bin', 'data/processed/validation.bin',
    ]}


def rng_samples(device):
    return {
        'python': torch.tensor([random.random() for _ in range(4)], dtype=torch.float64),
        'numpy': torch.from_numpy(np.random.random(4)),
        'torch': torch.rand(4),
        'device': torch.rand(4, device=device).cpu(),
    }


def recover(directory, device):
    manifest = json.loads((directory / 'checksums.json').read_text())
    for name, expected in manifest.items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata['identity'] != identity():
        raise ValueError('Checkpoint config/tokenizer/data identity mismatch')
    if metadata['device'] != device:
        raise ValueError('RNG recovery check requires the same device')
    torch.manual_seed(999)
    model = Transformer(ModelConfig(**metadata['config'])).to(device).eval()
    model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
    assert weight_hash(model) == metadata['weights_sha256']
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    state = torch.load(directory / 'state.pt', map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    assert optimizer.param_groups[0]['lr'] == metadata['optimizer_lr']
    assert scaler.state_dict() == state['scaler']
    for parameter, saved in zip(model.parameters(), state['optimizer']['state'].values(), strict=True):
        restored = optimizer.state[parameter]
        for name in ['step', 'exp_avg', 'exp_avg_sq']:
            torch.testing.assert_close(restored[name].cpu(), saved[name], rtol=0, atol=0)
    packed = PackedTokens(ROOT / 'data/processed/train.bin', metadata['config']['context'])
    packed.position = metadata['data_position']
    x, y = packed.batch(1, 'cpu')
    torch.testing.assert_close(x, state['next_x'], rtol=0, atol=0)
    torch.testing.assert_close(y, state['next_y'], rtol=0, atol=0)
    with torch.inference_mode():
        logits = model(state['input'].to(device)).cpu()
    torch.testing.assert_close(logits, state['logits'], rtol=0, atol=0)
    random.setstate(state['python_rng'])
    numpy_state = state['numpy_rng']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch_rng'])
    if device == 'cuda':
        torch.cuda.set_rng_state_all(state['device_rng'])
    samples = rng_samples(device)
    for name, expected in state['rng_samples'].items():
        torch.testing.assert_close(samples[name], expected, rtol=0, atol=0)
    assert metadata['optimizer_steps'] == metadata['tokens_trained'] == 0
    assert weight_hash(model) == metadata['weights_sha256']
    result = {
        'status': 'passed', 'device': device, 'parameters': sum(p.numel() for p in model.parameters()),
        'weights_sha256': weight_hash(model), 'optimizer_steps': 0, 'tokens_trained': 0,
        'exact_logits': True, 'rng_restored': list(samples), 'data_cursor_restored': True,
        'adamw_state': 'synthetic nonempty buffers; no optimization executed',
        'scaler_restored': device == 'cuda', 'fresh_process': True,
        'scope': 'random checkpoint recovery only; training/resume/retention not validated',
    }
    (directory / 'recovery.json').write_text(json.dumps(result, indent=2))
    logging.info(json.dumps(result))


def prepare(directory, device):
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    config = ModelConfig()
    model = Transformer(config).to(device).eval()
    original_hash = weight_hash(model)
    initial = ROOT / 'output/initial-3m.safetensors'
    if initial.exists():
        initial_weights = load_file(str(initial))
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter.cpu(), initial_weights[name], rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            'step': torch.tensor(0.0),
            'exp_avg': torch.full_like(parameter, 0.001),
            'exp_avg_sq': torch.full_like(parameter, 0.002),
        }
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    if device == 'cuda':
        scaler.scale(torch.ones((), device=device))
    tokens = torch.arange(16, device=device).reshape(1, 16)
    with torch.inference_mode():
        logits = model(tokens).cpu()
    packed = PackedTokens(ROOT / 'data/processed/train.bin', config.context)
    packed.batch(1, 'cpu')
    position = packed.position
    x, y = packed.batch(1, 'cpu')
    numpy_state = np.random.get_state()
    state = {
        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
        'python_rng': random.getstate(),
        'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'torch_rng': torch.get_rng_state(),
        'device_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [],
        'input': tokens.cpu(), 'logits': logits, 'next_x': x, 'next_y': y,
    }
    state['rng_samples'] = rng_samples(device)
    atomic_write(directory / 'model.safetensors', lambda path: save_file(
        {name: value.cpu().contiguous() for name, value in model.state_dict().items()}, str(path)))
    atomic_write(directory / 'state.pt', lambda path: torch.save(state, path))
    metadata = {
        'config': asdict(config), 'identity': identity(), 'device': device,
        'weights_sha256': original_hash, 'data_position': position,
        'optimizer_lr': optimizer.param_groups[0]['lr'], 'optimizer_steps': 0, 'tokens_trained': 0,
        'synthetic_optimizer_state': True,
    }
    atomic_write(directory / 'metadata.json', lambda path: path.write_text(json.dumps(metadata, indent=2)))
    checksums = {name: sha256(directory / name) for name in ['model.safetensors', 'state.pt', 'metadata.json']}
    atomic_write(directory / 'checksums.json', lambda path: path.write_text(json.dumps(checksums, indent=2)))
    assert weight_hash(model) == original_hash
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--recover', str(directory), '--device', device], check=True)
    with tempfile.TemporaryDirectory(prefix='checkpoint-corruption-', dir=ROOT / '.cache') as temporary:
        damaged_dir = Path(temporary)
        for name in [*checksums, 'checksums.json']:
            shutil.copy2(directory / name, damaged_dir / name)
        damaged = bytearray((damaged_dir / 'model.safetensors').read_bytes())
        damaged[-1] ^= 1
        (damaged_dir / 'model.safetensors').write_bytes(damaged)
        rejected = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--recover', str(damaged_dir), '--device', device], capture_output=True, text=True)
        assert rejected.returncode != 0 and 'Checkpoint checksum mismatch: model.safetensors' in rejected.stderr
        (directory / 'corruption-check.json').write_text(json.dumps({'corruption_rejected_before_load': True, 'returncode': rejected.returncode}))
    logging.info('PASS: safetensors export, fresh-process recovery and checksum detection; zero optimizer steps')


def main():
    parser = argparse.ArgumentParser(description='Recover random 3M weights and synthetic checkpoint state; never trains')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--recover', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; no CPU fallback')
    if args.recover:
        recover(args.recover.resolve(), args.device)
    else:
        prepare((args.output_dir or ROOT / 'output' / f'checkpoint-check-{uuid.uuid4().hex}').resolve(), args.device)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
