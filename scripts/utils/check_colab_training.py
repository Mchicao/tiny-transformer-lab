import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid

import numpy as np
from safetensors.torch import load_file
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model import ModelConfig, Transformer
from src.colab_training import Trainer, TrainingConfig, seed_all, weights_hash, write_json


def create(directory, seed=42, identity='fixture'):
    seed_all(seed)
    model = Transformer(ModelConfig(vocab_size=128, dim=32, layers=2, heads=2, context=16))
    config = TrainingConfig(microbatch=1, accumulation=2, max_tokens=224, warmup_updates=2)
    return Trainer(model, directory / 'train.bin', directory / 'validation.bin', config, 'cuda', {'fixture': identity})


def samples():
    return {'python': random.random(), 'numpy': np.random.random(), 'torch': torch.rand(4).tolist(),
            'cuda': torch.rand(4, device='cuda').cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description='Small authorized GPU training check: interrupted vs uninterrupted, RNG, validation and checkpoints')
    parser.add_argument('--recover', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for training recovery verification')
    if args.recover:
        directory = args.recover
        trainer = create(directory, seed=999)
        trainer.load(directory / 'split/checkpoints/iteration-000002')
        assert samples() == json.loads((directory / 'rng.json').read_text())
        losses = [trainer.step()['loss'] for _ in range(3)]
        trainer.save(directory / 'recovered', trainer.evaluate()['loss'])
        write_json(directory / 'recovered-losses.json', losses)
        return
    directory = ROOT / 'output' / f'training-check-{uuid.uuid4().hex}'
    directory.mkdir(parents=True, exist_ok=False)
    (np.arange(1024, dtype=np.uint16) % 128).astype('<u2').tofile(directory / 'train.bin')
    (np.arange(257, dtype=np.uint16) % 128).astype('<u2').tofile(directory / 'validation.bin')
    continuous = create(directory)
    initial = weights_hash(continuous.model)
    first = [continuous.step()['loss'] for _ in range(2)]
    expected_rng = samples()
    last = [continuous.step()['loss'] for _ in range(3)]
    continuous.save(directory / 'continuous', continuous.evaluate()['loss'])
    split = create(directory)
    actual_first = [split.step()['loss'] for _ in range(2)]
    split.model.train()
    position = split.train.position
    random_state, numpy_state, torch_state, cuda_state = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()
    evaluation = split.evaluate()
    assert evaluation['tokens'] == 256 and split.train.position == position and split.model.training
    assert random.getstate() == random_state and np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state) and torch.equal(torch.cuda.get_rng_state(), cuda_state)
    split.checkpoint(directory / 'split', split.evaluate()['loss'])
    assert samples() == expected_rng
    write_json(directory / 'rng.json', expected_rng)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--recover', str(directory)], check=True)
    recovered_weights = load_file(str(directory / 'recovered/model.safetensors'))
    for name, value in load_file(str(directory / 'continuous/model.safetensors')).items():
        torch.testing.assert_close(value, recovered_weights[name], rtol=1e-5, atol=1e-7)
    continuous_metadata = json.loads((directory / 'continuous/metadata.json').read_text())
    recovered_metadata = json.loads((directory / 'recovered/metadata.json').read_text())
    for field in ['iterations', 'updates', 'tokens', 'skipped', 'data_position', 'initial_weights_sha256']:
        assert continuous_metadata[field] == recovered_metadata[field], field
    torch.testing.assert_close(torch.tensor(first + last), torch.tensor(actual_first + json.loads((directory / 'recovered-losses.json').read_text())), rtol=1e-5, atol=1e-6)
    expected_state = torch.load(directory / 'continuous/state.pt', map_location='cpu', weights_only=True)
    recovered_state = torch.load(directory / 'recovered/state.pt', map_location='cpu', weights_only=True)
    for index, state in expected_state['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, recovered_state['optimizer']['state'][index][name], rtol=1e-5, atol=1e-7)
    assert expected_state['scaler'] == recovered_state['scaler']
    assert weights_hash(continuous.model) != initial and continuous.updates > 0
    wrong = create(directory, identity='different')
    try:
        wrong.load(directory / 'continuous')
        raise AssertionError('Accepted incompatible experiment')
    except ValueError as exc:
        assert 'identity/backend mismatch' in str(exc)
    damaged = directory / 'damaged'
    shutil.copytree(directory / 'continuous', damaged)
    with (damaged / 'model.safetensors').open('r+b') as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    try:
        create(directory).load(damaged)
        raise AssertionError('Accepted corrupt weights')
    except ValueError as exc:
        assert 'checksum mismatch' in str(exc)
    best = json.loads((directory / 'split/best.json').read_text())
    split.step()
    split.checkpoint(directory / 'split', best['validation_loss'] + 1)
    assert json.loads((directory / 'split/best.json').read_text()) == best
    assert json.loads((directory / 'split/latest.json').read_text())['tokens'] == split.tokens
    split.step()
    split.checkpoint(directory / 'split', best['validation_loss'] - 1)
    assert json.loads((directory / 'split/best.json').read_text())['tokens'] == split.tokens
    assert len(json.loads((directory / 'split/milestones.json').read_text())) == 3
    split.checkpoint(directory / 'new-segment', best['validation_loss'] + 2)
    assert json.loads((directory / 'new-segment/best.json').read_text())['tokens'] == split.tokens
    write_json(directory / 'result.json', {'status': 'passed', 'gpu': torch.cuda.get_device_name(0),
                                          'fresh_process_resume': True, 'rng_verified': True, 'evaluation_isolated': True,
                                          'adamw_and_scaler_restored': True, 'corruption_rejected': True,
                                          'identity_rejected': True, 'retention_verified': True, 'weight_rtol': 1e-5, 'weight_atol': 1e-7})
    print(json.dumps({'training_check': str(directory), 'status': 'passed'}))


if __name__ == '__main__':
    main()
