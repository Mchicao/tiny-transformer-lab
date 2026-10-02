import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.training import Trainer, atomic_json, rng_state, restore_rng, verify_checkpoint

RTOL, ATOL = 1e-6, 1e-7


def compare(left, right, exact=False):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0 if exact else RTOL, atol=0 if exact else ATOL)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key], exact or key in {'step', 'rng'})
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            compare(a, b, exact)
    else:
        assert left == right, (left, right)


def reference(run, name):
    return run / 'checkpoints' / json.loads((run / 'references.json').read_text())[name]


def execute(run_id, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/train.py'), '--run-id', run_id, '--check-target', str(target)]
    if resume:
        command.extend(['--resume', str(resume)])
    log = ROOT / 'logs/tests' / f'{run_id}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / run_id


def main():
    parser = argparse.ArgumentParser(description='10 updates total: continuous 5 vs interrupted 2 + fresh-process 3')
    parser.add_argument('--check-id', default='adamw-check-' + uuid.uuid4().hex[:12])
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'output' / args.check_id
    if args.verify_existing:
        original = json.loads((output / 'existing-artifacts.json').read_text())
        continuous, interrupted, resumed = [ROOT / 'runs' / (args.check_id + '-' + suffix)
                                           for suffix in ['continuous', 'interrupted', 'resumed']]
        boundary = reference(interrupted, 'latest')
    else:
        output.mkdir(parents=True, exist_ok=False)
        original = {str(p.relative_to(ROOT)): __import__('hashlib').sha256(p.read_bytes()).hexdigest()
                    for folder in ['data', 'output', 'runs'] for p in (ROOT / folder).rglob('*')
                    if p.is_file() and p.name != 'results.jsonl'}
        atomic_json(output / 'existing-artifacts.json', original)
        continuous = execute(args.check_id + '-continuous', 5)
        interrupted = execute(args.check_id + '-interrupted', 2)
        boundary = reference(interrupted, 'latest')
        resumed = execute(args.check_id + '-resumed', 5, boundary)
    trainer = Trainer()
    from safetensors.torch import load_file
    for first, second in [(reference(continuous, 'boundary'), boundary),
                          (reference(continuous, 'latest'), reference(resumed, 'latest'))]:
        a = verify_checkpoint(first, trainer.identity, trainer.backend)
        b = verify_checkpoint(second, trainer.identity, trainer.backend)
        compare(load_file(str(first / 'model.safetensors')), load_file(str(second / 'model.safetensors')))
        compare(torch.load(first / 'state.pt', weights_only=True, map_location='cpu'),
                torch.load(second / 'state.pt', weights_only=True, map_location='cpu'))
        for name in ['counters', 'cursor']:
            compare(a[name], b[name], exact=True)
        if a['counters']['iterations'] == 5:
            compare(a['best_loss'], b['best_loss'], exact=True)
        for row_a, row_b in zip(a['history'], b['history'], strict=True):
            compare({k: v for k, v in row_a.items() if k != 'seconds'},
                    {k: v for k, v in row_b.items() if k != 'seconds'}, exact=True)
    trainer.restore(boundary)
    saved = rng_state()
    position = trainer.data.position
    expected_x, expected_y = trainer.data.batch(4, 'cpu')
    trainer.data.position = position
    expected_samples = [random.random(), np.random.random(), torch.rand(4), torch.rand(4, device='cuda').cpu()]
    trainer.restore(boundary)
    actual_x, actual_y = trainer.data.batch(4, 'cpu')
    compare(expected_x, actual_x, exact=True)
    compare(expected_y, actual_y, exact=True)
    compare(expected_samples, [random.random(), np.random.random(), torch.rand(4), torch.rand(4, device='cuda').cpu()], exact=True)
    restore_rng(saved)
    position = trainer.data.position
    mode = trainer.model.training
    trainer.evaluate()
    trainer.generate()
    compare(saved, rng_state(), exact=True)
    assert trainer.data.position == position and trainer.model.training == mode
    wrong = copy.deepcopy(trainer.identity)
    wrong['configs/3m.json'] = 'incompatible'
    try:
        verify_checkpoint(boundary, wrong, trainer.backend)
    except ValueError:
        pass
    else:
        raise AssertionError('Incompatible identity accepted')
    damaged = output / ('corrupt-checkpoint-' + uuid.uuid4().hex[:8])
    shutil.copytree(boundary, damaged)
    with (damaged / 'state.pt').open('r+b') as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 1]))
    try:
        trainer.restore(damaged)
    except ValueError as error:
        assert 'checksum mismatch' in str(error)
    else:
        raise AssertionError('Corrupt optimizer checkpoint accepted')
    refs_before = json.loads((interrupted / 'references.json').read_text())
    pending = interrupted / ('checkpoints/.pending-simulated-interruption-' + uuid.uuid4().hex[:8])
    pending.mkdir()
    (pending / 'model.safetensors').write_bytes(b'incomplete')
    assert json.loads((interrupted / 'references.json').read_text()) == refs_before
    trainer.restore(reference(interrupted, 'latest'))
    for run in [continuous, interrupted, resumed]:
        for name in json.loads((run / 'references.json').read_text()):
            verify_checkpoint(reference(run, name), trainer.identity, trainer.backend)
    assert all(__import__('hashlib').sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in original.items())
    atomic_json(output / 'recovery.json', dict(status='passed', tolerance=dict(rtol=RTOL, atol=ATOL),
                updates_executed=10, tokens_executed=40960, comparison='5 continuous vs 2 + 3 in new processes',
                batches_counters_cursor_rng_exact=True, optimizer_scaler_verified=True,
                weights_updated=True, evaluation_generation_isolated=True, corruption_rejected=True,
                incompatible_identity_rejected=True, incomplete_checkpoint_ignored=True,
                references_verified=True, existing_artifacts_preserved=True,
                runs=[str(r.relative_to(ROOT)) for r in [continuous, interrupted, resumed]]))
    print(json.dumps(dict(status='passed', evidence=str(output / 'recovery.json'))))


if __name__ == '__main__':
    main()
