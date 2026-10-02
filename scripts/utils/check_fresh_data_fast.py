import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from scripts import fresh_data_experiment as fresh
from scripts.utils.check_training import compare, reference
from src.training import atomic_json, rng_state, sha256, weight_hash


def execute(identifier, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/fresh_data_experiment.py'), '--seed', str(fresh.SEED),
               '--run-id', identifier, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    with (ROOT / 'logs/tests' / (identifier + '.log')).open('x', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / identifier


def main():
    parser = argparse.ArgumentParser(description='E11 focused10-update recovery; historical inventory verified once separately')
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--check-id', required=True)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    if Path(args.check_id).name != args.check_id:
        parser.error('Unique check ID required')
    fresh.SEED = args.seed
    output = ROOT / 'output' / args.check_id
    if args.verify_existing:
        if not output.exists():
            raise ValueError('Existing verification output required')
        continuous, interrupted, resumed = [ROOT / 'runs' / (args.check_id + '-' + suffix)
                                            for suffix in ['continuous', 'interrupted', 'resumed']]
    else:
        output.mkdir(parents=True, exist_ok=False)
        continuous = execute(args.check_id + '-continuous', 5)
        interrupted = execute(args.check_id + '-interrupted', 2)
        resumed = execute(args.check_id + '-resumed', 5, reference(interrupted, 'latest'))
    for left, right in [(reference(continuous, 'boundary'), reference(interrupted, 'latest')),
                        (reference(continuous, 'latest'), reference(resumed, 'latest'))]:
        a, b = [json.loads((path / 'metadata.json').read_text()) for path in [left, right]]
        assert a['weights_sha256'] == b['weights_sha256']
        compare(load_file(str(left / 'model.safetensors')), load_file(str(right / 'model.safetensors')), exact=True)
        compare(torch.load(left / 'state.pt', weights_only=True, map_location='cpu'),
                torch.load(right / 'state.pt', weights_only=True, map_location='cpu'), exact=True)
        for key in ['counters', 'cursor']:
            compare(a[key], b[key], exact=True)
        if a['counters']['iterations'] == 5:
            compare(a['best_loss'], b['best_loss'], exact=True)
        for x, y in zip(a['history'], b['history'], strict=True):
            compare({k: v for k, v in x.items() if k != 'seconds'},
                    {k: v for k, v in y.items() if k != 'seconds'}, exact=True)
    trainer, metadata = fresh.restore(reference(resumed, 'latest'))
    assert trainer.counters['tokens'] == trainer.data.position == 20480
    assert trainer.counters['effective'] == 5 and trainer.counters['skipped'] == 0
    before = (weight_hash(trainer.model), copy.deepcopy(trainer.counters), trainer.data.position,
              rng_state(), trainer.model.training)
    trainer.evaluate()
    trainer.generate()
    compare(before, (weight_hash(trainer.model), trainer.counters, trainer.data.position,
                    rng_state(), trainer.model.training), exact=True)
    damaged = output / 'corrupt-checkpoint'
    shutil.copytree(reference(interrupted, 'latest'), damaged)
    with (damaged / 'state.pt').open('r+b') as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    try:
        fresh.restore(damaged)
    except ValueError as error:
        assert 'checksum mismatch' in str(error)
    else:
        raise AssertionError('Corrupt checkpoint accepted')
    wrong = 43 if args.seed != 43 else 42
    fresh.SEED = wrong
    try:
        fresh.restore(reference(interrupted, 'latest'))
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong seed accepted')
    finally:
        fresh.SEED = args.seed
    assert metadata['initial_weights_sha256'] != metadata['weights_sha256']
    atomic_json(output / 'recovery.json', dict(status='passed', seed=args.seed, coverage_format=1,
                no_training_wrap=True, bitwise_weights_equal=True, optimizer_scaler_rng_exact=True,
                batches_cursor_history_exact=True, evaluation_generation_isolated=True, corruption_rejected=True,
                incompatible_seed_rejected=True, updates_executed=10, tokens_executed=40960,
                entrypoint_sha256=sha256(ROOT / 'scripts/fresh_data_experiment.py'),
                preservation_inventory='output/e11-fresh-seed42-check-20261002-v1/existing-artifacts.json',
                runs=[r.relative_to(ROOT).as_posix() for r in [continuous, interrupted, resumed]]))
    print(json.dumps(dict(status='passed', evidence=str(output / 'recovery.json'))))


if __name__ == '__main__':
    main()
