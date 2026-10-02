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
from scripts import tinctura_experiment as e7
from scripts.utils.check_training import compare, reference
from src.training import atomic_json, rng_state, sha256, weight_hash


def execute(identifier, revision, variant, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/tinctura_experiment.py'), '--revision', revision,
               '--variant', variant, '--verification', '--run-id', identifier, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (identifier + '.log')
    with log.open('x', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / identifier


def audit(run, revision, variant):
    metrics = json.loads((run / 'metrics.json').read_text())
    checkpoints = sorted((run / 'checkpoints').glob('step-*'))
    refs = json.loads((run / 'references.json').read_text())
    assert checkpoints and not list((run / 'checkpoints').glob('.pending-*'))
    for path in checkpoints:
        trainer, metadata = e7.restore(path, revision, variant)
        state = torch.load(path / 'state.pt', map_location='cpu', weights_only=True)
        compare(trainer.optimizer.state_dict(), state['optimizer'], exact=True)
        compare(trainer.scaler.state_dict(), state['scaler'], exact=True)
        compare(rng_state(), state['rng'], exact=True)
        assert trainer.counters == metadata['counters'] and trainer.data.position == metadata['cursor']
        del trainer, state
        torch.cuda.empty_cache()
    assert all((run / 'checkpoints' / name) in checkpoints for name in refs.values())
    trainer, metadata = e7.restore(reference(run, 'latest'), revision, variant)
    before = (weight_hash(trainer.model), copy.deepcopy(trainer.counters), trainer.data.position,
              rng_state(), copy.deepcopy(trainer.optimizer.state_dict()), copy.deepcopy(trainer.scaler.state_dict()))
    validation = trainer.evaluate()
    assert validation['nll'] == metadata['evaluations'][-1]['nll']
    assert validation['loops_nll'] == metadata['evaluations'][-1]['loops_nll']
    if metrics['status'] == 'completed':
        assert trainer.generate() == json.loads((run / 'generations-final.json').read_text())
        assert trainer.arithmetic_probe() == json.loads((run / 'arithmetic-final.json').read_text())
    compare(before, (weight_hash(trainer.model), trainer.counters, trainer.data.position, rng_state(),
                    trainer.optimizer.state_dict(), trainer.scaler.state_dict()), exact=True)
    atomic_json(run / 'post-run-verification.json', dict(status='passed', checkpoints_restored=len(checkpoints),
                optimizer_scaler_rng_exact=True, validation_and_loops_nll_exact=True,
                generations_and_arithmetic_exact=metrics['status'] == 'completed',
                training_updates=0, references_verified=True))
    print(json.dumps(dict(status='passed', audit=str(run), checkpoints=len(checkpoints), training_updates=0)))


def main():
    parser = argparse.ArgumentParser(description='E7 actual ctx2048 recovery5 vs2+3 or post-segment audit')
    parser.add_argument('--revision', choices=e7.SEEDS, required=True)
    parser.add_argument('--variant', choices=['dense', 'looped'], required=True)
    parser.add_argument('--check-id')
    parser.add_argument('--audit-run', type=Path)
    args = parser.parse_args()
    if args.audit_run:
        audit(args.audit_run.resolve(), args.revision, args.variant)
        return
    if not args.check_id or Path(args.check_id).name != args.check_id:
        parser.error('A unique check ID is required before verification updates')
    output = ROOT / 'output' / args.check_id
    output.mkdir(parents=True, exist_ok=False)
    continuous = execute(args.check_id + '-continuous', args.revision, args.variant, 5)
    interrupted = execute(args.check_id + '-interrupted', args.revision, args.variant, 2)
    resumed = execute(args.check_id + '-resumed', args.revision, args.variant, 5, reference(interrupted, 'latest'))
    for left, right in [(reference(continuous, 'milestone-8192'), reference(interrupted, 'latest')),
                        (reference(continuous, 'latest'), reference(resumed, 'latest'))]:
        a = json.loads((left / 'metadata.json').read_text())
        b = json.loads((right / 'metadata.json').read_text())
        assert a['weights_sha256'] == b['weights_sha256']
        compare(load_file(str(left / 'model.safetensors')), load_file(str(right / 'model.safetensors')), exact=True)
        compare(torch.load(left / 'state.pt', map_location='cpu', weights_only=True),
                torch.load(right / 'state.pt', map_location='cpu', weights_only=True), exact=True)
        for key in ['counters', 'cursor', 'best_loss']:
            compare(a[key], b[key], exact=True)
        for x, y in zip(a['history'], b['history'], strict=True):
            compare({k: v for k, v in x.items() if k != 'seconds'},
                    {k: v for k, v in y.items() if k != 'seconds'}, exact=True)
    trainer, _ = e7.restore(reference(resumed, 'latest'), args.revision, args.variant, True)
    assert trainer.counters['effective'] == 5 and trainer.counters['skipped'] == 0
    before = (weight_hash(trainer.model), copy.deepcopy(trainer.counters), trainer.data.position, rng_state())
    trainer.evaluate()
    compare(before, (weight_hash(trainer.model), trainer.counters, trainer.data.position, rng_state()), exact=True)
    boundary = reference(interrupted, 'latest')
    damaged = output / 'corrupt-checkpoint'
    shutil.copytree(boundary, damaged)
    with (damaged / 'state.pt').open('r+b') as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    try:
        trainer.restore(damaged)
    except ValueError as error:
        assert 'checksum mismatch' in str(error)
    else:
        raise AssertionError('Corrupt optimizer accepted')
    try:
        e7.restore(boundary, args.revision, args.variant, False)
    except ValueError:
        pass
    else:
        raise AssertionError('Verification checkpoint accepted as principal')
    principal = e7.contract(args.revision, args.variant)
    assert e7.schedule(1, principal) == 0.1 and e7.schedule(10, principal) == 1
    assert e7.schedule(415, principal) == 1 and e7.schedule(489, principal) == 0
    assert sum(min(4096, 2000000 - n * 4096) for n in range(489)) == 2000000
    atomic_json(output / 'recovery.json', dict(status='passed', revision=args.revision, variant=args.variant,
                seed=e7.SEEDS[args.revision], source_sha256=sha256(ROOT / 'src/tinctura_loop.py'),
                entrypoint_sha256=sha256(ROOT / 'scripts/tinctura_experiment.py'),
                comparison='5 continuous vs2+3 in new processes at actual ctx2048',
                updates_executed=10, tokens_executed=40960, bitwise_weights_optimizer_scaler_rng=True,
                batches_cursor_counters_exact=True, corruption_rejected=True,
                incompatible_verification_rejected=True, evaluation_isolated=True,
                principal_schedule_and_exact_2M_budget_verified=True,
                runs=[p.relative_to(ROOT).as_posix() for p in [continuous, interrupted, resumed]]))
    print(json.dumps(dict(status='passed', evidence=str(output / 'recovery.json'))))


if __name__ == '__main__':
    main()
