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
from scripts import ppt_experiment as ppt
from scripts.utils import check_experiments
from scripts.utils.check_training import compare, reference
from src.synthetic_ppt import check_sequences, probe
from src.training import atomic_json, rng_state, sha256, weight_hash


def execute(identifier, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/ppt_experiment.py'), '--variant', ppt.VARIANT,
               '--seed', str(ppt.SEED), '--verification', '--run-id', identifier, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / f'{identifier}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / identifier


def pair(left, right):
    trainer, a = ppt.restore(left)
    _, b = ppt.restore(right)
    compare(load_file(str(left / 'model.safetensors')), load_file(str(right / 'model.safetensors')), exact=True)
    compare(torch.load(left / 'state.pt', weights_only=True, map_location='cpu'),
            torch.load(right / 'state.pt', weights_only=True, map_location='cpu'), exact=True)
    for key in ['counters', 'cursor', 'best_loss', 'initial_weights_sha256']:
        compare(a[key], b[key], exact=True)
    assert a['weights_sha256'] == b['weights_sha256']
    for x, y in zip(a['history'], b['history'], strict=True):
        compare({k: v for k, v in x.items() if k != 'seconds'},
                {k: v for k, v in y.items() if k != 'seconds'}, exact=True)
    return trainer


def main():
    parser = argparse.ArgumentParser(description='E9: fresh-process recovery across PPT/PT and inside PT')
    parser.add_argument('--variant', choices=ppt.ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--check-id')
    parser.add_argument('--audit-run', type=Path)
    parser.add_argument('--baseline-run', type=Path)
    args = parser.parse_args()
    ppt.VARIANT, ppt.SEED = args.variant, args.seed
    if args.audit_run:
        run = args.audit_run.resolve()
        check_experiments.restore = ppt.restore
        check_experiments.audit(run)
        trainer, _ = ppt.restore(reference(run, 'latest'))
        before = (weight_hash(trainer.model), copy.deepcopy(trainer.counters), trainer.data.position, rng_state())
        result = probe(trainer.model, ppt.DATA)
        assert result == probe(trainer.model, ppt.DATA)
        compare(before, (weight_hash(trainer.model), trainer.counters, trainer.data.position, rng_state()), exact=True)
        atomic_json(run / 'synthetic-probe-final.json', result)
        if args.baseline_run:
            from scripts import e8_decay_experiment as e8
            e8.SEED = args.seed
            baseline = args.baseline_run.resolve()
            model, _ = e8.restore(reference(baseline, 'latest'))
            atomic_json(run / 'synthetic-probe-e8-control.json', dict(run=str(baseline.relative_to(ROOT)),
                        seed=args.seed, probe=probe(model.model, ppt.DATA), training_updates=0))
        return
    if not args.check_id or Path(args.check_id).name != args.check_id:
        parser.error('A unique --check-id is required before verification updates')
    ppt.VERIFICATION = True
    output = ROOT / 'output' / args.check_id
    output.mkdir(parents=True, exist_ok=False)
    check_sequences()
    continuous = execute(args.check_id + '-continuous', 5)
    interrupted = execute(args.check_id + '-interrupted', 2)
    resumed = execute(args.check_id + '-resumed', 5, reference(interrupted, 'latest'))
    pair(reference(continuous, 'milestone-8192'), reference(interrupted, 'latest'))
    trainer = pair(reference(continuous, 'latest'), reference(resumed, 'latest'))
    pt_interrupted = execute(args.check_id + '-pt-interrupted', 3)
    pt_resumed = execute(args.check_id + '-pt-resumed', 5, reference(pt_interrupted, 'latest'))
    pair(reference(continuous, 'milestone-12288'), reference(pt_interrupted, 'latest'))
    pair(reference(continuous, 'latest'), reference(pt_resumed, 'latest'))
    assert trainer.counters['effective'] == 5 and trainer.counters['skipped'] == 0
    assert [r['phase'] for r in trainer.history] == ['ppt', 'ppt', 'pt', 'pt', 'pt']
    assert [r['cursor'] for r in trainer.history] == [4096, 8192, 4096, 8192, 12288]
    assert trainer.history[2]['lr'] == ppt.BASE_CONTRACT['lr'] / 10
    assert trainer.history[-1]['lr'] == 0
    pt_checkpoint = reference(continuous, 'milestone-12288')
    state = torch.load(pt_checkpoint / 'state.pt', weights_only=True, map_location='cpu')
    assert all(int(s['step'].item()) == 1 for s in state['optimizer']['state'].values())
    boundary = reference(interrupted, 'latest')
    bad = output / 'corrupt-checkpoint'
    shutil.copytree(boundary, bad)
    with (bad / 'state.pt').open('r+b') as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    try:
        ppt.restore(bad)
    except ValueError as error:
        assert 'checksum mismatch' in str(error)
    else:
        raise AssertionError('Corrupt checkpoint accepted')
    old_seed, old_arm = ppt.SEED, ppt.VARIANT
    for attribute, wrong in [('SEED', 43 if ppt.SEED != 43 else 42), ('VARIANT', 'grammatical' if ppt.VARIANT == 'retrieval' else 'retrieval')]:
        setattr(ppt, attribute, wrong)
        try:
            ppt.restore(boundary)
        except ValueError:
            pass
        else:
            raise AssertionError('Incompatible seed/arm accepted')
        finally:
            ppt.SEED, ppt.VARIANT = old_seed, old_arm
    ppt.VERIFICATION = False
    assert ppt.schedule(245) == 1 and ppt.schedule(246) == 0.1
    assert ppt.schedule(245 + 1275) == 1 and ppt.schedule(1745) == 0
    assert all(ppt.schedule(n) >= ppt.schedule(n + 1) for n in range(1520, 1745))
    try:
        ppt.restore(boundary)
    except ValueError:
        pass
    else:
        raise AssertionError('Verification weights accepted by principal contract')
    atomic_json(output / 'recovery.json', dict(status='passed', variant=args.variant, seed=args.seed,
                bitwise_weights_equal=True, optimizer_scaler_rng_exact=True, phase_reset_verified=True,
                comparison='5 vs 2+3 across transition; 5 vs 3+2 inside PT, all in new processes',
                updates_executed=15, tokens_executed=61440, corruption_rejected=True,
                incompatible_seed_arm_and_verification_rejected=True, decay_schedule_verified=True,
                synthetic_generator_verified=True, entrypoint_sha256=sha256(ROOT / 'scripts/ppt_experiment.py'),
                generator_sha256=sha256(ROOT / 'src/synthetic_ppt.py'),
                runs=[str(r.relative_to(ROOT)) for r in [continuous, interrupted, resumed, pt_interrupted, pt_resumed]]))
    print(json.dumps(dict(status='passed', evidence=str(output / 'recovery.json'))))


if __name__ == '__main__':
    main()
