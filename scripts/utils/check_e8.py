import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import e8_decay_experiment
from scripts.utils import check_experiments, check_training
from src.training import atomic_json


def execute(run_id, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/e8_decay_experiment.py'),
               '--variant', e8_decay_experiment.VARIANT, '--seed', str(e8_decay_experiment.SEED),
               '--iterations', '1500', '--run-id', run_id, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (run_id + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / run_id


def schedule_checks():
    start, end = e8_decay_experiment.COOLDOWN_START, e8_decay_experiment.ITERATIONS
    for step, expected in [(1, 0.1), (5, 0.5), (10, 1.0), (11, 1.0), (start, 1.0)]:
        assert abs(e8_decay_experiment.decay_factor(step) - expected) < 1e-12, step
    assert abs(e8_decay_experiment.decay_factor(end) - 0.0) < 1e-12
    midpoint = e8_decay_experiment.decay_factor((start + end) // 2)
    assert abs(midpoint - 0.5) < 0.01
    factors = [e8_decay_experiment.decay_factor(step) for step in range(start, end + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(factors, factors[1:]))
    assert math.isfinite(factors[-1])


def main():
    for flag, attribute in [('--variant', 'VARIANT'), ('--seed', 'SEED')]:
        if flag in sys.argv:
            index = sys.argv.index(flag)
            setattr(e8_decay_experiment, attribute, int(sys.argv[index + 1]) if flag == '--seed' else sys.argv[index + 1])
            del sys.argv[index:index + 2]
    if len(sys.argv) == 3 and sys.argv[1] == '--audit-run':
        check_experiments.restore = e8_decay_experiment.restore
        check_experiments.audit(Path(sys.argv[2]).resolve())
        return
    if '--check-id' not in sys.argv and not any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
        raise ValueError('An explicit --check-id is required before any verification updates')
    e8_decay_experiment.contract()
    check_training.Trainer = e8_decay_experiment.fresh
    check_training.execute = execute
    check_training.main()
    identifier = sys.argv[sys.argv.index('--check-id') + 1]
    output = ROOT / 'output' / identifier
    evidence = json.loads((output / 'recovery.json').read_text())
    continuous, interrupted, resumed = [ROOT / name for name in evidence['runs']]
    bitwise = []
    for left, right in [(check_training.reference(continuous, 'boundary'), check_training.reference(interrupted, 'latest')),
                        (check_training.reference(continuous, 'latest'), check_training.reference(resumed, 'latest'))]:
        a, b = [json.loads((path / 'metadata.json').read_text()) for path in [left, right]]
        bitwise.append(a['weights_sha256'] == b['weights_sha256'])
    schedule_checks()
    atomic_json(output / 'recovery.json', dict(**evidence, variant=e8_decay_experiment.VARIANT,
                seed=e8_decay_experiment.SEED, decay_schedule_verified=True,
                bitwise_weights_equal=all(bitwise)))


if __name__ == '__main__':
    main()
