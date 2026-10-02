import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import fresh_data_experiment as fresh
from scripts.utils import check_experiments, check_training
from src.training import atomic_json, sha256


def execute(identifier, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/fresh_data_experiment.py'), '--seed', str(fresh.SEED),
               '--run-id', identifier, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (identifier + '.log')
    with log.open('x', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / identifier


def main():
    if '--seed' in sys.argv:
        index = sys.argv.index('--seed')
        fresh.SEED = int(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    if '--audit-run' in sys.argv:
        check_experiments.restore = fresh.restore
        run = Path(sys.argv[sys.argv.index('--audit-run') + 1]).resolve()
        check_experiments.audit(run)
        return
    if '--check-id' not in sys.argv:
        raise ValueError('A unique --check-id is required for verification updates')
    check_training.Trainer = fresh.fresh
    check_training.execute = execute
    check_training.main()
    identifier = sys.argv[sys.argv.index('--check-id') + 1]
    path = ROOT / 'output' / identifier / 'recovery.json'
    proof = json.loads(path.read_text())
    continuous, interrupted, resumed = [ROOT / name for name in proof['runs']]
    pairs = [(check_training.reference(continuous, 'boundary'), check_training.reference(interrupted, 'latest')),
             (check_training.reference(continuous, 'latest'), check_training.reference(resumed, 'latest'))]
    assert all(json.loads((a / 'metadata.json').read_text())['weights_sha256'] ==
               json.loads((b / 'metadata.json').read_text())['weights_sha256'] for a, b in pairs)
    atomic_json(path, dict(**proof, seed=fresh.SEED, coverage_format=1, no_training_wrap=True,
                bitwise_weights_equal=True, entrypoint_sha256=sha256(ROOT / 'scripts/fresh_data_experiment.py')))


if __name__ == '__main__':
    main()
