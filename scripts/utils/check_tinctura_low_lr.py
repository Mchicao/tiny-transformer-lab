import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import tinctura_low_lr_experiment as low
from scripts.utils import check_tinctura_training
from src.training import atomic_json


def execute(identifier, revision, variant, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/tinctura_low_lr_experiment.py'), '--revision', revision,
               '--variant', variant, '--verification', '--run-id', identifier, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (identifier + '.log')
    with log.open('x', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / identifier


def main():
    previous = check_tinctura_training.execute
    try:
        check_tinctura_training.execute = execute
        with low.factories():
            check_tinctura_training.main()
    finally:
        check_tinctura_training.execute = previous
    if '--audit-run' in sys.argv:
        return
    identifier = sys.argv[sys.argv.index('--check-id') + 1]
    path = ROOT / 'output' / identifier / 'recovery.json'
    proof = json.loads(path.read_text())
    atomic_json(path, dict(**proof, **low.fp32.numeric_receipt(), **low.receipt()))


if __name__ == '__main__':
    main()
