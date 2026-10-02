import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import tinctura_fp32_experiment as numeric
from scripts.utils import check_tinctura_model, check_tinctura_training
from src.training import atomic_json
from src.tinctura_xsa_fp32 import FP32XSATinctura


def execute(identifier, revision, variant, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/tinctura_fp32_experiment.py'), '--revision', revision,
               '--variant', variant, '--verification', '--run-id', identifier, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (identifier + '.log')
    with log.open('x', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / identifier


def main():
    if '--model-check' in sys.argv:
        sys.argv.remove('--model-check')
        identifier = sys.argv[sys.argv.index('--output-id') + 1]
        previous = (check_tinctura_model.Tinctura, check_tinctura_model.torch)
        try:
            check_tinctura_model.Tinctura = FP32XSATinctura
            check_tinctura_model.torch = numeric.FullPrecisionTorch()
            check_tinctura_model.main()
        finally:
            check_tinctura_model.Tinctura, check_tinctura_model.torch = previous
        path = ROOT / 'output' / identifier / 'model-check.json'
        proof = json.loads(path.read_text())
        for revision in proof['revisions'].values():
            for comparison in revision['comparisons']:
                comparison['native_fp32_nll'] = comparison.pop('native_fp16_nll')
                comparison['fp32_nll_delta'] = comparison.pop('fp16_nll_delta')
        proof.pop('fp16_nll_drift_limit')
    else:
        previous = check_tinctura_training.execute
        try:
            check_tinctura_training.execute = execute
            with numeric.factories():
                check_tinctura_training.main()
        finally:
            check_tinctura_training.execute = previous
        if '--audit-run' in sys.argv:
            return
        identifier = sys.argv[sys.argv.index('--check-id') + 1]
        path = ROOT / 'output' / identifier / 'recovery.json'
        proof = json.loads(path.read_text())
    atomic_json(path, dict(**proof, **numeric.numeric_receipt()))


if __name__ == '__main__':
    main()
