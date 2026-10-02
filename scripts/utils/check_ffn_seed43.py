from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import ffn_seed43
from scripts.utils import check_ffn


def execute(run_id, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/ffn_seed43.py'), '--variant', 'adamw',
               '--iterations', '1500', '--run-id', run_id, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (run_id + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / run_id


if __name__ == '__main__':
    check_ffn.ffn_experiment = ffn_seed43
    check_ffn.execute = execute
    check_ffn.main()
