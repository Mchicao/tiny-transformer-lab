from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import ffn_seed44
from scripts.utils import check_experiments, check_ffn, check_training


def execute(run_id, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/ffn_seed44.py'), '--ffn', ffn_seed44.FFN,
               '--variant', 'adamw', '--iterations', '1500', '--run-id', run_id, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (run_id + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / run_id


def main():
    if '--ffn' not in sys.argv:
        raise ValueError('Explicit --ffn gelu/swiglu required')
    index = sys.argv.index('--ffn')
    ffn_seed44.FFN = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    ffn_seed44.contract()
    if len(sys.argv) == 3 and sys.argv[1] == '--audit-run':
        check_experiments.restore = ffn_seed44.restore
        check_experiments.audit(Path(sys.argv[2]).resolve())
        return
    if '--check-id' not in sys.argv:
        raise ValueError('Explicit --check-id required before verification updates')
    if ffn_seed44.FFN == 'swiglu':
        check_ffn.ffn_experiment = ffn_seed44
        check_ffn.execute = execute
        check_ffn.main()
    else:
        check_training.Trainer = ffn_seed44.fresh
        check_training.execute = execute
        check_training.main()


if __name__ == '__main__':
    main()
