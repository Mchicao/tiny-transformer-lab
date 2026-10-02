import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safetensors.torch import load_file
import torch
from scripts import ffn_experiment
from scripts.utils import check_experiments, check_training
from src.training import atomic_json


def execute(run_id, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/ffn_experiment.py'), '--variant', 'adamw',
               '--iterations', '1500', '--run-id', run_id, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (run_id + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / run_id


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--audit-run':
        check_experiments.restore = ffn_experiment.restore
        check_experiments.audit(Path(sys.argv[2]).resolve())
        return
    if '--check-id' not in sys.argv and not any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
        raise ValueError('An explicit --check-id is required before any verification updates')
    check_training.Trainer = ffn_experiment.fresh
    check_training.execute = execute
    check_training.main()
    identifier = sys.argv[sys.argv.index('--check-id') + 1]
    output = ROOT / 'output' / identifier
    evidence = json.loads((output / 'recovery.json').read_text())
    continuous, interrupted, resumed = [ROOT / name for name in evidence['runs']]
    initial = load_file(str(check_training.reference(continuous, 'initial') / 'model.safetensors'))
    final = load_file(str(check_training.reference(continuous, 'latest') / 'model.safetensors'))
    ffn_names = [name for name in initial if '.ffn.' in name]
    assert len(ffn_names) == 15 and all(not torch.equal(initial[name], final[name]) for name in ffn_names)
    bitwise = []
    for left, right in [(check_training.reference(continuous, 'boundary'), check_training.reference(interrupted, 'latest')),
                        (check_training.reference(continuous, 'latest'), check_training.reference(resumed, 'latest'))]:
        a, b = [json.loads((path / 'metadata.json').read_text()) for path in [left, right]]
        bitwise.append(a['weights_sha256'] == b['weights_sha256'])
    atomic_json(output / 'recovery.json', dict(**evidence, ffn='swiglu', parameters=3000384,
                ffn_gate_up_down_weights_updated=True, bitwise_weights_equal=all(bitwise)))


if __name__ == '__main__':
    main()
