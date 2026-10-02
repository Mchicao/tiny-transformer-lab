import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safetensors.torch import load_file
import torch
from scripts import looped_experiment
from scripts.utils import check_experiments, check_training
from src.training import atomic_json


def execute(run_id, target, resume=None):
    command = [sys.executable, str(ROOT / 'scripts/looped_experiment.py'),
               '--variant', looped_experiment.VARIANT, '--seed', str(looped_experiment.SEED),
               '--iterations', '1500', '--run-id', run_id, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / (run_id + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / run_id


def main():
    for flag, attribute in [('--variant', 'VARIANT'), ('--seed', 'SEED')]:
        if flag in sys.argv:
            index = sys.argv.index(flag)
            setattr(looped_experiment, attribute, int(sys.argv[index + 1]) if flag == '--seed' else sys.argv[index + 1])
            del sys.argv[index:index + 2]
    if len(sys.argv) == 3 and sys.argv[1] == '--audit-run':
        check_experiments.restore = looped_experiment.restore
        check_experiments.audit(Path(sys.argv[2]).resolve())
        return
    if '--check-id' not in sys.argv and not any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
        raise ValueError('An explicit --check-id is required before any verification updates')
    looped_experiment.contract()
    check_training.Trainer = looped_experiment.fresh
    check_training.execute = execute
    check_training.main()
    identifier = sys.argv[sys.argv.index('--check-id') + 1]
    output = ROOT / 'output' / identifier
    evidence = json.loads((output / 'recovery.json').read_text())
    continuous, interrupted, resumed = [ROOT / name for name in evidence['runs']]
    initial = load_file(str(check_training.reference(continuous, 'initial') / 'model.safetensors'))
    final = load_file(str(check_training.reference(continuous, 'latest') / 'model.safetensors'))
    looped = 'loop_embedding' in initial
    assert not looped or not torch.equal(initial['loop_embedding'], final['loop_embedding'])
    changed = sum(0 if torch.equal(initial[name], final[name]) else 1 for name in initial)
    bitwise = []
    for left, right in [(check_training.reference(continuous, 'boundary'), check_training.reference(interrupted, 'latest')),
                        (check_training.reference(continuous, 'latest'), check_training.reference(resumed, 'latest'))]:
        a, b = [json.loads((path / 'metadata.json').read_text()) for path in [left, right]]
        bitwise.append(a['weights_sha256'] == b['weights_sha256'])
    evidence = dict(**evidence, variant=looped_experiment.VARIANT,
                    seed=looped_experiment.SEED, parameters=len(initial),
                    updated_tensors=changed, bitwise_weights_equal=all(bitwise))
    if looped:
        evidence['loop_embedding_updated'] = True
    atomic_json(output / 'recovery.json', evidence)


if __name__ == '__main__':
    main()
