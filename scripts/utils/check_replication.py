import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts import replicate
from scripts.utils import check_experiments as checks
from scripts.utils.check_training import compare
from src.training import atomic_json, rng_state, weight_hash


def execute(identifier, variant, target, checkpoint=None):
    command = [sys.executable, str(ROOT / 'scripts/replicate.py'), '--variant', variant,
               '--run-id', identifier, '--iterations', '1500', '--stop-after', str(target)]
    if checkpoint:
        command += ['--resume', str(checkpoint)]
    log = ROOT / 'logs/tests' / (identifier + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / identifier


def continuation(output, checkpoint=None):
    if checkpoint:
        trainer, metadata = replicate.restore(checkpoint)
        expected = torch.load(checkpoint / 'state.pt', weights_only=True, map_location='cpu')
        compare(trainer.optimizer.state_dict(), expected['optimizer'], exact=True)
        compare(trainer.scaler.state_dict(), expected['scaler'], exact=True)
        compare(rng_state(), expected['rng'], exact=True)
        assert trainer.data.position == metadata['cursor'] and trainer.counters == metadata['counters']
        assert trainer.evaluate()['nll'] == metadata['evaluations'][-1]['nll']
        return
    output.mkdir(parents=True, exist_ok=False)
    source = checks.reference(replicate.LEGACY_RUN, 'latest')
    trainer, metadata = replicate.legacy(source)
    expected = dict(optimizer=copy.deepcopy(trainer.optimizer.state_dict()), scaler=copy.deepcopy(trainer.scaler.state_dict()),
                    rng=rng_state(), weights=weight_hash(trainer.model), cursor=trainer.data.position,
                    counters=copy.deepcopy(trainer.counters))
    assert trainer.evaluate()['nll'] == metadata['evaluations'][-1]['nll']
    replicate.campaign.configure(replicate.contract('adamw'))
    replicate.identity(trainer)
    checkpoint = trainer.save(output, ['latest', 'initial'])
    compare(expected, dict(optimizer=trainer.optimizer.state_dict(), scaler=trainer.scaler.state_dict(), rng=rng_state(),
                          weights=weight_hash(trainer.model), cursor=trainer.data.position, counters=trainer.counters), exact=True)
    with (output / 'fresh-process.log').open('w', encoding='utf-8') as handle:
        subprocess.run([sys.executable, str(Path(__file__)), '--continuation-check', str(output),
                        '--checkpoint', str(checkpoint)], stdout=handle, stderr=subprocess.STDOUT, check=True)
    atomic_json(output / 'recovery.json', dict(status='passed', training_updates=0,
                extension_restored_in_new_process=True, weights_optimizer_scaler_rng_counters_cursor_exact=True,
                validation_exact=True, source=str(source.relative_to(ROOT))))


def main():
    if '--continuation-check' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--continuation-check', type=Path, required=True)
        parser.add_argument('--checkpoint', type=Path)
        args = parser.parse_args()
        continuation(args.continuation_check.resolve(), args.checkpoint)
        return
    checks.restore = replicate.restore
    checks.execute = execute
    checks.main()


if __name__ == '__main__':
    main()
