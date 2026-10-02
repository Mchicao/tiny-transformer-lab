import argparse
import copy
import json
import logging
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.train import extend_budget, restore_trainer
from scripts.utils.check_training import compare
from src.data import PackedTokens
from src.training import EXPERIMENT, Trainer, atomic_json, rng_state, sha256, weight_hash


def main():
    parser = argparse.ArgumentParser(description='Verifica restauración y ampliación de presupuesto sin optimizer steps')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--extend-budget', action='store_true')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    trainer, metadata = restore_trainer(checkpoint)
    state = torch.load(checkpoint / 'state.pt', weights_only=True, map_location='cpu')
    compare(trainer.optimizer.state_dict(), state['optimizer'], exact=True)
    compare(trainer.scaler.state_dict(), state['scaler'], exact=True)
    compare(rng_state(), state['rng'], exact=True)
    assert trainer.counters == metadata['counters']
    assert trainer.data.position == metadata['cursor']
    for group in trainer.optimizer.param_groups:
        assert group['lr'] == 3e-4 and group['betas'] == (0.9, 0.95)
        for parameter in group['params']:
            assert parameter.dtype == torch.float32
            assert group['weight_decay'] == (0.1 if parameter.ndim == 2 else 0.0)
            if trainer.counters['effective']:
                assert trainer.optimizer.state[parameter]['step'].item() == trainer.counters['effective']
            else:
                assert not trainer.optimizer.state
    before = dict(weights=weight_hash(trainer.model), rng=rng_state(), cursor=trainer.data.position,
                  counters=copy.deepcopy(trainer.counters), optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
                  scaler=copy.deepcopy(trainer.scaler.state_dict()))
    data = PackedTokens(ROOT / 'data/processed/train.bin', 256)
    data.position = trainer.data.position
    x, y = data.batch(4, 'cpu')
    actual_x, actual_y = trainer.data.batch(4, 'cpu')
    compare(x, actual_x, exact=True)
    compare(y, actual_y, exact=True)
    trainer.data.position = before['cursor']
    validation = trainer.evaluate()
    assert validation['nll'] == metadata['evaluations'][-1]['nll']
    generations = trainer.generate()
    compare(before, dict(weights=weight_hash(trainer.model), rng=rng_state(), cursor=trainer.data.position,
                         counters=trainer.counters, optimizer=trainer.optimizer.state_dict(),
                         scaler=trainer.scaler.state_dict()), exact=True)
    output = ROOT / 'output' / ('continuation-check-' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    if args.extend_budget:
        extend_budget(trainer)
        assert EXPERIMENT['iterations'] - trainer.counters['iterations'] == 245
        assert EXPERIMENT['iterations'] * 4096 == 2007040
        assert trainer.counters['effective'] >= EXPERIMENT['warmup']
        saved = trainer.save(output, ['latest', 'best', 'initial'])
        compare(trainer.optimizer.state_dict(), before['optimizer'], exact=True)
        compare(rng_state(), before['rng'], exact=True)
        child = subprocess.run([sys.executable, str(Path(__file__)), '--checkpoint', str(saved)],
                               capture_output=True, text=True, check=True)
        (output / 'fresh-process.log').write_text(child.stdout + child.stderr, encoding='utf-8')
    atomic_json(output / 'recovery.json', dict(status='passed', checkpoint=str(checkpoint),
                checksums_sha256=sha256(checkpoint / 'checksums.json'), weights_sha256=before['weights'],
                counters=trainer.counters, cursor=trainer.data.position, experiment=EXPERIMENT,
                validation=validation, generations=generations, training_updates=0,
                optimizer_scaler_rng_exact=True, next_batch_exact=True,
                extension_roundtrip_fresh_process=args.extend_budget))
    logging.info(json.dumps(dict(status='passed', output=str(output), nll=validation['nll'], training_updates=0)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
