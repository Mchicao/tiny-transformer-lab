import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from scripts import tinctura_xsa_experiment as numeric
from src.training import EXPERIMENT, atomic_json, sha256, weight_hash


def probe(trainer, offset, dtype):
    source = np.memmap(ROOT / EXPERIMENT['corpus'] / 'train.bin', dtype='<u2', mode='r')
    tokens = torch.tensor(np.array(source[offset:offset + 2049], dtype=np.int64), device='cuda')[None]
    bad = []
    handles = []

    def hook(name):
        def observe(module, inputs, output):
            values = output if isinstance(output, (tuple, list)) else (output,)
            if not bad and any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item() for v in values):
                bad.append(name)
        return observe

    for name, module in trainer.model.named_modules():
        if name:
            handles.append(module.register_forward_hook(hook(name)))
    trainer.model.train()
    trainer.model.zero_grad(set_to_none=True)
    before = weight_hash(trainer.model)
    start = time.perf_counter()
    try:
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            logits = trainer.model(tokens[:, :-1])[-1]
        loss = F.cross_entropy(logits.float().flatten(0, 1), tokens[:, 1:].flatten())
        finite = torch.isfinite(loss).item()
        grad_finite = None
        if finite:
            loss.backward()
            grad_finite = all(torch.isfinite(p.grad).all().item() for p in trainer.model.parameters() if p.grad is not None)
        torch.cuda.synchronize()
        result = dict(dtype=str(dtype), offset=offset, supported=True, loss=loss.item() if finite else None,
                      finite_loss=finite, finite_gradients=grad_finite, first_nonfinite_module=bad,
                      seconds=time.perf_counter() - start, optimizer_updates=0)
    except RuntimeError as error:
        result = dict(dtype=str(dtype), offset=offset, supported=False, error=str(error), optimizer_updates=0)
    finally:
        for handle in handles:
            handle.remove()
    assert weight_hash(trainer.model) == before
    trainer.model.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Capture exact failed-state range and test BF16/FP32 without optimizer')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--revision', choices=['early', 'middle', 'final'], required=True)
    parser.add_argument('--variant', choices=['dense', 'looped'], required=True)
    parser.add_argument('--replay-updates', type=int, default=0)
    parser.add_argument('--offset', type=int, required=True)
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id or not 0 <= args.replay_updates <= 25:
        parser.error('Unique output ID and bounded replay0..25 required')
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    trainer, metadata = numeric.restore(args.checkpoint.resolve(), args.revision, args.variant)
    starting_tokens = trainer.counters['tokens']
    trainer.identity['scripts/utils/diagnose_tinctura_range.py'] = sha256(Path(__file__))
    EXPERIMENT['diagnostic_only'] = True
    for _ in range(args.replay_updates):
        trainer.step()
    state = output / 'diagnostic-state'
    state.mkdir()
    trainer.save(state, ['latest'])
    results = [probe(trainer, offset, dtype) for offset in [args.offset, args.offset + 2048]
               for dtype in [torch.float16, torch.bfloat16, torch.float32]]
    atomic_json(output / 'range-check.json', dict(source_checkpoint=str(args.checkpoint.resolve()),
                source_counters=metadata['counters'], reconstructed_counters=trainer.counters,
                diagnostic_replay_updates=args.replay_updates,
                diagnostic_replay_tokens=trainer.counters['tokens'] - starting_tokens,
                principal_checkpoint_reuse_forbidden=True, results=results))
    print(json.dumps(dict(reconstructed_counters=trainer.counters, results=results)))


if __name__ == '__main__':
    main()
