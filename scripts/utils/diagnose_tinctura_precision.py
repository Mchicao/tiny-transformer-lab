import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from scripts import tinctura_experiment as e7
from src.training import atomic_json, weight_hash


def replay(trainer, offset, amp, gradients, verify_backward):
    tokens = np.memmap(e7.CORPUS / 'train.bin', dtype='<u2', mode='r')
    batch = torch.tensor(np.array(tokens[offset:offset + 2049], dtype=np.int64), device='cuda')[None]
    records, handles = [], []
    first_invalid = []

    def hook(name):
        def observe(module, inputs, output):
            values = output if isinstance(output, tuple) else (output,)
            bad = any(not torch.isfinite(v).all().item() for v in values if isinstance(v, torch.Tensor))
            if bad and not first_invalid:
                first_invalid.append(name)
            if name.endswith('v_proj'):
                norms = output.reshape(1, 2048, 5, 64)
                maximum = norms.float().square().sum(-1).max().item()
                records.append(dict(module=name, dtype=str(output.dtype),
                                    max_v_norm_squared_fp32=maximum if np.isfinite(maximum) else None,
                                    multiply_dtype=str((norms * norms).dtype),
                                    v_norm_squared_nonfinite_in_current_dtype=(~torch.isfinite((norms * norms).sum(-1))).sum().item()))
        return observe

    for name, module in trainer.model.named_modules():
        if name.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn')):
            handles.append(module.register_forward_hook(hook(name)))
    before = weight_hash(trainer.model)
    trainer.model.zero_grad(set_to_none=True)
    trainer.model.train(gradients)
    try:
        with torch.set_grad_enabled(gradients), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = trainer.model(batch[:, :-1])[-1]
        finite = torch.isfinite(logits).all().item()
        loss_tensor = F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten())
        loss = loss_tensor.item()
        gradients_finite = None
        if gradients and verify_backward and finite:
            loss_tensor.backward()
            gradients_finite = all(torch.isfinite(p.grad).all().item() for p in trainer.model.parameters() if p.grad is not None)
        result = dict(offset=offset, precision='fp16_autocast' if amp else 'fp32', gradients_enabled=gradients, finite_logits=finite,
                      loss=loss if np.isfinite(loss) else None, first_nonfinite_module=first_invalid,
                      v_norms=records, backward_gradients_finite=gradients_finite)
    finally:
        for handle in handles:
            handle.remove()
    assert weight_hash(trainer.model) == before
    return result


def main():
    parser = argparse.ArgumentParser(description='Read-only replay of failed E7 batch at unchanged checkpoint weights')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-id', required=True)
    parser.add_argument('--expect-finite', action='store_true')
    parser.add_argument('--fp32-xsa', action='store_true')
    parser.add_argument('--verify-backward', action='store_true')
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    trainer, metadata = e7.restore(args.checkpoint.resolve(), 'early', 'dense')
    for label in [f"checkpoint{metadata['counters']['iterations']}", 'pretrained_early']:
        if label == 'pretrained_early':
            del trainer
            torch.cuda.empty_cache()
            trainer = e7.E7Trainer('early', 'dense')
        if args.fp32_xsa:
            from src.tinctura_xsa_fp32 import stabilize_xsa
            stabilize_xsa(trainer.model)
        results[label] = [replay(trainer, offset, amp, gradients, args.verify_backward) for offset in [1114112, 1116160]
                          for amp in [True, False] for gradients in [False, True]]
    atomic_json(output / 'diagnosis.json', dict(results=results, training_updates=0, weights_unchanged=True,
                xsa_reductions_fp32=args.fp32_xsa))
    summary = {k: [{key: r[key] for key in ['offset', 'precision', 'gradients_enabled', 'finite_logits', 'loss', 'first_nonfinite_module']} for r in v]
               for k, v in results.items()}
    print(json.dumps(summary), flush=True)
    if args.expect_finite:
        assert all(r['finite_logits'] and r['backward_gradients_finite'] is not False for cases in results.values() for r in cases), 'Failed E7 batch reproduces nonfinite FP16 logits/gradients'


if __name__ == '__main__':
    main()
