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
from src.tinctura_loop import Reference, Tinctura, load_weights, weights
from src.training import atomic_json, weight_hash, sha256


def main():
    parser = argparse.ArgumentParser(description='E7 original one-pass equivalence and immutable-weight backward benchmark')
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('HIP GPU required')
    corpus = ROOT / 'data/posttrain/fineweb-e7-v2/train.bin'
    tokens = np.fromfile(corpus, dtype='<u2', count=2049).astype(np.int64)
    x = torch.tensor(tokens[:-1], device='cuda')[None]
    y = torch.tensor(tokens[1:], device='cuda')[None]
    results = {}
    for revision, seed in [('early', 42), ('middle', 43), ('final', 44)]:
        state, info = weights(revision)
        reference = Reference().cuda().eval()
        load_weights(reference, state)
        native = Tinctura(2, seed).cuda().eval()
        load_weights(native, state)
        comparisons = []
        with torch.inference_mode():
            for length in [32, 256, 2048]:
                original = reference(x[:, :length], return_dict=False)[0]
                equivalent = native(x[:, :length], equivalent_one_pass=True)[0]
                torch.testing.assert_close(equivalent, original, rtol=2e-5, atol=2e-4)
                error = (equivalent - original).abs().max().item()
                original_nll = F.cross_entropy(original.float().flatten(0, 1), y[:, :length].flatten()).item()
                with torch.autocast('cuda', dtype=torch.float16):
                    mixed = native(x[:, :length], equivalent_one_pass=True)[0]
                mixed_nll = F.cross_entropy(mixed.float().flatten(0, 1), y[:, :length].flatten()).item()
                if abs(mixed_nll - original_nll) > 0.02:
                    raise AssertionError('FP16/FP32 one-pass NLL drift exceeds 0.02 nats')
                comparisons.append(dict(context=length, fp32_max_abs_error=error,
                                        original_fp32_nll=original_nll, native_fp16_nll=mixed_nll,
                                        fp16_nll_delta=mixed_nll - original_nll))
        del reference, original, equivalent, mixed
        torch.cuda.empty_cache()
        benchmarks = {}
        for loops in [1, 2]:
            native.loops = loops
            native.train()
            native.zero_grad(set_to_none=True)
            before = weight_hash(native)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                logits = native(x)[-1]
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            loss.backward()
            if not torch.isfinite(loss).item() or not all(torch.isfinite(p.grad).all().item() for p in native.parameters() if p.grad is not None):
                raise RuntimeError('Nonfinite E7 forward/backward at scale1')
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if weight_hash(native) != before:
                raise AssertionError('Benchmark changed weights')
            benchmarks[str(loops)] = dict(seconds=elapsed, tokens_s=2048 / elapsed,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, loss=loss.item(),
                    nonzero_gradient_tensors=sum(bool(p.grad is not None and p.grad.count_nonzero().item()) for p in native.parameters()),
                    weights_unchanged=True, optimizer_updates=0)
            del logits, loss
            native.zero_grad(set_to_none=True)
        results[revision] = dict(seed=seed, pretraining_tokens=info['tokens'], comparisons=comparisons,
                                 benchmarks=benchmarks)
        logging_row = dict(revision=revision, status='passed', comparisons=comparisons, benchmarks=benchmarks)
        print(json.dumps(logging_row), flush=True)
        del native, state
        torch.cuda.empty_cache()
    atomic_json(output / 'model-check.json', dict(status='passed', revisions=results,
                fp32_tolerance=dict(rtol=2e-5, atol=2e-4), fp16_nll_drift_limit=0.02,
                initial_scale=1.0, gradient_checkpointing=True, optimizer_updates=0,
                source_sha256=sha256(ROOT / 'src/tinctura_loop.py'),
                upstream_sha256=sha256(ROOT / 'data/pretrained/tinctura-v1/final-v1/modeling_cagliostro.py')))


if __name__ == '__main__':
    main()
