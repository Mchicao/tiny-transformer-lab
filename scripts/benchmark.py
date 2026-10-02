import argparse
from contextlib import nullcontext
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ModelConfig, Transformer
from src.data import PackedTokens


def main():
    parser = argparse.ArgumentParser(description='Forward/backward capability benchmark; never updates weights')
    parser.add_argument('--config', default='configs/3m.json')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        raise ValueError('Preparation benchmark is limited to 1-20 steps')
    raw = json.loads((ROOT / args.config).read_text())
    config = ModelConfig(**{field.name: raw[field.name] for field in fields(ModelConfig) if field.name in raw})
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    if manifest['vocab_size'] != config.vocab_size:
        raise ValueError('Config vocabulary differs from tokenizer; prepare matching artifacts first')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU unavailable; no silent CPU fallback')
    torch.manual_seed(raw['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Transformer(config).to(args.device)
    initial_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    data = PackedTokens(ROOT / 'data/processed/train.bin', config.context)
    batch, accumulation = raw['microbatch'], raw['gradient_accumulation']
    autocast = (lambda: torch.autocast('cuda', dtype=torch.float16)) if args.device == 'cuda' and raw['precision'] == 'fp16' else nullcontext

    def step():
        model.zero_grad(set_to_none=True)
        value = 0.0
        for _ in range(accumulation):
            x, y = data.batch(batch, args.device)
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            (loss / accumulation).backward()
            value += loss.detach().item() / accumulation
        if not all(torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise RuntimeError('Non-finite gradients')
        return value

    step()
    data.position = 0
    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = [step() for _ in range(args.steps)]
    if args.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    final_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    assert initial_hash == final_hash, 'Benchmark changed model weights'
    props = torch.cuda.get_device_properties(0) if args.device == 'cuda' else None
    result = dict(run_id=args.run_id, mode='forward_backward_no_optimizer', optimizer_steps=0, gpu=props.name if props else 'CPU', backend='HIP' if torch.version.hip else ('CUDA' if props else 'CPU'), torch=torch.__version__, runtime=torch.version.hip or torch.version.cuda, parameters=sum(p.numel() for p in model.parameters()), config=raw, precision='fp16_autocast_fp32_weights' if props else 'fp32', batch_size=batch * accumulation, microbatch=batch, gradient_accumulation=accumulation, tokens=args.steps * batch * accumulation * config.context, tokens_s=args.steps * batch * accumulation * config.context / elapsed, step_time_s=elapsed / args.steps, total_time_s=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated() if props else None, peak_reserved_bytes=torch.cuda.max_memory_reserved() if props else None, total_vram_bytes=props.total_memory if props else None, losses=losses, weights_sha256=initial_hash, tokenizer_sha256=manifest['tokenizer_sha256'], data_sha256=manifest['splits']['train']['shard_sha256'])
    target = ROOT / 'runs' / args.run_id
    target.mkdir(parents=True, exist_ok=False)
    (target / 'metrics.json').write_text(json.dumps(result, indent=2))
    with (ROOT / 'runs/results.jsonl').open('a') as handle:
        handle.write(json.dumps(result) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
