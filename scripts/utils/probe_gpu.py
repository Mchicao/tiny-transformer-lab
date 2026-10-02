import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model import ModelConfig, Transformer


def main():
    result = {'torch': torch.__version__, 'hip': torch.version.hip, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available(), 'features': {}}
    if not result['available']:
        raise RuntimeError('GPU unavailable; refusing CPU fallback for GPU probe')
    props = torch.cuda.get_device_properties(0)
    result.update(gpu=props.name, arch=getattr(props, 'gcnArchName', None), total_vram_bytes=props.total_memory)
    torch.manual_seed(42)
    for mode in ['matmul_fp16', 'manual', 'sdpa']:
        try:
            if mode == 'matmul_fp16':
                x = torch.randn(128, 128, device='cuda', dtype=torch.float16, requires_grad=True)
                loss = (x @ x.T).float().square().mean()
                loss.backward()
                finite = torch.isfinite(x.grad).all().item()
            else:
                model = Transformer(ModelConfig(context=32, attention=mode)).cuda()
                tokens = torch.randint(0, 4096, (2, 32), device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(tokens)
                    loss = F.cross_entropy(logits.float().flatten(0, 1), tokens.roll(-1, 1).flatten())
                loss.backward()
                finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
            torch.cuda.synchronize()
            if not finite:
                raise RuntimeError('Non-finite gradients')
            result['features'][mode] = {'status': 'passed', 'loss': loss.item()}
        except Exception as exc:
            result['features'][mode] = {'status': 'failed', 'error': str(exc)}
    result['features'].update({name: {'status': 'not_tested', 'enabled': False} for name in ['torch_compile', 'triton', 'flash_attention', 'hipblaslt', 'rocwmma', 'fp8']})
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    (ROOT / 'output/local-gpu-probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if any(v['status'] == 'failed' for v in result['features'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
