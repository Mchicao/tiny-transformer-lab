from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import ModelConfig, Transformer

torch.set_num_threads(2)
torch.manual_seed(42)
config = ModelConfig(vocab_size=128, dim=32, heads=2, layers=2, context=16)
manual = Transformer(config)
sdpa = Transformer(ModelConfig(**{**config.__dict__, 'attention': 'sdpa'}))
sdpa.load_state_dict(manual.state_dict())
x = torch.randint(0, 128, (2, 16))
changed = x.clone()
changed[:, 8:] = torch.randint(0, 128, (2, 8))
torch.testing.assert_close(manual(x)[:, :8], manual(changed)[:, :8], rtol=0, atol=0)
torch.testing.assert_close(manual(x), sdpa(x), rtol=2e-5, atol=2e-6)
manual(x).square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in manual.parameters())
assert sum(p.numel() for p in Transformer(ModelConfig()).parameters()) == 3000384
print('PASS: causal isolation, manual/SDPA parity, finite gradients, parameter budget')
