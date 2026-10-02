from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.model import ModelConfig


@dataclass
class SwiGLUConfig(ModelConfig):
    ffn: str = 'swiglu'
    ffn_hidden: int = 512

    def __post_init__(self):
        super().__post_init__()
        if self.ffn != 'swiglu' or self.ffn_hidden * 3 != self.dim * 8:
            raise ValueError('SwiGLU requires the same FFN parameter budget as GELU')


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def install(model, seed=42):
    config = SwiGLUConfig(**vars(model.config))
    generator = np.random.Generator(np.random.PCG64(seed))
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        for block in model.blocks:
            block.ffn = SwiGLU(config.dim, config.ffn_hidden).to(model.embedding.weight.device)
            for name, parameter in block.ffn.named_parameters():
                scale = 0.02 / math.sqrt(2 * config.layers) if name == 'down.weight' else 0.02
                integers = generator.integers(0, 2**32, size=tuple(parameter.shape), dtype=np.uint32)
                values = ((integers >> 8).astype(np.float32) * np.float32(2**-24) - np.float32(0.5)) * np.float32(scale * math.sqrt(12))
                parameter.copy_(torch.from_numpy(values))
    model.config = config
    return config
