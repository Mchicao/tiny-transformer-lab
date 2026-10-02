from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.model import ModelConfig, Transformer


@dataclass
class LoopedConfig(ModelConfig):
    loop_k: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.loop_k not in {2, 4}:
            raise ValueError('Authorized loop depths are 2 and 4')


class LoopedTransformer(nn.Module):
    def __init__(self, base, loops):
        super().__init__()
        if not isinstance(base, Transformer) or loops < 2:
            raise ValueError('Looped models wrap the baseline transformer with at least two loops')
        self.config = base.config
        self.loops = loops
        self.embedding = base.embedding
        self.blocks = base.blocks
        self.norm = base.norm
        self.loop_embedding = nn.Parameter(torch.zeros(loops, base.config.dim, device=base.embedding.weight.device))

    def states(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.config.context:
            raise ValueError('Expected batch x sequence within configured context')
        injected = self.embedding(tokens)
        state = torch.zeros_like(injected)
        states = []
        for index in range(self.loops):
            state = self.blocks(state + injected + self.loop_embedding[index])
            states.append(state)
        return states

    def forward_loops(self, tokens):
        return [F.linear(self.norm(state), self.embedding.weight) for state in self.states(tokens)]

    def forward(self, tokens):
        return self.forward_loops(tokens)[-1]


def loop_generator(seed, loops):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=seed, spawn_key=(loops,))))


def install(model, loops, seed):
    config = LoopedConfig(**vars(model.config), loop_k=loops)
    generator = loop_generator(seed, loops)
    looped = LoopedTransformer(model, loops)
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        integers = generator.integers(0, 2**32, size=tuple(looped.loop_embedding.shape), dtype=np.uint32)
        values = ((integers >> 8).astype(np.float32) * np.float32(2**-24) - np.float32(0.5)) * np.float32(0.02 * math.sqrt(12))
        looped.loop_embedding.copy_(torch.from_numpy(values).to(looped.loop_embedding.device))
    looped.config = config
    return looped
