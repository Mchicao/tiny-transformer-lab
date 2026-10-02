from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 192
    layers: int = 5
    heads: int = 6
    context: int = 256
    attention: str = 'manual'
    qk_norm: bool = False

    def __post_init__(self):
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError('Head dimension must be even and divide dim')
        if self.attention not in {'manual', 'sdpa'}:
            raise ValueError('Unknown attention implementation')


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.proj = nn.Linear(config.dim, config.dim, bias=False)
        head_dim = config.dim // config.heads
        angles = torch.outer(torch.arange(config.context), 10000 ** (-torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)
        self.register_buffer('causal', torch.ones(config.context, config.context, dtype=torch.bool).tril(), persistent=False)

    def rope(self, x):
        even, odd = x[..., ::2], x[..., 1::2]
        cos = self.cos[:x.shape[-2]].to(x.dtype)
        sin = self.sin[:x.shape[-2]].to(x.dtype)
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.config.heads, dim // self.config.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rope(q), self.rope(k)
        if self.config.qk_norm:
            q = F.normalize(q.float(), dim=-1).to(q.dtype) * math.sqrt(q.shape[-1])
            k = F.normalize(k.float(), dim=-1).to(k.dtype) * math.sqrt(k.shape[-1])
        if self.config.attention == 'sdpa':
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
            scores = scores.masked_fill(~self.causal[:length, :length], float('-inf'))
            attended = scores.softmax(-1).to(v.dtype) @ v
        return self.proj(attended.transpose(1, 2).reshape(batch, length, dim))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = nn.Sequential(nn.Linear(config.dim, 4 * config.dim, bias=False), nn.GELU(), nn.Linear(4 * config.dim, config.dim, bias=False))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = RMSNorm(config.dim)
        generator = np.random.Generator(np.random.PCG64(torch.initial_seed()))
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if parameter.ndim == 2:
                    scale = 0.02 / math.sqrt(2 * config.layers) if name.endswith(('attention.proj.weight', 'ffn.2.weight')) else 0.02
                    integers = generator.integers(0, 2**32, size=tuple(parameter.shape), dtype=np.uint32)
                    values = ((integers >> 8).astype(np.float32) * np.float32(2**-24) - np.float32(0.5)) * np.float32(scale * math.sqrt(12))
                    parameter.copy_(torch.from_numpy(values))

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.config.context:
            raise ValueError('Expected batch x sequence within configured context')
        return F.linear(self.norm(self.blocks(self.embedding(tokens))), self.embedding.weight)
