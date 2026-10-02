from types import MethodType

import torch
from torch.nn import functional as F

from src.tinctura_loop import Tinctura


def attention_fp32_xsa(self, x, cos, sin, attn_mask=None, v_first=None):
    if self.value_residual:
        raise ValueError('E7 tinctura does not enable value residual')
    batch, length, _ = x.shape
    q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim)
    k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
    v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
    q, k = self.q_norm(q), self.k_norm(k)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q = q * cos + torch.cat([-q2, q1], dim=-1) * sin
    k = k * cos + torch.cat([-k2, k1], dim=-1) * sin
    raw_v = v
    q = q.transpose(1, 2)
    k = k.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
    v = v.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True) if attn_mask is None else F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    if self.xsa:
        values = v.float()
        attended = out.float()
        numerator = (attended * values).sum(dim=-1, keepdim=True)
        denominator = (values * values).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out = attended - (numerator / denominator) * values
    out = out.transpose(1, 2).contiguous().view(batch, length, -1)
    return self.o_proj(out), raw_v


def stabilize_xsa(model):
    for layer in model.layers:
        layer.attn.forward = MethodType(attention_fp32_xsa, layer.attn)


class FP32XSATinctura(Tinctura):
    def __init__(self, loops, seed, checkpoint_layers=True):
        super().__init__(loops, seed, checkpoint_layers)
        stabilize_xsa(self)
