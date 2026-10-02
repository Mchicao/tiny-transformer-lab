import ast
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file

from src.training import sha256

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'data/pretrained/tinctura-v1/final-v1'
UPSTREAM_SHA256 = 'e444974f5a6dfdee854af70d9346768897ed38b2e887efbb70ab914abd8516cc'


def upstream_namespace(amp_safe=False):
    source = FINAL / 'modeling_cagliostro.py'
    if sha256(source) != UPSTREAM_SHA256:
        raise ValueError('Upstream architecture source changed')
    tree = ast.parse(source.read_text())
    names = {'RMSNorm', 'rope_freqs', 'rotate_half', 'apply_rope', 'Attention', 'SwiGLU',
             'ReLU2MLP', 'Canon', 'DecoderLayer'}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Incomplete original architecture')
    namespace = dict(torch=torch, nn=nn, F=F, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    if amp_safe:
        original = namespace['RMSNorm']

        class AMPRMSNorm(original):
            def forward(self, x):
                normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
                return normalized.to(x.dtype) * self.weight.to(x.dtype)

        namespace['RMSNorm'] = AMPRMSNorm
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CagliostroForCausalLM')
    methods = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name in {'_rope', '_causal_mask', 'forward'}]
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def reference_class(amp_safe=False):
    namespace = upstream_namespace(amp_safe)
    config = json.loads((FINAL / 'config.json').read_text())
    if (config['hidden_size'], config['num_hidden_layers'], config['num_attention_heads'],
            config['num_key_value_heads'], config['intermediate_size'], config['vocab_size'],
            config['mlp_type'], config['xsa'], config['canon'], config['value_residual'],
            config['layer_repeat'], config['tie_word_embeddings'], config['logit_cap']) != (
            640, 18, 10, 5, 1536, 32768, 'swiglu', True, False, False, 1, True, 15.0):
        raise ValueError('Unsupported tinctura architecture')

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32768, 640)
            self.layers = nn.ModuleList([namespace['DecoderLayer'](640, 10, 5, 64, 1536, 1e-6,
                                        True, 'swiglu', False, False) for _ in range(18)])
            self.norm_out = namespace['RMSNorm'](640, eps=1e-6)
            self.lm_head = nn.Linear(640, 32768, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
            self.head_dim, self.layer_repeat, self.logit_cap = 64, 1, 15.0
            self.rope_theta, self._rope_cache = 100000.0, None

    for method in ['_rope', '_causal_mask', 'forward']:
        setattr(Reference, method, namespace[method])
    return Reference


Reference = reference_class()
AMPReference = reference_class(amp_safe=True)


class Tinctura(AMPReference):
    def __init__(self, loops, seed, checkpoint_layers=True):
        super().__init__()
        if loops not in (1, 2) or seed not in (42, 43, 44):
            raise ValueError('E7 supports K1/K2 and seeds42/43/44 only')
        self.loops = loops
        self.checkpoint_layers = checkpoint_layers
        if loops == 2:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(7,))))
            values = rng.uniform(-math.sqrt(12) * 0.01, math.sqrt(12) * 0.01, (2, 640)).astype(np.float32)
            self.loop_embedding = nn.Parameter(torch.from_numpy(values))

    def _rope(self, seq_len, device, dtype):
        with torch.inference_mode(False), torch.no_grad():
            return super()._rope(seq_len, device, dtype)

    def forward(self, input_ids, equivalent_one_pass=False):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= 2048:
            raise ValueError('Expected token batches within context2048')
        embedded = self.embed_tokens(input_ids)
        rope_dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else embedded.dtype
        cos, sin = self._rope(input_ids.shape[1], embedded.device, rope_dtype)
        hidden = torch.zeros_like(embedded)
        outputs = []
        count = 1 if equivalent_one_pass else self.loops
        for loop in range(count):
            hidden = hidden + embedded
            if self.loops == 2 and not equivalent_one_pass:
                hidden = hidden + self.loop_embedding[loop]
            for layer in self.layers:
                if self.training and self.checkpoint_layers and torch.is_grad_enabled():
                    hidden, _ = checkpoint(layer, hidden, cos, sin, use_reentrant=False, preserve_rng_state=False)
                else:
                    hidden, _ = layer(hidden, cos, sin)
            logits = self.lm_head(self.norm_out(hidden))
            outputs.append(self.logit_cap * torch.tanh(logits / self.logit_cap))
        return outputs


def weights(revision):
    if revision not in {'early', 'middle', 'final'}:
        raise ValueError('Unknown maturity revision')
    directory = ROOT / 'data/pretrained/tinctura-v1' / (revision + '-v1')
    manifest = json.loads((directory / 'manifest.json').read_text())
    filename = 'model.safetensors' if revision == 'final' else 'latest.pt'
    path = directory / filename
    if sha256(path) != manifest['files'][filename]['sha256']:
        raise ValueError('Pretrained checkpoint checksum mismatch')
    if revision == 'final':
        state = load_file(str(path))
        state['lm_head.weight'] = state['embed_tokens.weight']
        metadata = dict(tokens=75000000000, checkpoint_step=762939)
    else:
        saved = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        if (saved['xsa'], saved['logit_cap'], saved['seq_len'], saved['vocab_size'], saved['config']) != (
                True, 15.0, 2048, 32768, 'sub100-A'):
            raise ValueError('Pretrained checkpoint architecture mismatch')
        state = saved['model']
        metadata = dict(tokens=saved['tokens'], checkpoint_step=saved['step'])
    if not torch.equal(state['embed_tokens.weight'], state['lm_head.weight']):
        raise ValueError('Pretrained embeddings are not tied')
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError('Nonfinite or incompatible pretrained tensors')
    return state, dict(**manifest, **metadata)


def load_weights(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = ['loop_embedding'] if isinstance(model, Tinctura) and model.loops == 2 else []
    if missing != expected or unexpected:
        raise ValueError(f'Pretrained state mismatch: missing={missing}, unexpected={unexpected}')
    expected_params = 96200064 + (1280 if expected else 0)
    if sum(p.numel() for p in model.parameters()) != expected_params:
        raise ValueError('Pretrained parameter count mismatch')
