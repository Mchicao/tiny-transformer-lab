from contextlib import contextmanager
from dataclasses import asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from src.data import PackedTokens
from src.model import ModelConfig, Transformer

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ['Once upon a time', 'The little girl found', 'One day, a dog']
EXPERIMENT = dict(seed=42, microbatch=4, accumulation=4, lr=3e-4,
                  betas=[0.9, 0.95], weight_decay=0.1, warmup=10,
                  clip=1.0, init_scale=1024.0, iterations=245,
                  evaluation_interval=50, milestones=[50, 100, 150, 200, 245])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_hash(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if json.loads(path.read_text(encoding='utf-8')) != value:
        raise RuntimeError(f'Atomic JSON readback failed: {path}')


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), *state[2:]),
                torch=torch.get_rng_state(), gpu=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['gpu'])


@contextmanager
def isolated_eval(model):
    state, mode = rng_state(), model.training
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        model.train(mode)
        restore_rng(state)


def verified_identity():
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    paths = ['configs/3m.json', 'data/processed/manifest.json',
             'data/processed/tokenizer.json', 'output/initial-3m.safetensors',
             'src/model.py', 'src/data.py', 'src/training.py']
    for split, info in manifest['splits'].items():
        shard = ROOT / f'data/processed/{split}.bin'
        raw = ROOT / f'data/raw/{split}.jsonl'
        if sha256(shard) != info['shard_sha256'] or sha256(raw) != info['raw_sha256']:
            raise ValueError(f'Dataset identity mismatch: {split}')
        tokens = np.memmap(shard, dtype='<u2', mode='r')
        if len(tokens) != info['tokens'] or tokens.max() >= 4096:
            raise ValueError(f'Invalid shard: {split}')
        paths.extend([str(shard.relative_to(ROOT)).replace('\\', '/'),
                      str(raw.relative_to(ROOT)).replace('\\', '/')])
    if manifest['vocab_size'] != 4096 or sha256(ROOT / paths[2]) != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer identity mismatch')
    return {name: sha256(ROOT / name) for name in paths}


def verify_checkpoint(directory, identity, backend):
    manifest = json.loads((directory / 'checksums.json').read_text())
    if set(manifest) != {'model.safetensors', 'state.pt', 'metadata.json'}:
        raise ValueError('Incomplete checkpoint manifest')
    for name, expected in manifest.items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata['identity'] != identity or metadata['backend'] != backend or metadata['experiment'] != EXPERIMENT:
        raise ValueError('Checkpoint identity/backend/experiment mismatch')
    return metadata


class Trainer:
    def __init__(self):
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError('HIP GPU unavailable; CPU fallback forbidden')
        props = torch.cuda.get_device_properties(0)
        if props.name != 'AMD Radeon RX 6750 GRE 10GB' or props.gcnArchName.split(':')[0] != 'gfx1031':
            raise RuntimeError(f'Unexpected GPU: {props.name} / {props.gcnArchName}')
        self.backend = dict(gpu=props.name, arch=props.gcnArchName, vram=props.total_memory,
                            torch=torch.__version__, hip=torch.version.hip)
        self.identity = verified_identity()
        raw = json.loads((ROOT / 'configs/3m.json').read_text())
        self.config = ModelConfig(**{f.name: raw[f.name] for f in fields(ModelConfig) if f.name in raw})
        if raw != {**asdict(self.config), 'seed': 42, 'precision': 'fp16',
                   'microbatch': 4, 'gradient_accumulation': 4, 'optimizer': 'adamw'} or self.config != ModelConfig():
            raise ValueError('Incompatible baseline config')
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = Transformer(self.config).cuda()
        if sum(p.numel() for p in self.model.parameters()) != 3000384:
            raise ValueError('Unexpected parameter count')
        initial = load_file(str(ROOT / 'output/initial-3m.safetensors'))
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), initial[name], rtol=0, atol=0)
        self.model.load_state_dict(initial, strict=True)
        self.initial_hash = weight_hash(self.model)
        if self.initial_hash != '194d4675760675037cfaef56e78f3e8aad74ca07d244068e782ac60d8ac1ef64':
            raise ValueError('Initial weights identity mismatch')
        groups = [dict(params=[p for p in self.model.parameters() if p.ndim == 2], weight_decay=0.1),
                  dict(params=[p for p in self.model.parameters() if p.ndim != 2], weight_decay=0.0)]
        self.optimizer = torch.optim.AdamW(groups, lr=EXPERIMENT['lr'], betas=tuple(EXPERIMENT['betas']), foreach=False)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=EXPERIMENT['init_scale'])
        self.data = PackedTokens(ROOT / 'data/processed/train.bin', 256)
        self.counters = dict(iterations=0, effective=0, skipped=0, tokens=0, consecutive_skips=0)
        self.train_seconds = 0.0
        self.best_loss = None
        self.history = []
        self.evaluations = []

    def step(self):
        if self.counters['iterations'] >= EXPERIMENT['iterations']:
            raise RuntimeError('Authorized budget exhausted')
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr = EXPERIMENT['lr'] * min((self.counters['effective'] + 1) / EXPERIMENT['warmup'], 1.0)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        losses, batches = [], hashlib.sha256()
        for _ in range(4):
            x, y = self.data.batch(4, 'cuda')
            batches.update(x.cpu().numpy().tobytes())
            batches.update(y.cpu().numpy().tobytes())
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(self.model(x).float().flatten(0, 1), y.flatten())
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite training loss; stopping')
            self.scaler.scale(loss / 4).backward()
            losses.append(loss.detach().item())
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale
        self.counters['iterations'] += 1
        self.counters['tokens'] += 4096
        self.counters['skipped' if skipped else 'effective'] += 1
        self.counters['consecutive_skips'] = self.counters['consecutive_skips'] + 1 if skipped else 0
        if self.counters['consecutive_skips'] >= 3:
            raise RuntimeError('Persistent instability: three consecutive skipped updates')
        if not all(torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise RuntimeError('Nonfinite weights; stopping')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.train_seconds += elapsed
        row = dict(**self.counters, loss=sum(losses) / 4, lr=lr,
                   grad_norm=norm if math.isfinite(norm) else None,
                   scale=self.scaler.get_scale(), batch_sha256=batches.hexdigest(),
                   cursor=self.data.position, seconds=elapsed)
        self.history.append(row)
        return row

    def evaluate(self):
        start = time.perf_counter()
        tokens = np.memmap(ROOT / 'data/processed/validation.bin', dtype='<u2', mode='r')
        total, count = 0.0, 0
        with isolated_eval(self.model):
            for position in range(0, len(tokens) - 1, 256):
                chunk = torch.tensor(np.array(tokens[position:position + 257], dtype=np.int64), device='cuda')[None]
                with torch.autocast('cuda', dtype=torch.float16):
                    nll = F.cross_entropy(self.model(chunk[:, :-1]).float().flatten(0, 1), chunk[:, 1:].flatten(), reduction='sum')
                total += nll.item()
                count += chunk.shape[1] - 1
        if count != len(tokens) - 1 or not math.isfinite(total):
            raise RuntimeError('Validation NLL/token count invalid')
        row = dict(iteration=self.counters['iterations'], tokens=self.counters['tokens'],
                   nll=total / count, evaluated_tokens=count, seconds=time.perf_counter() - start)
        self.evaluations.append(row)
        return row

    def generate(self):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(ROOT / 'data/processed/tokenizer.json'))
        generations = []
        with isolated_eval(self.model):
            for prompt in PROMPTS:
                ids = tokenizer.encode(prompt).ids
                for _ in range(64):
                    x = torch.tensor([ids[-256:]], device='cuda')
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = self.model(x)[0, -1].float()
                    next_id = logits.argmax().item()
                    ids.append(next_id)
                    if next_id == tokenizer.token_to_id('<eos>'):
                        break
                generations.append(dict(prompt=prompt, text=tokenizer.decode(ids), ids=ids))
        return dict(method='greedy', max_new_tokens=64, prompts=generations)

    def save(self, run, references):
        checkpoints = run / 'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        name = f"step-{self.counters['iterations']:04d}-{uuid.uuid4().hex[:8]}"
        staging = checkpoints / ('.pending-' + name)
        staging.mkdir()
        save_file({name: value.cpu().contiguous() for name, value in self.model.state_dict().items()}, str(staging / 'model.safetensors'))
        torch.save(dict(optimizer=self.optimizer.state_dict(), scaler=self.scaler.state_dict(), rng=rng_state()), staging / 'state.pt')
        for filename in ['model.safetensors', 'state.pt']:
            with (staging / filename).open('rb+') as handle:
                os.fsync(handle.fileno())
        metadata = dict(identity=self.identity, backend=self.backend, config=asdict(self.config), experiment=EXPERIMENT,
                        counters=self.counters, cursor=self.data.position, train_seconds=self.train_seconds,
                        best_loss=self.best_loss, history=self.history, evaluations=self.evaluations,
                        weights_sha256=weight_hash(self.model), initial_weights_sha256=self.initial_hash)
        atomic_json(staging / 'metadata.json', metadata)
        atomic_json(staging / 'checksums.json', {f: sha256(staging / f) for f in ['model.safetensors', 'state.pt', 'metadata.json']})
        verify_checkpoint(staging, self.identity, self.backend)
        destination = checkpoints / name
        os.replace(staging, destination)
        verify_checkpoint(destination, self.identity, self.backend)
        ref_path = run / 'references.json'
        refs = json.loads(ref_path.read_text()) if ref_path.exists() else {}
        refs.update({ref: name for ref in references})
        atomic_json(ref_path, refs)
        return destination

    def restore(self, directory):
        metadata = verify_checkpoint(directory, self.identity, self.backend)
        if metadata['config'] != asdict(self.config) or metadata['initial_weights_sha256'] != self.initial_hash:
            raise ValueError('Incompatible model identity')
        self.model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        if weight_hash(self.model) != metadata['weights_sha256']:
            raise ValueError('Restored weights hash mismatch')
        state = torch.load(directory / 'state.pt', map_location='cpu', weights_only=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        self.counters = metadata['counters']
        self.data.position = metadata['cursor']
        self.train_seconds = metadata['train_seconds']
        self.best_loss = metadata['best_loss']
        self.history = metadata['history']
        self.evaluations = metadata['evaluations']
        restore_rng(state['rng'])
