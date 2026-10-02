from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from src.data import PackedTokens
from src.model import Transformer


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def weights_hash(model: Transformer) -> str:
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def write_json(path: Path, value):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainingConfig:
    seed: int = 42
    microbatch: int = 4
    accumulation: int = 4
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_updates: int = 10
    clip_norm: float = 1.0
    max_tokens: int = 1003520
    precision: str = 'fp16'

    def __post_init__(self):
        if min(self.microbatch, self.accumulation, self.max_tokens) < 1:
            raise ValueError('Batch, accumulation and token budget must be positive')
        if self.learning_rate <= 0 or self.clip_norm <= 0 or self.weight_decay < 0 or self.warmup_updates < 0:
            raise ValueError('Invalid AdamW hyperparameters')
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError('AdamW betas must be in [0, 1)')
        if self.precision not in {'fp16', 'fp32'}:
            raise ValueError('Unknown training precision')


class Trainer:
    def __init__(self, model: Transformer, train_path: Path, validation_path: Path, config: TrainingConfig, device: str, identity: dict):
        if device not in {'cuda', 'cpu'}:
            raise ValueError('Unknown device')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested GPU unavailable; no CPU fallback')
        if config.precision == 'fp16' and device != 'cuda':
            raise ValueError('FP16 training requires GPU')
        self.model = model.to(device)
        self.config, self.device = config, device
        self.train = PackedTokens(train_path, model.config.context)
        self.validation = np.memmap(validation_path, dtype='<u2', mode='r')
        for tokens in [self.train.tokens, self.validation]:
            if len(tokens) < 2 or tokens.max() >= model.config.vocab_size:
                raise ValueError('Invalid token shard for model vocabulary')
        self.identity = {**identity, 'train': file_hash(train_path), 'validation': file_hash(validation_path)}
        matrices = [p for p in self.model.parameters() if p.ndim >= 2]
        norms = [p for p in self.model.parameters() if p.ndim < 2]
        self.optimizer = torch.optim.AdamW([
            {'params': matrices, 'weight_decay': config.weight_decay},
            {'params': norms, 'weight_decay': 0.0},
        ], lr=config.learning_rate, betas=(config.beta1, config.beta2), foreach=False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.precision == 'fp16')
        self.iterations = self.updates = self.tokens = self.skipped = 0
        self.training_s = self.evaluation_s = 0.0
        self.best_loss = None
        self.initial_weights_sha256 = weights_hash(self.model)
        self.contract = {
            'format': 1, 'model': asdict(model.config), 'training': asdict(config),
            'identity': self.identity, 'device': device, 'torch': str(torch.__version__),
            'runtime': torch.version.hip or torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
        }

    @property
    def tokens_per_iteration(self):
        return self.config.microbatch * self.config.accumulation * self.model.config.context

    def synchronize(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

    def autocast(self):
        return torch.autocast(self.device, dtype=torch.float16, enabled=self.config.precision == 'fp16')

    def step(self):
        if self.tokens + self.tokens_per_iteration > self.config.max_tokens:
            raise ValueError('Training would exceed the token budget')
        self.synchronize()
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        warmup = min((self.updates + 1) / max(1, self.config.warmup_updates), 1.0)
        for group in self.optimizer.param_groups:
            group['lr'] = self.config.learning_rate * warmup
        losses = []
        for _ in range(self.config.accumulation):
            x, y = self.train.batch(self.config.microbatch, self.device)
            with self.autocast():
                loss = F.cross_entropy(self.model(x).float().flatten(0, 1), y.flatten())
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite training loss')
            self.scaler.scale(loss / self.config.accumulation).backward()
            losses.append(loss.detach().item())
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
        if not self.scaler.is_enabled() and not torch.isfinite(norm).item():
            raise RuntimeError('Non-finite FP32 gradients')
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale_before
        self.skipped += int(skipped)
        self.updates += int(not skipped)
        self.iterations += 1
        self.tokens += self.tokens_per_iteration
        self.synchronize()
        elapsed = time.perf_counter() - started
        self.training_s += elapsed
        return {'iteration': self.iterations, 'tokens': self.tokens, 'updates': self.updates,
                'skipped': skipped, 'loss': sum(losses) / len(losses), 'learning_rate': self.optimizer.param_groups[0]['lr'],
                'grad_norm': norm.item() if torch.isfinite(norm).item() else None,
                'scale': self.scaler.get_scale(), 'training_s': elapsed}

    def evaluate(self):
        self.synchronize()
        started = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        total_nll, count = 0.0, 0
        try:
            with torch.inference_mode():
                for start in range(0, len(self.validation) - 1, self.model.config.context):
                    sequence = torch.from_numpy(np.array(self.validation[start:start + self.model.config.context + 1], dtype=np.int64)).to(self.device)
                    with self.autocast():
                        logits = self.model(sequence[:-1].unsqueeze(0))
                        nll = F.cross_entropy(logits.float().flatten(0, 1), sequence[1:], reduction='sum')
                    total_nll += nll.item()
                    count += sequence.numel() - 1
        finally:
            self.model.train(was_training)
        self.synchronize()
        self.evaluation_s += time.perf_counter() - started
        loss = total_nll / count
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite validation loss')
        return {'loss': loss, 'tokens': count, 'iteration': self.iterations, 'training_tokens': self.tokens}

    def save(self, directory: Path, validation_loss: float):
        if not math.isfinite(validation_loss):
            raise ValueError('Cannot publish non-finite validation loss')
        if directory.exists():
            raise FileExistsError(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = directory.with_name(f'.{directory.name}.{uuid.uuid4().hex}.tmp')
        temporary.mkdir(exist_ok=False)
        numpy_rng = np.random.get_state()
        state = {
            'optimizer': self.optimizer.state_dict(), 'scaler': self.scaler.state_dict(),
            'python_rng': random.getstate(), 'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            'torch_rng': torch.get_rng_state(), 'device_rng': torch.cuda.get_rng_state_all() if self.device == 'cuda' else [],
        }
        cpu_weights = {name: value.detach().cpu().contiguous() for name, value in self.model.state_dict().items()}
        if not all(torch.isfinite(value).all().item() for value in cpu_weights.values()):
            raise RuntimeError('Refusing to save non-finite model weights')
        save_file(cpu_weights, str(temporary / 'model.safetensors'))
        torch.save(state, temporary / 'state.pt')
        metadata = {
            'contract': self.contract, 'iterations': self.iterations, 'updates': self.updates,
            'tokens': self.tokens, 'skipped': self.skipped, 'data_position': self.train.position,
            'training_s': self.training_s, 'evaluation_s': self.evaluation_s,
            'validation_loss': validation_loss, 'best_loss': self.best_loss,
            'initial_weights_sha256': self.initial_weights_sha256, 'weights_sha256': weights_hash(self.model),
        }
        write_json(temporary / 'metadata.json', metadata)
        checksums = {name: file_hash(temporary / name) for name in ['model.safetensors', 'state.pt', 'metadata.json']}
        write_json(temporary / 'checksums.json', checksums)
        temporary.replace(directory)
        return checksums

    def load(self, directory: Path):
        checksums = json.loads((directory / 'checksums.json').read_text())
        if set(checksums) != {'model.safetensors', 'state.pt', 'metadata.json'}:
            raise ValueError('Checkpoint manifest has unexpected files')
        for name, expected in checksums.items():
            if file_hash(directory / name) != expected:
                raise ValueError(f'Checkpoint checksum mismatch: {name}')
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['contract'] != self.contract:
            raise ValueError('Checkpoint experiment identity/backend mismatch')
        if metadata['tokens'] > self.config.max_tokens:
            raise ValueError('Checkpoint exceeds requested budget')
        self.model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        if weights_hash(self.model) != metadata['weights_sha256']:
            raise ValueError('Checkpoint weight identity mismatch')
        state = torch.load(directory / 'state.pt', map_location='cpu', weights_only=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        for name in ['iterations', 'updates', 'tokens', 'skipped', 'training_s', 'evaluation_s', 'best_loss', 'initial_weights_sha256']:
            setattr(self, name, metadata[name])
        self.train.position = metadata['data_position']
        random.setstate(state['python_rng'])
        n = state['numpy_rng']
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
        torch.set_rng_state(state['torch_rng'])
        if self.device == 'cuda':
            torch.cuda.set_rng_state_all(state['device_rng'])

    def checkpoint(self, run: Path, validation_loss: float):
        improved = self.best_loss is None or validation_loss < self.best_loss or not (run / 'best.json').exists()
        if improved:
            self.best_loss = validation_loss
        directory = run / 'checkpoints' / f'iteration-{self.iterations:06d}'
        checksums = self.save(directory, validation_loss)
        reference = {'checkpoint': directory.relative_to(run).as_posix(), 'checksums': checksums,
                     'tokens': self.tokens, 'validation_loss': validation_loss}
        write_json(run / 'latest.json', reference)
        if improved:
            write_json(run / 'best.json', reference)
        milestones_path = run / 'milestones.json'
        milestones = json.loads(milestones_path.read_text()) if milestones_path.exists() else {}
        milestones[str(self.tokens)] = reference
        write_json(milestones_path, milestones)
        return directory
