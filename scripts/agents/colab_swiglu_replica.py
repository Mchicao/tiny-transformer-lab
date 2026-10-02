from contextlib import contextmanager
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from safetensors.torch import load_file
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import ffn_experiment
from src.data import PackedTokens
from src.model import ModelConfig, Transformer
from src.training import EXPERIMENT, Trainer, sha256, verified_identity, weight_hash


def cuda_initialize(self):
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.version.hip:
        raise RuntimeError('This replica requires a CUDA GPU; no HIP or CPU fallback')
    props = torch.cuda.get_device_properties(0)
    if props.name != 'Tesla T4':
        raise RuntimeError(f'Unexpected replica GPU: {props.name}')
    self.backend = dict(gpu=props.name, arch=f'sm_{props.major}{props.minor}', vram=props.total_memory,
                        torch=str(torch.__version__), cuda=torch.version.cuda)
    self.identity = verified_identity()
    raw = json.loads((ROOT / 'configs/3m.json').read_text())
    self.config = ModelConfig(**{field.name: raw[field.name] for field in fields(ModelConfig) if field.name in raw})
    if raw != {**asdict(self.config), 'seed': 42, 'precision': 'fp16', 'microbatch': 4,
               'gradient_accumulation': 4, 'optimizer': 'adamw'} or self.config != ModelConfig():
        raise ValueError('Incompatible seed42 baseline config')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    self.model = Transformer(self.config).cuda()
    initial = load_file(str(ROOT / 'output/initial-3m.safetensors'))
    for name, parameter in self.model.state_dict().items():
        torch.testing.assert_close(parameter.cpu(), initial[name], rtol=0, atol=0)
    self.model.load_state_dict(initial, strict=True)
    self.initial_hash = weight_hash(self.model)
    if self.initial_hash != '194d4675760675037cfaef56e78f3e8aad74ca07d244068e782ac60d8ac1ef64':
        raise ValueError('Original seed42 weights mismatch')
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


@contextmanager
def cuda_factories():
    original_initialize, original_fresh = Trainer.__init__, ffn_experiment.fresh

    def fresh(variant='adamw', iterations=1500):
        trainer = original_fresh(variant, iterations)
        trainer.identity['scripts/agents/colab_swiglu_replica.py'] = sha256(Path(__file__))
        if trainer.initial_hash != '5ef6ed7a14a4065340f737a2a401d7a0f1cd62929ca905a9e1555c7d35f4033a':
            raise ValueError('Replica must start from the archived SwiGLU seed42 weights')
        return trainer

    try:
        Trainer.__init__ = cuda_initialize
        ffn_experiment.fresh = fresh
        yield
    finally:
        Trainer.__init__, ffn_experiment.fresh = original_initialize, original_fresh


def verification_execute(run_id, target, resume=None):
    command = [sys.executable, str(Path(__file__).resolve()), '--variant', 'adamw',
               '--iterations', '1500', '--run-id', run_id, '--stop-after', str(target)]
    if resume:
        command += ['--resume', str(resume)]
    log = ROOT / 'logs/tests' / f'{run_id}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / run_id


def main():
    if '--prepare-initialization' in sys.argv or '--continue-baseline' in sys.argv:
        raise ValueError('The replica only uses the existing archived SwiGLU initialization')
    with cuda_factories():
        if '--cuda-check-id' in sys.argv:
            from scripts.utils import check_ffn
            position = sys.argv.index('--cuda-check-id')
            sys.argv[position] = '--check-id'
            check_ffn.execute = verification_execute
            check_ffn.main()
        elif '--audit-run' in sys.argv:
            from scripts.utils import check_ffn
            check_ffn.main()
        else:
            ffn_experiment.main()


if __name__ == '__main__':
    main()
