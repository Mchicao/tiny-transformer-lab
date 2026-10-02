from dataclasses import fields
import json
import logging
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safetensors.torch import save_file
import torch
from src.model import ModelConfig, Transformer
from src.training import atomic_json, sha256, weight_hash


def main():
    raw = json.loads((ROOT / 'configs/3m.json').read_text())
    config = ModelConfig(**{field.name: raw[field.name] for field in fields(ModelConfig) if field.name in raw})
    if config != ModelConfig():
        raise ValueError('Initialization requires the unchanged 3M baseline')
    torch.manual_seed(43)
    model = Transformer(config)
    if sum(p.numel() for p in model.parameters()) != 3000384:
        raise ValueError('Unexpected parameter count')
    if not all(torch.isfinite(value).all().item() for value in model.parameters()):
        raise RuntimeError('Nonfinite initial weights')
    output = ROOT / 'output/initializations' / ('3m-seed43-' + uuid.uuid4().hex[:12])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.pending-' + output.name)
    staging.mkdir()
    save_file({name: value.contiguous() for name, value in model.state_dict().items()}, str(staging / 'model.safetensors'))
    with (staging / 'model.safetensors').open('rb+') as handle:
        os.fsync(handle.fileno())
    atomic_json(staging / 'initialization.json', dict(seed=43, model=vars(config), parameters=3000384,
                weights_sha256=weight_hash(model), model_source_sha256=sha256(ROOT / 'src/model.py'),
                training_updates=0, source_config_sha256=sha256(ROOT / 'configs/3m.json')))
    checksums = {name: sha256(staging / name) for name in ['model.safetensors', 'initialization.json']}
    atomic_json(staging / 'checksums.json', checksums)
    os.replace(staging, output)
    assert all(sha256(output / name) == digest for name, digest in checksums.items())
    logging.info(json.dumps(dict(initialization=str(output), seed=43, weights_sha256=weight_hash(model), training_updates=0)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
