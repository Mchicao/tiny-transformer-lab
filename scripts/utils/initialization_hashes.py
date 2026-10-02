import hashlib
import json

import numpy as np
import torch
from src.model import ModelConfig, Transformer

torch.manual_seed(42)
model = Transformer(ModelConfig())
print(json.dumps({'numpy': np.__version__, 'seed': torch.initial_seed(), 'parameters': {name: hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest() for name, parameter in model.named_parameters()}}))
