import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import PackedTokens

train = [json.loads(line)['text'] for line in (ROOT / 'data/raw/train.jsonl').read_text(encoding='utf-8').splitlines()]
validation = [json.loads(line)['text'] for line in (ROOT / 'data/raw/validation.jsonl').read_text(encoding='utf-8').splitlines()]
manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
result = {'train_stories': len(train), 'validation_stories': len(validation), 'exact_overlap': len(set(train) & set(validation))}
assert result['exact_overlap'] == 0
for split, stories in [('train', train), ('validation', validation)]:
    shard = np.memmap(ROOT / f'data/processed/{split}.bin', dtype='<u2', mode='r')
    assert len(shard) == manifest['splits'][split]['tokens']
    assert len(stories) == manifest['splits'][split]['stories']
    assert shard.max() < manifest['vocab_size']
    result[f'{split}_tokens'] = len(shard)
packed = PackedTokens(ROOT / 'data/processed/train.bin', 256)
x, y = packed.batch(2, 'cpu')
assert x.shape == y.shape == (2, 256)
torch.testing.assert_close(x[:, 1:], y[:, :-1], rtol=0, atol=0)
torch.testing.assert_close(x[1, :1], y[0, -1:], rtol=0, atol=0)
(ROOT / 'output/data-audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
