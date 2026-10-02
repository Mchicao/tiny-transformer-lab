import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
local = json.loads((ROOT / 'runs/local-prepare-3m-manual-v4/metrics.json').read_text())
remote = json.loads((ROOT / 'runs/colab-prepare-3m-manual-v4/metrics.json').read_text())
for field in ['config', 'parameters', 'precision', 'tokens', 'weights_sha256', 'tokenizer_sha256', 'data_sha256']:
    if local[field] != remote[field]:
        raise RuntimeError(f'Experiments differ: {field}')
result = {'comparable_inputs': True, 'optimizer_steps': 0, 'max_loss_difference': max(abs(a-b) for a, b in zip(local['losses'], remote['losses'], strict=True)), 'local_tokens_s': local['tokens_s'], 'colab_tokens_s': remote['tokens_s'], 'local_over_colab_speed_ratio': local['tokens_s'] / remote['tokens_s'], 'scope': '3 timed forward/backward steps after 1 warmup; no convergence or quality/GPU-hour conclusions'}
(ROOT / 'output/comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
