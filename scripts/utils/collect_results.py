import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
results = [json.loads(path.read_text()) for path in sorted((ROOT / 'runs').glob('*/metrics.json'))]
target = ROOT / 'runs/results.jsonl'
temporary = target.with_suffix('.partial')
temporary.write_text(''.join(json.dumps(result) + '\n' for result in results))
temporary.replace(target)
print(f'Collected {len(results)} runs; optimizer steps: {sum(result["optimizer_steps"] for result in results)}')
