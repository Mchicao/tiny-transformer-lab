import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training import atomic_json


def main():
    parser = argparse.ArgumentParser(description='Bounded streaming verification of an immutable historical file inventory')
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output-id', required=True)
    parser.add_argument('--max-seconds', type=int, default=300)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id or not 30 <= args.max_seconds <= 450:
        parser.error('Unique output ID and bounded duration required')
    inventory_path = args.inventory.resolve()
    raw = inventory_path.read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    inventory = json.loads(raw)
    files = inventory['files'] if 'files' in inventory else inventory
    names = sorted(files)
    directory = ROOT / 'output' / args.output_id
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / 'progress.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(inventory_sha256=identity, checked=0, bytes=0)
    if status['inventory_sha256'] != identity:
        raise ValueError('Historical inventory changed during verification')
    start = time.perf_counter()
    while status['checked'] < len(names):
        if time.perf_counter() - start > args.max_seconds:
            break
        name = names[status['checked']]
        path = ROOT / name
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
                status['bytes'] += len(chunk)
        if digest.hexdigest() != files[name]:
            raise ValueError(f'Historical artifact modified: {name}')
        status['checked'] += 1
        if status['checked'] % 100 == 0:
            atomic_json(status_path, status)
    status['status'] = 'passed' if status['checked'] == len(names) else 'partial'
    status['total'] = len(names)
    atomic_json(status_path, status)
    print(json.dumps(status))


if __name__ == '__main__':
    main()
