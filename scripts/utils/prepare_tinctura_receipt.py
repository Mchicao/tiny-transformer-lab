import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description='E7 immutable source snapshot and historical artifact receipt')
    parser.add_argument('--output-id', required=True)
    parser.add_argument('--verify-preserved', action='store_true')
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    output = ROOT / 'output' / args.output_id
    if args.verify_preserved:
        saved = json.loads((output / 'preservation.json').read_text())
        for name, digest in saved['files'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Historical artifact modified: {name}')
        with (ROOT / 'runs/results.jsonl').open('rb') as handle:
            prefix = handle.read(saved['results_prefix_bytes'])
        assert hashlib.sha256(prefix).hexdigest() == saved['results_prefix_sha256']
        atomic_json(output / 'preservation-verified.json', dict(status='passed', files=len(saved['files']),
                    historical_results_prefix_preserved=True))
        print(json.dumps(dict(status='passed', files=len(saved['files']))))
        return
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for folder in ['src', 'scripts', 'configs', 'data', 'runs', 'output']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and output not in path.parents and path.name != 'results.jsonl':
                files[path.relative_to(ROOT).as_posix()] = sha256(path)
    results = ROOT / 'runs/results.jsonl'
    atomic_json(output / 'preservation.json', dict(files=files, results_prefix_bytes=results.stat().st_size,
                results_prefix_sha256=sha256(results)))
    for folder in ['src', 'scripts', 'configs']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json'}:
                destination = output / 'source-snapshot' / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    print(json.dumps(dict(status='prepared', output=str(output), historical_files=len(files), training_updates=0)))


if __name__ == '__main__':
    main()
