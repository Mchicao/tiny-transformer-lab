import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import ppt_experiment as ppt
from src.synthetic_ppt import prepare, verify
from src.training import atomic_json, sha256, weight_hash


def main():
    parser = argparse.ArgumentParser(description='Prepare deterministic E9 shards and preserve existing artifacts')
    parser.add_argument('--output-id', required=True)
    parser.add_argument('--verify-preserved', action='store_true')
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    output = ROOT / 'output' / args.output_id
    if args.verify_preserved:
        original = json.loads((output / 'preservation.json').read_text())
        for name, digest in original['files'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Existing artifact modified: {name}')
        results = ROOT / 'runs/results.jsonl'
        with results.open('rb') as handle:
            prefix = handle.read(original['results_prefix_bytes'])
        import hashlib
        if hashlib.sha256(prefix).hexdigest() != original['results_prefix_sha256']:
            raise ValueError('Historical results prefix modified')
        atomic_json(output / 'preservation-verified.json', dict(status='passed', files=len(original['files']),
                    results_prefix_preserved=True))
        print(json.dumps(dict(status='passed', existing_files_preserved=len(original['files']))))
        return
    output.mkdir(parents=True, exist_ok=False)
    originals = {}
    for folder in ['src', 'scripts', 'configs', 'data', 'runs', 'output']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and output not in path.parents and path.name != 'results.jsonl':
                originals[path.relative_to(ROOT).as_posix()] = sha256(path)
    results = ROOT / 'runs/results.jsonl'
    atomic_json(output / 'preservation.json', dict(files=originals, results_prefix_bytes=results.stat().st_size,
                results_prefix_sha256=sha256(results)))
    for folder in ['src', 'scripts', 'configs']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.ps1'}:
                target = output / 'source-snapshot' / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    if not ppt.DATA.exists():
        prepare(ppt.DATA)
    identities = verify(ppt.DATA)
    initializations = {}
    for seed, index in [(42, '022'), (43, '023'), (44, '024')]:
        ppt.SEED = seed
        trainer = ppt.fresh('retrieval')
        baseline = ROOT / 'runs' / f'e8d-3m-20261001-{index}-seed{seed}'
        refs = json.loads((baseline / 'references.json').read_text())
        checkpoint = baseline / 'checkpoints' / refs['initial']
        metadata = json.loads((checkpoint / 'metadata.json').read_text())
        actual = weight_hash(trainer.model)
        if actual != metadata['weights_sha256']:
            raise ValueError(f'E9 and E8 initial weights differ: seed{seed}')
        initializations[str(seed)] = dict(weights_sha256=actual, e8_control=baseline.relative_to(ROOT).as_posix(),
                                        e8_initial_checkpoint=checkpoint.relative_to(ROOT).as_posix())
    atomic_json(output / 'preparation.json', dict(status='prepared', experiment=ppt.contract(),
                initializations=initializations, synthetic_identity=identities,
                preserved_existing_files=len(originals), training_updates=0))
    print(json.dumps(dict(status='prepared', output=str(output), existing_files=len(originals),
                         initializations=initializations, training_updates=0)))


if __name__ == '__main__':
    main()
