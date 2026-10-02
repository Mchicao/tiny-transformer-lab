import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import requests
from tokenizers import Tokenizer
from src.training import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description='Prefixed disjoint development set for E7 LR diagnosis, no training')
    parser.add_argument('--dataset-id', required=True)
    args = parser.parse_args()
    if Path(args.dataset_id).name != args.dataset_id:
        parser.error('Dataset ID must be one directory name')
    directory = ROOT / 'data/posttrain' / args.dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    pages = directory / 'pages'
    pages.mkdir(exist_ok=True)
    if (directory / 'manifest.json').exists():
        print(json.dumps(dict(status='already_prepared', directory=str(directory))))
        return
    seen_ids, seen_text = set(), set()
    for split in ['train', 'validation']:
        for line in (ROOT / f'data/posttrain/fineweb-e7-v2/{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            seen_ids.add(row['id'])
            seen_text.add(row['text_sha256'])
    tokenizer_path = ROOT / 'data/pretrained/tinctura-v1/final-v1/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    selected, tokens, offset = [], [], 20000
    while len(tokens) < 65537:
        path = pages / f'rows-{offset}.json'
        url = ('https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu'
               f'&config=sample-10BT&split=train&offset={offset}&length=50')
        if path.exists():
            body = json.loads(path.read_text())
        else:
            response = requests.get(url, timeout=90)
            if response.status_code == 429:
                print(json.dumps(dict(status='rate_limited', offset=offset)))
                return
            response.raise_for_status()
            body = response.json()
            atomic_json(path, body)
        if not body['rows']:
            raise RuntimeError('Dev rows exhausted')
        for item in body['rows']:
            row = item['row']
            digest = hashlib.sha256(row['text'].encode()).hexdigest()
            if item.get('truncated_cells') or row['id'] in seen_ids or digest in seen_text:
                continue
            seen_ids.add(row['id'])
            seen_text.add(digest)
            selected.append(dict(id=row['id'], row_idx=item['row_idx'], text=row['text'], text_sha256=digest))
            ids = tokenizer.encode(row['text']).ids + [0]
            if max(ids) >= 32768:
                raise ValueError('Invalid dev vocabulary')
            tokens.extend(ids)
            if len(tokens) >= 65537:
                break
        offset += len(body['rows'])
    np.array(tokens[:65537], dtype='<u2').tofile(directory / 'dev.bin')
    with (directory / 'dev.jsonl').open('x', encoding='utf-8') as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    atomic_json(directory / 'manifest.json', dict(dataset='HuggingFaceFW/fineweb-edu', config='sample-10BT',
                policy='increasing rows from20000, reject train/val/internal IDs and text hashes', documents=len(selected),
                tokens=65537, targets=65536, tokenizer_sha256=sha256(tokenizer_path),
                dev_sha256=sha256(directory / 'dev.bin'), raw_sha256=sha256(directory / 'dev.jsonl'),
                disjoint_from_training_and_validation=True, selection_before_pilots=True, training_updates=0))
    print(json.dumps(dict(status='prepared', documents=len(selected), targets=65536, directory=str(directory))))


if __name__ == '__main__':
    main()
