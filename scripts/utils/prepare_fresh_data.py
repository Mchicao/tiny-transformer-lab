import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import requests
from tokenizers import Tokenizer
from src.training import atomic_json, sha256

BUDGET = 6144000
DATASET = 'roneneldan/TinyStories'


def revision():
    response = requests.get(f'https://huggingface.co/api/datasets/{DATASET}', timeout=60)
    response.raise_for_status()
    return response.json()['sha']


def page(offset, cache, expected_revision):
    path = cache / f'rows-{offset:07d}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['revision'] != expected_revision:
            raise ValueError('Cached page revision incompatible')
        return saved['rows']
    url = (f'https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default'
           f'&split=train&offset={offset}&length=100')
    response = requests.get(url, timeout=90)
    if response.status_code in {429, 502, 503, 504}:
        logging.info(json.dumps(dict(event='remote_retry', status=response.status_code, retry_after=response.headers.get('Retry-After'))))
        return None
    response.raise_for_status()
    body = response.json()
    if not body['rows']:
        raise RuntimeError('TinyStories source exhausted')
    rows = []
    for item in body['rows']:
        if item.get('truncated_cells'):
            raise ValueError('Truncated TinyStories text cannot be used')
        rows.append(dict(id=item['row_idx'], text=item['row']['text']))
    atomic_json(path, dict(revision=expected_revision, url=url, rows=rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description='E11 deterministic expanded TinyStories, original frozen BPE and validation')
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--max-seconds', type=int, default=450)
    args = parser.parse_args()
    if Path(args.dataset_id).name != args.dataset_id or not 30 <= args.max_seconds <= 480:
        parser.error('Single-directory dataset ID and time budget30..480 required')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    directory = ROOT / 'data/expanded-tinystories' / args.dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / 'pages'
    cache.mkdir(exist_ok=True)
    old = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    expected_revision = old['observed_dataset_revision']
    if revision() != expected_revision:
        raise ValueError('Source revision changed; same-source comparison requires a new explicit design')
    if (directory / 'manifest.json').exists():
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha256(directory / name) != digest:
                raise ValueError('Completed expanded dataset changed')
        print(json.dumps(dict(status='already_prepared', directory=str(directory), tokens=manifest['tokens'])))
        return
    tokenizer_path = ROOT / 'data/processed/tokenizer.json'
    if sha256(tokenizer_path) != old['tokenizer_sha256']:
        raise ValueError('Historical tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id('<eos>')
    baseline = [json.loads(line) for line in (ROOT / 'data/raw/train.jsonl').read_text(encoding='utf-8').splitlines()]
    validation = [json.loads(line) for line in (ROOT / 'data/raw/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    digest_text = lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest()
    val_hashes = {digest_text(row['text']) for row in validation}
    seen, selected, token_ids = set(), [], []

    def append(row):
        digest = digest_text(row['text'])
        if digest in val_hashes or digest in seen:
            return False
        seen.add(digest)
        selected.append(dict(**row, text_sha256=digest))
        ids = tokenizer.encode(row['text']).ids + [eos]
        if max(ids) >= 4096:
            raise ValueError('Expanded tokenizer IDs outside existing vocabulary')
        token_ids.extend(ids)
        return True

    for row in baseline:
        if not append(row):
            raise ValueError('Historical train prefix contains duplicates or validation overlap')
    prefix = np.array(token_ids, dtype='<u2')
    if hashlib.sha256(prefix.tobytes()).hexdigest() != old['splits']['train']['shard_sha256']:
        raise ValueError('Historical train prefix is not reproduced exactly')
    start, offset, rejected = time.perf_counter(), len(baseline), 0
    while len(token_ids) < BUDGET + 1:
        if time.perf_counter() - start > args.max_seconds:
            print(json.dumps(dict(status='partial', next_offset=offset, tokens_collected=len(token_ids), directory=str(directory))))
            return
        rows = page(offset, cache, expected_revision)
        if rows is None:
            print(json.dumps(dict(status='rate_limited', next_offset=offset, tokens_collected=len(token_ids), directory=str(directory))))
            return
        for row in rows:
            if not append(row):
                rejected += 1
            if len(token_ids) >= BUDGET + 1:
                break
        offset += len(rows)
        if offset % 2000 == 0:
            logging.info(json.dumps(dict(event='coverage', next_offset=offset, tokens=len(token_ids), documents=len(selected))))
    if revision() != expected_revision:
        raise ValueError('Source changed during preparation')
    if len({row['id'] for row in selected}) != len(selected) or len(seen) != len(selected):
        raise ValueError('Expanded document IDs/texts duplicated')
    temporary = directory / ('train.' + uuid.uuid4().hex + '.tmp')
    np.array(token_ids, dtype='<u2').tofile(temporary)
    temporary.rename(directory / 'train.bin')
    with (directory / 'train.jsonl').open('x', encoding='utf-8', newline='\n') as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    files = {name: sha256(directory / name) for name in ['train.bin', 'train.jsonl']}
    manifest = dict(format=1, dataset=DATASET, observed_dataset_revision=expected_revision,
                    source='rows API; revision checked before/after; archived page/raw hashes authoritative',
                    policy='historical first2000 prefix, then increasing train row IDs; reject duplicates/val overlap',
                    documents=len(selected), unique_documents=len(seen), tokens=len(token_ids), token_budget=BUDGET,
                    internal_duplicates_and_validation_overlap=0, rejected_rows=rejected,
                    historical_prefix_tokens=len(prefix), historical_prefix_sha256=old['splits']['train']['shard_sha256'],
                    tokenizer_sha256=old['tokenizer_sha256'], validation_sha256=old['splits']['validation']['shard_sha256'],
                    vocab_size=4096, dtype='uint16', files=files,
                    generator_sha256=sha256(Path(__file__)), training_updates=0)
    atomic_json(directory / 'manifest.json', manifest)
    print(json.dumps(dict(status='prepared', directory=str(directory), documents=len(selected), tokens=len(token_ids), rejected_rows=rejected)))


if __name__ == '__main__':
    main()
