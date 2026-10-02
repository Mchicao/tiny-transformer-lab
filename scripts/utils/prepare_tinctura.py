import argparse
import json
import logging
from pathlib import Path
import sys
import uuid
import hashlib
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import requests
import numpy as np
from tokenizers import Tokenizer
from src.training import atomic_json, sha256

REPO = 'bench-labs/tinctura-v1'
REVISIONS = {
    'early': dict(revision='b57a4f781c583bf0015cf6123fedab45d9c63436', step=22000, seed=42),
    'middle': dict(revision='f5a0a6822f6727d7a51459514f0e14fb6c809121', step=380000, seed=43),
    'final': dict(revision='f475b8a2698b0f2de14fbae584014908fec4f719', step=762939, seed=44),
}


def get_json(url):
    for attempt in range(4):
        response = requests.get(url, timeout=90)
        if response.status_code not in {429, 502, 503, 504} or attempt == 3:
            response.raise_for_status()
            return response.json()
        retry_after = response.headers.get('Retry-After', '')
        delay = int(retry_after) if retry_after.isdigit() else 30 * (attempt + 1)
        if delay > 120:
            response.raise_for_status()
        logging.info(json.dumps(dict(event='http_backoff', status=response.status_code, seconds=delay)))
        time.sleep(delay)
    raise RuntimeError('HTTP retry budget exhausted')


def download(url, destination, expected=None):
    temporary = ROOT / '.cache/tinctura' / (destination.name + '.' + uuid.uuid4().hex + '.part')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open('xb') as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                handle.write(chunk)
                count += len(chunk)
    digest = sha256(temporary)
    if expected and digest != expected:
        raise ValueError(f'Download hash mismatch: {destination.name}')
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    logging.info(json.dumps(dict(event='downloaded', name=destination.name, bytes=count, sha256=digest)))
    return dict(bytes=count, sha256=digest, url=url)


def corpus(artifact_id, reuse_train=None):
    directory = ROOT / 'data/posttrain' / ('fineweb-e7-' + artifact_id)
    directory.mkdir(parents=True, exist_ok=False)
    tokenizer_path = ROOT / 'data/pretrained/tinctura-v1/final-v1/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    seen_ids, seen_texts, splits = set(), set(), {}
    for split, offset, target in [('train', 0, 2000001), ('validation', 10000, 65537)]:
        tokens, documents, requests_count = [], 0, 0
        if split == 'train' and reuse_train:
            source = reuse_train.resolve()
            for line in (source / 'train.jsonl').read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                digest = hashlib.sha256(row['text'].encode()).hexdigest()
                if digest != row['text_sha256'] or row['id'] in seen_ids or digest in seen_texts:
                    raise ValueError('Reused training documents invalid or duplicated')
                seen_ids.add(row['id'])
                seen_texts.add(digest)
                tokens.extend(tokenizer.encode(row['text']).ids + [0])
                documents += 1
            shard = np.fromfile(source / 'train.bin', dtype='<u2')
            if len(shard) != target or not np.array_equal(shard, np.array(tokens[:target], dtype='<u2')):
                raise ValueError('Reused training shard does not match its raw documents')
            for name in ['train.jsonl', 'train.bin']:
                import shutil
                shutil.copy2(source / name, directory / name)
            splits[split] = dict(tokens=target, documents=documents, source_reused=source.relative_to(ROOT).as_posix(),
                                 shard_sha256=sha256(directory / 'train.bin'), raw_sha256=sha256(directory / 'train.jsonl'))
            continue
        with (directory / f'{split}.jsonl').open('x', encoding='utf-8') as handle:
            while len(tokens) < target:
                url = ('https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu'
                       f'&config=sample-10BT&split=train&offset={offset}&length=50')
                page = get_json(url)
                requests_count += 1
                if not page['rows']:
                    raise RuntimeError('FineWeb rows exhausted before fixed corpus budget')
                for item in page['rows']:
                    if item.get('truncated_cells'):
                        continue
                    row = item['row']
                    text_digest = hashlib.sha256(row['text'].encode()).hexdigest()
                    if row['id'] in seen_ids or text_digest in seen_texts:
                        continue
                    seen_ids.add(row['id'])
                    seen_texts.add(text_digest)
                    encoded = tokenizer.encode(row['text']).ids + [0]
                    if not encoded or max(encoded) >= 32768:
                        raise ValueError('Invalid FineWeb tokenizer result')
                    handle.write(json.dumps(dict(row_idx=item['row_idx'], id=row['id'], text=row['text'],
                                                 text_sha256=text_digest), ensure_ascii=False) + '\n')
                    tokens.extend(encoded)
                    documents += 1
                    if len(tokens) >= target:
                        break
                offset += len(page['rows'])
                if split == 'train' and offset >= 10000:
                    raise RuntimeError('Training rows would overlap the reserved validation range')
        shard = directory / f'{split}.bin'
        np.array(tokens[:target], dtype='<u2').tofile(shard)
        splits[split] = dict(tokens=target, documents=documents, api_requests=requests_count,
                             next_row_offset=offset, shard_sha256=sha256(shard),
                             raw_sha256=sha256(directory / f'{split}.jsonl'))
        logging.info(json.dumps(dict(event='corpus_split', split=split, **splits[split])))
    rng = np.random.Generator(np.random.PCG64(202610017))
    tasks = []
    for index in range(64):
        a, b = map(int, rng.integers(1, 100, 2))
        op = ['+', '-', '*', '/'][index % 4]
        if op == '+':
            answer = a + b
        elif op == '-':
            a, b = max(a, b), min(a, b)
            answer = a - b
        elif op == '*':
            a, b = a % 12 + 1, b % 12 + 1
            answer = a * b
        else:
            answer = a % 12 + 1
            a = answer * b
        options = {answer}
        while len(options) < 4:
            options.add(max(0, answer + int(rng.integers(-20, 21))))
        options = rng.permutation(sorted(options)).tolist()
        prompt = f'What is {a} {op} {b}?\nAnswer:'
        prefix = tokenizer.encode(prompt).ids
        candidates = []
        for option in options:
            full = tokenizer.encode(prompt + ' ' + str(option)).ids
            if full[:len(prefix)] != prefix or len(full) == len(prefix):
                raise ValueError('Arithmetic answer boundary is not tokenizer-stable')
            candidates.append(dict(value=option, ids=full, answer_start=len(prefix)))
        tasks.append(dict(prompt=prompt, answer=answer, candidates=candidates))
    atomic_json(directory / 'arithmetic-probe.json', dict(seed=202610017, tasks=tasks,
                selection='mean_answer_log_likelihood', official_arithmark=False, trained=False))
    atomic_json(directory / 'manifest.json', dict(dataset='HuggingFaceFW/fineweb-edu', config='sample-10BT',
                source_api='https://datasets-server.huggingface.co/rows', source_license='ODC-By',
                tokenizer_sha256=sha256(tokenizer_path), splits=splits, heldout_document_ids_and_texts_disjoint=True,
                arithmetic_probe_sha256=sha256(directory / 'arithmetic-probe.json'), training_updates=0))
    print(json.dumps(dict(status='prepared', corpus=str(directory), splits=splits, arithmetic_tasks=len(tasks),
                         training_updates=0)))


def main():
    parser = argparse.ArgumentParser(description='Download immutable public tinctura revisions, no training')
    parser.add_argument('--revision', choices=[*REVISIONS, 'corpus'], required=True)
    parser.add_argument('--artifact-id', default='v1')
    parser.add_argument('--reuse-train', type=Path)
    args = parser.parse_args()
    if Path(args.artifact_id).name != args.artifact_id:
        parser.error('Artifact ID must be one directory name')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.revision == 'corpus':
        corpus(args.artifact_id, args.reuse_train)
        return
    info = REVISIONS[args.revision]
    revision = info['revision']
    directory = ROOT / 'data/pretrained/tinctura-v1' / (args.revision + '-' + args.artifact_id)
    directory.mkdir(parents=True, exist_ok=False)
    tree = get_json(f'https://huggingface.co/api/models/{REPO}/tree/{revision}?recursive=false&expand=false')
    index = {entry['path']: entry for entry in tree}
    names = ['latest.pt'] if args.revision != 'final' else [
        'model.safetensors', 'config.json', 'modeling_cagliostro.py', 'configuration_cagliostro.py',
        'tokenizer.json', 'tokenizer_config.json', 'README.md']
    files = {}
    for name in names:
        entry = index[name]
        files[name] = download(f'https://huggingface.co/{REPO}/resolve/{revision}/{name}', directory / name,
                               entry.get('lfs', {}).get('oid'))
        if files[name]['bytes'] != entry['size']:
            raise ValueError(f'Download size mismatch: {name}')
    atomic_json(directory / 'manifest.json', dict(repo=REPO, **info, files=files, license='Apache-2.0',
                training_updates=0))
    print(json.dumps(dict(status='downloaded', revision=args.revision, directory=str(directory),
                         total_bytes=sum(f['bytes'] for f in files.values()), training_updates=0)))


if __name__ == '__main__':
    main()
