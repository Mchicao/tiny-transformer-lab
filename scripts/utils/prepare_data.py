import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = Path(__file__).resolve().parents[2]
DATASET = 'roneneldan/TinyStories'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(split, rows, raw_directory):
    path = raw_directory / f'{split}.jsonl'
    if path.exists():
        with path.open(encoding='utf-8') as handle:
            if sum(1 for _ in handle) != rows:
                raise ValueError('Cached subset size differs; preserve it and prepare a separate dataset directory')
        return path
    temporary = path.with_suffix('.partial')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        for offset in range(0, rows, 100):
            url = f'https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default&split={split}&offset={offset}&length={min(100, rows-offset)}'
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            if not payload.get('rows'):
                raise RuntimeError(f'No rows returned at {split}:{offset}')
            for row in payload['rows']:
                handle.write(json.dumps({'id': row['row_idx'], 'text': row['row']['text']}, ensure_ascii=False) + '\n')
    temporary.replace(path)
    return path


def texts(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)['text']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-stories', type=int, default=2000)
    parser.add_argument('--validation-stories', type=int, default=200)
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    args = parser.parse_args()
    if not 256 < args.vocab_size < 65536:
        raise ValueError('Vocabulary must fit uint16 and byte alphabet')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['vocab_size'] != args.vocab_size:
        raise ValueError('Existing tokenizer has a different vocabulary; preserve it and use a separate dataset directory')
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{DATASET}', timeout=30) as response:
        revision = json.load(response)['sha']
    train = download('train', args.train_stories, args.raw_dir)
    validation = download('validation', args.validation_stories, args.raw_dir)
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts(train), trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=['<unk>', '<eos>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    target = args.output_dir
    tokenizer.save(str(target / 'tokenizer.json'))
    manifest = {'dataset': DATASET, 'observed_dataset_revision': revision, 'source': 'datasets-server rows API; archived subset hashes are authoritative', 'vocab_size': tokenizer.get_vocab_size(), 'dtype': 'uint16', 'packing': 'sequential concatenation with eos; one-token shift', 'splits': {}}
    for name, source in [('train', train), ('validation', validation)]:
        count = 0
        stories = 0
        shard = target / f'{name}.bin'
        with shard.open('wb') as handle:
            for story in texts(source):
                ids = tokenizer.encode(story).ids + [tokenizer.token_to_id('<eos>')]
                handle.write(np.asarray(ids, dtype='<u2').tobytes())
                count += len(ids)
                stories += 1
        manifest['splits'][name] = {'stories': stories, 'tokens': count, 'raw_sha256': digest(source), 'shard_sha256': digest(shard)}
    manifest['tokenizer_sha256'] = digest(target / 'tokenizer.json')
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
