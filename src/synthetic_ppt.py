import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.training import atomic_json, isolated_eval, sha256

CONTEXT = 256
PPT_STEPS = 245
PPT_TOKENS = PPT_STEPS * 4096
ARMS = ('retrieval', 'grammatical')
PROBE_SEED = 202610019
PROBE_ROWS = 128
ANSWER_POSITIONS = np.arange(162, 208, 3)
SPEC = dict(format=1, context=CONTEXT, vocab_size=4096, terminals=[16, 4096],
            pairs=16, memory_start=0, query_start=160, query_end=208,
            markers=dict(memory=0, query=1), grammatical_shift=2039,
            rng='PCG64/SeedSequence(seed,spawn_key=(9,))',
            objective='full_sequence_next_token', answer_fraction=16 / 256)


def sequences(arm, seed, rows):
    if arm not in ARMS or rows < 1:
        raise ValueError('Invalid synthetic arm or sequence count')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(9,))))
    result = np.empty((rows, CONTEXT), dtype='<u2')
    for row in result:
        row[:] = rng.integers(16, 4096, CONTEXT, dtype=np.uint16)
        keys = rng.choice(np.arange(16, 4096), 16, replace=False)
        random_values = rng.integers(16, 4096, 16)
        values = random_values if arm == 'retrieval' else (keys - 16 + SPEC['grammatical_shift']) % 4080 + 16
        order = rng.permutation(16)
        row[:48].reshape(16, 3)[:] = np.stack((np.zeros(16, dtype=int), keys, values), axis=1)
        row[160:208].reshape(16, 3)[:] = np.stack((np.ones(16, dtype=int), keys[order], values[order]), axis=1)
    return result


def check_sequences():
    a = sequences('retrieval', 42, 16)
    b = sequences('grammatical', 42, 16)
    assert np.array_equal(a, sequences('retrieval', 42, 16))
    assert not np.array_equal(a, sequences('retrieval', 43, 16))
    assert np.array_equal(a[:, 48:160], b[:, 48:160])
    assert np.array_equal(a[:, 208:], b[:, 208:])
    assert np.array_equal(a[:, 161:208:3], b[:, 161:208:3])
    for arm, batch in [('retrieval', a), ('grammatical', b)]:
        assert batch.shape == (16, 256) and batch.dtype == np.dtype('<u2')
        assert batch.max() < 4096
        for row in batch:
            memory = row[:48].reshape(16, 3)
            queries = row[160:208].reshape(16, 3)
            table = dict(zip(memory[:, 1].tolist(), memory[:, 2].tolist(), strict=True))
            assert len(table) == 16 and len(set(queries[:, 1])) == 16
            assert all(table[int(key)] == int(value) for _, key, value in queries)
            if arm == 'grammatical':
                assert np.all(queries[:, 2] == (queries[:, 1].astype(np.int64) - 16 + 2039) % 4080 + 16)
    assert all(sequences(arm, PROBE_SEED, PROBE_ROWS).tobytes() != sequences(arm, 42, PROBE_ROWS).tobytes()
               for arm in ARMS)


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    check_sequences()
    files = {}
    for arm in ARMS:
        for seed in (42, 43, 44):
            path = directory / f'{arm}-seed{seed}.bin'
            batch = sequences(arm, seed, PPT_TOKENS // CONTEXT)
            with path.open('xb') as handle:
                handle.write(batch.tobytes())
                handle.write(np.array([0], dtype='<u2').tobytes())
            files[path.name] = dict(sha256=sha256(path), tokens=PPT_TOKENS + 1, arm=arm, seed=seed)
        path = directory / f'{arm}-probe.bin'
        path.write_bytes(sequences(arm, PROBE_SEED, PROBE_ROWS).tobytes())
        files[path.name] = dict(sha256=sha256(path), tokens=PROBE_ROWS * CONTEXT, arm=arm, seed=PROBE_SEED)
    atomic_json(directory / 'manifest.json', dict(spec=SPEC, generator_sha256=sha256(Path(__file__)),
                ppt_steps=PPT_STEPS, ppt_tokens=PPT_TOKENS, probe_rows=PROBE_ROWS,
                probe_seed=PROBE_SEED, files=files))


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['spec'] != SPEC or manifest['generator_sha256'] != sha256(Path(__file__)):
        raise ValueError('Synthetic generator/spec identity mismatch')
    if manifest['ppt_tokens'] != PPT_TOKENS or manifest['ppt_steps'] != PPT_STEPS:
        raise ValueError('Synthetic budget mismatch')
    expected = {f'{a}-seed{s}.bin' for a in ARMS for s in (42, 43, 44)} | {f'{a}-probe.bin' for a in ARMS}
    if set(manifest['files']) != expected:
        raise ValueError('Incomplete synthetic manifest')
    for name, info in manifest['files'].items():
        path = directory / name
        if sha256(path) != info['sha256'] or path.stat().st_size != info['tokens'] * 2:
            raise ValueError(f'Synthetic shard mismatch: {name}')
    return {str((directory / name).relative_to(Path(__file__).resolve().parents[1])).replace('\\', '/'): sha256(directory / name)
            for name in ['manifest.json', *manifest['files']]}


def probe(model, directory):
    results = {}
    with isolated_eval(model):
        positions = torch.tensor(ANSWER_POSITIONS - 1, device='cuda')
        for arm in ARMS:
            rows = np.fromfile(Path(directory) / f'{arm}-probe.bin', dtype='<u2').reshape(PROBE_ROWS, CONTEXT)
            loss, correct, exact = 0.0, 0, 0
            for start in range(0, len(rows), 4):
                batch = torch.tensor(rows[start:start + 4].astype(np.int64), device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(batch[:, :-1]).float()[:, positions]
                targets = batch[:, positions + 1]
                loss += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='sum').item()
                matches = logits.argmax(-1) == targets
                correct += matches.sum().item()
                exact += matches.all(-1).sum().item()
            count = PROBE_ROWS * len(ANSWER_POSITIONS)
            results[arm] = dict(answer_nll=loss / count, answer_accuracy=correct / count,
                                sequence_exact_match=exact / PROBE_ROWS, answers=count, rows=PROBE_ROWS)
    return results
