import argparse
import copy
import json
import logging
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from safetensors.torch import load_file
from src.model import Transformer
from src.training import EXPERIMENT, Trainer, atomic_json, sha256, weight_hash

BASE_EXPERIMENT = copy.deepcopy(EXPERIMENT)
CONTINUATION_EXPERIMENT = {**BASE_EXPERIMENT, 'iterations': 490,
                           'milestones': BASE_EXPERIMENT['milestones'] + [295, 345, 395, 445, 490]}
SEED43_EXPERIMENT = {**CONTINUATION_EXPERIMENT, 'seed': 43}


def make_trainer(seed=42, initialization=None):
    if seed not in {42, 43} or (seed == 43) != (initialization is not None):
        raise ValueError('Seed 43 requires its own initialization; seed 42 uses the original')
    trainer = Trainer()
    if seed == 42:
        return trainer
    if EXPERIMENT != SEED43_EXPERIMENT:
        raise ValueError('Seed 43 requires the explicit 490-iteration experiment')
    initialization = initialization.resolve()
    initialization.relative_to(ROOT / 'output' / 'initializations')
    checksums = json.loads((initialization / 'checksums.json').read_text())
    if set(checksums) != {'model.safetensors', 'initialization.json'}:
        raise ValueError('Incomplete initialization manifest')
    for name, expected in checksums.items():
        if sha256(initialization / name) != expected:
            raise ValueError(f'Initialization checksum mismatch: {name}')
    metadata = json.loads((initialization / 'initialization.json').read_text())
    if metadata['seed'] != 43 or metadata['model'] != vars(trainer.config) or metadata['parameters'] != 3000384:
        raise ValueError('Incompatible initialization')
    if metadata['model_source_sha256'] != trainer.identity['src/model.py']:
        raise ValueError('Initialization model source mismatch')
    if metadata['source_config_sha256'] != trainer.identity['configs/3m.json']:
        raise ValueError('Initialization config source mismatch')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    expected = Transformer(trainer.config)
    initial = load_file(str(initialization / 'model.safetensors'))
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    trainer.model.load_state_dict(initial, strict=True)
    old_hash = trainer.initial_hash
    trainer.initial_hash = weight_hash(trainer.model)
    if trainer.initial_hash == old_hash or trainer.initial_hash != metadata['weights_sha256']:
        raise ValueError('Seed 43 weights identity mismatch')
    if trainer.optimizer.state or trainer.data.position or any(trainer.counters.values()):
        raise RuntimeError('New initialization requires fresh optimizer and counters')
    for name in checksums:
        path = initialization / name
        trainer.identity[path.relative_to(ROOT).as_posix()] = sha256(path)
    trainer.identity['scripts/train.py'] = sha256(Path(__file__))
    return trainer


def configure_checkpoint_experiment(checkpoint):
    checksums = json.loads((checkpoint / 'checksums.json').read_text())
    if set(checksums) != {'model.safetensors', 'state.pt', 'metadata.json'}:
        raise ValueError('Incomplete checkpoint manifest')
    for name, expected in checksums.items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    if metadata['experiment'] not in [BASE_EXPERIMENT, CONTINUATION_EXPERIMENT, SEED43_EXPERIMENT]:
        raise ValueError('Unsupported checkpoint experiment')
    EXPERIMENT.update(copy.deepcopy(metadata['experiment']))
    return metadata


def restore_trainer(checkpoint):
    metadata = configure_checkpoint_experiment(checkpoint)
    seed = metadata['experiment']['seed']
    initialization = None
    if seed == 43:
        paths = [ROOT / name for name in metadata['identity']
                 if name.startswith('output/initializations/') and name.endswith('/initialization.json')]
        if len(paths) != 1:
            raise ValueError('Checkpoint must identify exactly one seed 43 initialization')
        initialization = paths[0].parent
    trainer = make_trainer(seed, initialization)
    trainer.restore(checkpoint)
    return trainer, metadata


def extend_budget(trainer):
    if trainer.counters['iterations'] != 245 or EXPERIMENT != BASE_EXPERIMENT:
        raise ValueError('Continuation requires the completed 245-iteration baseline')
    EXPERIMENT.update(copy.deepcopy(CONTINUATION_EXPERIMENT))


def curve(run, rows):
    rows = list({row['iteration']: row for row in rows}.values())
    low, high = min(r['nll'] for r in rows), max(r['nll'] for r in rows)
    end = max(r['iteration'] for r in rows) or 1
    points = ' '.join(f"{60 + r['iteration'] / end * 660:.1f},{330 - (r['nll'] - low) / max(high - low, 0.01) * 270:.1f}" for r in rows)
    labels = ''.join(f'<text x="{60 + r["iteration"] / end * 660:.1f}" y="{350}" text-anchor="middle">{r["iteration"]}</text>' for r in rows)
    (run / 'validation.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="780" height="400" viewBox="0 0 780 400"><rect width="780" height="400" fill="white"/><g font-family="sans-serif" font-size="14" fill="#222"><text x="60" y="25">Validation NLL por token — AdamW 3M</text><text x="5" y="65">{high:.3f}</text><text x="5" y="330">{low:.3f}</text><path d="M60 50 V330 H730" fill="none" stroke="#555"/><polyline points="{points}" fill="none" stroke="#165dcc" stroke-width="3"/>{labels}<text x="350" y="385">Iteraciones (4096 tokens cada una)</text></g></svg>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Baseline autorizado AdamW 3M, exclusivamente HIP local')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--additional-iterations', type=int, choices=[245])
    parser.add_argument('--seed', type=int, choices=[42, 43])
    parser.add_argument('--initialization', type=Path)
    parser.add_argument('--check-target', type=int, choices=[2, 5])
    args = parser.parse_args()
    if args.additional_iterations and (not args.resume or args.check_target):
        raise ValueError('Additional iterations require --resume and cannot run as a check')
    if args.resume and (args.initialization or args.seed is not None):
        raise ValueError('A resumed checkpoint supplies its own seed and initialization')
    if args.seed == 43 and (args.additional_iterations or args.check_target):
        raise ValueError('Seed 43 is a fresh 490-iteration run, not a continuation or training check')
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('Run ID must be a single directory name')
    run = ROOT / 'runs' / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    logs = ROOT / 'logs/app'
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(logs / f'{args.run_id}.jsonl', encoding='utf-8')])
    start = time.perf_counter()
    source = None
    if args.resume:
        trainer, source = restore_trainer(args.resume.resolve())
    else:
        seed = args.seed or 42
        EXPERIMENT.update(copy.deepcopy(SEED43_EXPERIMENT if seed == 43 else BASE_EXPERIMENT))
        trainer = make_trainer(seed, args.initialization)
    torch.cuda.reset_peak_memory_stats()
    if args.resume:
        if args.additional_iterations:
            extend_budget(trainer)
    else:
        trainer.evaluate()
        trainer.best_loss = trainer.evaluations[-1]['nll']
        trainer.save(run, ['latest', 'best', 'initial'])
    starting_counters = copy.deepcopy(trainer.counters)
    starting_seconds = trainer.train_seconds
    starting_hash = weight_hash(trainer.model)
    target = args.check_target or EXPERIMENT['iterations']
    if trainer.counters['iterations'] >= target:
        raise ValueError('Checkpoint has already exhausted the requested budget')
    mode = 'verification' if args.check_target else ('continuation' if args.additional_iterations else ('seed_replication' if EXPERIMENT['seed'] == 43 else 'principal'))
    atomic_json(run / 'config.json', dict(model=vars(trainer.config), experiment=EXPERIMENT, identity=trainer.identity,
                backend=trainer.backend, mode=mode, start_counters=starting_counters,
                source_checkpoint=str(args.resume.resolve()) if args.resume else None,
                source_checksums_sha256=sha256(args.resume.resolve() / 'checksums.json') if args.resume else None,
                initialization=str(args.initialization.resolve()) if args.initialization else None,
                trainer_cli_sha256=sha256(Path(__file__))))
    if args.resume and not args.check_target:
        boundary = trainer.evaluate()
        if boundary['nll'] != source['evaluations'][-1]['nll']:
            raise RuntimeError('Validation at resume differs from the source checkpoint')
        trainer.save(run, ['latest', 'best', 'initial'])
    if not args.check_target:
        atomic_json(run / 'generations-initial.json', trainer.generate())
        if args.additional_iterations or EXPERIMENT['seed'] == 43:
            from tokenizers import Tokenizer
            from scripts.utils.sample_checkpoint import sample
            tokenizer = Tokenizer.from_file(str(ROOT / 'data/processed/tokenizer.json'))
            atomic_json(run / 'sampling-initial.json', dict(temperature=0.8, top_p=0.9,
                        seed_per_prompt=42, max_new_tokens=64, rows=sample(trainer.model, tokenizer)))
    try:
        while trainer.counters['iterations'] < target:
            row = trainer.step()
            logging.info(json.dumps(dict(event='train', **row), allow_nan=False))
            evaluation_due = row['iterations'] in EXPERIMENT['milestones'] if EXPERIMENT['seed'] == 43 else (row['iterations'] - starting_counters['iterations']) % 50 == 0
            if evaluation_due or row['iterations'] == target:
                validation = trainer.evaluate()
                logging.info(json.dumps(dict(event='validation', **validation)))
                refs = ['latest']
                if validation['nll'] < trainer.best_loss:
                    trainer.best_loss = validation['nll']
                    refs.append('best')
                if row['iterations'] in EXPERIMENT['milestones'] or args.check_target:
                    refs.append(f"milestone-{row['tokens']}")
                trainer.save(run, refs)
            elif args.check_target and row['iterations'] == 2:
                trainer.save(run, ['latest', 'boundary'])
        final_hash = weight_hash(trainer.model)
        if trainer.counters['effective'] == starting_counters['effective'] or final_hash == starting_hash:
            raise RuntimeError('No observable learning update to weights')
        if not args.check_target:
            atomic_json(run / 'generations-final.json', trainer.generate())
            if args.additional_iterations or EXPERIMENT['seed'] == 43:
                atomic_json(run / 'sampling-final.json', dict(temperature=0.8, top_p=0.9,
                            seed_per_prompt=42, max_new_tokens=64, rows=sample(trainer.model, tokenizer)))
        segment = {key: trainer.counters[key] - starting_counters[key] for key in ['iterations', 'effective', 'skipped', 'tokens']}
        segment['train_seconds'] = trainer.train_seconds - starting_seconds
        segment['tokens_s'] = segment['tokens'] / segment['train_seconds']
        result = dict(status='completed', run_id=args.run_id, mode=mode,
                      backend=trainer.backend, config=vars(trainer.config), experiment=EXPERIMENT,
                      **trainer.counters, train_seconds=trainer.train_seconds,
                      wall_seconds=time.perf_counter() - start,
                      tokens_s=trainer.counters['tokens'] / trainer.train_seconds,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      initial_weights_sha256=trainer.initial_hash, final_weights_sha256=final_hash,
                      starting_weights_sha256=starting_hash, start_counters=starting_counters, segment=segment,
                      evaluations=trainer.evaluations, history=trainer.history)
        atomic_json(run / 'metrics.json', result)
        curve(run, trainer.evaluations)
        with (ROOT / 'runs/results.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({k: v for k, v in result.items() if k != 'history'}) + '\n')
        logging.info(json.dumps(dict(event='completed', run=str(run), effective=result['effective'], skipped=result['skipped'])))
    except Exception as error:
        atomic_json(run / 'failure.json', dict(error_type=type(error).__name__, error=str(error), counters=trainer.counters))
        raise


if __name__ == '__main__':
    main()
