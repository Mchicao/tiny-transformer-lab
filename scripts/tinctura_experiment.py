import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from scripts.experiment import configure
from src.data import PackedTokens
from src.tinctura_loop import FINAL, Tinctura, load_weights, weights
from src.training import EXPERIMENT, Trainer, atomic_json, isolated_eval, sha256, weight_hash

CORPUS = ROOT / 'data/posttrain/fineweb-e7-v2'
SEEDS = dict(early=42, middle=43, final=44)


@dataclass
class E7Config:
    dim: int = 640
    layers: int = 18
    heads: int = 10
    kv_heads: int = 5
    context: int = 2048
    vocab_size: int = 32768
    intermediate: int = 1536
    attention: str = 'torch_sdpa_xsa'
    precision: str = 'fp16_autocast_fp32_master'
    checkpoint_layers: bool = True
    loop_k: int = 1


def contract(revision, variant, verification=False):
    if revision not in SEEDS or variant not in {'dense', 'looped'}:
        raise ValueError('E7 requires early/middle/final maturity and dense/looped arms')
    budget = 20480 if verification else 2000000
    steps = math.ceil(budget / 4096)
    return dict(e7_format=1, revision=revision, seed=SEEDS[revision], variant=variant, verification=verification,
                loop_k=1 if variant == 'dense' else 2, loss_recipe='final', iterations=steps, token_budget=budget,
                tokens_per_update=4096, microbatch=1, accumulation=2, context=2048,
                lr=3e-4, betas=[0.9, 0.95], weight_decay=0.1, clip=1.0,
                warmup=10, cooldown_start=int(steps * 0.85), cooldown_end=steps, init_scale=1.0,
                gradient_checkpointing=True, corpus=CORPUS.relative_to(ROOT).as_posix(),
                maturity_seed_assignment=SEEDS, maturity_interaction_is_descriptive=True)


def schedule(step, value):
    if not 1 <= step <= value['iterations']:
        raise ValueError('LR step outside fixed budget')
    start, end = value['cooldown_start'], value['iterations']
    if step <= start:
        return min(step / value['warmup'], 1.0)
    return 0.5 * (1 + math.cos(math.pi * (step - start) / (end - start)))


def identity(revision, model_info):
    manifest = json.loads((CORPUS / 'manifest.json').read_text())
    if manifest['splits']['train']['tokens'] != 2000001 or manifest['splits']['validation']['tokens'] != 65537:
        raise ValueError('E7 corpus token counts incompatible')
    names = ['src/tinctura_loop.py', 'scripts/tinctura_experiment.py', 'src/training.py', 'src/data.py',
             'scripts/experiment.py', 'data/posttrain/fineweb-e7-v2/manifest.json',
             'data/posttrain/fineweb-e7-v2/arithmetic-probe.json']
    for split, info in manifest['splits'].items():
        for suffix, field in [('bin', 'shard_sha256'), ('jsonl', 'raw_sha256')]:
            name = f'data/posttrain/fineweb-e7-v2/{split}.{suffix}'
            if sha256(ROOT / name) != info[field]:
                raise ValueError('E7 corpus checksum mismatch')
            names.append(name)
    if sha256(FINAL / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('E7 corpus tokenizer mismatch')
    if sha256(CORPUS / 'arithmetic-probe.json') != manifest['arithmetic_probe_sha256']:
        raise ValueError('E7 arithmetic probe mismatch')
    for label in {revision, 'final'}:
        directory = ROOT / 'data/pretrained/tinctura-v1' / (label + '-v1')
        info = model_info if label == revision else json.loads((directory / 'manifest.json').read_text())
        names.append((directory / 'manifest.json').relative_to(ROOT).as_posix())
        for filename, metadata in info['files'].items():
            path = directory / filename
            if sha256(path) != metadata['sha256']:
                raise ValueError(f'E7 pretrained artifact checksum mismatch: {filename}')
            names.append(path.relative_to(ROOT).as_posix())
    return {name: sha256(ROOT / name) for name in names}


class E7Trainer(Trainer):
    def __init__(self, revision, variant, verification=False):
        value = contract(revision, variant, verification)
        configure(value)
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError('HIP GPU required, CPU fallback forbidden')
        props = torch.cuda.get_device_properties(0)
        if props.name != 'AMD Radeon RX 6750 GRE 10GB' or props.gcnArchName.split(':')[0] != 'gfx1031':
            raise RuntimeError('Unexpected E7 GPU')
        self.backend = dict(gpu=props.name, arch=props.gcnArchName, vram=props.total_memory,
                            torch=torch.__version__, hip=torch.version.hip)
        random.seed(value['seed'])
        np.random.seed(value['seed'])
        torch.manual_seed(value['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        self.config = E7Config(loop_k=value['loop_k'])
        state, model_info = weights(revision)
        self.identity = identity(revision, model_info)
        self.model = Tinctura(value['loop_k'], value['seed']).cuda()
        load_weights(self.model, state)
        self.initial_hash = weight_hash(self.model)
        groups = [dict(params=[p for p in self.model.parameters() if p.ndim == 2], weight_decay=0.1),
                  dict(params=[p for p in self.model.parameters() if p.ndim != 2], weight_decay=0.0)]
        self.optimizer = torch.optim.AdamW(groups, lr=value['lr'], betas=tuple(value['betas']), foreach=False)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=value['init_scale'])
        self.data = PackedTokens(CORPUS / 'train.bin', 2048)
        self.counters = dict(iterations=0, effective=0, skipped=0, tokens=0, consecutive_skips=0)
        self.train_seconds, self.best_loss = 0.0, None
        self.history, self.evaluations = [], []
        self.tokenizer = Tokenizer.from_file(str(FINAL / 'tokenizer.json'))

    def step(self):
        value = EXPERIMENT
        remaining = value['token_budget'] - self.counters['tokens']
        if remaining <= 0 or self.counters['iterations'] >= value['iterations']:
            raise RuntimeError('E7 fixed token budget exhausted')
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr = value['lr'] * schedule(self.counters['effective'] + 1, value)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        planned = min(4096, remaining)
        total_loss, batches = 0.0, hashlib.sha256()
        for offset in range(0, planned, 2048):
            length = min(2048, planned - offset)
            self.data.context = length
            x, y = self.data.batch(1, 'cuda')
            self.data.context = 2048
            batches.update(x.cpu().numpy().tobytes())
            batches.update(y.cpu().numpy().tobytes())
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(self.model(x)[-1].float().flatten(0, 1), y.flatten())
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite E7 training loss')
            self.scaler.scale(loss * (length / planned)).backward()
            total_loss += loss.item() * length
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale
        self.counters['iterations'] += 1
        self.counters['tokens'] += planned
        self.counters['skipped' if skipped else 'effective'] += 1
        if skipped or not math.isfinite(norm):
            raise RuntimeError('E7 gate requires finite gradient and zero omitted updates')
        if not all(torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise RuntimeError('Nonfinite E7 weights')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.train_seconds += elapsed
        row = dict(**self.counters, loss=total_loss / planned, lr=lr, grad_norm=norm,
                   scale=self.scaler.get_scale(), batch_sha256=batches.hexdigest(), cursor=self.data.position,
                   update_tokens=planned, seconds=elapsed)
        self.history.append(row)
        return row

    def evaluate(self):
        start = time.perf_counter()
        tokens = np.memmap(CORPUS / 'validation.bin', dtype='<u2', mode='r')
        totals, count = [0.0] * self.config.loop_k, 0
        with isolated_eval(self.model):
            for position in range(0, len(tokens) - 1, 2048):
                chunk = torch.tensor(np.array(tokens[position:position + 2049], dtype=np.int64), device='cuda')[None]
                with torch.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(chunk[:, :-1])
                for loop, logits in enumerate(outputs):
                    totals[loop] += F.cross_entropy(logits.float().flatten(0, 1), chunk[:, 1:].flatten(), reduction='sum').item()
                count += chunk.shape[1] - 1
        if count != 65536 or not all(math.isfinite(n) for n in totals):
            raise RuntimeError('E7 heldout NLL/count invalid')
        row = dict(iteration=self.counters['iterations'], tokens=self.counters['tokens'], nll=totals[-1] / count,
                   loops_nll=[n / count for n in totals], evaluated_tokens=count, seconds=time.perf_counter() - start)
        self.evaluations.append(row)
        return row

    def generate(self):
        results = []
        with isolated_eval(self.model):
            for prompt in ['The water cycle begins when', 'To solve 47 + 68, first', 'A prime number is a number that']:
                ids = self.tokenizer.encode(prompt).ids
                for _ in range(32):
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = self.model(torch.tensor([ids[-2048:]], device='cuda'))[-1][0, -1]
                    token = int(logits.argmax().item())
                    ids.append(token)
                    if token == 0:
                        break
                results.append(dict(prompt=prompt, ids=ids, text=self.tokenizer.decode(ids)))
        return dict(method='greedy', max_new_tokens=32, rows=results)

    def arithmetic_probe(self):
        tasks = json.loads((CORPUS / 'arithmetic-probe.json').read_text())['tasks']
        results = []
        with isolated_eval(self.model):
            for task in tasks:
                scores = []
                for option in task['candidates']:
                    ids = torch.tensor([option['ids']], device='cuda')
                    start = option['answer_start']
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = self.model(ids[:, :-1])[-1].float()[:, start - 1:]
                    loss = F.cross_entropy(logits.flatten(0, 1), ids[:, start:].flatten()).item()
                    scores.append(-loss)
                predicted = task['candidates'][max(range(4), key=scores.__getitem__)]['value']
                results.append(dict(prompt=task['prompt'], answer=task['answer'], predicted=predicted,
                                    correct=predicted == task['answer'], normalized_scores=scores))
        return dict(tasks=len(tasks), accuracy=sum(r['correct'] for r in results) / len(results),
                    official_arithmark=False, scoring='mean_answer_log_likelihood', rows=results)


def restore(checkpoint, revision, variant, verification=False):
    trainer = E7Trainer(revision, variant, verification)
    trainer.restore(checkpoint)
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    return trainer, metadata


def main():
    parser = argparse.ArgumentParser(description='E7 fixed 2M tokens, unique resumable segments, HIP eager')
    parser.add_argument('--revision', choices=SEEDS, required=True)
    parser.add_argument('--variant', choices=['dense', 'looped'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--chunk-steps', type=int, default=100)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--verification', action='store_true')
    parser.add_argument('--recovery-check', type=Path)
    parser.add_argument('--model-check', type=Path)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'} or not 1 <= args.chunk_steps <= 100:
        parser.error('A unique single-directory run ID and chunk steps1..100 are required')
    if not args.verification:
        if not args.recovery_check or not args.model_check:
            parser.error('Principal requires passing recovery and model-equivalence evidence')
        recovery = json.loads(args.recovery_check.read_text())
        model_check = json.loads(args.model_check.read_text())
        for proof in [recovery, model_check]:
            if proof['status'] != 'passed' or proof['source_sha256'] != sha256(ROOT / 'src/tinctura_loop.py'):
                raise ValueError('E7 model proof source mismatch')
        if (recovery['revision'], recovery['variant'], recovery['entrypoint_sha256']) != (
                args.revision, args.variant, sha256(Path(__file__))):
            raise ValueError('E7 continuity proof revision/variant/entrypoint mismatch')
    run = ROOT / 'runs' / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    log = ROOT / 'logs/app' / (args.run_id + '.jsonl')
    log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log, encoding='utf-8')])
    start = time.perf_counter()
    source = None
    if args.resume:
        trainer, source = restore(args.resume.resolve(), args.revision, args.variant, args.verification)
    else:
        trainer = E7Trainer(args.revision, args.variant, args.verification)
    value = copy.deepcopy(EXPERIMENT)
    initial = copy.deepcopy(trainer.counters)
    target = args.stop_after or min(value['iterations'], initial['iterations'] + args.chunk_steps)
    if not initial['iterations'] < target <= value['iterations']:
        raise ValueError('E7 target outside fixed budget or exhausted')
    starting_hash, starting_seconds = weight_hash(trainer.model), trainer.train_seconds
    atomic_json(run / 'config.json', dict(experiment=value, identity=trainer.identity, backend=trainer.backend,
                source_checkpoint=str(args.resume.resolve()) if args.resume else None, start_counters=initial,
                model=vars(trainer.config)))
    validation = trainer.evaluate()
    if source and validation['nll'] != source['evaluations'][-1]['nll']:
        raise RuntimeError('E7 restored validation differs from source')
    trainer.best_loss = min(validation['nll'], trainer.best_loss) if trainer.best_loss is not None else validation['nll']
    trainer.save(run, ['initial', 'latest'])
    torch.cuda.reset_peak_memory_stats()
    try:
        while trainer.counters['iterations'] < target:
            row = trainer.step()
            logging.info(json.dumps(dict(event='train', **row), allow_nan=False))
            if row['iterations'] % 50 == 0 or row['iterations'] == target or (args.verification and row['iterations'] == 2):
                validation = trainer.evaluate()
                refs = ['latest', f"milestone-{row['tokens']}"]
                if validation['nll'] < trainer.best_loss:
                    trainer.best_loss = validation['nll']
                    refs.append('best')
                trainer.save(run, refs)
                logging.info(json.dumps(dict(event='validation', **validation)))
        if weight_hash(trainer.model) == starting_hash:
            raise RuntimeError('No real E7 weight update in segment')
        finished = trainer.counters['tokens'] == value['token_budget']
        if finished and not args.verification:
            atomic_json(run / 'generations-final.json', trainer.generate())
            atomic_json(run / 'arithmetic-final.json', trainer.arithmetic_probe())
        result = dict(status='verification' if args.verification else ('completed' if finished else 'segment_completed'),
                      run_id=args.run_id, experiment=value, config=vars(trainer.config), backend=trainer.backend,
                      **trainer.counters, train_seconds=trainer.train_seconds, wall_seconds=time.perf_counter() - start,
                      initial_weights_sha256=trainer.initial_hash, final_weights_sha256=weight_hash(trainer.model),
                      evaluations=trainer.evaluations, history=trainer.history,
                      source_checkpoint=str(args.resume.resolve()) if args.resume else None,
                      segment=dict(iterations=trainer.counters['iterations'] - initial['iterations'],
                                   tokens=trainer.counters['tokens'] - initial['tokens'],
                                   train_seconds=trainer.train_seconds - starting_seconds),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic_json(run / 'metrics.json', result)
        with (ROOT / 'runs/results.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({k: v for k, v in result.items() if k != 'history'}) + '\n')
        logging.info(json.dumps(dict(event='completed', run=str(run), status=result['status'], nll=validation['nll'],
                                     tokens=trainer.counters['tokens'], seconds=result['wall_seconds'])))
    except Exception as error:
        atomic_json(run / 'failure.json', dict(error_type=type(error).__name__, error=str(error), counters=trainer.counters))
        raise


if __name__ == '__main__':
    main()
