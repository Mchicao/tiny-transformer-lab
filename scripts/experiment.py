import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from scripts.train import SEED43_EXPERIMENT, make_trainer, restore_trainer as restore_legacy, curve
from scripts.utils.sample_checkpoint import sample
from src.training import EXPERIMENT, atomic_json, sha256, weight_hash

INITIALIZATION = ROOT / 'output/initializations/3m-seed43-0f4e40f4b755'
SOURCE = ROOT / 'runs/adamw-3m-20261001-003-seed43/checkpoints/step-0490-28c7e446'
VARIANTS = ['adamw', 'muon', 'muon-qk']


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, model):
        hidden = [p for name, p in model.named_parameters() if p.ndim == 2 and name != 'embedding.weight']
        embedding = [model.embedding.weight]
        norms = [p for p in model.parameters() if p.ndim != 2]
        self.muon = torch.optim.Muon(hidden, lr=3e-4, weight_decay=0.1, momentum=0.95,
                                    nesterov=True, ns_steps=5, adjust_lr_fn='match_rms_adamw')
        self.adamw = torch.optim.AdamW([dict(params=embedding, weight_decay=0.1),
                                       dict(params=norms, weight_decay=0.0)],
                                      lr=3e-4, betas=(0.9, 0.95), foreach=False)
        super().__init__([*self.muon.param_groups, *self.adamw.param_groups], {})
        parameters = [p for group in self.param_groups for p in group['params']]
        if len(parameters) != len(set(parameters)) or set(parameters) != set(model.parameters()):
            raise ValueError('Optimizer groups must partition model parameters')

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are not supported in the AMP training loop')
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return dict(format=1, muon=self.muon.state_dict(), adamw=self.adamw.state_dict())

    def load_state_dict(self, state):
        if state['format'] != 1 or set(state) != {'format', 'muon', 'adamw'}:
            raise ValueError('Incompatible combined optimizer state')
        self.muon.load_state_dict(state['muon'])
        self.adamw.load_state_dict(state['adamw'])
        self.param_groups = [*self.muon.param_groups, *self.adamw.param_groups]


def contract(variant, iterations=980):
    if variant not in VARIANTS or type(iterations) is not int or not 5 <= iterations <= 1960:
        raise ValueError('Unsupported variant or bounded token budget')
    milestones = sorted(set(SEED43_EXPERIMENT['milestones'] + list(range(50, iterations, 50)) + [iterations]))
    return {**copy.deepcopy(SEED43_EXPERIMENT), 'iterations': iterations,
            'milestones': [value for value in milestones if value <= iterations],
            'campaign_format': 1, 'variant': variant, 'qk_norm': variant == 'muon-qk',
            'muon': None if variant == 'adamw' else dict(momentum=0.95, nesterov=True, ns_steps=5,
                ns_coefficients=[3.4445, -4.775, 2.0315], eps=1e-7, adjust_lr_fn='match_rms_adamw')}


def configure(value):
    # ponytail: one contract per process; concurrent runs require separate processes.
    EXPERIMENT.clear()
    EXPERIMENT.update(copy.deepcopy(value))


def campaign_identity(trainer):
    trainer.identity['scripts/experiment.py'] = sha256(Path(__file__))


def fresh(variant, iterations=980):
    configure(SEED43_EXPERIMENT)
    trainer = make_trainer(43, INITIALIZATION)
    if variant != 'adamw':
        trainer.optimizer = MuonAdamW(trainer.model)
    trainer.config.qk_norm = variant == 'muon-qk'
    configure(contract(variant, iterations))
    campaign_identity(trainer)
    return trainer


def restore(checkpoint):
    checksums = json.loads((checkpoint / 'checksums.json').read_text())
    if set(checksums) != {'model.safetensors', 'state.pt', 'metadata.json'}:
        raise ValueError('Incomplete checkpoint manifest')
    for name, expected in checksums.items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    value = metadata['experiment']
    if value != contract(value['variant'], value['iterations']):
        raise ValueError('Checkpoint experiment mismatch')
    trainer = fresh(value['variant'], value['iterations'])
    trainer.restore(checkpoint)
    return trainer, metadata


def plateau(evaluations):
    if len(evaluations) < 5:
        return False
    previous_best = min(row['nll'] for row in evaluations[:-4])
    recent_best = min(row['nll'] for row in evaluations[-4:])
    return previous_best - recent_best < 0.01


def main():
    parser = argparse.ArgumentParser(description='Campaña autorizada 3M, HIP eager, seed43')
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--iterations', type=int, default=980)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--continue-baseline', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--plateau-stop', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('Invalid run ID')
    if args.continue_baseline and (args.variant != 'adamw' or args.resume):
        raise ValueError('Baseline continuation requires AdamW and its legacy source')
    if args.resume:
        trainer, source = restore(args.resume.resolve())
        if EXPERIMENT['variant'] != args.variant or EXPERIMENT['iterations'] != args.iterations:
            raise ValueError('Resume arguments differ from saved contract')
    elif args.continue_baseline:
        trainer, source = restore_legacy(SOURCE)
        configure(contract('adamw', args.iterations))
        campaign_identity(trainer)
    else:
        trainer, source = fresh(args.variant, args.iterations), None
    if args.stop_after is not None and not trainer.counters['iterations'] < args.stop_after <= args.iterations:
        raise ValueError('Stop iteration must follow the cursor and remain inside the budget')
    target = args.stop_after or args.iterations
    if trainer.counters['iterations'] >= target:
        raise ValueError('No authorized updates remain')
    run = ROOT / 'runs' / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    log = ROOT / 'logs/app' / (args.run_id + '.jsonl')
    log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log, encoding='utf-8')])
    start = time.perf_counter()
    initial = copy.deepcopy(trainer.counters)
    starting_seconds = trainer.train_seconds
    starting_hash = weight_hash(trainer.model)
    atomic_json(run / 'config.json', dict(model=vars(trainer.config), experiment=EXPERIMENT,
                identity=trainer.identity, backend=trainer.backend, start_counters=initial,
                source_checkpoint=str(args.resume.resolve()) if args.resume else (str(SOURCE) if args.continue_baseline else None)))
    validation = trainer.evaluate()
    if source and source['evaluations'][-1]['iteration'] == trainer.counters['iterations'] and validation['nll'] != source['evaluations'][-1]['nll']:
        raise RuntimeError('Validation at restore differs from source')
    trainer.best_loss = min(validation['nll'], trainer.best_loss) if trainer.best_loss is not None else validation['nll']
    trainer.save(run, ['latest', 'best', 'initial'])
    diagnostics = args.stop_after is None
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/processed/tokenizer.json'))
    if diagnostics:
        atomic_json(run / 'generations-initial.json', trainer.generate())
        atomic_json(run / 'sampling-initial.json', sample(trainer.model, tokenizer))
    torch.cuda.reset_peak_memory_stats()
    stopped_on_plateau = False
    try:
        while trainer.counters['iterations'] < target:
            row = trainer.step()
            logging.info(json.dumps(dict(event='train', **row), allow_nan=False))
            if row['iterations'] in EXPERIMENT['milestones'] or row['iterations'] == target:
                validation = trainer.evaluate()
                logging.info(json.dumps(dict(event='validation', **validation)))
                refs = ['latest', f"milestone-{row['tokens']}"]
                if validation['nll'] < trainer.best_loss:
                    trainer.best_loss = validation['nll']
                    refs.append('best')
                trainer.save(run, refs)
                unique_evaluations = list({r['iteration']: r for r in trainer.evaluations}.values())
                if args.plateau_stop and plateau(unique_evaluations):
                    stopped_on_plateau = True
                    break
            elif args.stop_after and row['iterations'] == 2:
                trainer.save(run, ['boundary', 'latest'])
        final_hash = weight_hash(trainer.model)
        if final_hash == starting_hash or trainer.counters['effective'] == initial['effective']:
            raise RuntimeError('No real weight update')
        if diagnostics:
            atomic_json(run / 'generations-final.json', trainer.generate())
            atomic_json(run / 'sampling-final.json', sample(trainer.model, tokenizer))
        segment = {key: trainer.counters[key] - initial[key] for key in ['iterations', 'effective', 'skipped', 'tokens']}
        segment['train_seconds'] = trainer.train_seconds - starting_seconds
        segment['tokens_s'] = segment['tokens'] / segment['train_seconds']
        result = dict(status='plateau' if stopped_on_plateau else ('verification' if args.stop_after else 'completed'),
                      run_id=args.run_id, variant=args.variant, experiment=EXPERIMENT, config=vars(trainer.config),
                      backend=trainer.backend, **trainer.counters, segment=segment,
                      train_seconds=trainer.train_seconds, wall_seconds=time.perf_counter() - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      initial_weights_sha256=trainer.initial_hash, starting_weights_sha256=starting_hash,
                      final_weights_sha256=final_hash, evaluations=trainer.evaluations, history=trainer.history,
                      plateau_reached=stopped_on_plateau, plateau_policy=dict(min_delta=0.01, evaluations=4))
        atomic_json(run / 'metrics.json', result)
        curve(run, trainer.evaluations)
        chart = run / 'validation.svg'
        chart.write_text(chart.read_text(encoding='utf-8').replace('AdamW 3M', args.variant + ' 3M'), encoding='utf-8')
        with (ROOT / 'runs/results.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({k: v for k, v in result.items() if k != 'history'}) + '\n')
        logging.info(json.dumps(dict(event='completed', run=str(run), nll=validation['nll'], segment=segment)))
    except Exception as error:
        atomic_json(run / 'failure.json', dict(error_type=type(error).__name__, error=str(error), counters=trainer.counters))
        raise


if __name__ == '__main__':
    main()
