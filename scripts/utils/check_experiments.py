import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from scripts.experiment import MuonAdamW, fresh, restore
from scripts.utils.check_training import compare, reference, RTOL, ATOL
from scripts.utils.sample_checkpoint import sample
from src.training import atomic_json, rng_state, restore_rng, sha256, weight_hash, verify_checkpoint


def amp_check():
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(8, 4)
    model.hidden = torch.nn.Linear(4, 4, bias=False)
    model.norm = torch.nn.LayerNorm(4)
    model.cuda()
    optimizer = MuonAdamW(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    for group in optimizer.param_groups:
        group['lr'] = 0.00003
    before = [p.detach().clone() for p in model.parameters()]
    scaler.scale(sum(p.square().sum() for p in model.parameters())).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    changed = [not torch.equal(old, p) for old, p in zip(before, model.parameters(), strict=True)]
    assert changed[0] and changed[1]
    state = copy.deepcopy(optimizer.state_dict())
    reloaded = MuonAdamW(model)
    reloaded.load_state_dict(state)
    compare(reloaded.state_dict(), state, exact=True)
    optimizer = reloaded
    for group in optimizer.param_groups:
        group['lr'] = 0.0
    optimizer.zero_grad(set_to_none=True)
    before = [p.detach().clone() for p in model.parameters()]
    scaler.scale(sum(p.square().sum() for p in model.parameters())).backward()
    scaler.step(optimizer)
    scaler.update()
    assert all(torch.equal(old, p) for old, p in zip(before, model.parameters(), strict=True))
    for parameter in [model.hidden.weight, model.embedding.weight]:
        optimizer.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in model.parameters()]
        state = copy.deepcopy(optimizer.state_dict())
        scale = scaler.get_scale()
        scaler.scale(sum(p.square().sum() for p in model.parameters())).backward()
        parameter.grad.flatten()[0] = float('inf')
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() == scale / 2
        assert all(torch.equal(old, p) for old, p in zip(before, model.parameters(), strict=True))
        compare(optimizer.state_dict(), state, exact=True)
    return dict(synthetic_updates=2, synthetic_skips=2, transformer_updates=0,
                both_branches_skip_atomically=True, restored_lr_applies_to_both=True)


def execute(identifier, variant, target, checkpoint=None):
    command = [sys.executable, str(ROOT / 'scripts/experiment.py'), '--variant', variant,
               '--run-id', identifier, '--iterations', '1960', '--stop-after', str(target)]
    if checkpoint:
        command += ['--resume', str(checkpoint)]
    log = ROOT / 'logs/tests' / (identifier + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return ROOT / 'runs' / identifier


def audit(run):
    refs = json.loads((run / 'references.json').read_text())
    checkpoints = list((run / 'checkpoints').glob('step-*'))
    for checkpoint in checkpoints:
        trainer, metadata = restore(checkpoint)
        state = torch.load(checkpoint / 'state.pt', weights_only=True, map_location='cpu')
        compare(trainer.optimizer.state_dict(), state['optimizer'], exact=True)
        compare(trainer.scaler.state_dict(), state['scaler'], exact=True)
        compare(rng_state(), state['rng'], exact=True)
        assert trainer.counters == metadata['counters'] and trainer.data.position == metadata['cursor']
    assert all((run / 'checkpoints' / name) in checkpoints for name in refs.values())
    assert not list((run / 'checkpoints').glob('.pending-*'))
    trainer, metadata = restore(reference(run, 'latest'))
    before = dict(weights=weight_hash(trainer.model), rng=rng_state(), cursor=trainer.data.position,
                  counters=copy.deepcopy(trainer.counters), optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
                  scaler=copy.deepcopy(trainer.scaler.state_dict()), mode=trainer.model.training)
    validation = trainer.evaluate()
    assert validation['nll'] == metadata['evaluations'][-1]['nll']
    assert trainer.generate() == json.loads((run / 'generations-final.json').read_text())
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/processed/tokenizer.json'))
    samples = sample(trainer.model, tokenizer)
    assert samples == json.loads((run / 'sampling-final.json').read_text())
    compare(before, dict(weights=weight_hash(trainer.model), rng=rng_state(), cursor=trainer.data.position,
                         counters=trainer.counters, optimizer=trainer.optimizer.state_dict(),
                         scaler=trainer.scaler.state_dict(), mode=trainer.model.training), exact=True)
    atomic_json(run / 'post-run-verification.json', dict(status='passed', checkpoints_restored=len(checkpoints),
                references_verified=True, optimizer_scaler_rng_exact=True, validation_exact=True,
                greedy_and_sampling_exact=True, evaluation_generation_isolated=True, training_updates=0))


def main():
    parser = argparse.ArgumentParser(description='HIP recovery checks: 10 Transformer updates per new variant')
    parser.add_argument('--variant', choices=['muon', 'muon-qk'])
    parser.add_argument('--check-id')
    parser.add_argument('--verify-existing', action='store_true')
    parser.add_argument('--audit-run', type=Path)
    args = parser.parse_args()
    if args.audit_run:
        audit(args.audit_run.resolve())
        return
    if not args.variant:
        parser.error('--variant is required for continuity checks')
    identifier = args.check_id or (args.variant + '-check-' + uuid.uuid4().hex[:12])
    output = ROOT / 'output' / identifier
    if not args.verify_existing:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / 'amp.json', amp_check())
        continuous = execute(identifier + '-continuous', args.variant, 5)
        interrupted = execute(identifier + '-interrupted', args.variant, 2)
        resumed = execute(identifier + '-resumed', args.variant, 5, reference(interrupted, 'latest'))
    else:
        continuous, interrupted, resumed = [ROOT / 'runs' / (identifier + '-' + suffix)
                                            for suffix in ['continuous', 'interrupted', 'resumed']]
    boundary = reference(interrupted, 'latest')
    bitwise = []
    for first, second in [(reference(continuous, 'boundary'), boundary),
                          (reference(continuous, 'latest'), reference(resumed, 'latest'))]:
        trainer, left = restore(first)
        _, right = restore(second)
        compare(load_file(str(first / 'model.safetensors')), load_file(str(second / 'model.safetensors')))
        compare(torch.load(first / 'state.pt', weights_only=True, map_location='cpu'),
                torch.load(second / 'state.pt', weights_only=True, map_location='cpu'))
        compare(left['counters'], right['counters'], exact=True)
        assert left['cursor'] == right['cursor']
        for a, b in zip(left['history'], right['history'], strict=True):
            compare({k: v for k, v in a.items() if k != 'seconds'},
                    {k: v for k, v in b.items() if k != 'seconds'}, exact=True)
        bitwise.append(left['weights_sha256'] == right['weights_sha256'])
    trainer, _ = restore(boundary)
    saved, position = rng_state(), trainer.data.position
    expected_batch = trainer.data.batch(4, 'cpu')
    draws = [random.random(), np.random.random(), torch.rand(4), torch.rand(4, device='cuda').cpu()]
    trainer, _ = restore(boundary)
    compare(expected_batch, trainer.data.batch(4, 'cpu'), exact=True)
    compare(draws, [random.random(), np.random.random(), torch.rand(4), torch.rand(4, device='cuda').cpu()], exact=True)
    restore_rng(saved)
    trainer.data.position = position
    mode = trainer.model.training
    trainer.evaluate()
    trainer.generate()
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/processed/tokenizer.json'))
    samples = sample(trainer.model, tokenizer)
    assert samples == sample(trainer.model, tokenizer)
    compare(saved, rng_state(), exact=True)
    assert trainer.data.position == position and trainer.model.training == mode
    wrong = copy.deepcopy(trainer.identity)
    wrong['configs/3m.json'] = 'incompatible'
    try:
        verify_checkpoint(boundary, wrong, trainer.backend)
    except ValueError:
        pass
    else:
        raise AssertionError('Incompatible identity accepted')
    damaged = output / ('corrupt-checkpoint-' + uuid.uuid4().hex[:8])
    shutil.copytree(boundary, damaged)
    with (damaged / 'state.pt').open('r+b') as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    try:
        restore(damaged)
    except ValueError as error:
        assert 'checksum mismatch' in str(error)
    else:
        raise AssertionError('Corrupt checkpoint accepted')
    assert trainer.initial_hash != weight_hash(trainer.model)
    atomic_json(output / 'recovery.json', dict(status='passed', variant=args.variant,
                tolerance=dict(rtol=RTOL, atol=ATOL), bitwise_weights_equal=all(bitwise),
                updates_executed=10, tokens_executed=40960, comparison='5 vs 2 + 3 in new processes',
                batches_counters_cursor_rng_exact=True, optimizer_scaler_verified=True,
                weights_updated=True, evaluation_generation_isolated=True, sampling_repeat_exact=True,
                corruption_rejected=True, incompatible_identity_rejected=True,
                runs=[str(run.relative_to(ROOT)) for run in [continuous, interrupted, resumed]]))
    print(json.dumps(dict(status='passed', evidence=str(output / 'recovery.json'))))


if __name__ == '__main__':
    main()
