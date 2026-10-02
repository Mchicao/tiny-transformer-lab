import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts import tinctura_xsa_experiment as numeric
from scripts import tinctura_fp32_experiment as full_precision
from scripts.utils.check_training import compare, reference
from scripts.utils.sample_checkpoint import nucleus_probabilities
from src.training import atomic_json, isolated_eval, rng_state, weight_hash


def lineage(run):
    segments, current = [], run
    while current:
        metrics = json.loads((current / 'metrics.json').read_text())
        audit = json.loads((current / 'post-run-verification.json').read_text())
        assert audit['status'] == 'passed' and audit['training_updates'] == 0
        assert metrics['experiment']['e7_numeric_format'] in {2, 3} and metrics['skipped'] == 0
        segments.append((current, metrics, audit))
        source = metrics['source_checkpoint']
        current = Path(source).parents[1] if source else None
    segments.reverse()
    final = segments[-1][1]
    assert final['status'] == 'completed' and final['tokens'] == 2000000 and final['effective'] == 489
    assert len(final['history']) == 489 and final['history'][-1]['update_tokens'] == 1152
    assert sum(row['update_tokens'] for row in final['history']) == 2000000
    for i, (_, m, _) in enumerate(segments):
        assert m['experiment'] == final['experiment'] and m['backend'] == final['backend']
        assert m['history'] == final['history'][:m['iterations']]
        if i:
            assert m['iterations'] - m['segment']['iterations'] == segments[i - 1][1]['iterations']
    assert sum(m['segment']['tokens'] for _, m, _ in segments) == 2000000
    return segments


def sampled_generations(trainer):
    rows = []
    with isolated_eval(trainer.model):
        for prompt in ['The water cycle begins when', 'To solve 47 + 68, first', 'A prime number is a number that']:
            generator = torch.Generator(device='cuda').manual_seed(42)
            ids = trainer.tokenizer.encode(prompt).ids
            for _ in range(32):
                with torch.autocast('cuda', dtype=torch.float16, enabled=trainer.config.precision != 'float32_eager'):
                    logits = trainer.model(torch.tensor([ids[-2048:]], device='cuda'))[-1][0, -1]
                probabilities, indices = nucleus_probabilities(logits, 0.8, 0.9)
                token = int(indices[torch.multinomial(probabilities, 1, generator=generator)].item())
                ids.append(token)
                if token == 0:
                    break
            rows.append(dict(prompt=prompt, ids=ids, text=trainer.tokenizer.decode(ids)))
    return dict(temperature=0.8, top_p=0.9, seed_per_prompt=42, max_new_tokens=32, rows=rows)


def main():
    parser = argparse.ArgumentParser(description='Report audited E7-v2 lineages at exact2M tokens, with paired controls')
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    cases, histories = {}, {}
    numeric_versions = set()
    for path in args.runs:
        run = path.resolve()
        segments = lineage(run)
        metrics = segments[-1][1]
        value = metrics['experiment']
        numeric_versions.add(value['e7_numeric_format'])
        assert len(numeric_versions) == 1, 'Mixed numerical versions in a comparison'
        loader = full_precision if value['e7_numeric_format'] == 3 else numeric
        revision, variant = value['revision'], value['variant']
        key = revision + '-' + variant
        assert key not in cases
        trainer, _ = loader.restore(reference(run, 'latest'), revision, variant)
        before = (weight_hash(trainer.model), copy.deepcopy(trainer.counters), trainer.data.position, rng_state(),
                  copy.deepcopy(trainer.optimizer.state_dict()), copy.deepcopy(trainer.scaler.state_dict()), trainer.model.training)
        samples = sampled_generations(trainer)
        assert samples == sampled_generations(trainer)
        compare(before, (weight_hash(trainer.model), trainer.counters, trainer.data.position, rng_state(),
                        trainer.optimizer.state_dict(), trainer.scaler.state_dict(), trainer.model.training), exact=True)
        atomic_json(output / (key + '-sampling.json'), dict(**samples, training_updates=0, deterministic_repeat_exact=True))
        arithmetic = json.loads((run / 'arithmetic-final.json').read_text())
        final_eval = metrics['evaluations'][-1]
        del trainer, before
        torch.cuda.empty_cache()
        initial_trainer, _ = loader.restore(reference(segments[0][0], 'initial'), revision, variant)
        initial_before = (weight_hash(initial_trainer.model), copy.deepcopy(initial_trainer.counters),
                          initial_trainer.data.position, rng_state())
        initial_arithmetic = initial_trainer.arithmetic_probe()
        compare(initial_before, (weight_hash(initial_trainer.model), initial_trainer.counters,
                                initial_trainer.data.position, rng_state()), exact=True)
        atomic_json(output / (key + '-arithmetic-initial.json'), dict(**initial_arithmetic, training_updates=0))
        del initial_trainer, initial_before
        torch.cuda.empty_cache()
        histories[key] = metrics['history']
        cases[key] = dict(revision=revision, variant=variant, seed=value['seed'],
                          final_run=run.relative_to(ROOT).as_posix(),
                          segments=[p.relative_to(ROOT).as_posix() for p, _, _ in segments],
                          tokens=metrics['tokens'], final_nll=final_eval['nll'], loops_nll=final_eval['loops_nll'],
                          initial_nll=next(r['nll'] for r in metrics['evaluations'] if r['iteration'] == 0),
                          train_seconds=metrics['train_seconds'], tokens_s=2000000 / metrics['train_seconds'],
                          wall_seconds=sum(m['wall_seconds'] for _, m, _ in segments),
                          peak_mib=max(m['peak_allocated_bytes'] for _, m, _ in segments) / 2**20,
                          checkpoints_audited=sum(a['checkpoints_restored'] for _, _, a in segments),
                          nominal_layer_passes_per_token=18 * value['loop_k'],
                          layer_pass_tokens=2000000 * 18 * value['loop_k'],
                          arithmetic_accuracy=arithmetic['accuracy'], arithmetic_tasks=arithmetic['tasks'])
        cases[key]['initial_arithmetic_accuracy'] = initial_arithmetic['accuracy']
        cases[key]['nll_delta_vs_own_initial'] = cases[key]['final_nll'] - cases[key]['initial_nll']
    pairs = {}
    for revision in ['early', 'middle', 'final']:
        keys = [revision + '-' + arm for arm in ['dense', 'looped']]
        if not all(k in cases for k in keys):
            continue
        dense, looped = [cases[k] for k in keys]
        for a, b in zip(histories[keys[0]], histories[keys[1]], strict=True):
            for field in ['batch_sha256', 'cursor', 'lr', 'scale', 'tokens', 'update_tokens']:
                assert a[field] == b[field], (revision, a['iterations'], field)
        pairs[revision] = dict(delta_nats=looped['final_nll'] - dense['final_nll'],
                               arithmetic_delta=looped['arithmetic_accuracy'] - dense['arithmetic_accuracy'],
                               train_time_ratio=looped['train_seconds'] / dense['train_seconds'],
                               peak_mib_delta=looped['peak_mib'] - dense['peak_mib'],
                               all_489_batches_cursor_lr_scale_paired=True)
    complete = set(cases) == {r + '-' + a for r in ['early', 'middle', 'final'] for a in ['dense', 'looped']}
    result = dict(status='completed' if complete else 'partial_campaign', numeric_version=next(iter(numeric_versions)), cases=cases, pairs=pairs,
                  exploratory_wins=sum(p['delta_nats'] < 0 for p in pairs.values()),
                  maturity_interaction_causal=False, compute_matched=False, training_updates_during_report=0)
    atomic_json(output / 'comparison.json', result)
    lines = [f'# E7-v{result["numeric_version"]}: comparación a2M tokens por linaje', '',
             'XSA FP32 común a ambos brazos; todos los segmentos incluidos están auditados. ' 
             'Presupuesto igualado por tokens, no por cómputo. Madurez y seed no cruzadas.', '',
             '| Revisión | Brazo | NLL inicial | NLL final | Aritmética | Bucle train (s) | Pico MiB |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for revision in ['early', 'middle', 'final']:
        for variant in ['dense', 'looped']:
            if (key := revision + '-' + variant) in cases:
                c = cases[key]
                lines.append(f'| {revision} | {variant} | {c["initial_nll"]:.6f} | {c["final_nll"]:.6f} | {100*c["arithmetic_accuracy"]:.2f}% | {c["train_seconds"]:.2f} | {c["peak_mib"]:.2f} |')
    lines += ['', '## Diferencias pareadas', '']
    for revision, p in pairs.items():
        lines.append(f'- {revision}: looped−denso **{p["delta_nats"]:+.6f} nats**, aritmética **{100*p["arithmetic_delta"]:+.2f} puntos**, ' 
                     f'tiempo **{p["train_time_ratio"]:.2f}×**, VRAM **{p["peak_mib_delta"]:+.2f} MiB**. ' 
                     'Los489 batches, cursores, LR y escalas coinciden.')
    lines += ['', f'Estado: `{result["status"]}`. Ganancias exploratorias: {result["exploratory_wins"]}/{len(pairs)} revisiones completas. ' 
              'Una ganancia requiere réplica; no es adopción ni significancia. La sonda propia de64 tareas no es ArithMark oficial. ' 
              'No comparar estas NLL/tokenizer con TinyStories3M.', '',
              '## Cambio frente al propio checkpoint inicial', '']
    for key, c in cases.items():
        lines.append(f'- {key}: Δ NLL **{c["nll_delta_vs_own_initial"]:+.6f}**, aritmética inicial/final ' 
                     f'**{100*c["initial_arithmetic_accuracy"]:.2f}%/{100*c["arithmetic_accuracy"]:.2f}%**. ' 
                     'Ganar a un control degradado no demuestra superar al preentrenado original.')
    lines += ['',
              '## Procedencia', '', 'Los intentos descartados por overflow y sus replays son sobrecoste operativo y no entran en esta matriz. ' 
              'Sampling T0,8/top-p0,9 con repetición exacta y estado intacto está junto a este reporte. ' 
              'Generaciones greedy y sonda aritmética están en cada run final.', '']
    lines += [f'- `{c["final_run"]}` ({len(c["segments"])} segmentos, {c["checkpoints_audited"]} checkpoints auditados).' for c in cases.values()]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], pairs=pairs,
                         cases={k: {f: c[f] for f in ['final_nll', 'train_seconds', 'peak_mib', 'arithmetic_accuracy']} for k, c in cases.items()})))


if __name__ == '__main__':
    main()
