import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training import atomic_json, sha256

PREVIOUS = {
    42: ['adamw-3m-20261001-007-seed42-replica', 'adamw-swiglu-3m-20261001-009-seed42'],
    43: ['adamw-3m-20261001-004-convergence', 'adamw-swiglu-3m-20261001-010-seed43'],
}


def main():
    parser = argparse.ArgumentParser(description='Reporte pareado de tres seeds; no actualiza pesos')
    parser.add_argument('--gelu-run', type=Path, required=True)
    parser.add_argument('--swiglu-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    paths = {seed: [ROOT / 'runs' / name for name in names] for seed, names in PREVIOUS.items()}
    paths[44] = [args.gelu_run.resolve(), args.swiglu_run.resolve()]
    summaries, curves, samples, pairs = [], [], [], []
    common_keys = ['microbatch', 'accumulation', 'lr', 'betas', 'weight_decay', 'warmup', 'clip', 'init_scale']
    common = backend = None
    for seed, runs in paths.items():
        metrics = [json.loads((run / 'metrics.json').read_text()) for run in runs]
        a, b = metrics
        assert {k: v for k, v in b['config'].items() if k not in {'ffn', 'ffn_hidden'}} == a['config']
        assert b['config']['ffn'] == 'swiglu' and b['config']['ffn_hidden'] == 512
        trajectory = ['iterations', 'tokens', 'batch_sha256', 'cursor', 'lr', 'scale']
        assert [{k: row[k] for k in trajectory} for row in a['history']] == [{k: row[k] for k in trajectory} for row in b['history']]
        current = []
        for ffn, run, m in zip(['gelu', 'swiglu'], runs, metrics, strict=True):
            assert m['status'] in {'completed', 'plateau'} and m['experiment']['seed'] == seed and m['variant'] == 'adamw'
            assert m['tokens'] == 6144000 and m['effective'] == 1500 and m['skipped'] == 0
            if common is None:
                common, backend = {k: m['experiment'][k] for k in common_keys}, m['backend']
            assert common == {k: m['experiment'][k] for k in common_keys} and backend == m['backend']
            audit = json.loads((run / 'post-run-verification.json').read_text())
            assert audit['status'] == 'passed' and audit['training_updates'] == 0
            evaluations = list({e['iteration']: e for e in m['evaluations']}.values())
            assert all(e['evaluated_tokens'] == 39275 for e in evaluations)
            best, final = min(evaluations, key=lambda e: e['nll']), evaluations[-1]
            refs = json.loads((run / 'references.json').read_text())
            info = dict(seed=seed, ffn=ffn, run_id=m['run_id'], initial_nll=evaluations[0]['nll'],
                        final_nll=final['nll'], best_nll=best['nll'], best_tokens=best['tokens'],
                        final_perplexity=math.exp(final['nll']), train_seconds=m['train_seconds'],
                        wall_seconds=m['wall_seconds'], tokens_s=m['tokens']/m['train_seconds'],
                        peak_allocated_mib=m['peak_allocated_bytes']/2**20,
                        peak_reserved_mib=m['peak_reserved_bytes']/2**20,
                        checkpoints=audit['checkpoints_restored'], latest=refs['latest'], best=refs['best'])
            summaries.append(info)
            current.append(info)
            curves.extend(dict(seed=seed, ffn=ffn, **{k: e[k] for k in ['iteration', 'tokens', 'nll']}) for e in evaluations)
            samples.append(dict(seed=seed, ffn=ffn, greedy=json.loads((run/'generations-final.json').read_text()),
                                sampling=json.loads((run/'sampling-final.json').read_text())))
            if seed == 44:
                initialization = ROOT / 'output/initializations/3m-ffn-seed44-3f197002' / ffn
                initial = json.loads((initialization / 'initialization.json').read_text())
                assert m['starting_weights_sha256'] == m['initial_weights_sha256'] == initial['weights_sha256']
                recovery = json.loads((ROOT/f'output/ffn-{ffn}-seed44-check-20261001-3f197002/recovery.json').read_text())
                assert recovery['status'] == 'passed' and recovery['updates_executed'] == 10 and recovery['tokens_executed'] == 40960
        pairs.append(dict(seed=seed, final_swiglu_minus_gelu=current[1]['final_nll']-current[0]['final_nll'],
                          best_swiglu_minus_gelu=current[1]['best_nll']-current[0]['best_nll']))
    manifest = json.loads((output / 'existing-artifacts.json').read_text())
    assert all(sha256(ROOT / name) == digest for name, digest in manifest.items())
    prefix = json.loads((output / 'preservation-prefix.json').read_text())
    raw = (ROOT/'runs/results.jsonl').read_bytes()[:prefix['results_prefix_size']]
    assert hashlib.sha256(raw).hexdigest() == prefix['results_prefix_sha256']
    result = dict(status='passed', models=summaries, paired_seeds=pairs,
                  mean_final_gap=sum(p['final_swiglu_minus_gelu'] for p in pairs)/3,
                  mean_best_gap=sum(p['best_swiglu_minus_gelu'] for p in pairs)/3,
                  statistical_significance_tested=False, parameters_each=3000384,
                  equal_batches_cursor_lr_scale=True, common_hyperparameters_backend=True,
                  new_main_tokens=12288000, new_main_updates=3000, verification_tokens=81920,
                  verification_updates=20, historical_files_preserved=len(manifest), results_prefix_preserved=True)
    atomic_json(output/'comparison.json', result)
    atomic_json(output/'generations.json', dict(temperature=.8, top_p=.9, seed_per_prompt=42,
                                               max_new_tokens=64, models=samples))
    with (output/'curve.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['seed', 'ffn', 'iteration', 'tokens', 'nll'])
        writer.writeheader()
        writer.writerows(curves)
    chart = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="850"><rect width="1000" height="850" fill="white"/><g font-family="sans-serif" font-size="15" fill="#222"><text x="80" y="25">AdamW 3M; NLL por target — GELU azul / SwiGLU rojo; zoom 3–4,5</text>']
    for panel, seed in enumerate(paths):
        top = 65 + panel*255
        chart.append(f'<text x="80" y="{top-10}">Seed {seed}; valores superiores a 4,5 recortados</text>')
        for value in [3, 3.5, 4, 4.5]:
            y = top + (4.5-value)/1.5*185
            chart.append(f'<path d="M80 {y} H950" stroke="#ddd"/><text x="25" y="{y+5}">{value:.1f}</text>')
        for fraction in [0,.25,.5,.75,1]:
            chart.append(f'<text x="{80+fraction*870}" y="{top+207}" text-anchor="middle">{fraction*6.144:.3f}M</text>')
        for ffn, color in [('gelu','#1565c0'), ('swiglu','#c62828')]:
            points = ' '.join(f"{80+e['tokens']/6144000*870:.2f},{top+(4.5-min(4.5,max(3,e['nll'])))/1.5*185:.2f}" for e in curves if e['seed']==seed and e['ffn']==ffn)
            chart.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    chart.append('<text x="400" y="830">Tokens de entrenamiento</text></g></svg>')
    (output/'comparison.svg').write_text(''.join(chart), encoding='utf-8')
    ET.parse(output/'comparison.svg')
    sources = ['src/model.py','src/data.py','src/training.py','src/swiglu.py','configs/3m.json',
               'configs/3m-gelu-seed44.json','configs/3m-swiglu-seed44.json','scripts/train.py',
               'scripts/experiment.py','scripts/ffn_seed44.py','scripts/utils/check_training.py',
               'scripts/utils/check_experiments.py','scripts/utils/check_ffn.py','scripts/ffn_experiment.py',
               'scripts/utils/check_ffn_seed44.py','scripts/utils/sample_checkpoint.py','scripts/utils/report_ffn_seed44.py']
    for run in paths[44]:
        destination = run/'sources'
        destination.mkdir(exist_ok=False)
        for name in sources:
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        atomic_json(destination/'checksums.json', {name: sha256(destination/name) for name in sources})
    table = '\n'.join(f"| {s['seed']} | {s['ffn']} | {s['initial_nll']:.6f} | {s['final_nll']:.6f} | {s['best_nll']:.6f} | {s['final_perplexity']:.3f} |" for s in summaries)
    gaps = '\n'.join(f"| {p['seed']} | {p['final_swiglu_minus_gelu']:+.6f} | {p['best_swiglu_minus_gelu']:+.6f} |" for p in pairs)
    costs = '\n'.join(f"| {s['ffn']} | {s['train_seconds']:.2f} | {s['wall_seconds']:.2f} | {s['tokens_s']:.0f} | {s['peak_allocated_mib']:.1f}/{s['peak_reserved_mib']:.1f} | {s['checkpoints']} |" for s in summaries if s['seed']==44)
    checkpoints = '\n'.join(f"- {s['ffn']}: latest `{s['latest']}`, best `{s['best']}` a {s['best_tokens']:,} tokens; [auditoría](../../runs/{s['run_id']}/post-run-verification.json)." for s in summaries if s['seed']==44)
    diagnosis = '\n\n'.join(f"**{group['ffn']} — {row['prompt']}**\n\n> {row['text'].replace(chr(10), ' ')}" for group in samples if group['seed']==44 for row in group['sampling'])
    report = f'''# GELU / SwiGLU, tercera réplica seed44 — 2026-10-01

Se completaron ambos modelos desde pesos aleatorios propios seed44. En seed44 GELU obtuvo menor NLL final y del mejor checkpoint; SwiGLU empeoró después de su mínimo a 1.200 pasos. Media de diferencias SwiGLU−GELU entre seeds42/43/44: final **{result['mean_final_gap']:+.6f}**, mejor checkpoint **{result['mean_best_gap']:+.6f} nats/target**. Menor NLL es mejor. GELU gana al final en dos de tres seeds; SwiGLU gana al comparar mejores checkpoints en dos de tres. No hay una ventaja consistente en las tres seeds. Comparación descriptiva; no se probó significancia ni superioridad universal.

| Seed | FFN | NLL inicial | NLL final | Mejor NLL | Perplexity final |
| --- | --- | ---: | ---: | ---: | ---: |
{table}

| Seed | Brecha final SwiGLU−GELU | Brecha mejor checkpoint |
| --- | ---: | ---: |
{gaps}

![Tres curvas pareadas, sin suavizar](comparison.svg)

[CSV completo, incluye NLL inicial fuera del zoom](curve.csv), [métricas](comparison.json), [generaciones completas de las seis trayectorias](generations.json).

## Presupuesto, controles y recuperación

Cada modelo consumió 6.144.000 tokens, 1.500 actualizaciones efectivas y 0 omitidas. Nuevos principales: **12.288.000 tokens / 3.000 pasos**. Pruebas aparte: **81.920 tokens / 20 pasos** (10 por arquitectura); total nuevo 12.369.920 tokens / 3.020 pasos. Seeds42/43 se reutilizan conservadas, sin reentrenar. GELU seed43 detuvo por su meseta operativa a 1.500 dentro de un cap histórico de 1.960; los nuevos runs tienen cap fijo de 1.500.

Ambos tienen 3.000.384 parámetros. GELU: FFN 192→768→192. SwiGLU: gate/up/down interna 512, `down(silu(gate(x))*up(x))`, sin bias; `2*192*768 = 3*192*512`. Las 22 matrices/normas ajenas a FFN coinciden bit a bit en la inicialización seed44. Las FFN tienen formas distintas: GELU sigue PCG64 seed44 del modelo original, SwiGLU un stream PCG64 seed44 separado, misma distribución y escala residual. [Inicializaciones y hardware verificados](initialization-pair-verification.json).

{backend['gpu']}, {backend['arch']}; torch {backend['torch']}, HIP {backend['hip']}. Pesos FP32/autocast FP16/GradScaler 1024, atención manual eager, RoPE/RMSNorm/weight tying, contexto256, microbatch4 y acumulación4. AdamW LR3e-4, betas(0.9,0.95), decay0.1 matrices/0 normas, warmup10 y LR constante; clip global1.0 después de unscale. Sin cambios de drivers, servicios, datos, BPE4096 o fuentes anteriores. Los hashes de batches/cursor/LR/scaler de cada pareja son iguales.

Validation cada50 pasos, milestones históricos y final; NLL ponderada sobre39.275 targets, aislada de cursor/RNG de train. 5 pasos continuos frente a2+3 reanudados en procesos nuevos por arquitectura, rtol1e-6/atol1e-7, con hashes de pesos bit a bit iguales en ambas; batches/contadores/cursor/optimizer/scaler/RNG comprobados, rechazo de corrupción e identidad incompatible. El principal empieza desde el artefacto aleatorio, nunca desde las pruebas. [Resumen previo](pre-run-verification.json), [recuperación GELU](../ffn-gelu-seed44-check-20261001-3f197002/recovery.json), [recuperación SwiGLU](../ffn-swiglu-seed44-check-20261001-3f197002/recovery.json).

Checkpoints latest/best/milestones con escritura atómica, fsync, checksums y relectura en D:. Todos los checkpoints nuevos se restauraron en procesos nuevos; optimizer/scaler/RNG exactos, NLL/greedy/muestreo reproducidos e independientes del train. {len(manifest)} archivos anteriores y prefijo de results.jsonl preservados por SHA-256; snapshots de {len(sources)} fuentes/configs por run.

## Coste nuevo y checkpoints

| FFN seed44 | Train s | Wall s | Tokens/s train | Pico PyTorch asignado/reservado MiB | Checkpoints |
| --- | ---: | ---: | ---: | ---: | ---: |
{costs}

{checkpoints}

El tiempo train incluye sincronización, hashing y verificaciones CPU; el wall incluye evaluación/persistencia/generación y excluye preparación/auditorías. La memoria PyTorch no incluye memoria del driver. Mediciones secuenciales, no benchmark de rendimiento simultáneo.

## Generaciones diagnósticas

Checkpoint final, prompts fijos, temperatura0.8/top-p0.9, seed de generación42 reiniciada por prompt, máximo64 tokens nuevos; distinta de la seed de entrenamiento44. Greedy e iniciales también conservados. Tres prompts no certifican coherencia ni seguimiento de instrucciones.

{diagnosis}

## Límites y siguiente propuesta

Tres seeds y un único ajuste de hiperparámetros no permiten extrapolar a modelos grandes. El mejor checkpoint se selecciona con validation: su ventaja no equivale a medir un test externo. Sólo2.000 historias train/200 validation; 6.144.000 tokens equivalen aproximadamente a14,4 recorridos del shard train de427.856 tokens. Más pasos sobre el mismo corpus no son nuevos datos. Los resultados describen estas inicializaciones, precisión, schedule y backend.

Propuesta siguiente, no ejecutada: repetir la pareja seed44 con decay del learning rate tras warmup, manteniendo tamaño, datos y 6.144.000 tokens por modelo, para comprobar si el deterioro tardío depende del schedule constante. Cambia un único factor y requiere autorización; sus resultados no medirían mayor diversidad de datos.
'''
    (output/'report.md').write_text(report, encoding='utf-8')
    for run in paths[44]:
        (run/'report.md').write_text(f'# Réplica FFN seed44\n\n[Reporte, curva, checkpoints y generaciones](../../output/{output.name}/report.md).\n', encoding='utf-8')
    print(json.dumps(dict(status='passed', pairs=pairs, mean_final_gap=result['mean_final_gap'],
                          mean_best_gap=result['mean_best_gap'], historical_files=len(manifest))))


if __name__ == '__main__':
    main()
