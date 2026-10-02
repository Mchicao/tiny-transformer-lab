import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from scripts.ffn_seed43 import INITIALIZATION
from src.training import atomic_json, sha256

BASELINE = ROOT / 'runs/adamw-3m-20261001-004-convergence'


def main():
    parser = argparse.ArgumentParser(description='Comparación GELU/SwiGLU sin nuevas actualizaciones')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    runs = [BASELINE, run]
    metrics = [json.loads((path / 'metrics.json').read_text()) for path in runs]
    for m in metrics:
        assert m['status'] in {'completed', 'plateau'} and m['tokens'] == 6144000 and m['effective'] == 1500 and m['skipped'] == 0
        assert m['variant'] == 'adamw' and m['experiment']['seed'] == 43
    a, b = metrics
    assert {k: v for k, v in b['config'].items() if k not in {'ffn', 'ffn_hidden'}} == a['config']
    assert b['config']['ffn'] == 'swiglu' and b['config']['ffn_hidden'] == 512
    assert a['backend'] == b['backend']
    keys = ['seed', 'microbatch', 'accumulation', 'lr', 'betas', 'weight_decay', 'warmup', 'clip', 'init_scale']
    assert {k: a['experiment'][k] for k in keys} == {k: b['experiment'][k] for k in keys}
    trajectory = ['iterations', 'tokens', 'batch_sha256', 'cursor', 'lr', 'scale']
    assert [{k: r[k] for k in trajectory} for r in a['history']] == [{k: r[k] for k in trajectory} for r in b['history']]
    original = load_file(str(ROOT / 'output/initializations/3m-seed43-0f4e40f4b755/model.safetensors'))
    initial = load_file(str(INITIALIZATION / 'model.safetensors'))
    assert sum(t.numel() for t in original.values()) == sum(t.numel() for t in initial.values()) == 3000384
    shared = [name for name in original if '.ffn.' not in name]
    for name in shared:
        torch.testing.assert_close(original[name], initial[name], rtol=0, atol=0)
    manifest = json.loads((output / 'existing-artifacts.json').read_text())
    assert all(sha256(ROOT / name) == digest for name, digest in manifest.items())
    recovery = json.loads((ROOT / 'output/swiglu-seed43-check-20261001-b6cfe902/recovery.json').read_text())
    assert recovery['status'] == 'passed' and recovery['updates_executed'] == 10
    summaries, curves, samples = [], [], []
    for path, m, ffn in zip(runs, metrics, ['gelu', 'swiglu'], strict=True):
        audit = json.loads((path / 'post-run-verification.json').read_text())
        assert audit['status'] == 'passed'
        evaluations = list({e['iteration']: e for e in m['evaluations']}.values())
        best, final = min(evaluations, key=lambda e: e['nll']), evaluations[-1]
        refs = json.loads((path / 'references.json').read_text())
        info = dict(ffn=ffn, run_id=m['run_id'], initial_nll=evaluations[0]['nll'], final_nll=final['nll'],
                    best_nll=best['nll'], best_tokens=best['tokens'], final_perplexity=math.exp(final['nll']),
                    tokens=m['tokens'], effective=m['effective'], skipped=m['skipped'],
                    train_seconds=m['train_seconds'], train_hours=m['train_seconds']/3600,
                    tokens_s=m['tokens']/m['train_seconds'], wall_seconds_segment=m['wall_seconds'],
                    peak_allocated_mib=m['peak_allocated_bytes']/2**20,
                    peak_reserved_mib=m['peak_reserved_bytes']/2**20,
                    checkpoints=audit['checkpoints_restored'], latest=refs['latest'], best=refs['best'])
        summaries.append(info)
        curves.extend(dict(ffn=ffn, **{k:e[k] for k in ['iteration', 'tokens', 'nll']}) for e in evaluations)
        samples.append(dict(ffn=ffn, greedy=json.loads((path / 'generations-final.json').read_text()),
                            sampling=json.loads((path / 'sampling-final.json').read_text())))
    gap = summaries[1]['final_nll'] - summaries[0]['final_nll']
    best_gap = summaries[1]['best_nll'] - summaries[0]['best_nll']
    result = dict(status='passed', models=summaries, final_swiglu_minus_gelu=gap, best_swiglu_minus_gelu=best_gap,
                  swiglu_final_perplexity_reduction_pct=-math.expm1(gap)*100,
                  equal_batches_cursor_lr_scale=True, equal_common_hyperparameters_backend=True,
                  parameters_per_model=3000384, shared_initial_tensors_exact=len(shared),
                  different_ffn_initialization=True, historical_files_preserved=len(manifest),
                  new_main_tokens=6144000, new_main_updates=1500, verification_tokens=40960, verification_updates=10,
                  statistical_significance_tested=False, curve_points=len(curves))
    previous = json.loads((ROOT / 'output/swiglu-3m-20261001-01cfc7e0/comparison.json').read_text())
    paired = dict(seeds=[42, 43], final_gaps=[previous['final_swiglu_minus_gelu'], gap],
                  best_gaps=[previous['best_swiglu_minus_gelu'], best_gap],
                  mean_final_gap=(previous['final_swiglu_minus_gelu']+gap)/2,
                  mean_best_gap=(previous['best_swiglu_minus_gelu']+best_gap)/2,
                  statistical_significance_tested=False)
    atomic_json(output / 'paired-seeds.json', paired)
    result['paired_seeds'] = paired
    atomic_json(output / 'comparison.json', result)
    atomic_json(output / 'generations.json', dict(temperature=.8, top_p=.9, seed_per_prompt=42,
                max_new_tokens=64, models=samples))
    with (output / 'curve.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ffn', 'iteration', 'tokens', 'nll'])
        writer.writeheader()
        writer.writerows(curves)
    chart = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="700"><rect width="920" height="700" fill="white"/><g font-family="sans-serif" font-size="15" fill="#222">',
             '<text x="70" y="28" fill="#1565c0">GELU (azul)</text><text x="280" y="28" fill="#c62828">SwiGLU (rojo)</text><text x="520" y="28">AdamW 3M, seed43</text>']
    for panel, low, high in [(0, 3.0, 8.5), (1, 3.0, 4.5)]:
        top = 70 + panel*330
        chart.append(f'<text x="70" y="{top-12}">NLL/target — {"rango completo" if panel==0 else "zoom 3–4,5; valores superiores recortados"}</text>')
        for value in [low, (low+high)/2, high]:
            y = top + (high-value)/(high-low)*230
            chart.append(f'<path d="M70 {y} H870" stroke="#ddd"/><text x="20" y="{y+5}">{value:.2f}</text>')
        for fraction in [0,.25,.5,.75,1]:
            chart.append(f'<text x="{70+fraction*800}" y="{top+252}" text-anchor="middle">{fraction*6.144:.3f}M</text>')
        for ffn, color in [('gelu','#1565c0'), ('swiglu','#c62828')]:
            points = ' '.join(f"{70+e['tokens']/6144000*800:.2f},{top+(high-min(high,max(low,e['nll'])))/(high-low)*230:.2f}" for e in curves if e['ffn']==ffn)
            chart.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    chart.append('<text x="370" y="680">Tokens de entrenamiento</text></g></svg>')
    (output / 'comparison.svg').write_text(''.join(chart), encoding='utf-8')
    ET.parse(output / 'comparison.svg')
    source_dir = run / 'sources'
    source_dir.mkdir(exist_ok=False)
    sources = ['src/model.py', 'src/data.py', 'src/training.py', 'src/swiglu.py', 'configs/3m.json',
               'configs/3m-swiglu-seed43.json', 'scripts/train.py', 'scripts/experiment.py', 'scripts/ffn_seed43.py',
               'scripts/utils/check_ffn.py', 'scripts/utils/check_training.py', 'scripts/utils/check_experiments.py', 'scripts/utils/check_ffn_seed43.py',
               'scripts/utils/sample_checkpoint.py', 'scripts/utils/report_ffn_seed43.py']
    for name in sources:
        destination = source_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    atomic_json(source_dir / 'checksums.json', {name:sha256(source_dir/name) for name in sources})
    table = '\n'.join(f"| {s['ffn']} | {s['initial_nll']:.6f} | {s['final_nll']:.6f} | {s['best_nll']:.6f} | {s['final_perplexity']:.3f} | {s['train_seconds']:.2f} | {s['tokens_s']:.0f} |" for s in summaries)
    current = summaries[1]
    diagnosis = '\n\n'.join(f"**{group['ffn']} — {row['prompt']}**\n\n> {row['text'].replace(chr(10), ' ')}" for group in samples for row in group['sampling'])
    report = f'''# FFN GELU frente a SwiGLU — AdamW 3M, seed43

NLL final SwiGLU−GELU: **{gap:+.6f} nats/target**; diferencia del mejor checkpoint: **{best_gap:+.6f}**. Menor NLL es mejor. Esta pareja seed43, sin prueba de significancia estadística: resultado descriptivo para estos datos e hiperparámetros.

| FFN | NLL inicial | NLL final | Mejor NLL | Perplexity final | Train (s) | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Cada trayectoria contiene 6.144.000 tokens, 1.500 efectivas y 0 omitidas. GELU reutiliza el baseline histórico conservado: no se reentrenó. Nuevo principal SwiGLU: 6.144.000 tokens / 1.500 pasos. Pruebas separadas: 40.960 tokens / 10 pasos; total nuevo 6.184.960 tokens / 1.510 pasos. El run principal partió de los pesos aleatorios nuevos verificados, nunca de los pesos de prueba.

![Curva completa y zoom](comparison.svg)

[CSV sin suavizar](curve.csv), [métricas comparativas](comparison.json), [generaciones completas](generations.json), [recuperación previa](../swiglu-seed43-check-20261001-b6cfe902/recovery.json).

## Réplica pareada

Seeds42/43: brechas finales SwiGLU−GELU **{paired['final_gaps'][0]:+.6f} / {gap:+.6f}**, media **{paired['mean_final_gap']:+.6f}**. Brechas de mejores checkpoints **{paired['best_gaps'][0]:+.6f} / {best_gap:+.6f}**. Dos seeds: descripción de variabilidad, sin significancia estadística demostrada. La selección del mejor checkpoint usa validation y no equivale a medir un test externo. GELU seed43 consumió 1.500 pasos y detuvo por la meseta operativa dentro de un cap de 1.960; SwiGLU usa exactamente ese consumo, con cap fijo de 1.500.

[Resultados pareados](paired-seeds.json), [réplica seed42 conservada](../swiglu-3m-20261001-01cfc7e0/report.md).

## Control de la comparación

3.000.384 parámetros en ambos. GELU usa dos matrices con dimensión interna 768; SwiGLU usa gate/up/down con dimensión 512: `2*192*768 = 3*192*512`. SwiGLU calcula `down(silu(gate(x)) * up(x))`, sin bias. Atención manual eager, RoPE, RMSNorm, weight tying, contexto 256, microbatch 4 y acumulación 4 sin cambios. Pesos FP32, autocast FP16 y GradScaler inicial 1024. AdamW LR 3e-4, betas (0.9,0.95), decay matrices 0.1 / normas 0, warmup 10 y LR constante, clip global 1.0 después de unscale. GPU {b['backend']['gpu']}, {b['backend']['arch']}, torch {b['backend']['torch']}, HIP {b['backend']['hip']}; sin CPU de entrenamiento ni compile/Triton.

Los 22 tensores ajenos a la FFN coinciden bit a bit con el checkpoint aleatorio original seed43. Las FFN tienen formas diferentes y no pueden compartir todos sus pesos: SwiGLU se inicializó mediante un stream PCG64 seed43 separado, misma distribución uniforme y escala de proyección residual. Su hash inicial es `{b['initial_weights_sha256']}`. La identidad incluye la configuración y fuentes nuevas y el artefacto inicial propio. Los cargadores y fuentes anteriores no se modificaron.

Los 1.500 hashes de batches/cursor/LR/scaler coinciden con GELU. BPE 4096 y shards originales, sin preparación nueva. Validation cada 50 pasos, milestones históricos y final, NLL total dividida por 39.275 targets; no consume RNG/cursor de train. Dos mil historias train y 200 validation limitan la interpretación. Igual número de parámetros no garantiza igual coste de kernels; tiempo train incluye sincronización, hashing y controles CPU. GELU acumula segmentos anteriores: no es un benchmark simultáneo.

## Estabilidad y recuperación

5 pasos continuos frente a 2 + 3 en procesos nuevos: rtol=1e-6, atol=1e-7; hashes finales bit a bit iguales: {recovery['bitwise_weights_equal']}. Batches, contadores, cursor, optimizer, scaler y RNG verificados; las 15 matrices gate/up/down cambiaron. Rechazo de corrupción e identidad incompatible y evaluación/generación aisladas. Ninguna inestabilidad persistente.

{current['checkpoints']} checkpoints del run principal recuperados en un proceso nuevo, optimizer/scaler/RNG exactos; NLL, greedy y muestreo finales reproducidos exactamente. Escritura atómica, fsync, checksums y relectura en D:. {len(manifest)} archivos históricos/data/config/fuentes preservados por SHA-256. Snapshot de las fuentes/configs del run nuevo con checksums.

Wall SwiGLU: {current['wall_seconds_segment']:.2f} s; train {current['train_seconds']:.2f} s ({current['train_hours']:.6f} horas), {current['tokens_s']:.0f} tokens/s. Pico asignado/reservado PyTorch: {current['peak_allocated_mib']:.1f}/{current['peak_reserved_mib']:.1f} MiB; excluye memoria del driver. Latest `{current['latest']}`; best `{current['best']}` a {current['best_tokens']:,} tokens. [Run y auditoría](../../runs/{run.name}/post-run-verification.json).

## Generaciones diagnósticas

Checkpoints finales, temperatura 0.8, top-p 0.9, seed42 reiniciada por prompt, máximo 64 tokens nuevos. También se conservaron greedy y generaciones iniciales. La NLL no certifica coherencia, gramática ni seguimiento de instrucciones; tres prompts no son un benchmark de calidad.

{diagnosis}

## Siguiente comprobación propuesta

Una tercera seed pareada, al mismo presupuesto, permitiría observar si la diferencia se sostiene frente a variación entre inicializaciones. No se ejecutó; requiere autorización. Dos seeds y un único ajuste de hiperparámetros no prueban superioridad general.
'''
    (output / 'report.md').write_text(report, encoding='utf-8')
    (run / 'report.md').write_text(f"# AdamW SwiGLU 3M seed43\n\nNLL {current['initial_nll']:.6f} → {current['final_nll']:.6f}; mejor {current['best_nll']:.6f}. 6.144.000 tokens / 1.500 efectivas / 0 omitidas.\n\n[Reporte, curva, coste, checkpoints y generaciones](../../output/{output.name}/report.md).\n", encoding='utf-8')
    print(json.dumps(dict(status='passed', final_gap=gap, best_gap=best_gap, historical_files_preserved=len(manifest))))


if __name__ == '__main__':
    main()
