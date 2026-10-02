import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.utils.check_training import compare, reference
from src.training import atomic_json, sha256

OLD_RUNS = ['adamw-3m-20261001-004-convergence', 'muon-3m-20261001-005-comparison']


def main():
    parser = argparse.ArgumentParser(description='Réplica pareada AdamW/Muon, seeds 42 y 43')
    parser.add_argument('--runs', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    runs = [p.resolve() for p in args.runs] + [ROOT / 'runs' / name for name in OLD_RUNS]
    metrics = [json.loads((p / 'metrics.json').read_text()) for p in runs]
    assert [(m['experiment']['seed'], m['variant']) for m in metrics] == [(42, 'adamw'), (42, 'muon'), (43, 'adamw'), (43, 'muon')]
    assert all(m['tokens'] == 6144000 and m['iterations'] == 1500 and not m['config']['qk_norm'] for m in metrics)
    trajectory = ['iterations', 'tokens', 'batch_sha256', 'cursor', 'lr', 'scale']
    expected = [{k: row[k] for k in trajectory} for row in metrics[0]['history']]
    assert all([{k: row[k] for k in trajectory} for row in m['history']] == expected for m in metrics)
    hyperparameters = ['microbatch', 'accumulation', 'lr', 'betas', 'weight_decay', 'warmup', 'clip', 'init_scale']
    assert all({k: m['experiment'][k] for k in hyperparameters} == {k: metrics[0]['experiment'][k] for k in hyperparameters} for m in metrics)
    assert all(m['config'] == metrics[0]['config'] and m['backend'] == metrics[0]['backend'] for m in metrics)
    assert metrics[0]['initial_weights_sha256'] == metrics[1]['initial_weights_sha256']
    assert metrics[2]['initial_weights_sha256'] == metrics[3]['initial_weights_sha256']
    assert metrics[0]['initial_weights_sha256'] != metrics[2]['initial_weights_sha256']
    legacy = ROOT / 'runs/adamw-3m-20261001-002-continuation'
    initial, previous = reference(runs[0], 'initial'), reference(legacy, 'latest')
    compare(torch.load(initial / 'state.pt', map_location='cpu', weights_only=True),
            torch.load(previous / 'state.pt', map_location='cpu', weights_only=True), exact=True)
    a, b = [json.loads((p / 'metadata.json').read_text()) for p in [initial, previous]]
    assert all(a[k] == b[k] for k in ['weights_sha256', 'counters', 'cursor'])
    manifest = json.loads((output / 'existing-artifacts.json').read_text())
    assert all(sha256(ROOT / name) == digest for name, digest in manifest.items())
    columns = ['seed', 'variant', 'iteration', 'tokens', 'nll']
    rows, summaries, samples, plots = [], [], [], []
    for index, (run, m) in enumerate(zip(runs, metrics, strict=True)):
        checks = json.loads((run / 'post-run-verification.json').read_text())
        assert checks['status'] == 'passed'
        if index < 2:
            source_dir = run / 'sources'
            source_dir.mkdir(exist_ok=False)
            sources = ['src/model.py', 'src/data.py', 'src/training.py', 'scripts/train.py', 'scripts/experiment.py',
                       'scripts/replicate.py', 'scripts/utils/check_experiments.py', 'scripts/utils/check_replication.py',
                       'scripts/utils/sample_checkpoint.py', 'scripts/utils/report_replication.py']
            for name in sources:
                destination = source_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            atomic_json(source_dir / 'checksums.json', {name: sha256(source_dir / name) for name in sources})
        evals = list({e['iteration']: e for e in m['evaluations']}.values())
        best, final = min(evals, key=lambda e: e['nll']), evals[-1]
        refs = json.loads((run / 'references.json').read_text())
        thresholds = {}
        for target in [3.5, 3.3]:
            hit = next((e for e in evals if e['nll'] <= target), None)
            thresholds[str(target)] = None if hit is None else dict(tokens=hit['tokens'], iteration=hit['iteration'],
                train_seconds=sum(r['seconds'] for r in m['history'] if r['iterations'] <= hit['iteration']))
        info = dict(seed=m['experiment']['seed'], variant=m['variant'], run_id=m['run_id'],
                    initial_nll=evals[0]['nll'], final_nll=final['nll'], best_nll=best['nll'], best_tokens=best['tokens'],
                    effective=m['effective'], skipped=m['skipped'], tokens=m['tokens'], train_seconds=m['train_seconds'],
                    train_loop_hours=m['train_seconds']/3600, tokens_s=m['tokens']/m['train_seconds'],
                    wall_seconds_segment=m['wall_seconds'], peak_allocated_mib=m['peak_allocated_bytes']/2**20,
                    peak_reserved_mib=m['peak_reserved_bytes']/2**20, segment=m['segment'], checkpoints=checks['checkpoints_restored'],
                    latest=refs['latest'], best=refs['best'], thresholds=thresholds)
        summaries.append(info)
        samples.append(dict(seed=info['seed'], variant=info['variant'],
                            greedy=json.loads((run / 'generations-final.json').read_text()),
                            sampling=json.loads((run / 'sampling-final.json').read_text())))
        for e in evals:
            rows.append(dict(seed=info['seed'], variant=info['variant'], iteration=e['iteration'], tokens=e['tokens'], nll=e['nll']))
        color = '#1565c0' if info['variant'] == 'adamw' else '#c62828'
        dash = '' if info['seed'] == 42 else 'stroke-dasharray="7 5"'
        for panel, low, high in [(0, 3.0, 8.5), (1, 3.0, 4.5)]:
            points = ' '.join(f"{70 + e['tokens']/6144000*800:.2f},{70+panel*330+(high-min(high,max(low,e['nll'])))/(high-low)*230:.2f}" for e in evals)
            plots.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" {dash}/>')
        if index < 2:
            text = f"# {info['variant']} 3M seed42 — réplica\n\nNLL del linaje {info['initial_nll']:.6f} → {info['final_nll']:.6f}; mejor {info['best_nll']:.6f} a {info['best_tokens']:,} tokens. {m['tokens']:,} tokens, {m['effective']} efectivas / {m['skipped']} omitidas. Train acumulado {m['train_seconds']:.2f} s; segmento {m['segment']['train_seconds']:.2f} s / wall {m['wall_seconds']:.2f} s. {info['tokens_s']:.0f} tokens/s; pico asignado/reservado {info['peak_allocated_mib']:.1f}/{info['peak_reserved_mib']:.1f} MiB.\n\n{checks['checkpoints_restored']} checkpoints recuperados. Latest: {refs['latest']}; best: {refs['best']}. [Verificación](post-run-verification.json), [curva](validation.svg), [métricas](metrics.json), [greedy](generations-final.json), [muestreo](sampling-final.json).\n\n[Comparación de ambas seeds](../../output/{output.name}/report.md).\n"
            (run / 'report.md').write_text(text, encoding='utf-8')
    gaps = [dict(seed=seed, final_muon_minus_adamw=next(s['final_nll'] for s in summaries if s['seed']==seed and s['variant']=='muon')-next(s['final_nll'] for s in summaries if s['seed']==seed and s['variant']=='adamw'),
                 best_muon_minus_adamw=next(s['best_nll'] for s in summaries if s['seed']==seed and s['variant']=='muon')-next(s['best_nll'] for s in summaries if s['seed']==seed and s['variant']=='adamw')) for seed in [42,43]]
    mean_best = sum(g['best_muon_minus_adamw'] for g in gaps)/2
    axes = []
    for panel, low, high in [(0, 3.0, 8.5), (1, 3.0, 4.5)]:
        top = 70 + panel*330
        axes.append(f'<text x="70" y="{top-12}">NLL por target — {"rango completo" if panel==0 else "zoom 3–4,5; valores superiores recortados"}</text>')
        for value in [low, (low+high)/2, high]:
            y=top+(high-value)/(high-low)*230
            axes.append(f'<path d="M70 {y} H870" stroke="#ddd"/><text x="20" y="{y+5}">{value:.2f}</text>')
        for fraction in [0,.25,.5,.75,1]:
            axes.append(f'<text x="{70+fraction*800}" y="{top+252}" text-anchor="middle">{fraction*6.144:.3f}M</text>')
    legend='<text x="70" y="28" fill="#1565c0">AdamW (azul)</text><text x="280" y="28" fill="#c62828">Muon (rojo)</text><text x="480" y="28">seed42: sólido; seed43: discontinuo</text>'
    chart=output/'comparison.svg'
    chart.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="920" height="700"><rect width="920" height="700" fill="white"/><g font-family="sans-serif" font-size="15" fill="#222">'+legend+''.join(axes+plots)+'<text x="370" y="680">Tokens de entrenamiento</text></g></svg>',encoding='utf-8')
    ET.parse(chart)
    with (output/'curve.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output/'comparison.json',dict(status='passed',models=summaries,paired_differences=gaps,
        mean_best_muon_minus_adamw=mean_best,equal_batches_cursor_lr_scale=True,equal_hyperparameters_config_backend=True,
        equal_initial_weights_within_seed=True,adamw_extension_state_exact=True,historical_files_preserved=len(manifest),
        new_main_tokens=sum(m['segment']['tokens'] for m in metrics[:2]),new_main_updates=sum(m['segment']['iterations'] for m in metrics[:2]),
        verification_tokens=40960,verification_updates=10,curve_points=len(rows)))
    atomic_json(output/'generations.json',dict(temperature=.8,top_p=.9,seed_per_prompt=42,max_new_tokens=64,models=samples))
    print(json.dumps(dict(status='passed',paired_differences=gaps,mean_best_muon_minus_adamw=mean_best,
                          historical_files_preserved=len(manifest))))


if __name__ == '__main__':
    main()
