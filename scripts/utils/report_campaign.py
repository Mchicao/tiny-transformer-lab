import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.sample_checkpoint import repetition
from scripts.utils.check_training import compare, reference
from src.training import atomic_json, sha256
from tokenizers import Tokenizer
import torch


def main():
    parser = argparse.ArgumentParser(description='Reporte comparativo de la campaña 3M local')
    parser.add_argument('--runs', nargs=3, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    runs = [path.resolve() for path in args.runs]
    metrics = [json.loads((run / 'metrics.json').read_text()) for run in runs]
    assert [m['variant'] for m in metrics] == ['adamw', 'muon', 'muon-qk']
    assert len({m['tokens'] for m in metrics}) == 1
    assert len({m['initial_weights_sha256'] for m in metrics}) == 1
    for m in metrics:
        assert m['effective'] + m['skipped'] == m['iterations']
        assert m['tokens'] == m['iterations'] * 4096
    trajectory = ['iterations', 'tokens', 'batch_sha256', 'cursor', 'lr']
    expected = [{k: row[k] for k in trajectory} for row in metrics[0]['history']]
    assert all([{k: row[k] for k in trajectory} for row in m['history']] == expected for m in metrics)
    legacy = ROOT / 'runs/adamw-3m-20261001-003-seed43'
    first = reference(runs[0], 'initial')
    previous = reference(legacy, 'latest')
    compare(torch.load(first / 'state.pt', map_location='cpu', weights_only=True),
            torch.load(previous / 'state.pt', map_location='cpu', weights_only=True), exact=True)
    before = json.loads((previous / 'metadata.json').read_text())
    after = json.loads((first / 'metadata.json').read_text())
    for name in ['weights_sha256', 'counters', 'cursor']:
        assert before[name] == after[name]
    manifest = json.loads((output / 'existing-artifacts.json').read_text())
    assert all(sha256(ROOT / name) == digest for name, digest in manifest.items())
    colors = ['#1565c0', '#c62828', '#00876c']
    low_nll = min(3.0, math.floor(min(e['nll'] for m in metrics for e in m['evaluations']) * 4) / 4)
    paths, rows, summary, generations = [], [], [], []
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/processed/tokenizer.json'))
    sources = ['src/model.py', 'src/data.py', 'src/training.py', 'scripts/train.py',
               'scripts/experiment.py', 'scripts/utils/sample_checkpoint.py', 'scripts/utils/check_experiments.py',
               'scripts/utils/report_campaign.py']
    for run, m, color in zip(runs, metrics, colors, strict=True):
        assert json.loads((run / 'post-run-verification.json').read_text())['status'] == 'passed'
        source_dir = run / 'sources'
        source_dir.mkdir(exist_ok=False)
        for name in sources:
            destination = source_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        atomic_json(source_dir / 'checksums.json', {name: sha256(source_dir / name) for name in sources})
        evals = list({r['iteration']: r for r in m['evaluations']}.values())
        best = min(evals, key=lambda r: r['nll'])
        final = evals[-1]
        thresholds = {}
        for threshold in [3.5, 3.3]:
            hit = next((r for r in evals if r['nll'] <= threshold), None)
            if hit:
                elapsed = sum(r['seconds'] for r in m['history'] if r['iterations'] <= hit['iteration'])
                thresholds[str(threshold)] = dict(iteration=hit['iteration'], tokens=hit['tokens'], train_seconds=elapsed,
                                                train_loop_hours=elapsed / 3600, observed_nll=hit['nll'])
            else:
                thresholds[str(threshold)] = None
        refs = json.loads((run / 'references.json').read_text())
        info = dict(variant=m['variant'], run_id=m['run_id'], initial_nll=evals[0]['nll'], final_nll=final['nll'],
                    best_nll=best['nll'], best_tokens=best['tokens'], tokens=m['tokens'], effective=m['effective'],
                    skipped=m['skipped'], train_seconds=m['train_seconds'], train_loop_hours=m['train_seconds'] / 3600,
                    tokens_s=m['tokens'] / m['train_seconds'], wall_seconds=m['wall_seconds'], segment=m['segment'],
                    peak_allocated_mib=m['peak_allocated_bytes'] / 2**20, peak_reserved_mib=m['peak_reserved_bytes'] / 2**20,
                    checkpoints=json.loads((run / 'post-run-verification.json').read_text())['checkpoints_restored'],
                    latest=refs['latest'], best=refs['best'], plateau_reached=m['plateau_reached'], thresholds=thresholds)
        if m['variant'] == 'adamw':
            prior = json.loads((ROOT / 'runs/adamw-3m-20261001-003-seed43/metrics.json').read_text())
            info['wall_seconds_lineage'] = m['wall_seconds'] + prior['wall_seconds']
        else:
            info['wall_seconds_lineage'] = m['wall_seconds']
        summary.append(info)
        for e in evals:
            rows.append(dict(variant=m['variant'], iteration=e['iteration'], tokens=e['tokens'], nll=e['nll']))
        for panel, low, high in [(0, low_nll, 8.5), (1, low_nll, 5.0)]:
            points = ' '.join(f"{70 + e['tokens'] / m['tokens'] * 800:.2f},{70 + panel * 330 + (high - min(high, max(low, e['nll']))) / (high - low) * 230:.2f}" for e in evals)
            paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        greedy = json.loads((run / 'generations-final.json').read_text())['prompts']
        sampled = json.loads((run / 'sampling-final.json').read_text())
        entries = []
        for a, b in zip(greedy, sampled, strict=True):
            offset = len(tokenizer.encode(a['prompt']).ids)
            entries.append(dict(prompt=a['prompt'], greedy=a['text'], sampling=b['text'],
                                greedy_repetition=repetition(a['ids'][offset:]), sampling_repetition=repetition(b['ids'][offset:])))
        generations.append(dict(variant=m['variant'], samples=entries))
        table = f"| NLL inicial / final / mejor | {info['initial_nll']:.6f} / {info['final_nll']:.6f} / {info['best_nll']:.6f} |\n| Tokens / efectivas / omitidas | {m['tokens']:,} / {m['effective']} / {m['skipped']} |\n| Train total / segmento / wall segmento | {m['train_seconds']:.2f} s / {m['segment']['train_seconds']:.2f} s / {m['wall_seconds']:.2f} s |\n| Throughput total | {info['tokens_s']:.0f} tokens/s |\n| Pico asignado / reservado | {info['peak_allocated_mib']:.1f} / {info['peak_reserved_mib']:.1f} MiB |\n| Checkpoints nuevos recuperados | {info['checkpoints']} |\n| Latest / best | {refs['latest']} / {refs['best']} |"
        samples_text = '\n\n'.join(f"**{e['prompt']}**\n\nGreedy: {e['greedy']}\n\nMuestreo: {e['sampling']}" for e in entries)
        (run / 'report.md').write_text(f"# {m['variant']} 3M — campaña local\n\n| Métrica | Valor |\n| --- | --- |\n{table}\n\nCurva: [validation.svg](validation.svg). Estado completo: [metrics.json](metrics.json). Verificación: [post-run-verification.json](post-run-verification.json). Comparación y límites: [reporte de campaña](../../output/campaign-3m-20261001/report.md).\n\nPrompts fijos, greedy y muestreo T=0.8 / top-p=0.9 / seed 42 por prompt / máximo 64 tokens. No actualizan pesos.\n\n{samples_text}\n", encoding='utf-8')
    axes = []
    for panel, low, high in [(0, low_nll, 8.5), (1, low_nll, 5.0)]:
        top = 70 + panel * 330
        axes.append(f'<text x="70" y="{top - 12}">NLL por target — {"rango completo" if panel == 0 else "zoom; valores mayores que 5 recortados"}</text>')
        for value in [low, (low + high) / 2, high]:
            y = top + (high - value) / (high - low) * 230
            axes.append(f'<path d="M70 {y} H870" stroke="#ddd"/><text x="20" y="{y+5}">{value:.2f}</text>')
        for fraction in [0, .25, .5, .75, 1]:
            x = 70 + fraction * 800
            axes.append(f'<text x="{x}" y="{top+252}" text-anchor="middle">{fraction * metrics[0]["tokens"] / 1e6:.3f}M</text>')
    legend = ''.join(f'<text x="{70+i*270}" y="35" fill="{color}">{html.escape(m["variant"])}</text>' for i, (m, color) in enumerate(zip(metrics, colors, strict=True)))
    (output / 'comparison.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="920" height="700"><rect width="920" height="700" fill="white"/><g font-family="sans-serif" font-size="15" fill="#222">' + legend + ''.join(axes+paths) + '<text x="380" y="680">Tokens de entrenamiento</text></g></svg>', encoding='utf-8')
    with (output / 'curve.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['variant', 'iteration', 'tokens', 'nll'])
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / 'comparison.json', dict(models=summary, equal_batches_cursor_lr=True,
                equal_initial_weights=True, adamw_extension_state_exact=True,
                historical_files_verified=len(manifest), historical_files_preserved=True,
                additional_verification_updates=20, additional_verification_tokens=81920,
                new_main_updates=sum(m['segment']['iterations'] for m in metrics),
                new_main_tokens=sum(m['segment']['tokens'] for m in metrics)))
    atomic_json(output / 'generations.json', dict(temperature=.8, top_p=.9, seed_per_prompt=42, variants=generations))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
