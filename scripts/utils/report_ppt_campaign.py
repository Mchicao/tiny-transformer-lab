import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training import atomic_json


def main():
    parser = argparse.ArgumentParser(description='Paired three-seed E9 campaign gates, audited runs only')
    parser.add_argument('--runs', nargs=6, type=Path, required=True)
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    runs = {}
    restored = 0
    for path in args.runs:
        path = path.resolve()
        m = json.loads((path / 'metrics.json').read_text())
        a = json.loads((path / 'post-run-verification.json').read_text())
        control = json.loads((path / 'synthetic-probe-e8-control.json').read_text())
        baseline = json.loads((ROOT / control['run'] / 'metrics.json').read_text())
        assert m['status'] == 'completed' and m['tokens'] == 7147520 and m['skipped'] == 0
        assert a['status'] == 'passed' and a['training_updates'] == 0
        pt = [r for r in m['history'] if r['phase'] == 'pt']
        assert len(pt) == len(baseline['history']) == 1500
        assert all(all(x[k] == y[k] for k in ['batch_sha256', 'cursor', 'lr', 'scale'])
                   for x, y in zip(pt, baseline['history'], strict=True))
        key = (m['variant'], m['experiment']['seed'])
        assert key not in runs
        phases = json.loads((path / 'phase-accounting.json').read_text())['phases']
        runs[key] = dict(run=path.relative_to(ROOT).as_posix(), final_nll=m['evaluations'][-1]['nll'],
                         control_nll=baseline['evaluations'][-1]['nll'],
                         delta=m['evaluations'][-1]['nll'] - baseline['evaluations'][-1]['nll'],
                         phases=phases, wall_seconds=m['wall_seconds'], audit=a,
                         probe=json.loads((path / 'synthetic-probe-final.json').read_text()))
        restored += a['checkpoints_restored']
    assert set(runs) == {(a, s) for a in ['retrieval', 'grammatical'] for s in [42, 43, 44]}
    summary = {}
    for arm in ['retrieval', 'grammatical']:
        values = [runs[(arm, seed)] for seed in [42, 43, 44]]
        delta = statistics.mean(v['delta'] for v in values)
        wins = sum(v['delta'] < 0 for v in values)
        summary[arm] = dict(mean_final_nll=statistics.mean(v['final_nll'] for v in values),
                            mean_delta=delta, wins=wins, numerical_001_2of3_gate=delta <= -0.01 and wins >= 2,
                            per_seed=values)
    contrast = [runs[('retrieval', s)]['final_nll'] - runs[('grammatical', s)]['final_nll'] for s in [42, 43, 44]]
    result = dict(status='completed', summary=summary, mean_e8_final=statistics.mean(runs[('retrieval', s)]['control_nll'] for s in [42, 43, 44]),
                  retrieval_minus_grammatical=contrast, mean_retrieval_minus_grammatical=statistics.mean(contrast),
                  ppt_retrieval_adopted=summary['retrieval']['numerical_001_2of3_gate'],
                  mechanism_confirmed=False, checkpoints_audited=restored, main_tokens=6 * 7147520,
                  ppt_extra_tokens_per_run=1003520, validation_batch_lr_scale_paired=True,
                  train_seconds=sum(v['phases']['ppt']['train_seconds'] + v['phases']['pt']['train_seconds'] for v in runs.values()),
                  wall_seconds=sum(v['wall_seconds'] for v in runs.values()), training_updates_during_report=0)
    atomic_json(output / 'comparison.json', result)
    lines = ['# E9 completado: retrieval no supera el gate', '',
             'Seis principales, tres seeds pareadas contra E8; todos auditados, cero omisiones. ' 
             'No se adopta PPT-retrieval ni se confirma su mecanismo.', '',
             '| Seed | E8 final | Retrieval final | Δ retrieval−E8 | Grammatical final | Δ grammatical−E8 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for seed in [42, 43, 44]:
        b, c = runs[('retrieval', seed)], runs[('grammatical', seed)]
        lines.append(f'| {seed} | {b["control_nll"]:.6f} | {b["final_nll"]:.6f} | {b["delta"]:+.6f} | {c["final_nll"]:.6f} | {c["delta"]:+.6f} |')
    lines += ['', f'Medias: E8 **{result["mean_e8_final"]:.6f}**, retrieval **{summary["retrieval"]["mean_final_nll"]:.6f}**, ' 
              f'grammatical **{summary["grammatical"]["mean_final_nll"]:.6f}**.', '',
              f'- Retrieval: Δ medio **{summary["retrieval"]["mean_delta"]:+.6f}**, gana {summary["retrieval"]["wins"]}/3. Gate ≥0,01 y ≥2/3: falla.',
              f'- Grammatical: Δ medio **{summary["grammatical"]["mean_delta"]:+.6f}**, gana {summary["grammatical"]["wins"]}/3. ' 
              'Resultado descriptivo del control; no fue el brazo prefijado para adopción.',
              f'- Retrieval−grammatical medio: **{result["mean_retrieval_minus_grammatical"]:+.6f}** nats; no respalda ventaja retrieval.', '',
              '## Coste y límites', '',
              f'{result["main_tokens"]:,} tokens principales, con 16,33% adicional por run frente a E8; ' 
              f'{result["train_seconds"]/60:.2f} min de bucle y {result["wall_seconds"]/60:.2f} min wall de principales. ' 
              'Pruebas, auditorías y preparación se contabilizan aparte.', '',
              f'{restored} checkpoints auditados. Los 1.500 batches, cursores, LR y escalas coinciden con E8 para cada principal.', '',
              'La sonda tras PPT no demuestra adquisición de retrieval. El control local no iguala dificultad/entropía; ' 
              'la mejora grammatical tampoco prueba un prior gramatical causal. Tres seeds permiten una lectura descriptiva, no significancia ni transferencia a 10M.', '',
              '## Runs', '']
    lines += [f'- `{v["run"]}`' for v in runs.values()]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'mean_e8_final', 'mean_retrieval_minus_grammatical', 'ppt_retrieval_adopted', 'checkpoints_audited', 'train_seconds', 'wall_seconds']}))


if __name__ == '__main__':
    main()
