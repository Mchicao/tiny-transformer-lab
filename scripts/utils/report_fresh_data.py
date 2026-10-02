import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.check_training import reference
from src.training import atomic_json


def main():
    parser = argparse.ArgumentParser(description='E11 same-budget new-document coverage against paired E8 controls')
    parser.add_argument('--runs', nargs=3, type=Path, required=True)
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    cases = {}
    for path in args.runs:
        run = path.resolve()
        m = json.loads((run / 'metrics.json').read_text())
        audit = json.loads((run / 'post-run-verification.json').read_text())
        seed = m['experiment']['seed']
        assert seed in [42, 43, 44] and seed not in cases
        assert m['status'] == 'completed' and m['tokens'] == 6144000 and m['effective'] == 1500 and m['skipped'] == 0
        assert audit['status'] == 'passed' and audit['training_updates'] == 0
        baseline_run = ROOT / 'runs' / f'e8d-3m-20261001-{ {42:"022",43:"023",44:"024"}[seed]}-seed{seed}'
        baseline = json.loads((baseline_run / 'metrics.json').read_text())
        initial = json.loads((reference(baseline_run, 'initial') / 'metadata.json').read_text())
        assert m['initial_weights_sha256'] == initial['weights_sha256']
        assert len(m['history']) == len(baseline['history']) == 1500
        assert all(a['lr'] == b['lr'] and a['scale'] == b['scale'] for a, b in zip(m['history'], baseline['history'], strict=True))
        assert all(a['batch_sha256'] == b['batch_sha256'] for a, b in zip(m['history'][:104], baseline['history'][:104], strict=True))
        assert all(row['cursor'] == row['tokens'] == (index + 1) * 4096 for index, row in enumerate(m['history']))
        final, old = m['evaluations'][-1]['nll'], baseline['evaluations'][-1]['nll']
        cases[str(seed)] = dict(run=run.relative_to(ROOT).as_posix(), baseline=baseline_run.relative_to(ROOT).as_posix(),
                                final_nll=final, control_final_nll=old, delta_nats=final - old,
                                best_nll=min(r['nll'] for r in m['evaluations']), train_seconds=m['train_seconds'],
                                wall_seconds=m['wall_seconds'], tokens_s=m['segment']['tokens_s'],
                                checkpoints_audited=audit['checkpoints_restored'], initial_weights_sha256=m['initial_weights_sha256'])
    assert set(cases) == {'42', '43', '44'}
    mean_new = statistics.mean(c['final_nll'] for c in cases.values())
    mean_old = statistics.mean(c['control_final_nll'] for c in cases.values())
    delta = mean_new - mean_old
    result = dict(status='completed', cases=cases, mean_final_nll=mean_new, mean_control_final_nll=mean_old,
                  mean_delta_nats=delta, wins=sum(c['delta_nats'] < 0 for c in cases.values()),
                  perplexity_reduction_percent=100 * (1 - math.exp(delta)), new_main_tokens=18432000,
                  train_wrap=False, first104_batches_identical=True, lr_scale_paired_all1500=True,
                  recommendation='prioritize_unique_data_at_this_budget', significance_claim=False,
                  historical_time_comparison_confirmatory=False, training_updates_during_report=0)
    atomic_json(output / 'comparison.json', result)
    lines = ['# E11: datos nuevos frente a repetidos, tres seeds', '',
             'Mismo denso3M,6.144M tokens, AdamW/cooldown, BPE4096 y39.275 targets val. Sólo cambia cobertura train. ' 
             'Todos los principales y checkpoints auditados, cero omisiones.', '',
             '| Seed | E8 repetido | Datos ampliados | Δ nats | Bucle train (s) |', '| --- | ---: | ---: | ---: | ---: |']
    for seed in ['42', '43', '44']:
        c = cases[seed]
        lines.append(f'| {seed} | {c["control_final_nll"]:.6f} | {c["final_nll"]:.6f} | {c["delta_nats"]:+.6f} | {c["train_seconds"]:.2f} |')
    lines += ['', f'Media nueva **{mean_new:.6f}**, E8 **{mean_old:.6f}**, Δ **{delta:+.6f} nats**; ' 
              f'{result["wins"]}/3 seeds favorables, **{result["perplexity_reduction_percent"]:.2f}% menos perplexity**.', '',
              '## Verificación y alcance', '',
              '- Inicializaciones reales iguales a los checkpoints iniciales E8; campo histórico incorrecto de seeds43/44 no utilizado.',
              '- Primeros104 batches idénticos por seed antes de que el control repita; LR/escalas coinciden en los1.500 updates.',
              '- Cursor nuevo crece0→6.144.000 sin wrap.26.864 historias train únicas, sin overlap exacto val, misma revisión/fuente y tokenizer.',
              '- No hay cambio de parámetros, FFN, precisión, batch, schedule ni validación. Cobertura train aumenta; controles históricos preservados.',
              '- Tiempos históricos sólo descriptivos, no prueba de aceleración; costos de preparación/continuidad aparte.',
              '- Tres seeds y una validación permiten evidencia local descriptiva, no una conclusión universal ni significancia.', '',
              '## Decisión', '', 'Priorizar cobertura de datos únicos antes de nuevas técnicas en esta escala. ' 
              'Usar estos linajes como controles de futuras comparaciones con este corpus; no mezclar la frontera nueva con baselines de datos repetidos. ' 
              'Receta numérica sigue GELU+AdamW+cooldown15%.', '',
              '## Runs', '']
    lines += [f'- `{c["run"]}`; {c["checkpoints_audited"]} checkpoints auditados.' for c in cases.values()]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'mean_final_nll', 'mean_control_final_nll', 'mean_delta_nats', 'wins', 'perplexity_reduction_percent']}))


if __name__ == '__main__':
    main()
