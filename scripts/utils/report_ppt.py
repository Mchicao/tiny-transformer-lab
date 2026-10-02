import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import ppt_experiment as ppt
from scripts.utils.check_training import reference
from src.synthetic_ppt import probe
from src.training import atomic_json, rng_state, weight_hash
from scripts.utils.check_training import compare


def retained_probe(trainer):
    before = (weight_hash(trainer.model), dict(trainer.counters), trainer.data.position, rng_state())
    result = probe(trainer.model, ppt.DATA)
    compare(before, (weight_hash(trainer.model), trainer.counters, trainer.data.position, rng_state()), exact=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Report one audited E9 run against its paired E8 control')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    run = args.run.resolve()
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    metrics = json.loads((run / 'metrics.json').read_text())
    audit = json.loads((run / 'post-run-verification.json').read_text())
    control = json.loads((run / 'synthetic-probe-e8-control.json').read_text())
    baseline_run = ROOT / control['run']
    baseline = json.loads((baseline_run / 'metrics.json').read_text())
    assert audit['status'] == 'passed' and audit['training_updates'] == 0
    assert metrics['status'] == 'completed' and metrics['effective'] == 1745 and metrics['skipped'] == 0
    assert metrics['tokens'] == 7147520
    history = [row for row in metrics['history'] if row['phase'] == 'pt']
    assert len(history) == len(baseline['history']) == 1500
    for a, b in zip(history, baseline['history'], strict=True):
        for field in ['batch_sha256', 'cursor', 'lr', 'scale']:
            assert a[field] == b[field], (a['phase_iteration'], field)
    ppt.SEED, ppt.VARIANT = metrics['experiment']['seed'], metrics['variant']
    trainer = ppt.fresh(ppt.VARIANT)
    init_probe = retained_probe(trainer)
    trainer, _ = ppt.restore(reference(run, 'milestone-1003520'))
    ppt_probe = retained_probe(trainer)
    final_probe = json.loads((run / 'synthetic-probe-final.json').read_text())
    nll, control_nll = metrics['evaluations'][-1]['nll'], baseline['evaluations'][-1]['nll']
    delta = nll - control_nll
    phases = json.loads((run / 'phase-accounting.json').read_text())['phases']
    result = dict(status='one_seed_only', run=run.relative_to(ROOT).as_posix(),
                  control_run=baseline_run.relative_to(ROOT).as_posix(), variant=ppt.VARIANT, seed=ppt.SEED,
                  final_nll=nll, control_final_nll=control_nll, paired_delta_nats=delta,
                  perplexity_reduction_percent=100 * (1 - math.exp(delta)), phases=phases,
                  pt_batch_cursor_lr_scale_identical_to_e8=True, audit=audit,
                  wall_seconds=metrics['wall_seconds'], peak_mib=metrics['peak_allocated_bytes'] / 2**20,
                  probe=dict(random_initial=init_probe, after_ppt=ppt_probe, after_pt=final_probe,
                             e8_control=control['probe']), adoption_gate_evaluable=False,
                  remaining_principals=5, training_updates_during_report=0)
    atomic_json(output / 'comparison.json', result)
    lines = [f'# E9 {ppt.VARIANT}, seed{ppt.SEED}: primera observación', '',
             f'NLL final **{nll:.6f}**, control E8 **{control_nll:.6f}**, diferencia **{delta:+.6f} nats**.',
             f'Reducción de perplexity: {result["perplexity_reduction_percent"]:.3f}%. Sólo una seed; no adopción de PPT.', '',
             '## Coste y verificación', '',
             f'- PPT: {phases["ppt"]["tokens"]:,} tokens, {phases["ppt"]["train_seconds"]:.2f} s de bucle.',
             f'- TinyStories: {phases["pt"]["tokens"]:,} tokens, {phases["pt"]["train_seconds"]:.2f} s de bucle.',
             f'- Wall principal: {metrics["wall_seconds"]:.2f} s; pico asignado: {result["peak_mib"]:.2f} MiB.',
             f'- Auditoría: {audit["checkpoints_restored"]} checkpoints, NLL/generaciones y estado exactos; cero actualizaciones.',
             '- Los 1.500 batches, cursores, LR y escalas PT son idénticos a E8; 1.745 efectivas, cero omisiones.', '',
             '## Sonda retenida', '',
             '| Estado | Retrieval NLL respuesta | Retrieval precisión | Grammatical NLL respuesta | Grammatical precisión |',
             '| --- | ---: | ---: | ---: | ---: |']
    for label, p in [('Inicial aleatorio', init_probe), ('Tras PPT', ppt_probe), ('Tras TinyStories', final_probe), ('Control E8', control['probe'])]:
        lines.append(f'| {label} | {p["retrieval"]["answer_nll"]:.6f} | {100*p["retrieval"]["answer_accuracy"]:.4f}% | {p["grammatical"]["answer_nll"]:.6f} | {100*p["grammatical"]["answer_accuracy"]:.4f}% |')
    lines += ['', '2.048 respuestas por brazo. La sonda no se usa para seleccionar hiperparámetros. ' 
              'No inferir adquisición de retrieval desde una mejora de NLL TinyStories.', '',
              '## Pendiente', '', 'Cinco principales pendientes de autorización individual: grammatical seed42; ambos brazos seeds43/44. ' 
              'El PPT suma 16,33% de tokens frente a E8. El control grammatical no iguala dificultad/entropía. ' 
              'No hay todavía comparación B–C ni gate de tres seeds evaluable.', '',
              f'Run: `{result["run"]}`. Control: `{result["control_run"]}`.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['status', 'final_nll', 'control_final_nll', 'paired_delta_nats', 'wall_seconds', 'peak_mib']}))


if __name__ == '__main__':
    main()
