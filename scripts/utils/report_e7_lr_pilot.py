import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import tinctura_fp32_experiment as high_lr
from scripts import tinctura_low_lr_experiment as low_lr
from scripts.utils.check_training import compare, reference
from scripts.utils.diagnose_e7_adaptation import fixed_loss
from src.training import atomic_json, rng_state, sha256, weight_hash


def main():
    parser = argparse.ArgumentParser(description='Fixed development gate for LR-only E7 FP32 pilots, no optimizer')
    parser.add_argument('--high', type=Path, required=True)
    parser.add_argument('--low', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    dev = args.dev.resolve()
    manifest = json.loads((dev / 'manifest.json').read_text())
    if sha256(dev / 'dev.bin') != manifest['dev_sha256'] or manifest['targets'] != 65536:
        raise ValueError('Development identity/count mismatch')
    train_positions = np.linspace(0, 409600 - 2049, 32, dtype=np.int64).tolist()
    positions = list(range(0, 65536, 2048))
    metrics = [json.loads((p.resolve() / 'metrics.json').read_text()) for p in [args.high, args.low]]
    assert all(m['iterations'] == 100 and m['tokens'] == 409600 and m['skipped'] == 0 for m in metrics)
    for a, b in zip(metrics[0]['history'], metrics[1]['history'], strict=True):
        assert all(a[k] == b[k] for k in ['batch_sha256', 'cursor', 'scale', 'update_tokens', 'tokens'])
        assert abs(a['lr'] / 10 - b['lr']) < 1e-15
    results = {}
    cases = [('initial', args.high.resolve(), 'initial', high_lr),
             ('high_lr', args.high.resolve(), 'latest', high_lr),
             ('low_lr', args.low.resolve(), 'latest', low_lr)]
    for label, run, ref, loader in cases:
        if label != 'initial':
            assert json.loads((run / 'post-run-verification.json').read_text())['status'] == 'passed'
        trainer, _ = loader.restore(reference(run, ref), 'early', 'dense')
        before = (weight_hash(trainer.model), copy.deepcopy(trainer.counters), trainer.data.position, rng_state())
        results[label] = dict(dev=fixed_loss(trainer.model, dev / 'dev.bin', positions),
                             train_seen=fixed_loss(trainer.model, ROOT / 'data/posttrain/fineweb-e7-v2/train.bin', train_positions),
                             validation=fixed_loss(trainer.model, ROOT / 'data/posttrain/fineweb-e7-v2/validation.bin', positions))
        compare(before, (weight_hash(trainer.model), trainer.counters, trainer.data.position, rng_state()), exact=True)
        del trainer
        torch.cuda.empty_cache()
    initial, high, low = [results[k]['dev']['nll'] for k in ['initial', 'high_lr', 'low_lr']]
    passed = low <= initial + 0.01 and low <= high - 0.005
    result = dict(status='completed', compute_dtype='float32', results=results,
                  lower_lr_gate_passed=passed, dev_low_minus_initial=low - initial,
                  dev_low_minus_high=low - high, choice='3e-5' if passed else 'no_lower_lr_adoption',
                  paired100_batches_cursors_scales=True, main_tokens=819200, training_updates_during_report=0,
                  dev_identity=manifest, high_run=args.high.as_posix(), low_run=args.low.as_posix())
    atomic_json(output / 'comparison.json', result)
    lines = ['# E7: piloto pareado LR en FP32', '',
             'Mismos pesos iniciales, datos,100 updates, schedule de2M y precisión; sólo cambia LR. ' 
             'Dev disjunto fijado antes de los pilotos; val histórica descriptiva. Cero updates en reporte.', '',
             '| Punto | Train fijo | Dev | Val histórica |', '| --- | ---: | ---: | ---: |']
    for label, values in results.items():
        lines.append(f'| {label} | {values["train_seen"]["nll"]:.6f} | {values["dev"]["nll"]:.6f} | {values["validation"]["nll"]:.6f} |')
    lines += ['', f'Gate LR bajo: **{"pasa" if passed else "falla"}**. Dev bajo−inicial {low-initial:+.6f}; ' 
              f'bajo−alto {low-high:+.6f}. Criterio fijado: no deteriorar más de0,01 y mejorar≥0,005 frente a3e-4.', '',
              'Es diagnóstico exploratorio de una seed; no LR óptimo ni réplica confirmatoria. ' 
              'No mezclar el control FP16 histórico con esta comparación FP32.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='completed', lower_lr_gate_passed=passed, results=results)))


if __name__ == '__main__':
    main()
