import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from scripts import tinctura_xsa_experiment as numeric
from scripts.utils.check_training import compare, reference
from src.training import atomic_json, isolated_eval, rng_state, weight_hash


def fixed_loss(model, path, positions):
    tokens = np.memmap(path, dtype='<u2', mode='r')
    total, count = 0.0, 0
    with isolated_eval(model):
        for position in positions:
            chunk = torch.tensor(np.array(tokens[position:position + 2049], dtype=np.int64), device='cuda')[None]
            with torch.autocast('cuda', enabled=False):
                logits = model(chunk[:, :-1])[-1].float()
            total += F.cross_entropy(logits.flatten(0, 1), chunk[:, 1:].flatten(), reduction='sum').item()
            count += chunk.shape[1] - 1
    if not np.isfinite(total) or count != len(positions) * 2048:
        raise RuntimeError('Fixed diagnostic sample invalid')
    return dict(nll=total / count, targets=count)


def main():
    parser = argparse.ArgumentParser(description='Read-only fixed train/val samples on E7-v2 initial/intermediate/final weights')
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id:
        parser.error('Output ID must be one directory name')
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    train_positions = np.linspace(0, 409600 - 2049, 32, dtype=np.int64).tolist()
    val_positions = list(range(0, 65536, 2048))
    results = {}
    for variant, index in [('dense', '032'), ('looped', '033')]:
        prefix = f'e7x-early-{variant}-20261001-{index}-seg'
        cases = [('initial', ROOT / 'runs' / (prefix + '01'), 'initial'),
                 ('100', ROOT / 'runs' / (prefix + '01'), 'latest'),
                 ('200', ROOT / 'runs' / (prefix + '02'), 'latest'),
                 ('final', ROOT / 'runs' / (prefix + '05'), 'latest')]
        values = []
        for label, run, ref in cases:
            checkpoint = reference(run, ref)
            trainer, metadata = numeric.restore(checkpoint, 'early', variant)
            before = (weight_hash(trainer.model), copy.deepcopy(trainer.counters), trainer.data.position,
                      rng_state(), trainer.model.training)
            train = fixed_loss(trainer.model, ROOT / 'data/posttrain/fineweb-e7-v2/train.bin', train_positions)
            validation = fixed_loss(trainer.model, ROOT / 'data/posttrain/fineweb-e7-v2/validation.bin', val_positions)
            compare(before, (weight_hash(trainer.model), trainer.counters, trainer.data.position,
                            rng_state(), trainer.model.training), exact=True)
            values.append(dict(label=label, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                               iteration=metadata['counters']['iterations'], fixed_train=train, validation=validation))
            del trainer
            torch.cuda.empty_cache()
        results[variant] = values
    atomic_json(output / 'diagnosis.json', dict(compute_dtype='float32', source_numeric_version=2,
                fixed_train_positions=train_positions, fixed_validation_positions=val_positions,
                train_sample_within_first_100_updates=True, results=results, training_updates=0,
                weights_rng_cursor_counters_mode_unchanged=True))
    lines = ['# Diagnóstico E7: mismos targets, sin optimizador', '',
             'Pesos v2 evaluados en FP32; muestra train fija dentro de los primeros409.600 tokens y validación completa. ' 
             'Cero updates; pesos/RNG/cursor/contadores/modo intactos.', '',
             '| Brazo | Punto | Train fijo NLL | Val NLL |', '| --- | --- | ---: | ---: |']
    for variant, values in results.items():
        for value in values:
            lines.append(f'| {variant} | {value["label"]} | {value["fixed_train"]["nll"]:.6f} | {value["validation"]["nll"]:.6f} |')
    lines += ['', 'Comparar diferencias frente al propio inicio sobre los mismos targets. ' 
              'Si también empeora train visto, investigar optimización/flujo antes de explicar sólo generalización. ' 
              'Estas observaciones no identifican por sí solas LR como causa. La pareja v2 no se mezcla con principales FP32-v3.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='completed', results=results, training_updates=0)))


if __name__ == '__main__':
    main()
