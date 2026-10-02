import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from src.training import PROMPTS, Trainer, atomic_json, isolated_eval, rng_state, sha256, weight_hash
from scripts.utils.check_training import compare


def nucleus_probabilities(logits, temperature, top_p):
    if not 0 < temperature < float('inf') or not 0 < top_p <= 1:
        raise ValueError('Temperature must be finite and positive; top_p must be in (0, 1]')
    if logits.ndim != 1 or not torch.isfinite(logits).all().item():
        raise ValueError('Expected a finite vector of logits')
    probabilities, indices = (logits.float() / temperature).softmax(-1).sort(descending=True)
    excluded = probabilities.cumsum(-1) > top_p
    excluded[1:] = excluded[:-1].clone()
    excluded[0] = False
    probabilities = probabilities.masked_fill(excluded, 0)
    probabilities /= probabilities.sum()
    return probabilities, indices


def sample(model, tokenizer):
    rows = []
    with isolated_eval(model):
        for prompt in PROMPTS:
            generator = torch.Generator(device='cuda').manual_seed(42)
            ids = tokenizer.encode(prompt).ids
            prompt_tokens = len(ids)
            for _ in range(64):
                x = torch.tensor([ids[-256:]], device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x)[0, -1]
                probabilities, indices = nucleus_probabilities(logits, 0.8, 0.9)
                token = indices[torch.multinomial(probabilities, 1, generator=generator)].item()
                ids.append(token)
                if token == tokenizer.token_to_id('<eos>'):
                    break
            rows.append(dict(prompt=prompt, text=tokenizer.decode(ids), ids=ids,
                             generated_tokens=len(ids) - prompt_tokens))
    return rows


def repetition(ids):
    triples = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return 1 - len(set(triples)) / len(triples) if triples else 0.0


def main():
    parser = argparse.ArgumentParser(description='Muestreo fijo de un checkpoint local, sin entrenamiento')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/adamw-3m-20261001-001')
    args = parser.parse_args()
    probabilities, indices = nucleus_probabilities(torch.tensor([0.6, 0.25, 0.15]).log(), 1.0, 0.7)
    torch.testing.assert_close(probabilities, torch.tensor([0.6 / 0.85, 0.25 / 0.85, 0.0]))
    assert indices.tolist() == [0, 1, 2]
    probabilities, _ = nucleus_probabilities(torch.tensor([0.6, 0.25, 0.15]).log(), 1.0, 0.5)
    torch.testing.assert_close(probabilities, torch.tensor([1.0, 0.0, 0.0]))
    run = args.run_dir.resolve()
    references = json.loads((run / 'references.json').read_text())
    checkpoint = run / 'checkpoints' / references['latest']
    hashes = {str(path.relative_to(run)): sha256(path) for path in run.rglob('*') if path.is_file()}
    from scripts.train import restore_trainer
    trainer, _ = restore_trainer(checkpoint)
    before = dict(weights=weight_hash(trainer.model), rng=rng_state(), counters=copy.deepcopy(trainer.counters),
                  cursor=trainer.data.position, optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
                  scaler=copy.deepcopy(trainer.scaler.state_dict()), mode=trainer.model.training)
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/processed/tokenizer.json'))
    rows = sample(trainer.model, tokenizer)
    assert rows == sample(trainer.model, tokenizer)
    after = dict(weights=weight_hash(trainer.model), rng=rng_state(), counters=trainer.counters,
                 cursor=trainer.data.position, optimizer=trainer.optimizer.state_dict(),
                 scaler=trainer.scaler.state_dict(), mode=trainer.model.training)
    compare(before, after, exact=True)
    assert all(sha256(run / name) == expected for name, expected in hashes.items())
    greedy = json.loads((run / 'generations-final.json').read_text())['prompts']
    comparisons = []
    for original, sampled in zip(greedy, rows, strict=True):
        assert original['prompt'] == sampled['prompt']
        offset = len(tokenizer.encode(sampled['prompt']).ids)
        comparisons.append(dict(prompt=sampled['prompt'],
                                greedy_repeated_token_trigrams=repetition(original['ids'][offset:]),
                                sampling_repeated_token_trigrams=repetition(sampled['ids'][offset:])))
    output = ROOT / 'output' / ('sampling-adamw-3m-' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'samples.json', dict(checkpoint=str(checkpoint.relative_to(ROOT)),
                weights_sha256=before['weights'], temperature=0.8, top_p=0.9, seed_per_prompt=42,
                max_new_tokens=64, rows=rows, comparisons=comparisons,
                deterministic_repeat_exact=True, weights_optimizer_scaler_rng_cursor_unchanged=True,
                source_run_files_unchanged=True, training_updates=0))
    table = '\n'.join(f"| {r['prompt']} | {r['greedy_repeated_token_trigrams']:.1%} | {r['sampling_repeated_token_trigrams']:.1%} |" for r in comparisons)
    samples = '\n\n'.join(f"**{r['prompt']}** ({r['generated_tokens']} tokens nuevos)\n\n> {r['text'].replace(chr(10), ' ')}" for r in rows)
    (output / 'report.md').write_text(f'''# Diagnóstico de muestreo — mismo checkpoint AdamW 3M

Temperatura 0.8, top-p 0.9, seed 42 reiniciada por prompt, máximo 64 tokens nuevos. HIP eager FP16; generador CUDA separado. Repetición de la ejecución: IDs exactos. Cero actualizaciones de entrenamiento; pesos, optimizer, scaler, RNG, cursor y archivos del run original intactos.

Proporción de trigramas de tokens generados que duplican un trigrama previo (excluye el prompt; denominador real por muestra). Diagnóstico descriptivo sobre tres muestras con una sola seed; no mide coherencia ni prueba un efecto general del método.

| Prompt | Greedy: repetición | Muestreo: repetición |
| --- | ---: | ---: |
{table}

{samples}
''', encoding='utf-8')
    logging.info(json.dumps(dict(output=str(output), comparisons=comparisons, samples=[dict(prompt=r['prompt'], text=r['text']) for r in rows]), ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
