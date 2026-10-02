import argparse
from dataclasses import fields
import json
import logging
from pathlib import Path
import sys
import time

from safetensors.torch import load_file
from tokenizers import Tokenizer
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ModelConfig, Transformer
from src.colab_training import Trainer, TrainingConfig, file_hash, seed_all, weights_hash, write_json


def main():
    parser = argparse.ArgumentParser(description='Explicitly authorized AdamW training with validation and resumable checkpoints')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/3m.json')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--max-tokens', type=int, default=1003520)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--eval-every', type=int, default=50)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id or not all(c.isalnum() or c in '-_' for c in args.run_id):
        raise ValueError('Run ID must contain ASCII letters, digits, hyphens or underscores')
    if args.eval_every < 1 or (args.stop_after is not None and args.stop_after < 1):
        raise ValueError('Evaluation interval and stop iteration must be positive')
    raw = json.loads(args.config.read_text())
    model_config = ModelConfig(**{field.name: raw[field.name] for field in fields(ModelConfig) if field.name in raw})
    config = TrainingConfig(seed=raw['seed'], microbatch=raw['microbatch'], accumulation=raw['gradient_accumulation'],
                            precision=raw['precision'] if args.device == 'cuda' else 'fp32', max_tokens=args.max_tokens)
    if config.max_tokens % (config.microbatch * config.accumulation * model_config.context):
        raise ValueError('Token budget must be a multiple of tokens per iteration')
    processed = ROOT / 'data/processed'
    manifest = json.loads((processed / 'manifest.json').read_text())
    if manifest['vocab_size'] != model_config.vocab_size:
        raise ValueError('Tokenizer vocabulary does not match model')
    if file_hash(processed / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    for split in ['train', 'validation']:
        if file_hash(processed / f'{split}.bin') != manifest['splits'][split]['shard_sha256']:
            raise ValueError(f'Data checksum mismatch: {split}')
    seed_all(config.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Transformer(model_config)
    initial = ROOT / 'output/initial-3m.safetensors'
    model.load_state_dict(load_file(str(initial)), strict=True)
    trainer = Trainer(model, processed / 'train.bin', processed / 'validation.bin', config, args.device,
                      {'config': file_hash(args.config), 'tokenizer': manifest['tokenizer_sha256'],
                       'initial': file_hash(initial), 'trainer_source': file_hash(ROOT / 'src/colab_training.py')})
    if args.resume:
        trainer.load(args.resume.resolve())
    run = ROOT / 'runs' / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / 'config.json', trainer.contract)
    started = time.perf_counter()
    validations = []
    curve = []
    consecutive_skips = 0

    def evaluate_and_save():
        validation = trainer.evaluate()
        validations.append(validation)
        checkpoint = trainer.checkpoint(run, validation['loss'])
        write_json(run / 'progress.json', {'status': 'running', 'iteration': trainer.iterations,
                                          'tokens': trainer.tokens, 'validation': validation,
                                          'checkpoint': checkpoint.relative_to(run).as_posix()})
        logging.info(json.dumps({'event': 'validation', **validation}))

    evaluate_and_save()
    while trainer.tokens < config.max_tokens and (args.stop_after is None or trainer.iterations < args.stop_after):
        step = trainer.step()
        curve.append(step)
        consecutive_skips = consecutive_skips + 1 if step['skipped'] else 0
        if consecutive_skips > 8:
            raise RuntimeError('More than eight consecutive optimizer updates skipped')
        if trainer.iterations % args.eval_every == 0 or trainer.tokens == config.max_tokens or trainer.iterations == args.stop_after:
            evaluate_and_save()
            write_json(run / 'curve.json', curve)
    if validations[-1]['iteration'] != trainer.iterations:
        evaluate_and_save()
    status = 'completed' if trainer.tokens == config.max_tokens else 'paused'
    tokenizer = Tokenizer.from_file(str(processed / 'tokenizer.json'))
    generations = []
    trainer.model.eval()
    with torch.inference_mode():
        for prompt in ['Once upon a time, there was a little girl', 'Tom wanted to help his friend']:
            tokens = torch.tensor([tokenizer.encode(prompt).ids], device=args.device)
            for _ in range(64):
                with trainer.autocast():
                    next_token = trainer.model(tokens[:, -model_config.context:])[:, -1].float().argmax(-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
                if next_token.item() == tokenizer.token_to_id('<eos>'):
                    break
            generations.append({'prompt': prompt, 'method': 'greedy', 'text': tokenizer.decode(tokens[0].tolist())})
    final_hash = weights_hash(trainer.model)
    if trainer.updates and final_hash == trainer.initial_weights_sha256:
        raise RuntimeError('Optimizer reported updates but model weights did not change')
    result = {'run_id': args.run_id, 'status': status, 'parameters': sum(p.numel() for p in model.parameters()),
              'resume_from': str(args.resume.resolve()) if args.resume else None, 'best_scope': 'current run segment',
              'contract': trainer.contract, 'iterations': trainer.iterations, 'optimizer_steps': trainer.updates,
              'skipped_updates': trainer.skipped, 'tokens': trainer.tokens, 'data_position': trainer.train.position,
              'training_s': trainer.training_s, 'evaluation_s': trainer.evaluation_s, 'wall_s': time.perf_counter() - started,
              'training_tokens_s': trainer.tokens / trainer.training_s if trainer.training_s else None,
              'training_compute_window_gpu_hours': trainer.training_s / 3600,
              'initial_weights_sha256': trainer.initial_weights_sha256, 'final_weights_sha256': final_hash,
              'validation': validations, 'generations': generations,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved() if args.device == 'cuda' else None}
    write_json(run / 'curve.json', curve)
    write_json(run / 'metrics.json', result)
    write_json(run / 'progress.json', {'status': status, 'tokens': trainer.tokens, 'iteration': trainer.iterations})
    logging.info(json.dumps({'event': 'finished', 'run_id': args.run_id, 'status': status, 'tokens': trainer.tokens,
                             'optimizer_steps': trainer.updates, 'validation_loss': validations[-1]['loss']}))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
