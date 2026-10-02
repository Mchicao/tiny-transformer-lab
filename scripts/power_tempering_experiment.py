import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from src.power_tempering import Sampler
from src.tinctura_loop import FINAL, load_weights, weights
from src.tinctura_xsa_fp32 import FP32XSATinctura
from src.training import atomic_json, isolated_eval, rng_state, sha256, weight_hash
from scripts.utils.check_training import compare

METHODS = ['greedy', 'standard', 'best_of_n', 'single_power', 'ppt']


def main():
    parser = argparse.ArgumentParser(description='E10 bounded inference pilot, original final96M, zero training')
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--output-id', required=True)
    args = parser.parse_args()
    if Path(args.output_id).name != args.output_id or not 0 <= args.start < 16 or not 1 <= args.count <= 4 or args.start + args.count > 16:
        parser.error('Unique output ID, fixed16-task slice and count1..4 required')
    if datetime.now() >= datetime(2026, 10, 2, 8, 50):
        raise RuntimeError('No new inference batches after08:50 before09:00 deadline')
    output = ROOT / 'output' / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('HIP GPU required')
    props = torch.cuda.get_device_properties(0)
    if props.name != 'AMD Radeon RX 6750 GRE 10GB':
        raise RuntimeError('Unexpected inference GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    state, provenance = weights('final')
    model = FP32XSATinctura(1, 42, checkpoint_layers=False).cuda().eval()
    load_weights(model, state)
    del state
    tokenizer = Tokenizer.from_file(str(FINAL / 'tokenizer.json'))
    probe_path = ROOT / 'data/posttrain/fineweb-e7-v2/arithmetic-probe.json'
    tasks = json.loads(probe_path.read_text())['tasks'][:16]
    identity = dict(weights_sha256=weight_hash(model), pretrained_revision=provenance['revision'],
                    tokenizer_sha256=sha256(FINAL / 'tokenizer.json'), tasks_sha256=sha256(probe_path),
                    sampler_sha256=sha256(ROOT / 'src/power_tempering.py'), entrypoint_sha256=sha256(Path(__file__)),
                    numeric_source_sha256=sha256(ROOT / 'src/tinctura_xsa_fp32.py'),
                    compute_dtype='float32', horizon=24, search_forward_token_cap=16000,
                    candidate_extraction='first_signed_integer_in_generated_suffix', training_updates=0)
    before = (weight_hash(model), rng_state(), model.training)
    atomic_json(output / 'config.json', dict(**vars(args), identity=identity,
                backend=dict(torch=torch.__version__, hip=torch.version.hip, gpu=props.name)))
    rows = []
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with isolated_eval(model):
        for index in range(args.start, args.start + args.count):
            task = tasks[index]
            prompt = tokenizer.encode(task['prompt']).ids

            def callback(ids):
                return model(torch.tensor([ids], device='cuda'))[-1][0, -1].float()

            engine = Sampler(callback, prompt, seed=args.seed + index * 1000003,
                             horizon=24, eos=0, token_budget=16000)
            torch.cuda.synchronize()
            started = time.perf_counter()
            record = engine.run(args.method)
            torch.cuda.synchronize()
            text = tokenizer.decode(record.tokens)
            found = re.search(r'(?<!\d)[+-]?\d+', text)
            answer = int(found.group()) if found else None
            row = dict(task_index=index, prompt=task['prompt'], expected=task['answer'], predicted=answer,
                       correct=answer == task['answer'], generated_ids=record.tokens, generated_text=text,
                       returned_base_logp=record.logp(), seconds=time.perf_counter() - started,
                       method=args.method, seed=args.seed, **engine.accounting())
            rows.append(row)
            atomic_json(output / 'rows.json', rows)
            print(json.dumps({k: row[k] for k in ['task_index', 'method', 'seed', 'correct', 'seconds',
                                                'processed_forward_tokens', 'forward_calls', 'proposals', 'swaps_accepted']}), flush=True)
    compare(before, (weight_hash(model), rng_state(), model.training), exact=True)
    result = dict(status='completed', method=args.method, seed=args.seed, task_start=args.start, tasks=args.count,
                  accuracy=sum(row['correct'] for row in rows) / len(rows), rows=rows, identity=identity,
                  wall_seconds=time.perf_counter() - start_time, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  weights_rng_mode_unchanged=True, training_updates=0,
                  reference_not_compute_matched=args.method in ['greedy', 'standard'])
    atomic_json(output / 'results.json', result)
    print(json.dumps(dict(status='completed', output=str(output), accuracy=result['accuracy'], training_updates=0)))


if __name__ == '__main__':
    main()
