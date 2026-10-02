import json

lines = [json.loads(l) for l in open('runs/results.jsonl', encoding='utf-8') if l.strip()]

name_map = {
    'adamw-3m-20261001-004-convergence': ('Baseline GELU · s43', '#38bdf8'),
    'muon-3m-20261001-005-comparison': ('Muon · s43', '#fb923c'),
    'muon-qk-3m-20261001-006-comparison': ('Muon+QK · s43', '#f87171'),
    'adamw-3m-20261001-007-seed42-replica': ('Baseline GELU · s42', '#38bdf8'),
    'muon-3m-20261001-008-seed42-replica': ('Muon · s42', '#fb923c'),
    'adamw-swiglu-3m-20261001-009-seed42': ('SwiGLU · s42', '#a78bfa'),
    'adamw-swiglu-3m-20261001-010-seed43': ('SwiGLU · s43', '#a78bfa'),
    'adamw-gelu-3m-20261001-011-seed44': ('Baseline GELU · s44', '#38bdf8'),
    'adamw-swiglu-3m-20261001-012-seed44': ('SwiGLU · s44', '#a78bfa'),
    'looped2f-3m-20261001-013-seed42': ('Looped-K2 final · s42', '#818cf8'),
    'looped2f-3m-20261001-014-seed43': ('Looped-K2 final · s43', '#818cf8'),
    'looped2f-3m-20261001-015-seed44': ('Looped-K2 final · s44', '#818cf8'),
    'looped2d-3m-20261001-016-seed42': ('Looped-K2 deep · s42', '#c084fc'),
    'looped2d-3m-20261001-017-seed43': ('Looped-K2 deep · s43', '#c084fc'),
    'looped2d-3m-20261001-018-seed44': ('Looped-K2 deep · s44', '#c084fc'),
    'dense10-3m-20261001-019-seed42': ('Dense-10L · s42', '#f59e0b'),
    'dense10-3m-20261001-020-seed43': ('Dense-10L · s43', '#f59e0b'),
    'dense10-3m-20261001-021-seed44': ('Dense-10L · s44', '#f59e0b'),
    'e8d-3m-20261001-022-seed42': ('E8 Decay · s42', '#34d399'),
    'e8d-3m-20261001-023-seed43': ('E8 Decay · s43', '#34d399'),
    'e8d-3m-20261001-024-seed44': ('E8 Decay · s44', '#34d399'),
    'e9-retrieval-3m-20261001-025-seed42': ('E9 Retrieval · s42', '#06b6d4'),
    'e9-grammatical-3m-20261001-026-seed42': ('E9 Grammatical · s42', '#10b981'),
    'e9-retrieval-3m-20261001-027-seed43': ('E9 Retrieval · s43', '#06b6d4'),
    'e9-grammatical-3m-20261001-028-seed43': ('E9 Grammatical · s43', '#10b981'),
    'e9-retrieval-3m-20261001-029-seed44': ('E9 Retrieval · s44', '#06b6d4'),
    'e9-grammatical-3m-20261001-030-seed44': ('E9 Grammatical · s44', '#10b981'),
    'e11-fresh-3m-20261002-037-seed42': ('E11 Fresh · s42', '#4ade80'),
    'e11-fresh-3m-20261002-038-seed43': ('E11 Fresh · s43', '#4ade80'),
    'e11-fresh-3m-20261002-039-seed44': ('E11 Fresh · s44', '#4ade80')
}

full_runs = [x for x in lines if x.get('iterations',0) >= 1500]
results = []
for r in full_runs:
    rid = r['run_id']
    name, color = name_map.get(rid, (rid, '#94a3b8'))
    b_ev = min(r['evaluations'], key=lambda e: e['nll'])
    f_ev = r['evaluations'][-1]
    results.append({
        'name': name,
        'run': rid,
        'color': color,
        'best': round(b_ev['nll'], 6),
        'bestAt': b_ev['iteration'],
        'final': round(f_ev['nll'], 6),
        'tokens': r['tokens'],
        'eff': r['iterations'],
        'trainS': round(r['train_seconds'], 2),
        'wallS': round(r['wall_seconds'], 2),
        'mib': round(r['peak_allocated_bytes'] / (1024 * 1024), 1)
    })

results.sort(key=lambda x: x['best'])

curves = {}
for r in lines:
    rid = r.get('run_id')
    if rid in ['e11-fresh-3m-20261002-037-seed42', 'e11-fresh-3m-20261002-038-seed43', 'e11-fresh-3m-20261002-039-seed44',
               'e8d-3m-20261001-022-seed42', 'e8d-3m-20261001-023-seed43', 'e8d-3m-20261001-024-seed44']:
        curves[rid] = [[e['iteration'], round(e['nll'], 4)] for e in r['evaluations']]

payload = {
    'results': results,
    'curves': curves
}

with open('output/dashboard_payload.json', 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2)

with open('output/results_js.txt', 'w', encoding='utf-8') as f:
    f.write(json.dumps(results))

with open('output/curves_js.txt', 'w', encoding='utf-8') as f:
    f.write(json.dumps(curves))

print('Wrote files successfully.')
