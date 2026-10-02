import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
code = "import torch; x=torch.randn(32,device='cuda',requires_grad=True); f=torch.compile(lambda z:z.sin().square().sum()); y=f(x); y.backward(); torch.cuda.synchronize(); assert torch.isfinite(x.grad).all(); print('COMPILE_GPU_OK')"
start = time.perf_counter()
try:
    process = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=45)
    result = {'status': 'passed' if process.returncode == 0 else 'failed', 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr, 'elapsed_s': time.perf_counter() - start}
except subprocess.TimeoutExpired:
    result = {'status': 'timeout', 'limit_s': 45}
(ROOT / 'output/local-compile-probe.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
