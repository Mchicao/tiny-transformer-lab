import base64
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    files = [*ROOT.glob('src/*.py'), ROOT / 'scripts/benchmark.py', ROOT / 'scripts/utils/check_model.py', *ROOT.glob('configs/*.json'), ROOT / 'configs/requirements-common.lock.txt', *ROOT.glob('data/processed/*')]
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    payload = buffer.getvalue()
    (ROOT / 'output/colab-bundle.zip').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    code = f'''import base64, hashlib, io, json, pathlib, subprocess, sys, zipfile
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
root = pathlib.Path('/content/tiny-transformer-lab')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for entry in archive.infolist():
        assert not pathlib.PurePosixPath(entry.filename).is_absolute() and '..' not in pathlib.PurePosixPath(entry.filename).parts
    archive.extractall(root)
for name in ['runs', 'output', 'logs/tests']:
    (root / name).mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--require-hashes', '-r', str(root / 'configs/requirements-common.lock.txt')], check=True)
import torch
print(json.dumps({{'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'bundle_sha256': hashlib.sha256(payload).hexdigest()}}))
subprocess.run([sys.executable, 'scripts/utils/check_model.py'], cwd=root, check=True)
subprocess.run([sys.executable, 'scripts/benchmark.py', '--run-id', 'colab-prepare-3m-manual', '--steps', '3'], cwd=root, check=True)
'''
    (ROOT / 'output/colab-setup-cell.py').write_text(code)
    print(f'Bundle: {len(payload)} bytes; SHA256: {digest}')


if __name__ == '__main__':
    main()
