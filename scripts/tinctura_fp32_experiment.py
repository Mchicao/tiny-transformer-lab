from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import tinctura_experiment as base
from src.tinctura_xsa_fp32 import FP32XSATinctura
from src.training import sha256

BASE_CONTRACT = base.contract
BASE_TRAINER = base.E7Trainer
NUMERIC_SOURCES = ['src/tinctura_xsa_fp32.py', 'scripts/tinctura_fp32_experiment.py']


class FullPrecisionTorch:
    def __getattr__(self, name):
        return getattr(torch, name)

    def autocast(self, device_type, **kwargs):
        return torch.autocast(device_type, enabled=False)


def contract(revision, variant, verification=False):
    return {**BASE_CONTRACT(revision, variant, verification), 'e7_numeric_format': 3,
            'compute_dtype': 'float32', 'autocast': False, 'xsa_multiply_and_reduction_dtype': 'float32'}


class FP32Trainer(BASE_TRAINER):
    def __init__(self, revision, variant, verification=False):
        super().__init__(revision, variant, verification)
        if not isinstance(self.model, FP32XSATinctura):
            raise RuntimeError('FP32 trainer requires its process-scoped factories')
        self.config.precision = 'float32_eager'
        self.identity.update({name: sha256(ROOT / name) for name in NUMERIC_SOURCES})

    def step(self):
        with factories():
            return super().step()

    def evaluate(self):
        with factories():
            return super().evaluate()

    def generate(self):
        with factories():
            return super().generate()

    def arithmetic_probe(self):
        with factories():
            return super().arithmetic_probe()


@contextmanager
def factories():
    # ponytail: one precision per process preserves frozen trainers; future trainer APIs should accept precision explicitly.
    replacements = dict(contract=contract, E7Trainer=FP32Trainer, Tinctura=FP32XSATinctura, torch=FullPrecisionTorch())
    previous = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def restore(checkpoint, revision, variant, verification=False):
    with factories():
        return base.restore(checkpoint, revision, variant, verification)


def numeric_receipt():
    return dict(e7_numeric_format=3, compute_dtype='float32', numeric_source_sha256=sha256(ROOT / NUMERIC_SOURCES[0]),
                numeric_entrypoint_sha256=sha256(ROOT / NUMERIC_SOURCES[1]))


def main():
    if '--verification' not in sys.argv and not any(arg in {'-h', '--help'} for arg in sys.argv):
        for flag in ['--recovery-check', '--model-check']:
            if flag not in sys.argv:
                raise ValueError('E7 FP32 requires model and recovery evidence')
            proof = json.loads(Path(sys.argv[sys.argv.index(flag) + 1]).read_text())
            if any(proof.get(k) != v for k, v in numeric_receipt().items()):
                raise ValueError('E7 FP32 evidence source/version mismatch')
    with factories():
        base.main()


if __name__ == '__main__':
    main()
