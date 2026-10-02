from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import tinctura_experiment as base
from src.tinctura_xsa_fp32 import FP32XSATinctura
from src.training import sha256

BASE_CONTRACT = base.contract
BASE_TRAINER = base.E7Trainer
NUMERIC_SOURCES = ['src/tinctura_xsa_fp32.py', 'scripts/tinctura_xsa_experiment.py']


def contract(revision, variant, verification=False):
    return {**BASE_CONTRACT(revision, variant, verification),
            'e7_numeric_format': 2, 'xsa_multiply_and_reduction_dtype': 'float32'}


class FP32XSATrainer(BASE_TRAINER):
    def __init__(self, revision, variant, verification=False):
        super().__init__(revision, variant, verification)
        if not isinstance(self.model, FP32XSATinctura):
            raise RuntimeError('E7 numeric-v2 trainer requires its process-scoped factories')
        self.identity.update({name: sha256(ROOT / name) for name in NUMERIC_SOURCES})


@contextmanager
def factories():
    replacements = dict(contract=contract, E7Trainer=FP32XSATrainer, Tinctura=FP32XSATinctura)
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
    return dict(e7_numeric_format=2, numeric_source_sha256=sha256(ROOT / NUMERIC_SOURCES[0]),
                numeric_entrypoint_sha256=sha256(ROOT / NUMERIC_SOURCES[1]))


def main():
    if '--verification' not in sys.argv and not any(arg in {'-h', '--help'} for arg in sys.argv):
        expected = numeric_receipt()
        for flag in ['--recovery-check', '--model-check']:
            if flag not in sys.argv:
                raise ValueError('E7 numeric-v2 requires fresh model and recovery evidence')
            proof = json.loads(Path(sys.argv[sys.argv.index(flag) + 1]).read_text())
            if any(proof.get(key) != value for key, value in expected.items()):
                raise ValueError('E7 numeric-v2 evidence source/version mismatch')
    with factories():
        base.main()


if __name__ == '__main__':
    main()
