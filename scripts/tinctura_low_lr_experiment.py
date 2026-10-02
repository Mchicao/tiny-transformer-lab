from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import tinctura_fp32_experiment as fp32
from src.training import sha256

LR = 3e-5


def contract(revision, variant, verification=False):
    return {**fp32.contract(revision, variant, verification), 'lr': LR, 'e7_lr_format': 1,
            'lr_selection': 'fixed_low_lr_3e-5_vs3e-4_pilot', 'pilot_stop': 100}


class LowLRTrainer(fp32.FP32Trainer):
    def __init__(self, revision, variant, verification=False):
        super().__init__(revision, variant, verification)
        self.identity['scripts/tinctura_low_lr_experiment.py'] = sha256(Path(__file__))


@contextmanager
def factories():
    with fp32.factories():
        previous = fp32.base.contract, fp32.base.E7Trainer
        try:
            fp32.base.contract = contract
            fp32.base.E7Trainer = LowLRTrainer
            yield
        finally:
            fp32.base.contract, fp32.base.E7Trainer = previous


def restore(checkpoint, revision, variant, verification=False):
    with factories():
        return fp32.base.restore(checkpoint, revision, variant, verification)


def receipt():
    return dict(e7_lr_format=1, lr=LR, lr_entrypoint_sha256=sha256(Path(__file__)))


def main():
    if '--verification' not in sys.argv and not any(a in {'-h', '--help'} for a in sys.argv):
        for flag in ['--recovery-check', '--model-check']:
            if flag not in sys.argv:
                raise ValueError('Low-LR FP32 requires model and recovery proofs')
            proof = json.loads(Path(sys.argv[sys.argv.index(flag) + 1]).read_text())
            if any(proof.get(k) != v for k, v in fp32.numeric_receipt().items()):
                raise ValueError('FP32 architecture proof mismatch')
            if flag == '--recovery-check' and any(proof.get(k) != v for k, v in receipt().items()):
                raise ValueError('Low-LR continuity proof mismatch')
    with factories():
        fp32.base.main()


if __name__ == '__main__':
    main()
