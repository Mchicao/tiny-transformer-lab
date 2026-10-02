from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import e8_decay_experiment as e8
from scripts import experiment as campaign
from src.data import PackedTokens
from src.training import EXPERIMENT, sha256, weight_hash

SEED = 42
VARIANT = 'fresh-data'
DATA = ROOT / 'data/expanded-tinystories/e11-20261002-v1'
BASE_DECAY_TRAINER = e8.DecayTrainer


def contract(variant=None, iterations=1500):
    if variant not in (None, VARIANT) or iterations != 1500 or SEED not in (42, 43, 44):
        raise ValueError('E11 authorizes fresh-data GELU3M, seeds42/43/44, fixed6.144M tokens')
    e8.SEED = SEED
    return {**e8.contract(), 'variant': VARIANT, 'coverage_format': 1,
            'dataset': DATA.relative_to(ROOT).as_posix(), 'train_wrap_allowed': False,
            'comparison': 'E8 existing small-shard repetition vs expanded same-source unique targets'}


class CoverageTrainer(BASE_DECAY_TRAINER):
    def step(self):
        if self.data.position != self.counters['tokens'] or self.data.position + 4097 > len(self.data.tokens):
            raise RuntimeError('E11 budget requires unique targets and no train wrap')
        row = super().step()
        if row['skipped'] or self.data.position != self.counters['tokens']:
            raise RuntimeError('E11 gate requires zero omissions and no repeated target windows')
        return row


def fresh(variant=None, iterations=1500):
    value = contract(variant, iterations)
    previous = e8.DecayTrainer
    try:
        e8.DecayTrainer = CoverageTrainer
        trainer = e8.fresh()
    finally:
        e8.DecayTrainer = previous
    manifest = json.loads((DATA / 'manifest.json').read_text())
    old = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    if (manifest['token_budget'] != 6144000 or manifest['tokens'] < 6144001
            or manifest['internal_duplicates_and_validation_overlap'] != 0
            or manifest['tokenizer_sha256'] != old['tokenizer_sha256']
            or manifest['validation_sha256'] != old['splits']['validation']['shard_sha256']
            or manifest['observed_dataset_revision'] != old['observed_dataset_revision']
            or manifest['historical_prefix_sha256'] != old['splits']['train']['shard_sha256']):
        raise ValueError('Expanded corpus incompatible with E8 comparison')
    for name, digest in manifest['files'].items():
        if sha256(DATA / name) != digest:
            raise ValueError('Expanded data checksum mismatch')
        trainer.identity[(DATA / name).relative_to(ROOT).as_posix()] = digest
    trainer.identity[(DATA / 'manifest.json').relative_to(ROOT).as_posix()] = sha256(DATA / 'manifest.json')
    trainer.identity['scripts/fresh_data_experiment.py'] = sha256(Path(__file__))
    trainer.data = PackedTokens(DATA / 'train.bin', 256)
    trainer.initial_hash = weight_hash(trainer.model)
    campaign.configure(value)
    return trainer


@contextmanager
def factories():
    replacements = dict(contract=contract, fresh=fresh, VARIANTS=[VARIANT])
    previous = {name: getattr(campaign, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(campaign, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(campaign, name, value)


def restore(checkpoint):
    with factories():
        return campaign.restore(checkpoint)


def main():
    global SEED
    if '--seed' in sys.argv:
        index = sys.argv.index('--seed')
        SEED = int(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    if '--continue-baseline' in sys.argv or '--plateau-stop' in sys.argv:
        raise ValueError('E11 requires fresh weights and full fixed token budget')
    if '--iterations' not in sys.argv:
        sys.argv += ['--iterations', '1500']
    if '--variant' not in sys.argv:
        sys.argv += ['--variant', VARIANT]
    if '--stop-after' not in sys.argv:
        if '--recovery-check' not in sys.argv:
            raise ValueError('Principal requires passing seed-specific recovery evidence')
        index = sys.argv.index('--recovery-check')
        proof = json.loads(Path(sys.argv[index + 1]).read_text())
        del sys.argv[index:index + 2]
        if (proof['status'] != 'passed' or proof['seed'] != SEED
                or proof['entrypoint_sha256'] != sha256(Path(__file__))):
            raise ValueError('E11 continuity evidence mismatch')
    with factories():
        campaign.main()


if __name__ == '__main__':
    main()
