from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import experiment as campaign
from scripts.train import BASE_EXPERIMENT, CONTINUATION_EXPERIMENT
from src.training import Trainer, sha256

_CONTRACT = campaign.contract
_IDENTITY = campaign.campaign_identity
_RESTORE_LEGACY = campaign.restore_legacy
LEGACY_RUN = ROOT / 'runs/adamw-3m-20261001-002-continuation'
ITERATIONS = 1500


def contract(variant, iterations=ITERATIONS):
    if variant not in {'adamw', 'muon'} or iterations != ITERATIONS:
        raise ValueError('Seed42 replication authorizes only AdamW/Muon and 6,144,000 tokens each')
    return {**_CONTRACT(variant, iterations), 'seed': 42, 'replica_format': 1}


def identity(trainer):
    _IDENTITY(trainer)
    trainer.identity['scripts/replicate.py'] = sha256(Path(__file__))


def fresh(variant, iterations=ITERATIONS):
    value = contract(variant, iterations)
    campaign.configure(BASE_EXPERIMENT)
    trainer = Trainer()
    if variant == 'muon':
        trainer.optimizer = campaign.MuonAdamW(trainer.model)
    campaign.configure(value)
    identity(trainer)
    return trainer


def legacy(checkpoint):
    campaign.configure(BASE_EXPERIMENT)
    trainer, metadata = _RESTORE_LEGACY(checkpoint)
    if metadata['experiment'] != CONTINUATION_EXPERIMENT or trainer.counters['iterations'] != 490:
        raise ValueError('Replication continuation requires the complete seed42 490-step baseline')
    return trainer, metadata


@contextmanager
def factories():
    # ponytail: process-scoped factories preserve hashed legacy sources; parallel runs need separate processes.
    source = LEGACY_RUN / 'checkpoints' / json.loads((LEGACY_RUN / 'references.json').read_text())['latest']
    replacements = dict(contract=contract, fresh=fresh, campaign_identity=identity,
                        restore_legacy=legacy, SOURCE=source)
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
    with factories():
        campaign.main()


if __name__ == '__main__':
    main()
