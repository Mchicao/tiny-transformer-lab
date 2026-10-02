import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from google.colab import drive


def main():
    started = time.monotonic()
    result = {'optimizer_steps': 0, 'mount_timeout_ms': 60000}
    try:
        drive.mount('/content/drive', timeout_ms=60000)
        result['mount'] = 'passed'
        target = Path('/content/drive/MyDrive/Proyectos_B/tiny-transformer-lab/persistence-checks')
        target.mkdir(parents=True, exist_ok=True)
        run_id = f'prep-{uuid.uuid4().hex}'
        path = target / f'{run_id}.json'
        payload = json.dumps({'project': 'tiny-transformer-lab', 'run_id': run_id, 'optimizer_steps': 0}, sort_keys=True).encode()
        digest = hashlib.sha256(payload).hexdigest()
        with path.open('xb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        result.update(path=str(path), sha256=digest, bytes=len(payload), immediate_readback=True)
        drive.flush_and_unmount()
        assert not Path('/content/drive/MyDrive').exists()
        drive.mount('/content/drive', timeout_ms=60000)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        result['readback_after_remount'] = True
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        known_errors = ['mount failed', 'credential propagation was unsuccessful', 'Message timed out', 'domain policy has disabled Drive File Stream']
        result['error_class'] = next((message for message in known_errors if message in str(exc)), 'other failure')
        result['mount'] = result.get('mount', 'failed')
        phrases = ['timeout', 'timed out', 'credential', 'unauthenticated', 'permission_denied',
                   'invalid_grant', 'access denied', 'network', 'domain policy', 'oauth', 'error']
        result['drivefs_logs'] = []
        for base in [Path('/root/.config/Google/DriveFS/Logs'), Path('/tmp/DriveFS/Logs')]:
            if not base.exists():
                continue
            for log in sorted(base.glob('*.txt')):
                text = log.read_text(errors='replace')
                result['drivefs_logs'].append({
                    'file': log.name, 'bytes': log.stat().st_size,
                    'diagnostic_phrase_counts': {phrase: len(re.findall(re.escape(phrase), text, re.IGNORECASE)) for phrase in phrases},
                })
    result['elapsed_s'] = round(time.monotonic() - started, 3)
    scratch = Path('/content/tiny-transformer-preparation')
    scratch.mkdir(exist_ok=True)
    (scratch / f'drive-check-{uuid.uuid4().hex}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result.get('readback_after_remount'), 'Drive persistence not proven'


if __name__ == '__main__':
    main()
