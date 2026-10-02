import argparse
import json
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Call the official MCP through the active local session')
parser.add_argument('tool', help='Tool name, list, or stop')
parser.add_argument('--arguments-file', type=Path)
parser.add_argument('--session-id')
args = parser.parse_args()
if args.session_id and (not args.session_id.isascii() or not all(c.isalnum() or c in '-_' for c in args.session_id)):
    raise ValueError('Session ID must contain only ASCII letters, digits, hyphens or underscores')
queue = ROOT / '.cache/colab-queue'
if args.session_id:
    queue = queue / args.session_id
if not queue.exists():
    raise SystemExit('Start scripts/agents/start-colab.ps1 first')
request = queue / f'{uuid.uuid4().hex}.request.json'
response = request.with_name(request.name.replace('.request.json', '.response.json'))
command = {'action': args.tool} if args.tool in {'list', 'stop'} else {'action': 'call', 'tool': args.tool, 'arguments': json.loads(args.arguments_file.read_text()) if args.arguments_file else {}}
request.write_text(json.dumps(command))
deadline = time.monotonic() + 150
while not response.exists():
    if time.monotonic() >= deadline:
        raise SystemExit('No MCP response within 150s; request remains available for inspection')
    time.sleep(0.25)
result = json.loads(response.read_text())
print(json.dumps(result, indent=2))
if isinstance(result, dict) and (result.get('error') or result.get('is_error')):
    raise SystemExit(1)
