import argparse
import asyncio
import json
import logging
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / '.cache' / 'colab-queue'


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-id')
    args = parser.parse_args()
    if args.session_id and (not args.session_id.isascii() or not all(c.isalnum() or c in '-_' for c in args.session_id)):
        raise ValueError('Session ID must contain only ASCII letters, digits, hyphens or underscores')
    queue = QUEUE / args.session_id if args.session_id else QUEUE
    output = ROOT / 'output/colab-sessions' / args.session_id if args.session_id else ROOT / 'output'
    queue.mkdir(parents=True, exist_ok=not bool(args.session_id))
    output.mkdir(parents=True, exist_ok=not bool(args.session_id))
    server = ROOT / '.cache/vendor/colab-mcp/.venv/Scripts/colab-mcp.exe'
    transport = StdioTransport(command=str(server), args=['--log', str(ROOT / 'logs/app')])
    async with Client(transport, timeout=120) as client:
        tools = await client.list_tools()
        (output / 'colab-tools.json').write_text(json.dumps([t.model_dump() for t in tools], indent=2))
        logging.info('Official MCP initialized; opening browser connection')
        result = await client.call_tool('open_colab_browser_connection', {})
        (output / 'colab-connection.json').write_text(json.dumps(result.data))
        while True:
            for request in sorted(queue.glob('*.request.json')):
                response = request.with_name(request.name.replace('.request.json', '.response.json'))
                if response.exists():
                    continue
                try:
                    command = json.loads(request.read_text())
                    if command['action'] == 'list':
                        value = [t.model_dump() for t in await client.list_tools()]
                    elif command['action'] == 'stop':
                        response.write_text('{"stopped": true}')
                        return
                    else:
                        result = await client.call_tool(command['tool'], command.get('arguments', {}))
                        value = {'data': result.data, 'content': [c.model_dump() for c in result.content], 'is_error': result.is_error}
                    response.write_text(json.dumps(value, indent=2, default=str))
                except Exception as exc:
                    response.write_text(json.dumps({'error': str(exc)}))
            await asyncio.sleep(0.5)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
