"""Serve the viewer locally; automatically try another port when one is occupied."""
import argparse
import http.server
import socketserver
from pathlib import Path
from functools import partial

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, max-age=0')
        super().end_headers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    handler = partial(NoCache, directory=str(Path(__file__).resolve().parent))
    for port in range(args.port, min(args.port + 20, 65536)):
        try:
            server = socketserver.TCPServer(('127.0.0.1', port), handler)
            break
        except OSError as error:
            if error.errno not in (48, 98):
                raise
    else:
        raise SystemExit('No free port found. Try --port 9000.')
    print(f'Open http://localhost:{port} (Control+C to stop)', flush=True)
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
