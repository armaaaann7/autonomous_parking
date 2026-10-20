"""Static server with caching disabled, so edits and re-exported episodes always show up."""
import http.server, socketserver

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", 8000), NoCache) as s:
    print("open http://localhost:8000  (Ctrl+C to stop)")
    s.serve_forever()
