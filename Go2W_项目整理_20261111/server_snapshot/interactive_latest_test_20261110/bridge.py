"""Loopback-only input bridge, accessed through an SSH tunnel."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


def serve(controller):
    page = Path(__file__).with_name('controls_panel.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path == '/status':
                data, kind = json.dumps(controller.status(),ensure_ascii=False).encode(), 'application/json; charset=utf-8'
            elif self.path == '/':
                data, kind = page, 'text/html; charset=utf-8'
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed_origins={f'http://{host}:{port}' for host in ('127.0.0.1','localhost')
                             for port in (8765,controller.cfg['bridge_port'])}
            if origin and origin not in allowed_origins:
                self.send_error(403); return
            try:
                length = int(self.headers.get('Content-Length',0))
                if length > 4096: raise ValueError('oversized')
                msg = json.loads(self.rfile.read(length))
                source = 'bridge:'+str(msg['session'])[:80]
                if self.path == '/release': controller.release(source)
                elif self.path == '/input':
                    if not controller.feed(source,msg['keys']):
                        self.send_response(409);self.end_headers();return
                else: self.send_error(404); return
                self.send_response(204); self.end_headers()
            except (ValueError,KeyError,TypeError): self.send_error(400)
    server = ThreadingHTTPServer(('127.0.0.1',controller.cfg['bridge_port']),Handler)
    Thread(target=server.serve_forever,daemon=True).start()
    return server
