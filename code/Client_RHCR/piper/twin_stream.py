"""Loopback-only JPEG frame cache. Rendering stays on the MuJoCo main thread."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class TwinFrameServer:
    def __init__(self, port=50556):
        self.jpeg = None
        self.lock = threading.Lock()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/frame.jpg":
                    self.send_error(404); return
                with owner.lock:
                    jpeg = owner.jpeg
                if jpeg is None:
                    self.send_error(503); return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(jpeg)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def publish(self, jpeg):
        with self.lock:
            self.jpeg = bytes(jpeg)
    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
