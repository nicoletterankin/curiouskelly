import http.server
import socketserver
import os
import sys

PORT = 8000

class UnityHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve uncompressed files normally
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

print(f"🚀 Kelly Asset Server (Simple) Running on port {PORT}")

socketserver.TCPServer.allow_reuse_address = True
try:
    with socketserver.TCPServer(("", PORT), UnityHandler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Server stopped.")
    sys.exit(0)
except OSError as e:
    print(f"❌ Error: Port {PORT} is busy. {e}")



























