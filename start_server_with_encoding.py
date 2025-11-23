import http.server
import socketserver
import os
import sys

PORT = 8000

class UnityHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Handle Brotli compressed Unity files
        if self.path.endswith('.br'):
            # Determine actual file path
            path = self.translate_path(self.path)
            
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        # Determine content type based on double extension
                        ctype = 'application/octet-stream'
                        if self.path.endswith('.js.br'):
                            ctype = 'application/javascript'
                        elif self.path.endswith('.wasm.br'):
                            ctype = 'application/wasm'
                        elif self.path.endswith('.data.br'):
                            ctype = 'application/octet-stream'
                        
                        fs = os.fstat(f.fileno())
                        
                        self.send_response(200)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Encoding", "br")
                        self.send_header("Content-Length", str(fs.st_size))
                        self.end_headers()
                        
                        self.copyfile(f, self.wfile)
                        return
                except Exception as e:
                    print(f"Error serving {path}: {e}")
                    self.send_error(500, "Internal Server Error")
                    return
            else:
                self.send_error(404, "File not found")
                return

        # Default behavior for non-br files
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

print(f"🚀 Starting Curious Kelly Server at http://localhost:{PORT}")
print("✨ Serving with Brotli (.br) support for Unity WebGL")

# Allow reusing address to avoid "Address already in use" if we restart quickly
socketserver.TCPServer.allow_reuse_address = True

try:
    with socketserver.TCPServer(("", PORT), UnityHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")
            sys.exit(0)
except OSError as e:
    print(f"❌ Error: Port {PORT} is busy. Please stop other servers or try a different port.")
