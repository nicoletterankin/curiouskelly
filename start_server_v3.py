import http.server
import socketserver
import os
import sys

PORT = 8000

class UnityHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"Request for: {self.path}")
        
        # Map uncompressed requests to compressed files if they don't exist
        if self.path.endswith('.data') and not os.path.exists(self.translate_path(self.path)):
             br_path = self.path + ".br"
             if os.path.exists(self.translate_path(br_path)):
                 print(f"➡️ Redirecting {self.path} to {br_path} (Brotli)")
                 self.path = br_path
                 
        # Handle Brotli compressed Unity files
        if self.path.endswith('.br'):
            path = self.translate_path(self.path)
            
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        # Determine content type based on original extension
                        ctype = 'application/octet-stream'
                        if '.js.br' in self.path:
                            ctype = 'application/javascript'
                        elif '.wasm.br' in self.path:
                            ctype = 'application/wasm'
                        elif '.data.br' in self.path:
                            ctype = 'application/octet-stream'
                        
                        fs = os.fstat(f.fileno())
                        
                        self.send_response(200)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Encoding", "br")
                        self.send_header("Content-Length", str(fs.st_size))
                        # Add CORS and caching headers
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        
                        self.copyfile(f, self.wfile)
                        return
                except Exception as e:
                    print(f"Error serving {path}: {e}")
                    self.send_error(500, "Internal Server Error")
                    return
            else:
                self.send_error(404, f"File not found: {path}")
                return

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

print(f"🚀 Kelly Asset Server Running on port {PORT}")
print("✨ Handling .br files with Content-Encoding: br")

socketserver.TCPServer.allow_reuse_address = True
try:
    with socketserver.TCPServer(("", PORT), UnityHandler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Server stopped.")
    sys.exit(0)
except OSError as e:
    print(f"❌ Error: Port {PORT} is busy. {e}")














































