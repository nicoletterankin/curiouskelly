import http.server
import socketserver
import os
import sys

PORT = 8000

class UnityHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Special handling for the UnityWeb Compressed Content file
        # We renamed it to .data, but it contains the custom header
        # We want the browser to download it as-is, and let the Unity Loader handle it?
        # OR we want to strip the header?
        
        # Wait, if the header is "UnityWeb Compressed Content", the Unity Loader handles it!
        # The Unity Loader (kelly-v1.loader.js) likely reads this header and decompresses it 
        # using a WASM decompressor included in the framework.
        
        # So we should serve it as a plain binary file WITHOUT Content-Encoding: br
        # because it's NOT standard HTTP Brotli stream, it's a Unity-specific container.
        
        if self.path.endswith('.data'):
             self.send_response(200)
             self.send_header('Content-Type', 'application/octet-stream')
             # DO NOT send Content-Encoding: br
             self.send_header('Access-Control-Allow-Origin', '*')
             self.end_headers()
             
             # Serve the file
             path = self.translate_path(self.path)
             with open(path, 'rb') as f:
                 self.copyfile(f, self.wfile)
             return

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

print(f"🚀 Kelly Asset Server (Unity Native) Running on port {PORT}")
print("✨ Serving .data as application/octet-stream (No Browser Decompression)")

socketserver.TCPServer.allow_reuse_address = True
try:
    with socketserver.TCPServer(("", PORT), UnityHandler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\n🛑 Server stopped.")
    sys.exit(0)
except OSError as e:
    print(f"❌ Error: Port {PORT} is busy. {e}")




















