#!/usr/bin/env python3
"""
Simple HTTP Server for Voice Library Manager
Serves the HTML interface and voice library data locally
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

def serve_voice_library(port=8000):
    """Serve the voice library manager on the specified port"""
    
    print("🎵 Starting Voice Library Manager Server")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if required files exist
    html_file = "voice_library_manager_enhanced.html"
    data_file = "voice_library_data.json"
    
    if not Path(html_file).exists():
        print(f"❌ Error: {html_file} not found!")
        return
    
    if not Path(data_file).exists():
        print(f"⚠️  Warning: {data_file} not found. Using sample data.")
    
    # Create custom handler to serve files
    class VoiceLibraryHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers for local development
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
        
        def do_GET(self):
            # Redirect root to the HTML file
            if self.path == '/':
                self.path = f'/{html_file}'
            return super().do_GET()
    
    try:
        # Start the server
        with socketserver.TCPServer(("", port), VoiceLibraryHandler) as httpd:
            print(f"🌐 Server running at: http://localhost:{port}")
            print(f"📁 Serving directory: {script_dir}")
            print(f"🎵 Voice Library Manager: http://localhost:{port}")
            print("\nPress Ctrl+C to stop the server")
            
            # Try to open the browser automatically
            try:
                webbrowser.open(f'http://localhost:{port}')
                print("🚀 Opening browser automatically...")
            except:
                print("💡 Open your browser and navigate to the URL above")
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python serve_voice_library.py --port {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Serve Voice Library Manager locally")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to serve on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    if args.no_browser:
        # Temporarily disable webbrowser
        import webbrowser
        webbrowser.open = lambda x: None
    
    serve_voice_library(args.port)

if __name__ == "__main__":
    main()
