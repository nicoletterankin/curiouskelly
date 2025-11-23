"""Simple HTTP server to serve status as JSON for the dashboard"""
import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
import psycopg2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path, override=True)

DATABASE_URL = os.getenv('DATABASE_URL')

class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            try:
                conn = psycopg2.connect(DATABASE_URL)
                cursor = conn.cursor()
                
                # Get counts
                cursor.execute("SELECT COUNT(*) FROM core_lessons")
                lessons = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM lesson_atoms")
                atoms = cursor.fetchone()[0]
                
                cursor.close()
                conn.close()
                
                # Calculate progress
                target = 21900
                progress = (atoms / target) * 100
                remaining = target - atoms
                hours_left = (remaining * 3.6) / 3600
                
                status_data = {
                    "lessons": lessons,
                    "atoms": atoms,
                    "target": target,
                    "progress": round(progress, 2),
                    "remaining": remaining,
                    "eta_hours": round(hours_left, 1),
                    "status": "running" if atoms < target else "complete"
                }
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(status_data).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress log messages

if __name__ == '__main__':
    PORT = 5500
    server = HTTPServer(('localhost', PORT), StatusHandler)
    print(f"✅ Status server running on http://localhost:{PORT}/status")
    print("   Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.shutdown()






