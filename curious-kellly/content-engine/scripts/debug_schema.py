import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

try:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    
    print("🔍 Inspecting core_lessons columns:")
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'core_lessons';
    """)
    for row in cur.fetchall():
        print(f" - {row[0]} ({row[1]})")
        
    cur.close()
    conn.close()

except Exception as e:
    print(e)






