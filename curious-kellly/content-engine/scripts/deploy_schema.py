import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def deploy_schema():
    schema_path = os.path.join("database", "schema.sql")
    with open(schema_path, "r") as f:
        sql = f.read()

    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        
        print("🔥 Dropping old tables...")
        cur.execute("DROP TABLE IF EXISTS lesson_atoms CASCADE;")
        cur.execute("DROP TABLE IF EXISTS core_lessons CASCADE;")
        
        print("🚀 Deploying New Schema...")
        cur.execute(sql)
        conn.commit()
        
        print("✅ Tables Created Successfully!")
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Deployment Failed: {e}")

if __name__ == "__main__":
    deploy_schema()
