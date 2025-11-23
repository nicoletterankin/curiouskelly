import os
import sys
from dotenv import load_dotenv
import psycopg2

# Load .env from current directory (content-engine)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
print(f"📄 Loading .env from: {env_path}")
load_dotenv(env_path, override=True)

DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in .env!")
    sys.exit(1)

print(f"🔍 Connecting to: {DATABASE_URL[:60]}...")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    
    tables = cursor.fetchall()
    print(f"\n✅ Connected! Found {len(tables)} table(s):")
    for table in tables:
        print(f"  📊 {table[0]}")
    
    if not tables:
        print("\n❌ NO TABLES FOUND!")
        print("   You need to run: python scripts/deploy_schema.py")
    else:
        # Check counts
        print("\n📈 Row Counts:")
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"  {table_name}: {count} rows")
            except Exception as e:
                print(f"  {table_name}: Error - {e}")
    
    cursor.close()
    conn.close()
    
    print("\n✅ Database check complete!")
    
except Exception as e:
    print(f"\n❌ Connection failed!")
    print(f"   Error: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Check your DATABASE_URL in .env")
    print("   2. Verify Supabase project is running")
    print("   3. Check if password has special characters (might need URL encoding)")
    sys.exit(1)






