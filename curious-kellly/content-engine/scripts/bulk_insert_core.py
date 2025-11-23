import os
import sys
import json
import psycopg2
from dotenv import load_dotenv

# Add parent directory to path to find modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Path to the curriculum file
CURRICULUM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'curriculum_365.json')

def bulk_insert_core():
    print("🚀 Starting Bulk Insert of Core Lessons...")

    # 1. Connect to Database
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            print("❌ Error: DATABASE_URL is not set in .env")
            return

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        print("✅ Connected to Database")
    except Exception as e:
        print(f"❌ Database Connection Failed: {e}")
        return

    # 2. Load JSON Data
    try:
        with open(CURRICULUM_PATH, 'r', encoding='utf-8') as f:
            lessons = json.load(f)
        print(f"📄 Loaded {len(lessons)} lessons from {CURRICULUM_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Curriculum file not found at {CURRICULUM_PATH}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {CURRICULUM_PATH}")
        return

    # 3. Insert Data
    inserted_count = 0
    updated_count = 0

    try:
        for lesson in lessons:
            day = lesson.get('day_number')
            topic = lesson.get('topic')
            truth = lesson.get('universal_truth')
            desc = lesson.get('description', '')

            if not day or not topic or not truth:
                print(f"⚠️ Skipping invalid lesson entry: {lesson}")
                continue

            # Check if exists
            cur.execute("SELECT id FROM core_lessons WHERE day_number = %s", (day,))
            existing = cur.fetchone()

            if existing:
                # Update
                cur.execute("""
                    UPDATE core_lessons 
                    SET topic = %s, universal_truth = %s, description = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE day_number = %s
                """, (topic, truth, desc, day))
                updated_count += 1
            else:
                # Insert
                cur.execute("""
                    INSERT INTO core_lessons (day_number, topic, universal_truth, description)
                    VALUES (%s, %s, %s, %s)
                """, (day, topic, truth, desc))
                inserted_count += 1
        
        conn.commit()
        print(f"✅ Success! Inserted: {inserted_count}, Updated: {updated_count}")
        print(f"🎉 Total Core Lessons in DB: {inserted_count + updated_count}")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error during insertion: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    bulk_insert_core()
