import os
import json
import glob
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Connect to DB
try:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    print("✅ Connected to Database")
except Exception as e:
    print(f"❌ Database Connection Failed: {e}")
    exit(1)

def migrate_lessons():
    # Path to existing JSON files
    # Adjust path relative to where script is run
    json_path = os.path.join("..", "backend", "config", "lessons", "*.json")
    files = glob.glob(json_path)
    
    print(f"Found {len(files)} lesson files to migrate...")

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Skip if not a valid DNA file
                if 'title' not in data or 'id' not in data:
                    continue

                # 1. Insert into Core Lessons
                # Mapping existing JSON fields to our new Schema
                lesson_id = data['id']
                title = data['title']
                day = data.get('calendar', {}).get('day', 0)
                
                # Extract Universal Facts (using keyPoints from age variants as a proxy for now)
                # Ideally, we'd have a separate field, but we'll grab from the oldest age group
                universal_facts = {
                    "description": data.get('description', ''),
                    "tags": data.get('tags', []),
                    "learning_objectives": data.get('learningObjectives', [])
                }

                print(f"Migrating: {title} (Day {day})")

                # Upsert Core Lesson
                cur.execute("""
                    INSERT INTO core_lessons (id, topic, day_number, universal_facts)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE 
                    SET topic = EXCLUDED.topic, 
                        universal_facts = EXCLUDED.universal_facts;
                """, (lesson_id, title, day, json.dumps(universal_facts)))

                # 2. (Optional) Migrate existing static shards?
                # For now, we might just want to migrate the Core, 
                # and let the Engine regenerate the Atoms freshly using the new Archetype system.
                # Migrating the old static text into the dynamic Atom system is messy 
                # because the old text isn't split by "Hook/Fact/Wisdom".
                
            except Exception as e:
                print(f"❌ Error migrating {file_path}: {e}")

    conn.commit()
    print("✅ Migration Complete!")
    cur.close()
    conn.close()

if __name__ == "__main__":
    migrate_lessons()






