import os
import sys
import json
import psycopg2
from dotenv import load_dotenv

# Fix Import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.persona_generator import PersonaGenerator

load_dotenv()

def generate_day1():
    # 1. Connect to DB
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        print("✅ Connected to Database")
    except Exception as e:
        print(f"❌ Database Connection Failed: {e}")
        return

    # 2. Initialize Generator
    generator = PersonaGenerator()
    
    # 3. Fetch Day 1 (The Sun)
    cur.execute("SELECT id, topic, universal_truth FROM core_lessons WHERE day_number = 1;")
    lesson = cur.fetchone()
    
    if not lesson:
        print("❌ Day 1 Lesson not found in Core. Checking for any lesson...")
        cur.execute("SELECT id, topic, universal_truth FROM core_lessons LIMIT 1;")
        lesson = cur.fetchone()
        if not lesson:
             print("❌ No lessons found. Run migration first!")
             return
        
    lesson_id, topic, raw_truth = lesson
    
    # Handle text vs json
    core_fact = raw_truth
    if isinstance(raw_truth, dict):
        core_fact = raw_truth.get('description', str(raw_truth))

    print(f"🌞 Generating Atoms for: {topic} (ID: {lesson_id})")
    print(f"ℹ️ Core Fact: {core_fact[:50]}...")

    # 4. Generate for 3 Archetypes
    archetypes = ["The Survivor", "The Mystic", "The Scientist"]
    phases = ["Hook", "Fact1", "Wisdom"] 
    
    for arch in archetypes:
        print(f"\n--- 🎭 Archetype: {arch} ---")
        
        for phase in phases:
            print(f"   Generating {phase}...")
            
            atom = generator.generate_atom(
                topic=topic,
                core_fact=core_fact,
                archetype=arch,
                phase=phase
            )
            
            if atom:
                # Save to DB (matching actual schema: core_lesson_id, archetype, phase, content)
                cur.execute("""
                    INSERT INTO lesson_atoms 
                    (core_lesson_id, archetype, phase, content)
                    VALUES (%s, %s, %s, %s);
                """, (lesson_id, arch, phase, json.dumps({
                    "script": atom.script,
                    "options": [
                        {"text": o.text, "response": o.response} for o in atom.options
                    ],
                    "asl_gloss": atom.asl_gloss
                })))
                print(f"   ✅ Saved {phase} Atom")
            else:
                print(f"   ❌ Failed to generate {phase}")

    conn.commit()
    print("\n✅ IGNITION COMPLETE. Atoms are live in Supabase.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    generate_day1()
