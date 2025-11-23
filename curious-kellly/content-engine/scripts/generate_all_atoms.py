import os
import sys
import json
import psycopg2
import time
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generators.persona_generator import PersonaGenerator

load_dotenv()

ARCHETYPES = [
    "The Survivor", "The MacGyver", "The Provider", 
    "The Empath", "The Storyteller", "The Diplomat", 
    "The Scientist", "The Explorer", "The Mystic", 
    "The Architect", "The Strategist", "The Rebel"
]
PHASES = ["Hook", "Fact1", "Fact2", "Fact3", "Wisdom"]

def atom_exists(cur, lesson_id, archetype, phase):
    """Check if atom already exists in database"""
    cur.execute("""
        SELECT COUNT(*) FROM lesson_atoms 
        WHERE core_lesson_id = %s 
        AND archetype = %s 
        AND phase = %s
    """, (lesson_id, archetype, phase))
    return cur.fetchone()[0] > 0

def generate_all():
    # 1. Connect to DB
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

    # 2. Initialize Generator
    try:
        generator = PersonaGenerator() # Defaults to gemini-2.5-flash
        print("✅ Generator Initialized")
    except Exception as e:
        print(f"❌ Generator Init Failed: {e}")
        return

    # 3. Fetch all lessons
    cur.execute("SELECT id, topic, universal_truth FROM core_lessons ORDER BY day_number;")
    lessons = cur.fetchall()
    
    if not lessons:
        print("❌ No lessons found in core_lessons table. Run bulk_insert_core.py first!")
        return

    # Calculate total work
    total_atoms = len(lessons) * len(ARCHETYPES) * len(PHASES)
    print(f"🚀 Starting Generation Factory")
    print(f"   - Lessons: {len(lessons)}")
    print(f"   - Archetypes: {len(ARCHETYPES)}")
    print(f"   - Phases: {len(PHASES)}")
    print(f"   - Total Atoms: {total_atoms}")
    
    # 4. Main Loop
    with tqdm(total=total_atoms) as pbar:
        for lesson_id, topic, fact in lessons:
            for arch in ARCHETYPES:
                for phase in PHASES:
                    try:
                        # Skip if exists
                        if atom_exists(cur, lesson_id, arch, phase):
                            # pbar.write(f"⏭️  Skipping existing: {topic} / {arch} / {phase}")
                            pbar.update(1)
                            continue

                        # Generate
                        atom = generator.generate_atom(topic, fact, arch, phase)
                        
                        if atom:
                            # Serialize options
                            options_json = [
                                {
                                    "text": o.text, 
                                    "response": o.response, 
                                    "learning_value": o.learning_value
                                } for o in atom.options
                            ]
                            
                            content_json = json.dumps({
                                "script": atom.script,
                                "options": options_json,
                                "asl_gloss": atom.asl_gloss
                            })

                            # Insert
                            cur.execute("""
                                INSERT INTO lesson_atoms (core_lesson_id, archetype, phase, content)
                                VALUES (%s, %s, %s, %s);
                            """, (lesson_id, arch, phase, content_json))
                            conn.commit()
                            
                            # Rate limit protection (simple)
                            time.sleep(0.5) 
                        else:
                            pbar.write(f"⚠️  Generation returned None for {topic}/{arch}/{phase}")

                    except Exception as e:
                        conn.rollback()
                        pbar.write(f"❌ Failed: {topic}/{arch}/{phase} - {e}")
                    
                    pbar.update(1)
    
    cur.close()
    conn.close()
    print("✅ FACTORY COMPLETE")

if __name__ == "__main__":
    generate_all()
