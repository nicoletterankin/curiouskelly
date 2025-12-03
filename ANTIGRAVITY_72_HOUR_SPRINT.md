# 72-HOUR PRODUCTION SPRINT: 365 DAYS OF CONTENT

## MISSION BRIEFING
You have 72 hours to generate production-ready Atomic Shards for 365 lessons × 12 Archetypes × 5 Phases = **21,900 Atoms**.

Current Status:
- ✅ 60+ Core Lessons migrated to Supabase
- ✅ 8 Atoms generated (Proof of Concept)
- ❌ 305 Core Lessons missing (Need to draft)
- ❌ 21,892 Atoms missing (Need to generate)

## HOUR 0-12: THE CONTENT DRAFT (MISSING 305 LESSONS)

### Task 1A: Draft the 365-Day Curriculum Map
Create `src/data/curriculum_365.json`:
```json
{
  "1": {"topic": "The Sun", "fact": "The Sun is a star..."},
  "2": {"topic": "The Moon", "fact": "The Moon orbits Earth..."},
  ...
  "365": {"topic": "The Year Ahead", "fact": "Reflection and renewal..."}
}
```

**Rules:**
- Use the existing 60 lessons as Day 1-60.
- Draft the remaining 305 lessons (Days 61-365).
- Keep `fact` to 1-2 sentences (The Universal Truth).
- Themes: Science (Days 1-100), Humanities (101-200), Creativity (201-300), Life Skills (301-365).

### Task 1B: Mass-Insert Core Lessons
Create `src/scripts/bulk_insert_core.py`:
1. Read `curriculum_365.json`.
2. Insert all 365 lessons into `core_lessons` table.
3. Handle duplicates gracefully (UPSERT).
4. **Output:** "✅ 365 Core Lessons Ready"

**Deadline:** Hour 12

---

## HOUR 12-48: THE GENERATION FACTORY (21,900 ATOMS)

### Task 2A: The Batch Generator
Create `src/scripts/generate_all_atoms.py`:

**Logic:**
1. Fetch all `core_lessons` (365 rows).
2. For each lesson:
   - Loop through 12 Archetypes.
   - Loop through 5 Phases (Hook, Fact1, Fact2, Fact3, Wisdom).
   - Call `PersonaGenerator.generate_atom()`.
   - Insert into `lesson_atoms`.
3. Use `tqdm` progress bar to track completion.
4. Implement retry logic for API failures.
5. Save checkpoint every 100 atoms (in case of crash).

**Optimization:**
- Batch API calls (10 at a time) to maximize throughput.
- Use `gemini-2.5-flash` (fastest model).
- Estimated time: ~36 hours at 10 atoms/min.

### Task 2B: The Error Handler
Create `src/scripts/fix_failed_atoms.py`:
- Query `lesson_atoms` to find missing combinations.
- Re-run generation for failed atoms only.

**Deadline:** Hour 48

---

## HOUR 48-60: QUALITY ASSURANCE

### Task 3A: The Validation Script
Create `src/scripts/validate_atoms.py`:
1. Check that all 21,900 atoms exist.
2. Validate JSON structure (script, options, responses).
3. Flag atoms with broken ASL markers.
4. Output: CSV report of issues.

### Task 3B: The Sample Audit
Manually review 10 random atoms (one per archetype + phase combo).
- Do they follow the Kelly Constitution?
- Are metaphors age-appropriate?
- Are responses validating ("Yes, and...")?

**Deadline:** Hour 60

---

## HOUR 60-72: DEPLOYMENT & SMOKE TEST

### Task 4A: Frontend Connection Test
Create a minimal test page that:
1. Connects to Supabase.
2. Fetches one random atom.
3. Renders the script + options.
4. Logs the response.

### Task 4B: The Launch Checklist
- [ ] All 365 core_lessons in DB
- [ ] All 21,900 lesson_atoms in DB
- [ ] Validation report shows <1% error rate
- [ ] Frontend can query and render atoms
- [ ] Backup of Supabase DB created

**Deadline:** Hour 72

---

## EXECUTION PRIORITY

**DO THIS FIRST:**
1. Run `bulk_insert_core.py` (Get to 365 lessons).
2. Run `generate_all_atoms.py` (Start the factory).
3. Let it run overnight (Hour 12-48).
4. Monitor for API rate limits or crashes.

**DO NOT:**
- Try to parallelize beyond 10 concurrent calls (Gemini will throttle).
- Generate atoms for lessons that don't exist in `core_lessons`.
- Stop the script manually unless it's erroring repeatedly.

---

## CODE TEMPLATE: BULK GENERATOR

```python
import os
import json
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm
from generators.persona_generator import PersonaGenerator

load_dotenv()

ARCHETYPES = ["The Survivor", "The MacGyver", "The Provider", "The Empath", 
              "The Storyteller", "The Diplomat", "The Scientist", "The Explorer",
              "The Mystic", "The Architect", "The Strategist", "The Rebel"]
PHASES = ["Hook", "Fact1", "Fact2", "Fact3", "Wisdom"]

def generate_all():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    generator = PersonaGenerator()
    
    # Fetch all lessons
    cur.execute("SELECT id, topic, universal_truth FROM core_lessons ORDER BY day_number;")
    lessons = cur.fetchall()
    
    total = len(lessons) * len(ARCHETYPES) * len(PHASES)
    print(f"🚀 Generating {total} atoms...")
    
    with tqdm(total=total) as pbar:
        for lesson_id, topic, fact in lessons:
            for arch in ARCHETYPES:
                for phase in PHASES:
                    try:
                        atom = generator.generate_atom(topic, fact, arch, phase)
                        if atom:
                            cur.execute("""
                                INSERT INTO lesson_atoms (core_lesson_id, archetype, phase, content)
                                VALUES (%s, %s, %s, %s);
                            """, (lesson_id, arch, phase, json.dumps({
                                "script": atom.script,
                                "options": [{"text": o.text, "response": o.response} for o in atom.options],
                                "asl_gloss": atom.asl_gloss
                            })))
                            conn.commit()
                    except Exception as e:
                        print(f"❌ Failed: {topic}/{arch}/{phase} - {e}")
                    
                    pbar.update(1)
    
    cur.close()
    conn.close()
    print("✅ FACTORY COMPLETE")

if __name__ == "__main__":
    generate_all()
```

---

## YOUR FIRST ACTION

Run this NOW:
`python src/scripts/bulk_insert_core.py`

Then immediately start:
`python src/scripts/generate_all_atoms.py`

Report progress every 6 hours.

**GO. THE CLOCK IS TICKING.**




















