#!/usr/bin/env python3
"""
Curious Kelly Content Generation System
Generates lesson atoms and shards for all 365 days
"""

import os
import json
import time
import psycopg2
from datetime import datetime
from typing import Dict, List, Optional
import google.generativeai as genai

# Database connection
DB_URL = "postgresql://antigravity:antigravity123@localhost:5432/antigravity_dev"

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
else:
    print("⚠️  GEMINI_API_KEY not set. Using fallback content generation.")
    model = None

# Constants
PHASES = ["welcome", "q1", "q2", "q3", "wisdom"]
ARCHETYPES = [
    "Survivor", "Caregiver", "Explorer", "Rebel", "Lover", "Creator",
    "Jester", "Sage", "Magician", "Hero", "Everyman", "Ruler"
]
AGE_GROUPS = [5, 10, 15, 25, 40, 65]
TONES = ["curious", "playful", "serious"]

# Kelly Constitution principles
KELLY_CONSTITUTION = """
Kelly MUST embody these 5 principles:

1. Graceful Authority - Confident without being arrogant
   ✅ "Here's something fascinating..."
   ❌ "Let me teach you..."

2. Radical Curiosity - Genuinely excited to explore WITH the learner
   ✅ "I wonder what would happen if..."
   ❌ "The answer is..."

3. Warm Neutrality - Present multiple perspectives without agenda
   ✅ "Some think X, others believe Y. What resonates with you?"
   ❌ "The right answer is X."

4. Concise Poetics - Beautiful, memorable language without waste
   ✅ "Water remembers every shape it's held."
   ❌ "Water is a liquid that can take the shape of its container..."

5. "Yes, And..." - Build on every response, never shut down thinking
   ✅ "That's interesting! And what if we also considered..."
   ❌ "That's not quite right."
"""


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DB_URL)


def get_core_lesson(day_number: int) -> Optional[Dict]:
    """Fetch core lesson data for a given day"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, topic, universal_truth, marketing_headline, marketing_tagline
        FROM core_lessons
        WHERE day_number = %s
    """, (day_number,))
    row = cur.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "topic": row[1],
            "truth": row[2],
            "headline": row[3],
            "tagline": row[4]
        }
    return None


def generate_atom_content(topic: str, truth: str, phase: str, archetype: str) -> Dict:
    """Generate content for a single lesson atom using Gemini API"""
    
    if not model:
        # Fallback content if no API key
        return generate_fallback_atom(topic, phase, archetype)
    
    prompt = f"""Generate content for Curious Kelly lesson atom.

TOPIC: {topic}
UNIVERSAL TRUTH: {truth}
PHASE: {phase}
ARCHETYPE: {archetype}

{KELLY_CONSTITUTION}

PHASE DESCRIPTIONS:
- welcome: Kelly introduces the topic warmly (30-45 seconds)
- q1/q2/q3: Interactive questions with 3 choices (each 60-90 seconds)
- wisdom: Kelly's closing reflection (45-60 seconds)

ARCHETYPE LENS:
Apply the {archetype} archetype perspective to this content.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "title": "Engaging phase title",
  "script": "Kelly's spoken text (conversational, warm, {archetype} perspective)",
  "prompt": "{('Reflection question' if phase in ['q1', 'q2', 'q3'] else 'null')}",
  "options": [{('array of 3 options with responses' if phase in ['q1', 'q2', 'q3'] else 'null')}]
}}

{('For question phases, include:' if phase in ['q1', 'q2', 'q3'] else '')}
{('"options": [' if phase in ['q1', 'q2', 'q3'] else '')}
{('  {"text": "Choice A", "response": "Kelly responds to A with Yes-And energy"},' if phase in ['q1', 'q2', 'q3'] else '')}
{('  {"text": "Choice B", "response": "Kelly responds to B with curiosity"},' if phase in ['q1', 'q2', 'q3'] else '')}
{('  {"text": "Choice C", "response": "Kelly responds to C with warmth"}' if phase in ['q1', 'q2', 'q3'] else '')}
{(']' if phase in ['q1', 'q2', 'q3'] else '')}
"""
    
    try:
        response = model.generate_content(prompt)
        content_text = response.text.strip()
        
        # Clean up markdown code blocks if present
        if content_text.startswith("```"):
            content_text = content_text.split("```")[1]
            if content_text.startswith("json"):
                content_text = content_text[4:]
            content_text = content_text.strip()
        
        content = json.loads(content_text)
        
        # Validate structure
        if "script" not in content:
            raise ValueError("Missing script field")
        
        return content
        
    except Exception as e:
        print(f"⚠️  Gemini API error for {topic}/{phase}/{archetype}: {e}")
        return generate_fallback_atom(topic, phase, archetype)


def generate_fallback_atom(topic: str, phase: str, archetype: str) -> Dict:
    """Generate fallback content when API is unavailable"""
    
    base = {
        "title": f"{topic}: {phase.title()}",
        "script": f"Welcome to today's lesson about {topic}. Let's explore this together with a {archetype} perspective.",
        "prompt": None,
        "options": None
    }
    
    if phase in ["q1", "q2", "q3"]:
        base["prompt"] = f"What interests you most about {topic}?"
        base["options"] = [
            {"text": "Option A", "response": "That's a fascinating perspective! Let's explore that."},
            {"text": "Option B", "response": "I love that thinking! Here's what's interesting..."},
            {"text": "Option C", "response": "Great choice! Let me share something about that."}
        ]
    
    return base


def insert_lesson_atom(core_lesson_id: str, day_number: int, archetype: str, phase: str, content: Dict):
    """Insert a lesson atom into the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO lesson_atoms (core_lesson_id, day_number, archetype, phase, content, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
    """, (core_lesson_id, day_number, archetype, phase, json.dumps(content)))
    
    conn.commit()
    conn.close()


def generate_shard_content(topic: str, truth: str, age: int, tone: str) -> Dict:
    """Generate complete lesson shard for specific age/tone"""
    
    if not model:
        return generate_fallback_shard(topic, age, tone)
    
    age_descriptions = {
        5: "ages 2-7: Simple words, playful, lots of encouragement",
        10: "ages 8-12: Curious, building knowledge, interactive",
        15: "ages 13-17: More sophisticated, respects intelligence",
        25: "ages 18-35: Adult perspective, real-world applications",
        40: "ages 36-55: Life experience, practical wisdom",
        65: "ages 56+: Reflective, legacy, intergenerational"
    }
    
    prompt = f"""Generate a complete lesson for Curious Kelly.

TOPIC: {topic}
UNIVERSAL TRUTH: {truth}
AGE GROUP: {age_descriptions[age]}
TONE: {tone}

{KELLY_CONSTITUTION}

Generate a complete 5-phase lesson. Return ONLY valid JSON:
{{
  "age_group": {age},
  "tone": "{tone}",
  "language": "en",
  "phases": {{
    "welcome": {{
      "script": "Kelly's warm introduction (30-45 sec)",
      "duration_estimate": 35
    }},
    "q1": {{
      "script": "Kelly poses first question",
      "prompt": "The question",
      "options": [
        {{"text": "Choice A", "response": "Kelly's response"}},
        {{"text": "Choice B", "response": "Kelly's response"}},
        {{"text": "Choice C", "response": "Kelly's response"}}
      ],
      "duration_estimate": 75
    }},
    "q2": {{
      "script": "Kelly poses second question",
      "prompt": "The question",
      "options": [
        {{"text": "Choice A", "response": "Kelly's response"}},
        {{"text": "Choice B", "response": "Kelly's response"}},
        {{"text": "Choice C", "response": "Kelly's response"}}
      ],
      "duration_estimate": 75
    }},
    "q3": {{
      "script": "Kelly poses third question",
      "prompt": "The question",
      "options": [
        {{"text": "Choice A", "response": "Kelly's response"}},
        {{"text": "Choice B", "response": "Kelly's response"}},
        {{"text": "Choice C", "response": "Kelly's response"}}
      ],
      "duration_estimate": 75
    }},
    "wisdom": {{
      "script": "Kelly's closing wisdom (45-60 sec)",
      "duration_estimate": 50
    }}
  }}
}}

Make it age-appropriate, {tone}, and embody Kelly's voice.
"""
    
    try:
        response = model.generate_content(prompt)
        content_text = response.text.strip()
        
        # Clean up markdown
        if content_text.startswith("```"):
            content_text = content_text.split("```")[1]
            if content_text.startswith("json"):
                content_text = content_text[4:]
            content_text = content_text.strip()
        
        content = json.loads(content_text)
        return content
        
    except Exception as e:
        print(f"⚠️  Gemini API error for shard {topic}/{age}/{tone}: {e}")
        return generate_fallback_shard(topic, age, tone)


def generate_fallback_shard(topic: str, age: int, tone: str) -> Dict:
    """Generate fallback shard content"""
    return {
        "age_group": age,
        "tone": tone,
        "language": "en",
        "phases": {
            "welcome": {
                "script": f"Hello! Today we're exploring {topic}. Let's dive in together!",
                "duration_estimate": 30
            },
            "q1": {
                "script": f"Here's an interesting question about {topic}...",
                "prompt": f"What do you think about {topic}?",
                "options": [
                    {"text": "Option A", "response": "Great thinking!"},
                    {"text": "Option B", "response": "I love that!"},
                    {"text": "Option C", "response": "Interesting choice!"}
                ],
                "duration_estimate": 60
            },
            "q2": {
                "script": "Let's go deeper...",
                "prompt": "What else interests you?",
                "options": [
                    {"text": "Option A", "response": "Fascinating!"},
                    {"text": "Option B", "response": "Yes, and..."},
                    {"text": "Option C", "response": "I wonder..."}
                ],
                "duration_estimate": 60
            },
            "q3": {
                "script": "One more thing to explore...",
                "prompt": "How does this connect?",
                "options": [
                    {"text": "Option A", "response": "Beautiful connection!"},
                    {"text": "Option B", "response": "That's insightful!"},
                    {"text": "Option C", "response": "I see what you mean!"}
                ],
                "duration_estimate": 60
            },
            "wisdom": {
                "script": f"What I find beautiful about {topic} is how it connects us all.",
                "duration_estimate": 45
            }
        }
    }


def insert_lesson_shard(core_lesson_id: str, day_number: int, age: int, tone: str, content: Dict):
    """Insert a lesson shard into the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO lesson_shards (core_lesson_id, day_number, age, region, tone, script_content, created_at)
        VALUES (%s, %s, %s, 'en', %s, %s, NOW())
    """, (core_lesson_id, day_number, age, tone, json.dumps(content)))
    
    conn.commit()
    conn.close()


def generate_atoms_for_day(day_number: int):
    """Generate all atoms for a single day (60 atoms: 5 phases × 12 archetypes)"""
    lesson = get_core_lesson(day_number)
    if not lesson:
        print(f"❌ No core lesson found for day {day_number}")
        return 0
    
    count = 0
    for phase in PHASES:
        for archetype in ARCHETYPES:
            content = generate_atom_content(lesson["topic"], lesson["truth"], phase, archetype)
            insert_lesson_atom(lesson["id"], day_number, archetype, phase, content)
            count += 1
            print(f"  ✓ Day {day_number} | {phase} | {archetype}")
            time.sleep(0.5)  # Rate limiting
    
    return count


def generate_shards_for_day(day_number: int):
    """Generate all shards for a single day (18 shards: 6 ages × 3 tones)"""
    lesson = get_core_lesson(day_number)
    if not lesson:
        print(f"❌ No core lesson found for day {day_number}")
        return 0
    
    count = 0
    for age in AGE_GROUPS:
        for tone in TONES:
            content = generate_shard_content(lesson["topic"], lesson["truth"], age, tone)
            insert_lesson_shard(lesson["id"], day_number, age, tone, content)
            count += 1
            print(f"  ✓ Day {day_number} | Age {age} | {tone}")
            time.sleep(0.5)  # Rate limiting
    
    return count


def check_progress():
    """Check current generation progress"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM core_lessons")
    lessons = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM lesson_atoms")
    atoms = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM lesson_shards")
    shards = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT day_number) FROM lesson_atoms")
    days_with_atoms = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT day_number) FROM lesson_shards")
    days_with_shards = cur.fetchone()[0]
    
    conn.close()
    
    print("\n" + "="*60)
    print("CURIOUS KELLY CONTENT GENERATION PROGRESS")
    print("="*60)
    print(f"Core Lessons:    {lessons:>6} / 365     ({lessons/365*100:.1f}%)")
    print(f"Lesson Atoms:    {atoms:>6} / 21,900  ({atoms/21900*100:.1f}%)")
    print(f"  Days complete: {days_with_atoms} / 365")
    print(f"Lesson Shards:   {shards:>6} / 6,570   ({shards/6570*100:.1f}%)")
    print(f"  Days complete: {days_with_shards} / 365")
    print("="*60 + "\n")
    
    return {
        "lessons": lessons,
        "atoms": atoms,
        "shards": shards,
        "days_with_atoms": days_with_atoms,
        "days_with_shards": days_with_shards
    }


def generate_batch(start_day: int, end_day: int, generate_atoms: bool = True, generate_shards: bool = True):
    """Generate content for a batch of days"""
    print(f"\n🚀 Starting batch generation: Days {start_day}-{end_day}")
    print(f"   Atoms: {'✓' if generate_atoms else '✗'} | Shards: {'✓' if generate_shards else '✗'}")
    
    total_atoms = 0
    total_shards = 0
    start_time = time.time()
    
    for day in range(start_day, end_day + 1):
        print(f"\n📅 Day {day}:")
        
        if generate_atoms:
            atoms = generate_atoms_for_day(day)
            total_atoms += atoms
        
        if generate_shards:
            shards = generate_shards_for_day(day)
            total_shards += shards
    
    elapsed = time.time() - start_time
    print(f"\n✅ Batch complete!")
    print(f"   Generated: {total_atoms} atoms, {total_shards} shards")
    print(f"   Time: {elapsed/60:.1f} minutes")
    
    check_progress()


if __name__ == "__main__":
    import sys
    
    print("\n" + "⭐"*30)
    print("CURIOUS KELLY CONTENT GENERATOR")
    print("⭐"*30)
    
    # Check current status
    progress = check_progress()
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python generate_curious_kelly_content.py check")
        print("  python generate_curious_kelly_content.py batch START END [atoms|shards|both]")
        print("  python generate_curious_kelly_content.py launch  # Generate days 1-30")
        print("\nExamples:")
        print("  python generate_curious_kelly_content.py launch")
        print("  python generate_curious_kelly_content.py batch 1 10 both")
        print("  python generate_curious_kelly_content.py batch 31 100 atoms")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "check":
        pass  # Already displayed
    
    elif command == "launch":
        print("\n🎯 LAUNCH MODE: Generating days 1-30 (complete)")
        generate_batch(1, 30, generate_atoms=True, generate_shards=True)
    
    elif command == "batch":
        if len(sys.argv) < 4:
            print("❌ Usage: batch START END [atoms|shards|both]")
            sys.exit(1)
        
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        mode = sys.argv[4] if len(sys.argv) > 4 else "both"
        
        gen_atoms = mode in ["atoms", "both"]
        gen_shards = mode in ["shards", "both"]
        
        generate_batch(start, end, gen_atoms, gen_shards)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)






