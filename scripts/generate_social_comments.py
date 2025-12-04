#!/usr/bin/env python3
"""
Generate Social Comments for Lessons - v2
==========================================
Creates humble, growth-mindset focused comments for each lesson phase.

Philosophy:
- Normalize struggle (confusion is normal, asking is good)
- No hyperbole (no "MIND BLOWN", "BEST EVER")
- Specific to lesson content (reference actual topic/insights)
- Diverse global perspectives

Usage:
    python scripts/generate_social_comments.py --day 1
    python scripts/generate_social_comments.py --range 1-30
    python scripts/generate_social_comments.py --all
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

SUPABASE_URL = os.getenv("PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("PUBLIC_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANT_API_KEY")

# Phases in lesson order
PHASES = ["welcome", "hook", "q1", "q2", "q3", "wisdom", "complete"]

# Comments per phase
COMMENTS_PER_PHASE = {
    "welcome": 8,
    "hook": 10,
    "q1": 12,
    "q2": 12,
    "q3": 12,
    "wisdom": 10,
    "complete": 8,
}

# ═══════════════════════════════════════════════════════════════════
# DIVERSE PERSONA POOL (60 personas)
# ═══════════════════════════════════════════════════════════════════

PERSONAS = [
    # North America
    {"name": "Emma", "country": "US", "flag": "🇺🇸", "age": 28, "style": "analytical"},
    {"name": "Marcus", "country": "US", "flag": "🇺🇸", "age": 16, "style": "curious"},
    {"name": "Sarah", "country": "CA", "flag": "🇨🇦", "age": 45, "style": "supportive"},
    {"name": "Joe", "country": "US", "flag": "🇺🇸", "age": 72, "style": "wise"},
    {"name": "Maya", "country": "MX", "flag": "🇲🇽", "age": 34, "style": "creative"},
    
    # Europe
    {"name": "James", "country": "GB", "flag": "🇬🇧", "age": 31, "style": "methodical"},
    {"name": "Charlotte", "country": "GB", "flag": "🇬🇧", "age": 8, "style": "wonder"},
    {"name": "Marie", "country": "FR", "flag": "🇫🇷", "age": 52, "style": "reflective"},
    {"name": "Lucas", "country": "FR", "flag": "🇫🇷", "age": 19, "style": "questioning"},
    {"name": "Hans", "country": "DE", "flag": "🇩🇪", "age": 67, "style": "scholarly"},
    {"name": "Lena", "country": "DE", "flag": "🇩🇪", "age": 24, "style": "dedicated"},
    {"name": "Isabella", "country": "IT", "flag": "🇮🇹", "age": 38, "style": "passionate"},
    {"name": "Sven", "country": "SE", "flag": "🇸🇪", "age": 29, "style": "creative"},
    {"name": "Nina", "country": "NO", "flag": "🇳🇴", "age": 41, "style": "scientific"},
    {"name": "Olga", "country": "UA", "flag": "🇺🇦", "age": 33, "style": "efficient"},
    
    # Asia
    {"name": "Yuki", "country": "JP", "flag": "🇯🇵", "age": 26, "style": "artistic"},
    {"name": "Haruto", "country": "JP", "flag": "🇯🇵", "age": 12, "style": "curious"},
    {"name": "Sakura", "country": "JP", "flag": "🇯🇵", "age": 58, "style": "contemplative"},
    {"name": "Priya", "country": "IN", "flag": "🇮🇳", "age": 22, "style": "driven"},
    {"name": "Arjun", "country": "IN", "flag": "🇮🇳", "age": 35, "style": "nurturing"},
    {"name": "Ananya", "country": "IN", "flag": "🇮🇳", "age": 9, "style": "imaginative"},
    {"name": "Wei", "country": "CN", "flag": "🇨🇳", "age": 44, "style": "pragmatic"},
    {"name": "Mei", "country": "CN", "flag": "🇨🇳", "age": 17, "style": "focused"},
    {"name": "Jin", "country": "KR", "flag": "🇰🇷", "age": 27, "style": "creative"},
    {"name": "Soo-yeon", "country": "KR", "flag": "🇰🇷", "age": 63, "style": "patient"},
    
    # Middle East
    {"name": "Ahmed", "country": "EG", "flag": "🇪🇬", "age": 30, "style": "storyteller"},
    {"name": "Fatima", "country": "EG", "flag": "🇪🇬", "age": 21, "style": "investigative"},
    {"name": "Omar", "country": "AE", "flag": "🇦🇪", "age": 39, "style": "efficient"},
    {"name": "Layla", "country": "AE", "flag": "🇦🇪", "age": 14, "style": "experimental"},
    
    # Africa
    {"name": "Kofi", "country": "GH", "flag": "🇬🇭", "age": 25, "style": "practical"},
    {"name": "Ama", "country": "GH", "flag": "🇬🇭", "age": 48, "style": "mentoring"},
    {"name": "Aisha", "country": "KE", "flag": "🇰🇪", "age": 20, "style": "passionate"},
    {"name": "Thabo", "country": "ZA", "flag": "🇿🇦", "age": 36, "style": "artistic"},
    {"name": "Naledi", "country": "ZA", "flag": "🇿🇦", "age": 11, "style": "playful"},
    {"name": "Adebayo", "country": "NG", "flag": "🇳🇬", "age": 42, "style": "innovative"},
    
    # South America
    {"name": "Maria", "country": "BR", "flag": "🇧🇷", "age": 28, "style": "compassionate"},
    {"name": "Pedro", "country": "BR", "flag": "🇧🇷", "age": 55, "style": "experiential"},
    {"name": "Carlos", "country": "AR", "flag": "🇦🇷", "age": 32, "style": "analytical"},
    {"name": "Diego", "country": "CL", "flag": "🇨🇱", "age": 18, "style": "wonder"},
    {"name": "Valentina", "country": "CO", "flag": "🇨🇴", "age": 7, "style": "playful"},
    
    # Oceania & SE Asia
    {"name": "Lisa", "country": "AU", "flag": "🇦🇺", "age": 37, "style": "observant"},
    {"name": "Jack", "country": "NZ", "flag": "🇳🇿", "age": 23, "style": "adventurous"},
    {"name": "Linh", "country": "VN", "flag": "🇻🇳", "age": 29, "style": "steady"},
    {"name": "Ling", "country": "SG", "flag": "🇸🇬", "age": 45, "style": "structured"},
    {"name": "Kai", "country": "TH", "flag": "🇹🇭", "age": 19, "style": "open-minded"},
    {"name": "Putri", "country": "ID", "flag": "🇮🇩", "age": 31, "style": "nurturing"},
]

# Comment types
COMMENT_TYPES = ["insight", "question", "reflection", "struggle", "connection"]

# ═══════════════════════════════════════════════════════════════════
# SUPABASE CLIENT
# ═══════════════════════════════════════════════════════════════════

def get_supabase():
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except ImportError:
        print("❌ supabase-py not installed. Run: pip install supabase")
        sys.exit(1)

def get_lesson(supabase, day_number: int) -> Optional[Dict]:
    """Fetch lesson data."""
    try:
        result = supabase.table("core_lessons").select("*").eq("day_number", day_number).single().execute()
        return result.data
    except Exception as e:
        print(f"⚠️ Could not fetch lesson {day_number}: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════
# CLAUDE PROMPT - HUMBLE, GROWTH-MINDSET
# ═══════════════════════════════════════════════════════════════════

def get_claude_prompt(lesson: Dict, phase: str, count: int) -> str:
    """Generate prompt for Claude that produces humble, authentic comments."""
    
    topic = lesson.get("topic", "today's lesson")
    universal_truth = lesson.get("universal_truth", "")
    big_truth = lesson.get("big_truth", "")
    hook = lesson.get("hook", "")
    
    phase_context = {
        "welcome": f"""
Phase: WELCOME - The lesson is starting. People are settling in.
Topic being introduced: {topic}

Generate comments like someone joining a live stream. Simple greetings, 
casual acknowledgment of the topic. Some are new, some are regulars.
Examples of TONE (don't copy these exactly):
- "Morning everyone"
- "Don't know much about {topic} yet"
- "Day X, let's go"
- "Watching with my kids today"
""",
        "hook": f"""
Phase: HOOK - Kelly is introducing the topic and the key idea.
Topic: {topic}
Opening insight: {hook or big_truth}

Generate comments showing genuine engagement. Some are intrigued,
some confused (which is GOOD - normalize confusion), some connecting
to their own experience.
Examples of TONE:
- "I never thought of it that way"
- "Wait, I'm not following yet"
- "This connects to something at work"
- "Interesting framing"
""",
        "q1": f"""
Phase: QUESTION 1 - Kelly asked a question about {topic}.
Learners are thinking and choosing an answer.

Generate comments showing the thinking process. Uncertainty is NORMAL.
Some are confident, some unsure, some asking for clarification.
Examples of TONE:
- "Not sure about this one"
- "Both seem reasonable"  
- "Going with my gut"
- "Changed my mind twice"
""",
        "q2": f"""
Phase: QUESTION 2 - Second question about {topic}.
The lesson is building on earlier content.

Generate comments showing deeper engagement. Reference earlier parts.
Some struggling, some connecting dots.
Examples of TONE:
- "Based on the intro, I think..."
- "This is harder than Q1"
- "Anyone else confused here?"
- "The clue was in what Kelly said earlier"
""",
        "q3": f"""
Phase: QUESTION 3 - Final question about {topic}.
This often ties everything together.

Generate comments showing synthesis. Some confident, some still working
through it. Appreciate the challenge.
Examples of TONE:
- "Okay, putting it all together"
- "I think I finally get the pattern"
- "Good challenge"
- "Not 100% sure but learning either way"
""",
        "wisdom": f"""
Phase: WISDOM - Kelly shares the key insight/takeaway.
Universal truth: {universal_truth or big_truth}

Generate comments showing genuine appreciation WITHOUT HYPERBOLE.
No "mind blown" or "best ever". Instead: thoughtful, practical, personal.
Examples of TONE:
- "That's a helpful way to think about it"
- "I'll remember that"
- "This applies to a lot of things"
- "Going to think about this more"
""",
        "complete": f"""
Phase: COMPLETE - The lesson is ending.
Topic was: {topic}

Generate simple closing comments. Satisfaction, gratitude, looking forward.
Keep it humble and genuine.
Examples of TONE:
- "Good lesson today"
- "See everyone tomorrow"
- "Thanks, learned something"
- "Day X done"
"""
    }
    
    return f"""Generate {count} diverse social learning comments for an educational app.

{phase_context.get(phase, phase_context["welcome"])}

CRITICAL RULES:
1. HUMBLE: No hyperbole. No "mind blown", "best ever", "amazing", "incredible"
2. GROWTH MINDSET: Normalize struggle. Confusion is okay. Questions are good.
3. AUTHENTIC: Sound like real people, not marketing copy
4. DIVERSE: Mix ages (kids to seniors), backgrounds, learning styles
5. SHORT: Under 60 characters. Natural, not formal.
6. SPECIFIC: Reference the actual topic ({topic}) when relevant

AVOID THESE PHRASES (too hyperbolic):
- "Mind blown" / "Mind = blown"
- "Best teacher/lesson ever"
- "I'm obsessed/addicted"
- "Game changer"
- "This is everything"
- "Screenshotting this"

GOOD COMMENT TYPES:
- insight: Adds understanding or makes a connection
- question: Asks for clarification (normalizes not knowing)
- reflection: Thoughtful response to content
- struggle: Admits confusion (growth mindset!)
- connection: Relates to personal experience

Return ONLY a JSON array:
[
  {{"persona_name": "Emma", "persona_country": "US", "persona_flag": "🇺🇸", "comment_text": "I never connected X to Y before", "comment_type": "insight"}}
]"""


# ═══════════════════════════════════════════════════════════════════
# COMMENT GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_with_claude(lesson: Dict, phase: str, count: int) -> List[Dict]:
    """Generate comments using Claude."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        print(f"⚠️ Claude not available: {e}")
        return generate_fallback(lesson, phase, count)
    
    prompt = get_claude_prompt(lesson, phase, count)
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        import re
        content = response.content[0].text
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            comments = json.loads(json_match.group())
            # Validate and clean
            return [c for c in comments if validate_comment(c)]
        
    except Exception as e:
        print(f"⚠️ Claude generation failed: {e}")
    
    return generate_fallback(lesson, phase, count)


def validate_comment(comment: Dict) -> bool:
    """Validate comment isn't hyperbolic."""
    text = comment.get("comment_text", "").lower()
    
    # Reject hyperbolic phrases
    bad_phrases = [
        "mind blown", "mind = blown", "best ever", "best teacher",
        "obsessed", "addicted", "game changer", "screenshotting",
        "this is everything", "i'm literally", "incredible", "amazing"
    ]
    
    for phrase in bad_phrases:
        if phrase in text:
            return False
    
    return len(text) > 0 and len(text) < 100


def generate_fallback(lesson: Dict, phase: str, count: int) -> List[Dict]:
    """Generate comments without AI."""
    topic = lesson.get("topic", "this topic")
    day = lesson.get("day_number", 1)
    
    templates = {
        "welcome": [
            "Morning everyone 👋",
            f"Day {day}, here we go",
            "Coffee ready, let's learn",
            f"Don't know much about {topic} yet",
            "Ready for today's lesson",
            "Back again",
            "Watching with my daughter today",
        ],
        "hook": [
            "I never thought of it that way",
            "Interesting approach",
            "This connects to something from work",
            "Not sure I follow yet",
            "Huh, didn't know that",
            "Good way to explain it",
        ],
        "q1": [
            "Hmm, not sure about this one",
            "Let me think...",
            "Going with my gut",
            "Both seem reasonable",
            "Changed my mind twice",
        ],
        "q2": [
            "This one's harder",
            "Based on the intro, I think...",
            "Anyone else unsure here?",
            "Okay, connecting the dots",
        ],
        "q3": [
            "Putting it all together",
            "Good challenge",
            "Not 100% sure but learning",
            "Think I finally see the pattern",
        ],
        "wisdom": [
            "That's helpful",
            "I'll remember that",
            "Good takeaway",
            "This applies to a lot of things",
            "Going to think about this more",
            f"Makes sense now about {topic}",
        ],
        "complete": [
            "Good lesson today",
            "See everyone tomorrow",
            f"Day {day} done ✓",
            "Short but good",
            "Thanks, learned something",
        ],
    }
    
    phase_templates = templates.get(phase, templates["welcome"])
    comments = []
    used_personas = set()
    
    for i in range(count):
        persona = random.choice(PERSONAS)
        while persona["name"] in used_personas and len(used_personas) < len(PERSONAS):
            persona = random.choice(PERSONAS)
        used_personas.add(persona["name"])
        
        template = random.choice(phase_templates)
        
        comments.append({
            "persona_name": persona["name"],
            "persona_country": persona["country"],
            "persona_flag": persona["flag"],
            "comment_text": template,
            "comment_type": random.choice(COMMENT_TYPES)
        })
    
    return comments


# ═══════════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ═══════════════════════════════════════════════════════════════════

def insert_comments(supabase, lesson_day: int, phase: str, comments: List[Dict]) -> int:
    """Insert comments into Supabase."""
    if not comments:
        return 0
    
    records = []
    for c in comments:
        records.append({
            "lesson_day": lesson_day,
            "phase": phase,
            "option_context": None,
            "persona_name": c.get("persona_name"),
            "persona_country": c.get("persona_country"),
            "persona_flag": c.get("persona_flag"),
            "comment_text": c.get("comment_text"),
            "comment_type": c.get("comment_type", "reflection"),
        })
    
    try:
        result = supabase.table("lesson_comments").insert(records).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        return 0


def delete_comments(supabase, lesson_day: int):
    """Delete existing comments for a lesson."""
    try:
        supabase.table("lesson_comments").delete().eq("lesson_day", lesson_day).execute()
        print(f"🗑️ Cleared comments for day {lesson_day}")
    except Exception as e:
        print(f"⚠️ Could not delete: {e}")


# ═══════════════════════════════════════════════════════════════════
# MAIN GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_for_lesson(supabase, day: int, regenerate: bool = False) -> int:
    """Generate all comments for a lesson."""
    print(f"\n📚 Generating comments for Day {day}...")
    
    lesson = get_lesson(supabase, day)
    if not lesson:
        print(f"⚠️ Lesson {day} not found")
        return 0
    
    topic = lesson.get("topic", "Unknown")
    print(f"   Topic: {topic}")
    
    if regenerate:
        delete_comments(supabase, day)
    
    total = 0
    
    for phase in PHASES:
        count = COMMENTS_PER_PHASE.get(phase, 8)
        print(f"   📝 {phase}: generating {count} comments...")
        
        comments = generate_with_claude(lesson, phase, count)
        inserted = insert_comments(supabase, day, phase, comments)
        total += inserted
        print(f"      ✅ Inserted {inserted}")
    
    print(f"✅ Day {day}: {total} comments total")
    return total


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate social learning comments")
    parser.add_argument("--day", type=int, help="Generate for specific day")
    parser.add_argument("--range", type=str, help="Generate for range (e.g., 1-30)")
    parser.add_argument("--all", action="store_true", help="Generate for all 365 days")
    parser.add_argument("--regenerate", action="store_true", help="Delete and regenerate")
    
    args = parser.parse_args()
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Missing SUPABASE_URL or SUPABASE_KEY")
        sys.exit(1)
    
    print("🚀 Social Comments Generator v2")
    print("   Philosophy: Humble • Growth-Mindset • Authentic")
    print("=" * 50)
    
    supabase = get_supabase()
    total = 0
    
    if args.day:
        total = generate_for_lesson(supabase, args.day, args.regenerate)
    elif args.range:
        start, end = map(int, args.range.split("-"))
        for day in range(start, end + 1):
            total += generate_for_lesson(supabase, day, args.regenerate)
    elif args.all:
        print("⚠️ Generating for ALL 365 days. This takes time and API credits.")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != "y":
            sys.exit(0)
        for day in range(1, 366):
            total += generate_for_lesson(supabase, day, args.regenerate)
    else:
        parser.print_help()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print(f"🎉 Complete! {total} comments generated")


if __name__ == "__main__":
    main()

