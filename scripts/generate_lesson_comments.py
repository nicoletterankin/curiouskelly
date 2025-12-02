#!/usr/bin/env python3
"""
Generate AI Comments for All Lessons
=====================================
Generates diverse, global persona comments for each lesson phase.
Per CURIOUS-KELLY-COMPLETE-SYSTEM-SPEC.md

Usage:
    python scripts/generate_lesson_comments.py --day 1
    python scripts/generate_lesson_comments.py --all
    python scripts/generate_lesson_comments.py --range 1-30
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

SUPABASE_URL = os.getenv("PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
# Prefer anon key (service role key may be truncated in .env)
SUPABASE_KEY = os.getenv("PUBLIC_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANT_API_KEY")

# Global names pool with flags
GLOBAL_PERSONAS = [
    {"name": "Emma", "country": "US", "flag": "🇺🇸"},
    {"name": "Liam", "country": "US", "flag": "🇺🇸"},
    {"name": "James", "country": "GB", "flag": "🇬🇧"},
    {"name": "Charlotte", "country": "GB", "flag": "🇬🇧"},
    {"name": "Marie", "country": "FR", "flag": "🇫🇷"},
    {"name": "Lucas", "country": "FR", "flag": "🇫🇷"},
    {"name": "Hans", "country": "DE", "flag": "🇩🇪"},
    {"name": "Lena", "country": "DE", "flag": "🇩🇪"},
    {"name": "Yuki", "country": "JP", "flag": "🇯🇵"},
    {"name": "Haruto", "country": "JP", "flag": "🇯🇵"},
    {"name": "Priya", "country": "IN", "flag": "🇮🇳"},
    {"name": "Arjun", "country": "IN", "flag": "🇮🇳"},
    {"name": "Wei", "country": "CN", "flag": "🇨🇳"},
    {"name": "Mei", "country": "CN", "flag": "🇨🇳"},
    {"name": "Sofia", "country": "MX", "flag": "🇲🇽"},
    {"name": "Diego", "country": "MX", "flag": "🇲🇽"},
    {"name": "Maria", "country": "BR", "flag": "🇧🇷"},
    {"name": "Pedro", "country": "BR", "flag": "🇧🇷"},
    {"name": "Ahmed", "country": "EG", "flag": "🇪🇬"},
    {"name": "Fatima", "country": "EG", "flag": "🇪🇬"},
    {"name": "Jin", "country": "KR", "flag": "🇰🇷"},
    {"name": "Soo-yeon", "country": "KR", "flag": "🇰🇷"},
    {"name": "Isabella", "country": "IT", "flag": "🇮🇹"},
    {"name": "Marco", "country": "IT", "flag": "🇮🇹"},
    {"name": "Kofi", "country": "GH", "flag": "🇬🇭"},
    {"name": "Ama", "country": "GH", "flag": "🇬🇭"},
    {"name": "Thabo", "country": "ZA", "flag": "🇿🇦"},
    {"name": "Naledi", "country": "ZA", "flag": "🇿🇦"},
    {"name": "Omar", "country": "AE", "flag": "🇦🇪"},
    {"name": "Layla", "country": "AE", "flag": "🇦🇪"},
    {"name": "Sven", "country": "SE", "flag": "🇸🇪"},
    {"name": "Elsa", "country": "SE", "flag": "🇸🇪"},
    {"name": "Carlos", "country": "AR", "flag": "🇦🇷"},
    {"name": "Luciana", "country": "AR", "flag": "🇦🇷"},
    {"name": "Sarah", "country": "CA", "flag": "🇨🇦"},
    {"name": "Michael", "country": "CA", "flag": "🇨🇦"},
    {"name": "Olga", "country": "UA", "flag": "🇺🇦"},
    {"name": "Ivan", "country": "RU", "flag": "🇷🇺"},
    {"name": "Chen", "country": "TW", "flag": "🇹🇼"},
    {"name": "Aisha", "country": "KE", "flag": "🇰🇪"},
]

COMMENT_TYPES = ["insight", "reaction", "question", "funny"]

PHASES = ["welcome", "q1", "q2", "q3", "hook", "complete"]

# ═══════════════════════════════════════════════════════════════════
# SUPABASE CLIENT
# ═══════════════════════════════════════════════════════════════════

def get_supabase():
    """Initialize Supabase client."""
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except ImportError:
        print("❌ supabase-py not installed. Run: pip install supabase")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# LESSON DATA FETCHING
# ═══════════════════════════════════════════════════════════════════

def get_lesson(supabase, day_number: int) -> Optional[Dict]:
    """Fetch lesson data from core_lessons."""
    try:
        result = supabase.table("core_lessons").select("*").eq("day_number", day_number).single().execute()
        return result.data
    except Exception as e:
        print(f"⚠️ Could not fetch lesson {day_number}: {e}")
        return None


def get_lesson_atoms(supabase, core_lesson_id: str) -> List[Dict]:
    """Fetch lesson atoms for content context."""
    try:
        result = supabase.table("lesson_atoms").select("*").eq("core_lesson_id", core_lesson_id).execute()
        return result.data or []
    except Exception as e:
        print(f"⚠️ Could not fetch atoms: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
# COMMENT GENERATION (Claude/Anthropic)
# ═══════════════════════════════════════════════════════════════════

def generate_comments_with_claude(lesson: Dict, phase: str, count: int = 8) -> List[Dict]:
    """Generate comments using Claude API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        print("⚠️ anthropic not installed, using fallback generation")
        return generate_fallback_comments(lesson, phase, count)
    except Exception as e:
        print(f"⚠️ Claude init failed: {e}, using fallback")
        return generate_fallback_comments(lesson, phase, count)
    
    topic = lesson.get("topic", "today's lesson")
    universal_truth = lesson.get("universal_truth", "")
    
    phase_context = {
        "welcome": "The lesson is just starting. People are greeting each other and showing excitement.",
        "q1": "Kelly asked the first question. People are thinking and discussing.",
        "q2": "Second question phase. Engagement is high.",
        "q3": "Third question. People are getting into it.",
        "hook": "Kelly just revealed the key insight/wisdom. This is the 'aha moment'.",
        "complete": "Lesson is complete! People are celebrating and saying goodbye."
    }
    
    prompt = f"""Generate {count} diverse social media-style comments for an educational app.

LESSON TOPIC: {topic}
UNIVERSAL TRUTH: {universal_truth}
PHASE: {phase} - {phase_context.get(phase, '')}

Generate comments from diverse global personas (different countries, ages, backgrounds).
Mix of types:
- insight: Adds educational value or connection
- reaction: Emotional response (excitement, surprise)
- question: Prompts thinking
- funny: Light humor (keep it family-friendly)

Rules:
- Keep comments SHORT (under 80 characters)
- Include emojis naturally (don't overdo it)
- Make them feel like real TikTok/Instagram comments
- No offensive or controversial content
- Age-appropriate for 2-102 years old

Return ONLY a JSON array with objects containing:
- persona_name (first name)
- persona_country (2-letter code)
- persona_flag (emoji flag)
- comment_text (the comment)
- comment_type (insight/reaction/question/funny)

Example format:
[
  {{"persona_name": "Yuki", "persona_country": "JP", "persona_flag": "🇯🇵", "comment_text": "Mind = blown 🤯", "comment_type": "reaction"}}
]"""

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Fast and cheap
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON from response
        content = response.content[0].text
        # Extract JSON array
        import re
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            comments = json.loads(json_match.group())
            return comments
        else:
            print(f"⚠️ No JSON found in response for {phase}")
            return generate_fallback_comments(lesson, phase, count)
            
    except Exception as e:
        print(f"⚠️ Claude generation failed: {e}")
        return generate_fallback_comments(lesson, phase, count)


def generate_fallback_comments(lesson: Dict, phase: str, count: int = 8) -> List[Dict]:
    """Generate comments without AI (fallback templates)."""
    topic = lesson.get("topic", "today's lesson")
    
    templates = {
        "welcome": [
            "Ready to learn! 🎓",
            "Good morning everyone ☀️",
            "Let's do this! 💪",
            "Excited for today's lesson",
            "Kelly is the best teacher 💙",
            "Day {day} here we go!",
        ],
        "q1": [
            "Hmm good question 🤔",
            "I think I know this!",
            "Wait let me think...",
            "This is tricky",
            "Going with my gut 🎯",
        ],
        "q2": [
            "This one's harder",
            "I changed my answer twice 😅",
            "Both seem right?",
            "Trust the process",
        ],
        "q3": [
            "Last question!",
            "I'm confident now 💪",
            "Learning so much today",
        ],
        "hook": [
            "Mind = blown 🤯",
            "I never thought of it that way!",
            "This is so deep 💎",
            "Screenshotting this 📸",
            "Words to live by 🙏",
            f"I love learning about {topic}",
        ],
        "complete": [
            "Great lesson! 🎉",
            "See you tomorrow! 👋",
            "One day closer to 365! 💪",
            "Thanks Kelly! 💙",
            "Can't wait for tomorrow",
        ],
    }
    
    day = lesson.get("day_number", 1)
    phase_templates = templates.get(phase, templates["welcome"])
    
    comments = []
    used_personas = set()
    
    for i in range(min(count, len(phase_templates) * 2)):
        # Pick unique persona
        persona = random.choice(GLOBAL_PERSONAS)
        while persona["name"] in used_personas and len(used_personas) < len(GLOBAL_PERSONAS):
            persona = random.choice(GLOBAL_PERSONAS)
        used_personas.add(persona["name"])
        
        # Pick template
        template = random.choice(phase_templates)
        text = template.format(day=day, topic=topic)
        
        comments.append({
            "persona_name": persona["name"],
            "persona_country": persona["country"],
            "persona_flag": persona["flag"],
            "comment_text": text,
            "comment_type": random.choice(COMMENT_TYPES)
        })
    
    return comments


# ═══════════════════════════════════════════════════════════════════
# DATABASE INSERTION
# ═══════════════════════════════════════════════════════════════════

def insert_comments(supabase, lesson_day: int, comments: List[Dict]) -> int:
    """Insert comments into Supabase."""
    if not comments:
        return 0
    
    try:
        result = supabase.table("lesson_comments").insert(comments).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        return 0


def delete_existing_comments(supabase, lesson_day: int):
    """Delete existing comments for a lesson (for regeneration)."""
    try:
        supabase.table("lesson_comments").delete().eq("lesson_day", lesson_day).execute()
        print(f"🗑️ Deleted existing comments for day {lesson_day}")
    except Exception as e:
        print(f"⚠️ Could not delete existing comments: {e}")


# ═══════════════════════════════════════════════════════════════════
# MAIN GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════════

def generate_for_lesson(supabase, day_number: int, regenerate: bool = False) -> int:
    """Generate all comments for a single lesson."""
    print(f"\n📚 Generating comments for Day {day_number}...")
    
    # Fetch lesson
    lesson = get_lesson(supabase, day_number)
    if not lesson:
        print(f"⚠️ Lesson {day_number} not found, skipping")
        return 0
    
    print(f"   Topic: {lesson.get('topic', 'Unknown')}")
    
    # Delete existing if regenerating
    if regenerate:
        delete_existing_comments(supabase, day_number)
    
    total_inserted = 0
    
    # Generate for each phase
    for phase in PHASES:
        print(f"   📝 Generating {phase} comments...")
        
        # Generate 8-12 comments per phase
        count = random.randint(8, 12)
        comments = generate_comments_with_claude(lesson, phase, count)
        
        # Add metadata
        for comment in comments:
            comment["lesson_day"] = day_number
            comment["phase"] = phase
            comment["option_context"] = None  # General phase comments
        
        # Insert
        inserted = insert_comments(supabase, day_number, comments)
        total_inserted += inserted
        print(f"      ✅ Inserted {inserted} comments")
    
    # Generate option-specific comments for question phases
    for phase in ["q1", "q2", "q3"]:
        for option in ["A", "B"]:
            print(f"   📝 Generating {phase} option {option} comments...")
            
            comments = generate_comments_with_claude(lesson, phase, count=3)
            
            for comment in comments:
                comment["lesson_day"] = day_number
                comment["phase"] = phase
                comment["option_context"] = option
            
            inserted = insert_comments(supabase, day_number, comments)
            total_inserted += inserted
    
    print(f"✅ Day {day_number}: {total_inserted} total comments generated")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Generate lesson comments")
    parser.add_argument("--day", type=int, help="Generate for specific day")
    parser.add_argument("--range", type=str, help="Generate for range (e.g., 1-30)")
    parser.add_argument("--all", action="store_true", help="Generate for all 365 days")
    parser.add_argument("--regenerate", action="store_true", help="Delete and regenerate existing")
    parser.add_argument("--dry-run", action="store_true", help="Preview without inserting")
    
    args = parser.parse_args()
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Missing SUPABASE_URL or SUPABASE_KEY environment variables")
        sys.exit(1)
    
    print("🚀 Lesson Comments Generator")
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
        print("⚠️ Generating for ALL 365 days. This may take a while and cost API credits.")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            sys.exit(0)
        
        for day in range(1, 366):
            total += generate_for_lesson(supabase, day, args.regenerate)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print(f"🎉 Complete! Total comments generated: {total}")


if __name__ == "__main__":
    main()

