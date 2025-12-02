#!/usr/bin/env python3
"""
Generate Essential Content - Fast Mode
=======================================
Generates the minimum viable content for all lessons quickly:
- Atoms: 3 archetypes × 5 phases = 15 atoms/lesson
- Shards: 3 ages × 2 tones = 6 shards/lesson
- No translations (do later)

This reduces API calls from ~90/lesson to ~21/lesson

Usage:
    python scripts/generate_essential_content.py --days 1-365 --yes
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from supabase import create_client

from content_generator.config import (
    SUPABASE_URL, SUPABASE_SERVICE_KEY, BATCH_SIZE
)
from content_generator.generator import ContentGenerator

# Essential subset for fast generation
ESSENTIAL_ARCHETYPES = [
    "The Scientist",
    "The Explorer", 
    "The Storyteller",
]

ESSENTIAL_AGE_BUCKETS = [
    {"age": 9, "label": "elementary", "range": "6-12", "birth_year": 2016},
    {"age": 26, "label": "young_adult", "range": "18-35", "birth_year": 1999},
    {"age": 72, "label": "wisdom_years", "range": "61-102", "birth_year": 1953},
]

ESSENTIAL_TONES = ["curious", "inspiring"]

PHASES = ["Hook", "Fact1", "Fact2", "Fact3", "Wisdom"]


def parse_day_range(day_range: str) -> tuple:
    if "-" in day_range:
        parts = day_range.split("-")
        return int(parts[0]), int(parts[1])
    else:
        day = int(day_range)
        return day, day


def get_lessons_from_supabase(start_day: int = 1, end_day: int = 365) -> list:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    result = supabase.table("core_lessons")\
        .select("*")\
        .gte("day_number", start_day)\
        .lte("day_number", end_day)\
        .order("day_number")\
        .execute()
    return result.data if result.data else []


def get_existing_atoms(supabase, lesson_ids: list) -> set:
    if not lesson_ids:
        return set()
    result = supabase.table("lesson_atoms")\
        .select("core_lesson_id")\
        .in_("core_lesson_id", lesson_ids)\
        .execute()
    return set(r["core_lesson_id"] for r in (result.data or []))


def main():
    parser = argparse.ArgumentParser(description="Generate essential lesson content (fast mode)")
    parser.add_argument("--days", type=str, default="1-365", help="Day range")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--atoms-only", action="store_true", help="Generate only atoms")
    parser.add_argument("--shards-only", action="store_true", help="Generate only shards")
    
    args = parser.parse_args()
    start_day, end_day = parse_day_range(args.days)
    
    generate_atoms = not args.shards_only
    generate_shards = not args.atoms_only
    
    print("")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   ⚡ ESSENTIAL CONTENT GENERATION (FAST MODE)                     ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"📅 Days: {start_day} - {end_day} ({end_day - start_day + 1} lessons)")
    print(f"🎭 Archetypes: {len(ESSENTIAL_ARCHETYPES)} (essential)")
    print(f"📚 Phases: {len(PHASES)}")
    print(f"👤 Age buckets: {len(ESSENTIAL_AGE_BUCKETS)} (essential)")
    print(f"🎨 Tones: {len(ESSENTIAL_TONES)} (essential)")
    print("")
    
    num_lessons = end_day - start_day + 1
    atoms_per_lesson = len(ESSENTIAL_ARCHETYPES) * len(PHASES) if generate_atoms else 0
    shards_per_lesson = len(ESSENTIAL_AGE_BUCKETS) * len(ESSENTIAL_TONES) if generate_shards else 0
    
    total_atoms = num_lessons * atoms_per_lesson
    total_shards = num_lessons * shards_per_lesson
    total_calls = total_atoms + total_shards
    
    print(f"📊 Generation plan:")
    print(f"   Atoms: {total_atoms:,} ({atoms_per_lesson}/lesson)")
    print(f"   Shards: {total_shards:,} ({shards_per_lesson}/lesson)")
    print(f"   Total API calls: {total_calls:,}")
    print(f"   Estimated time: ~{total_calls * 1.5 / 60:.0f} minutes")
    print("")
    
    if not args.yes:
        try:
            response = input("Proceed? [y/N]: ")
            if response.lower() != "y":
                print("Cancelled.")
                return
        except EOFError:
            print("Auto-proceeding...")
    
    # Initialize
    generator = ContentGenerator(dry_run=False)
    
    if not generator.client:
        print("❌ OpenAI client not initialized. Check OPENAI_API_KEY.")
        return
    
    # Fetch lessons
    print("\n📥 Fetching lessons...")
    lessons = get_lessons_from_supabase(start_day, end_day)
    print(f"   Found {len(lessons)} lessons")
    
    if not lessons:
        print("❌ No lessons found!")
        return
    
    # Get existing
    lesson_ids = [l["id"] for l in lessons]
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    existing_atoms = get_existing_atoms(supabase, lesson_ids) if args.skip_existing else set()
    print(f"   Existing atoms: {len(existing_atoms)} lessons")
    
    # Process
    start_time = datetime.now()
    all_atoms = []
    all_shards = []
    
    for i, lesson in enumerate(lessons):
        day = lesson.get("day_number", i + 1)
        topic = lesson.get("topic", "Unknown")
        
        print(f"\n[{i+1}/{len(lessons)}] Day {day}: {topic[:40]}")
        
        # Generate atoms
        if generate_atoms and lesson["id"] not in existing_atoms:
            print(f"   🧬 Generating {atoms_per_lesson} atoms...")
            atoms = generator.generate_all_atoms_for_lesson(
                lesson, 
                archetypes=ESSENTIAL_ARCHETYPES, 
                phases=PHASES
            )
            all_atoms.extend(atoms)
            
            # Upload batch
            if len(all_atoms) >= 50:
                uploaded = generator.upload_atoms(all_atoms)
                print(f"   📤 Uploaded {uploaded} atoms")
                all_atoms = []
        else:
            print("   ⏭️ Skipping atoms")
        
        # Generate shards
        if generate_shards:
            print(f"   👤 Generating {shards_per_lesson} shards...")
            shards = generator.generate_all_shards_for_lesson(
                lesson,
                age_buckets=ESSENTIAL_AGE_BUCKETS,
                tones=ESSENTIAL_TONES
            )
            all_shards.extend(shards)
            
            # Upload batch
            if len(all_shards) >= 50:
                uploaded = generator.upload_shards(all_shards)
                print(f"   📤 Uploaded {uploaded} shards")
                all_shards = []
        
        # Progress
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(lessons) - i - 1) / rate if rate > 0 else 0
        print(f"   ⏱️ {rate:.2f} lessons/min | ETA: {remaining:.0f} min")
    
    # Final upload
    if all_atoms:
        uploaded = generator.upload_atoms(all_atoms)
        print(f"\n📤 Final: {uploaded} atoms")
    if all_shards:
        uploaded = generator.upload_shards(all_shards)
        print(f"📤 Final: {uploaded} shards")
    
    # Stats
    generator.print_stats()
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
    print("\n✅ ESSENTIAL CONTENT COMPLETE!")


if __name__ == "__main__":
    main()

