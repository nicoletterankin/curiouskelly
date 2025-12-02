#!/usr/bin/env python3
"""
Generate All Content - Master Script
=====================================
Generates atoms, age variants, and translations for all 365 lessons.

Usage:
    python scripts/generate_all_content.py                    # Full generation
    python scripts/generate_all_content.py --days 1-31        # January only
    python scripts/generate_all_content.py --days 1-31 --atoms-only
    python scripts/generate_all_content.py --days 1-31 --shards-only
    python scripts/generate_all_content.py --dry-run          # Preview without uploading
    python scripts/generate_all_content.py --skip-existing    # Skip lessons with content
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()
load_dotenv("daily-lesson-marketing/.env")

from supabase import create_client

from content_generator.config import (
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    ARCHETYPES, PHASES, AGE_BUCKETS, TONES, BATCH_SIZE
)
from content_generator.generator import ContentGenerator


def parse_day_range(day_range: str) -> tuple:
    """Parse day range string like '1-31' into (start, end)."""
    if "-" in day_range:
        parts = day_range.split("-")
        return int(parts[0]), int(parts[1])
    else:
        day = int(day_range)
        return day, day


def get_lessons_from_supabase(start_day: int = 1, end_day: int = 365) -> list:
    """Fetch lessons from Supabase."""
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    
    result = supabase.table("core_lessons")\
        .select("*")\
        .gte("day_number", start_day)\
        .lte("day_number", end_day)\
        .order("day_number")\
        .execute()
    
    return result.data if result.data else []


def get_existing_atoms(supabase, lesson_ids: list) -> set:
    """Get lesson IDs that already have atoms."""
    if not lesson_ids:
        return set()
    
    result = supabase.table("lesson_atoms")\
        .select("core_lesson_id")\
        .in_("core_lesson_id", lesson_ids)\
        .execute()
    
    return set(r["core_lesson_id"] for r in (result.data or []))


def get_existing_shards(supabase, lesson_ids: list) -> set:
    """Get lesson IDs that already have shards."""
    if not lesson_ids:
        return set()
    
    result = supabase.table("lesson_shards")\
        .select("core_lesson_id")\
        .in_("core_lesson_id", lesson_ids)\
        .execute()
    
    return set(r["core_lesson_id"] for r in (result.data or []))


def main():
    parser = argparse.ArgumentParser(description="Generate lesson content")
    parser.add_argument("--days", type=str, default="1-365", help="Day range (e.g., '1-31' or '1')")
    parser.add_argument("--atoms-only", action="store_true", help="Generate only atoms")
    parser.add_argument("--shards-only", action="store_true", help="Generate only shards")
    parser.add_argument("--translations-only", action="store_true", help="Generate only translations")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    parser.add_argument("--skip-existing", action="store_true", help="Skip lessons with existing content")
    parser.add_argument("--archetypes", type=str, help="Comma-separated archetypes to generate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Lessons per batch")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm without prompt")
    
    args = parser.parse_args()
    
    # Parse day range
    start_day, end_day = parse_day_range(args.days)
    
    # Determine what to generate
    generate_atoms = not args.shards_only and not args.translations_only
    generate_shards = not args.atoms_only and not args.translations_only
    generate_translations = not args.atoms_only and not args.shards_only or args.translations_only
    
    # Parse archetypes if specified
    archetypes = args.archetypes.split(",") if args.archetypes else ARCHETYPES
    
    print("")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   🚀 CONTENT GENERATION PIPELINE                                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"📅 Days: {start_day} - {end_day} ({end_day - start_day + 1} lessons)")
    print(f"🎭 Archetypes: {len(archetypes)}")
    print(f"📚 Phases: {len(PHASES)}")
    print(f"👤 Age buckets: {len(AGE_BUCKETS)}")
    print(f"🎨 Tones: {len(TONES)}")
    print("")
    print(f"Generate atoms: {'✅' if generate_atoms else '❌'}")
    print(f"Generate shards: {'✅' if generate_shards else '❌'}")
    print(f"Generate translations: {'✅' if generate_translations else '❌'}")
    print(f"Dry run: {'✅' if args.dry_run else '❌'}")
    print(f"Skip existing: {'✅' if args.skip_existing else '❌'}")
    print("")
    
    # Estimate work
    num_lessons = end_day - start_day + 1
    atoms_per_lesson = len(archetypes) * len(PHASES)
    shards_per_lesson = len(AGE_BUCKETS) * len(TONES)
    translations_per_shard = 2  # ES + FR
    
    total_atoms = num_lessons * atoms_per_lesson if generate_atoms else 0
    total_shards = num_lessons * shards_per_lesson if generate_shards else 0
    total_translations = total_shards * translations_per_shard if generate_translations else 0
    
    print(f"📊 Estimated generation:")
    print(f"   Atoms: {total_atoms:,} ({atoms_per_lesson}/lesson)")
    print(f"   Shards: {total_shards:,} ({shards_per_lesson}/lesson)")
    print(f"   Translations: {total_translations:,}")
    print(f"   Total API calls: ~{total_atoms + total_shards + total_translations:,}")
    print("")
    
    # Confirm (auto-proceed if --yes flag or non-interactive)
    if not args.dry_run and not args.yes:
        try:
            response = input("Proceed? [y/N]: ")
            if response.lower() != "y":
                print("Cancelled.")
                return
        except EOFError:
            # Non-interactive mode, auto-proceed
            print("Auto-proceeding (non-interactive mode)...")
    
    # Initialize generator
    generator = ContentGenerator(dry_run=args.dry_run)
    
    if not generator.client:
        print("❌ OpenAI client not initialized. Check OPENAI_API_KEY.")
        return
    
    # Fetch lessons
    print("\n📥 Fetching lessons from Supabase...")
    lessons = get_lessons_from_supabase(start_day, end_day)
    print(f"   Found {len(lessons)} lessons")
    
    if not lessons:
        print("❌ No lessons found!")
        return
    
    # Get existing content if skip_existing
    lesson_ids = [l["id"] for l in lessons]
    existing_atom_lessons = set()
    existing_shard_lessons = set()
    
    if args.skip_existing and not args.dry_run:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        existing_atom_lessons = get_existing_atoms(supabase, lesson_ids)
        existing_shard_lessons = get_existing_shards(supabase, lesson_ids)
        print(f"   Lessons with atoms: {len(existing_atom_lessons)}")
        print(f"   Lessons with shards: {len(existing_shard_lessons)}")
    
    # Process lessons
    start_time = datetime.now()
    all_atoms = []
    all_shards = []
    all_translations = []
    
    for i, lesson in enumerate(lessons):
        day = lesson.get("day_number", i + 1)
        topic = lesson.get("topic", "Unknown")
        
        print(f"\n{'='*60}")
        print(f"📖 Day {day}: {topic}")
        print(f"{'='*60}")
        
        # Generate atoms
        if generate_atoms:
            if args.skip_existing and lesson["id"] in existing_atom_lessons:
                print("   ⏭️ Skipping atoms (already exist)")
            else:
                print(f"   🧬 Generating atoms ({atoms_per_lesson} total)...")
                atoms = generator.generate_all_atoms_for_lesson(lesson, archetypes, PHASES)
                all_atoms.extend(atoms)
                
                # Upload batch
                if not args.dry_run and len(all_atoms) >= args.batch_size * atoms_per_lesson:
                    uploaded = generator.upload_atoms(all_atoms)
                    print(f"   📤 Uploaded {uploaded} atoms")
                    all_atoms = []
        
        # Generate shards
        if generate_shards:
            if args.skip_existing and lesson["id"] in existing_shard_lessons:
                print("   ⏭️ Skipping shards (already exist)")
            else:
                print(f"   👤 Generating shards ({shards_per_lesson} total)...")
                shards = generator.generate_all_shards_for_lesson(lesson, AGE_BUCKETS, TONES)
                all_shards.extend(shards)
                
                # Generate translations for these shards
                if generate_translations and shards:
                    print(f"   🌍 Generating translations...")
                    translations = generator.generate_translations_for_shards(shards, ["es", "fr"])
                    all_shards.extend(translations)
                
                # Upload batch
                if not args.dry_run and len(all_shards) >= args.batch_size * shards_per_lesson:
                    uploaded = generator.upload_shards(all_shards)
                    print(f"   📤 Uploaded {uploaded} shards")
                    all_shards = []
        
        # Progress
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(lessons) - i - 1) / rate if rate > 0 else 0
        print(f"   ⏱️ Progress: {i+1}/{len(lessons)} | Rate: {rate:.2f} lessons/sec | ETA: {remaining/60:.1f} min")
    
    # Upload remaining
    if not args.dry_run:
        if all_atoms:
            uploaded = generator.upload_atoms(all_atoms)
            print(f"\n📤 Final upload: {uploaded} atoms")
        if all_shards:
            uploaded = generator.upload_shards(all_shards)
            print(f"📤 Final upload: {uploaded} shards")
    
    # Final stats
    generator.print_stats()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
    print("\n✅ GENERATION COMPLETE!")


if __name__ == "__main__":
    main()

