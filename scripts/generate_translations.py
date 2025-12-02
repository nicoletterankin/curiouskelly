#!/usr/bin/env python3
"""
Generate Translations for Existing Shards
==========================================
Translates all English shards to Spanish and French.

Run AFTER generate_essential_content.py completes.

Usage:
    python scripts/generate_translations.py --days 1-365 --lang es --yes
    python scripts/generate_translations.py --days 1-365 --lang fr --yes
    python scripts/generate_translations.py --days 1-365 --lang es,fr --yes
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from supabase import create_client

from content_generator.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from content_generator.generator import ContentGenerator


def parse_day_range(day_range: str) -> tuple:
    if "-" in day_range:
        parts = day_range.split("-")
        return int(parts[0]), int(parts[1])
    return int(day_range), int(day_range)


def get_english_shards(supabase, lesson_ids: list) -> list:
    """Get all English shards for given lessons."""
    if not lesson_ids:
        return []
    
    all_shards = []
    for i in range(0, len(lesson_ids), 50):
        batch = lesson_ids[i:i+50]
        result = supabase.table("lesson_shards")\
            .select("*")\
            .in_("core_lesson_id", batch)\
            .eq("region", "en")\
            .execute()
        if result.data:
            all_shards.extend(result.data)
    
    return all_shards


def main():
    parser = argparse.ArgumentParser(description="Generate translations for shards")
    parser.add_argument("--days", type=str, default="1-365", help="Day range")
    parser.add_argument("--lang", type=str, default="es,fr", help="Target languages (comma-separated)")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm")
    
    args = parser.parse_args()
    start_day, end_day = parse_day_range(args.days)
    target_langs = [l.strip() for l in args.lang.split(",")]
    
    print("")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   🌍 TRANSLATION GENERATION                                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"📅 Days: {start_day} - {end_day}")
    print(f"🌐 Languages: {', '.join(target_langs)}")
    print("")
    
    # Get lesson IDs
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    
    print("📥 Fetching lessons...")
    lessons = supabase.table("core_lessons")\
        .select("id, day_number, topic")\
        .gte("day_number", start_day)\
        .lte("day_number", end_day)\
        .order("day_number")\
        .execute()
    
    if not lessons.data:
        print("❌ No lessons found!")
        return
    
    lesson_ids = [l["id"] for l in lessons.data]
    print(f"   Found {len(lesson_ids)} lessons")
    
    # Get English shards
    print("📥 Fetching English shards...")
    en_shards = get_english_shards(supabase, lesson_ids)
    print(f"   Found {len(en_shards)} English shards")
    
    if not en_shards:
        print("❌ No English shards found! Run generate_essential_content.py first.")
        return
    
    total_translations = len(en_shards) * len(target_langs)
    est_time = total_translations * 1.5 / 60
    
    print(f"\n📊 Translation plan:")
    print(f"   English shards: {len(en_shards)}")
    print(f"   Languages: {len(target_langs)}")
    print(f"   Total translations: {total_translations:,}")
    print(f"   Estimated time: ~{est_time:.0f} minutes")
    print("")
    
    if not args.yes:
        try:
            response = input("Proceed? [y/N]: ")
            if response.lower() != "y":
                print("Cancelled.")
                return
        except EOFError:
            print("Auto-proceeding...")
    
    # Initialize generator
    generator = ContentGenerator(dry_run=False)
    
    if not generator.client:
        print("❌ OpenAI client not initialized.")
        return
    
    # Process translations
    start_time = datetime.now()
    translated_shards = []
    
    for i, shard in enumerate(en_shards):
        print(f"[{i+1}/{len(en_shards)}] Translating shard...")
        
        for lang in target_langs:
            translated_content = generator.translate_content(shard["script_content"], lang)
            
            if translated_content:
                import uuid
                new_shard = {
                    "id": str(uuid.uuid4()),
                    "core_lesson_id": shard["core_lesson_id"],
                    "age": shard["age"],
                    "region": lang,
                    "tone": shard["tone"],
                    "birth_year": shard.get("birth_year"),
                    "script_content": translated_content,
                    "created_at": datetime.utcnow().isoformat(),
                }
                translated_shards.append(new_shard)
            
            time.sleep(0.5)
        
        # Upload batch
        if len(translated_shards) >= 50:
            try:
                result = supabase.table("lesson_shards").insert(translated_shards).execute()
                print(f"   📤 Uploaded {len(result.data)} translations")
            except Exception as e:
                print(f"   ❌ Upload error: {e}")
            translated_shards = []
        
        # Progress
        if (i + 1) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(en_shards) - i - 1) / rate if rate > 0 else 0
            print(f"   ⏱️ {rate*60:.1f}/min | ETA: {remaining:.0f} min")
    
    # Final upload
    if translated_shards:
        try:
            result = supabase.table("lesson_shards").insert(translated_shards).execute()
            print(f"\n📤 Final: {len(result.data)} translations")
        except Exception as e:
            print(f"❌ Final upload error: {e}")
    
    generator.print_stats()
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
    print("\n✅ TRANSLATIONS COMPLETE!")


if __name__ == "__main__":
    main()



