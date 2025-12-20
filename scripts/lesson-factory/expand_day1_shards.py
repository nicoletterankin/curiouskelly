#!/usr/bin/env python3
"""
Day 1 Shard Expansion
=====================
- Ensures English shards exist for tones: curious, playful, serious
- Generates Spanish and French translations for every tone/age combo
"""

import os
import sys
import uuid
import time
from datetime import datetime
from typing import Dict, Tuple

from supabase import create_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from content_generator.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    AGE_BUCKETS,
    RATE_LIMIT_DELAY,
)
from content_generator.generator import ContentGenerator

TARGET_DAY = 1
TARGET_TONES = ["curious", "playful", "serious"]
TARGET_LANGS = ["es", "fr"]


def log(msg: str):
    print(msg, flush=True)


def fetch_lesson(supabase):
    res = (
        supabase.table("core_lessons")
        .select("id, day_number, topic, marketing_headline, universal_truth")
        .eq("day_number", TARGET_DAY)
        .single()
        .execute()
    )
    if res.data is None:
        raise RuntimeError(f"Day {TARGET_DAY} lesson not found.")
    return res.data


def fetch_existing_shards(supabase, lesson_id: str):
    res = (
        supabase.table("lesson_shards")
        .select("id, age, region, tone, script_content, birth_year")
        .eq("core_lesson_id", lesson_id)
        .execute()
    )
    return res.data or []


def ensure_english_shards(generator: ContentGenerator, supabase, lesson: dict, existing: list):
    english_map: Dict[Tuple[int, str], dict] = {
        (row["age"], row["tone"]): row
        for row in existing
        if row["region"] == "en"
    }

    missing = []
    for bucket in AGE_BUCKETS:
        for tone in TARGET_TONES:
            key = (bucket["age"], tone)
            if key not in english_map:
                missing.append((bucket, tone))

    if not missing:
        log("✅ English shards already cover curious/playful/serious.")
        return existing, english_map

    log(f"🧠 Generating {len(missing)} missing English shards...")
    new_shards = []
    for bucket, tone in missing:
        log(f"   • Age {bucket['age']} tone {tone}")
        shard = generator.generate_shard(lesson, bucket, tone)
        if shard:
            new_shards.append(shard)
            english_map[(bucket["age"], tone)] = shard
        time.sleep(RATE_LIMIT_DELAY)

    if new_shards:
        uploaded = generator.upload_shards(new_shards)
        log(f"   ↳ Uploaded {uploaded} English shards.")
        existing.extend(new_shards)
    else:
        log("   ⚠️ No new English shards generated.")

    return existing, english_map


def ensure_translations(generator: ContentGenerator, supabase, lesson: dict, existing: list, english_map: Dict[Tuple[int, str], dict]):
    existing_keys = {
        (row["age"], row["tone"], row["region"])
        for row in existing
    }

    translations = []

    for bucket in AGE_BUCKETS:
        for tone in TARGET_TONES:
            source = english_map.get((bucket["age"], tone))
            if not source:
                log(f"   ⚠️ Missing English shard for Age {bucket['age']} tone {tone}, skipping translations.")
                continue

            for lang in TARGET_LANGS:
                key = (bucket["age"], tone, lang)
                if key in existing_keys:
                    continue

                log(f"   • Translating Age {bucket['age']} tone {tone} → {lang}")
                translated = generator.translate_content(source["script_content"], lang)
                if translated:
                    translations.append({
                        "id": str(uuid.uuid4()),
                        "core_lesson_id": lesson["id"],
                        "age": bucket["age"],
                        "region": lang,
                        "tone": tone,
                        "birth_year": bucket["birth_year"],
                        "script_content": translated,
                        "created_at": datetime.utcnow().isoformat(),
                    })
                time.sleep(RATE_LIMIT_DELAY)

    if translations:
        uploaded = generator.upload_shards(translations)
        log(f"   ↳ Uploaded {uploaded} translations.")
        existing.extend(translations)
    else:
        log("   ✅ No translation gaps.")

    return existing


def main():
    log("╔══════════════════════════════════════════════╗")
    log("║   DAY 1 SHARD EXPANSION (TONES + LANGS)      ║")
    log("╚══════════════════════════════════════════════╝")

    generator = ContentGenerator(dry_run=False)
    if not generator.client:
        raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY.")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    lesson = fetch_lesson(supabase)
    log(f"📘 Lesson: Day {lesson['day_number']} — {lesson.get('topic')}")

    existing = fetch_existing_shards(supabase, lesson["id"])
    log(f"📊 Existing shards: {len(existing)}")

    existing, english_map = ensure_english_shards(generator, supabase, lesson, existing)
    existing = ensure_translations(generator, supabase, lesson, existing, english_map)

    # Final counts per language/tone
    counts = {}
    for row in existing:
        key = f"{row['region']}-{row['tone']}"
        counts[key] = counts.get(key, 0) + 1

    log("📈 Final shard counts by language+tone:")
    for key in sorted(counts.keys()):
        log(f"   - {key}: {counts[key]}")

    generator.print_stats()
    log("✅ Day 1 shard expansion complete.")


if __name__ == "__main__":
    main()
















