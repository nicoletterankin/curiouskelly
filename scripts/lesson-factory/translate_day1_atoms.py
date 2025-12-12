#!/usr/bin/env python3
"""
Translate Day 1 lesson atoms (script + options/responses) into ES & FR.
Stores translations inside lesson_atoms.content.translations[lang].
"""

import os
import sys
import time
from typing import Dict, Any

from supabase import create_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from content_generator.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    RATE_LIMIT_DELAY,
)
from content_generator.generator import ContentGenerator

TARGET_DAY = 1
TARGET_LANGS = ["es", "fr"]
TARGET_ARCHETYPES = [
    "The Architect",
    "The Empath",
    "The MacGyver",
    "The Explorer",
    "The Mystic",
    "The Provider",
    "The Rebel",
    "The Scientist",
    "The Storyteller",
    "The Strategist",
    "The Survivor",
    "The Diplomat"
]


def log(msg: str):
    print(msg, flush=True)


def fetch_lesson(supabase):
    res = (
        supabase.table("core_lessons")
        .select("id, day_number, topic")
        .eq("day_number", TARGET_DAY)
        .single()
        .execute()
    )
    if res.data is None:
        raise RuntimeError(f"Day {TARGET_DAY} lesson not found.")
    return res.data


def fetch_atoms(supabase, lesson_id: str):
    res = (
        supabase.table("lesson_atoms")
        .select("id, archetype, phase, content")
        .eq("core_lesson_id", lesson_id)
        .in_("archetype", TARGET_ARCHETYPES)
        .execute()
    )
    return res.data or []


def build_payload(content: Dict[str, Any]) -> Dict[str, Any]:
    options_payload = []
    for opt in content.get("options", []):
        options_payload.append({
            "text": opt.get("text", ""),
            "response": opt.get("response", ""),
        })
    return {
        "script": content.get("script", ""),
        "options": options_payload,
    }


def apply_translation(content: Dict[str, Any], lang: str, translated: Dict[str, Any]):
    translations = content.get("translations") or {}
    translations[lang] = {
        "script": translated.get("script", ""),
        "options": translated.get("options", []),
    }
    content["translations"] = translations
    return content


def main():
    log("╔══════════════════════════════════════════╗")
    log("║  DAY 1 LESSON ATOM TRANSLATIONS (ES/FR)  ║")
    log("╚══════════════════════════════════════════╝")

    generator = ContentGenerator(dry_run=False)
    if not generator.client:
        raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    lesson = fetch_lesson(supabase)
    atoms = fetch_atoms(supabase, lesson["id"])

    log(f"📘 Lesson: Day {lesson['day_number']} — {lesson.get('topic')}")
    log(f"🧩 Atoms selected: {len(atoms)} ({', '.join(TARGET_ARCHETYPES)})")

    updates = 0
    for atom in atoms:
        content = atom.get("content") or {}
        translations = content.get("translations") or {}
        payload = build_payload(content)

        for lang in TARGET_LANGS:
            if payload["script"].strip() == "":
                continue
            if lang in translations and translations[lang].get("script"):
                continue

            log(f"   • Translating {atom['archetype']} {atom['phase']} → {lang}")
            translated = generator.translate_content(payload, lang)
            if not translated:
                log("     ⚠️ Translation failed, skipping.")
                continue

            content = apply_translation(content, lang, translated)
            updates += 1
            time.sleep(RATE_LIMIT_DELAY)

        # Persist if new translations inserted
        supabase.table("lesson_atoms").update({"content": content}).eq("id", atom["id"]).execute()

    generator.print_stats()
    log(f"✅ Updated {updates} translation entries.")


if __name__ == "__main__":
    main()

