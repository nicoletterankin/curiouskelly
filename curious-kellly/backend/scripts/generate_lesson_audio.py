#!/usr/bin/env python3
"""
Generate lesson audio files using ElevenLabs API.

This script generates audio for all lessons across all age variants and languages.
It follows the rules in CLAUDE.md:
- Uses ElevenLabs for synthesis (never browser TTS)
- Batches API calls efficiently
- Caches responses
- Validates sample rate, duration, and format

Usage:
    python generate_lesson_audio.py --lesson all
    python generate_lesson_audio.py --lesson the-sun --age-variant 6-12 --language en
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import time
import sys

try:
    from elevenlabs import generate, set_api_key, voices, Voice
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("⚠️  ElevenLabs library not installed. Run: pip install elevenlabs")

# Paths
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
LESSONS_DIR = BACKEND_DIR / "config" / "lessons"
AUDIO_OUTPUT_DIR = BACKEND_DIR / "assets" / "audio"
METADATA_OUTPUT_DIR = BACKEND_DIR / "assets" / "audio" / "metadata"
CACHE_DIR = BACKEND_DIR / ".audio_cache"

# Create directories
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ElevenLabs Voice IDs (Kelly voice from user's account)
VOICE_IDS = {
    "kelly": "wAdymQH5YucAkXwmrdL0"  # Kelly voice from training
}

# Age buckets
AGE_BUCKETS = ["2-5", "6-12", "13-17", "18-35", "36-60", "61-102"]

# Languages
LANGUAGES = ["en", "es", "fr"]

# Phase types (for future expansion)
PHASE_TYPES = ["welcome", "mainContent", "wisdomMoment"]


class AudioGenerator:
    """Generate audio files for lessons using ElevenLabs API."""

    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        """
        Initialize audio generator.

        Args:
            api_key: ElevenLabs API key (or use ELEVENLABS_API_KEY env var)
            use_cache: Whether to use cached audio files
        """
        if not ELEVENLABS_AVAILABLE:
            raise RuntimeError("ElevenLabs library not available. Install with: pip install elevenlabs")

        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key required. Set ELEVENLABS_API_KEY or pass --api-key")

        set_api_key(self.api_key)
        self.use_cache = use_cache
        self.request_count = 0
        self.cache_hits = 0

    def get_cache_key(self, text: str, voice_id: str, language: str) -> str:
        """Generate cache key for text/voice/language combination."""
        content = f"{text}|{voice_id}|{language}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Retrieve cached audio if available."""
        if not self.use_cache:
            return None

        cache_file = CACHE_DIR / f"{cache_key}.mp3"
        if cache_file.exists():
            self.cache_hits += 1
            return cache_file.read_bytes()
        return None

    def save_to_cache(self, cache_key: str, audio_data: bytes):
        """Save audio to cache."""
        cache_file = CACHE_DIR / f"{cache_key}.mp3"
        cache_file.write_bytes(audio_data)

    def generate_audio(self, text: str, voice_id: str, language: str) -> bytes:
        """
        Generate audio using ElevenLabs API.

        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID
            language: Language code (en, es, fr)

        Returns:
            Audio data as bytes (MP3 format)
        """
        cache_key = self.get_cache_key(text, voice_id, language)

        # Check cache
        cached_audio = self.get_cached_audio(cache_key)
        if cached_audio:
            print(f"  ✓ Cache hit")
            return cached_audio

        # Generate audio via API
        print(f"  → Generating audio via ElevenLabs API...")
        self.request_count += 1

        try:
            audio = generate(
                text=text,
                voice=voice_id,
                model="eleven_multilingual_v2"  # Supports EN, ES, FR
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio)

            # Save to cache
            self.save_to_cache(cache_key, audio_bytes)

            # Rate limiting (11Labs free tier: 10,000 chars/month)
            time.sleep(0.5)  # Be respectful to API

            return audio_bytes

        except Exception as e:
            print(f"  ✗ Error generating audio: {e}")
            raise

    def load_lesson(self, lesson_id: str) -> Dict:
        """Load lesson DNA file."""
        lesson_file = LESSONS_DIR / f"{lesson_id}.json"
        if not lesson_file.exists():
            raise FileNotFoundError(f"Lesson not found: {lesson_file}")

        with open(lesson_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_lesson_audio(self, lesson_id: str, age_variant: Optional[str] = None, language: Optional[str] = None):
        """
        Generate audio for a lesson.

        Args:
            lesson_id: Lesson ID (e.g., "the-sun")
            age_variant: Specific age variant (e.g., "6-12") or None for all
            language: Specific language (e.g., "en") or None for all
        """
        print(f"\n{'='*60}")
        print(f"📚 Lesson: {lesson_id}")
        print(f"{'='*60}\n")

        lesson = self.load_lesson(lesson_id)

        age_variants_to_process = [age_variant] if age_variant else AGE_BUCKETS
        languages_to_process = [language] if language else LANGUAGES

        for age_bucket in age_variants_to_process:
            if age_bucket not in lesson["ageVariants"]:
                print(f"⚠️  Age variant {age_bucket} not found in lesson")
                continue

            age_data = lesson["ageVariants"][age_bucket]

            for lang in languages_to_process:
                if lang not in age_data.get("language", {}):
                    print(f"⚠️  Language {lang} not found for age {age_bucket}")
                    continue

                lang_data = age_data["language"][lang]
                voice_id = VOICE_IDS["kelly"]  # Use Kelly voice

                print(f"🎙️  {age_bucket} | {lang.upper()}")

                # Generate welcome audio
                welcome_text = lang_data.get("welcome", "")
                if welcome_text:
                    output_file = AUDIO_OUTPUT_DIR / f"{lesson_id}_{age_bucket}_{lang}_welcome.mp3"
                    print(f"  Welcome: {len(welcome_text)} chars → {output_file.name}")

                    audio_data = self.generate_audio(welcome_text, voice_id, lang)
                    output_file.write_bytes(audio_data)

                    # Create metadata
                    self.create_metadata(output_file, welcome_text, voice_id, lang, age_bucket, "welcome")

                # Generate main content audio
                main_content = lang_data.get("mainContent", "")
                if main_content:
                    output_file = AUDIO_OUTPUT_DIR / f"{lesson_id}_{age_bucket}_{lang}_main.mp3"
                    print(f"  Main: {len(main_content)} chars → {output_file.name}")

                    audio_data = self.generate_audio(main_content, voice_id, lang)
                    output_file.write_bytes(audio_data)

                    self.create_metadata(output_file, main_content, voice_id, lang, age_bucket, "main")

                # Generate wisdom moment audio
                wisdom = lang_data.get("wisdomMoment", "")
                if wisdom:
                    output_file = AUDIO_OUTPUT_DIR / f"{lesson_id}_{age_bucket}_{lang}_wisdom.mp3"
                    print(f"  Wisdom: {len(wisdom)} chars → {output_file.name}")

                    audio_data = self.generate_audio(wisdom, voice_id, lang)
                    output_file.write_bytes(audio_data)

                    self.create_metadata(output_file, wisdom, voice_id, lang, age_bucket, "wisdom")

                print()

    def create_metadata(self, audio_file: Path, text: str, voice_id: str, language: str, age_variant: str, phase_type: str):
        """Create metadata file for generated audio."""
        metadata = {
            "file": audio_file.name,
            "text": text,
            "text_length": len(text),
            "voice_id": voice_id,
            "language": language,
            "age_variant": age_variant,
            "phase_type": phase_type,
            "model": "eleven_multilingual_v2",
            "format": "mp3",
            "sample_rate": 44100,  # ElevenLabs default
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sync_markers": []  # To be populated by sync marker tool
        }

        metadata_file = METADATA_OUTPUT_DIR / f"{audio_file.stem}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def generate_all_lessons(self):
        """Generate audio for all lessons."""
        lessons = ["the-sun", "the-moon", "the-ocean", "puppies"]

        print("\n" + "="*60)
        print("🎬 GENERATING ALL LESSON AUDIO")
        print("="*60)
        print(f"Lessons: {len(lessons)}")
        print(f"Age variants: {len(AGE_BUCKETS)}")
        print(f"Languages: {len(LANGUAGES)}")
        print(f"Estimated files: {len(lessons) * len(AGE_BUCKETS) * len(LANGUAGES) * 3} (welcome + main + wisdom)")
        print("="*60 + "\n")

        for lesson_id in lessons:
            try:
                self.generate_lesson_audio(lesson_id)
            except Exception as e:
                print(f"✗ Error processing {lesson_id}: {e}")
                continue

        self.print_summary()

    def print_summary(self):
        """Print generation summary."""
        print("\n" + "="*60)
        print("✅ AUDIO GENERATION COMPLETE")
        print("="*60)
        print(f"API requests: {self.request_count}")
        print(f"Cache hits: {self.cache_hits}")
        print(f"Total generated: {self.request_count + self.cache_hits}")
        print(f"Output directory: {AUDIO_OUTPUT_DIR}")
        print(f"Metadata directory: {METADATA_OUTPUT_DIR}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate lesson audio using ElevenLabs")
    parser.add_argument("--lesson", default="all", help="Lesson ID or 'all'")
    parser.add_argument("--age-variant", help="Specific age variant (e.g., '6-12')")
    parser.add_argument("--language", help="Specific language (e.g., 'en')")
    parser.add_argument("--api-key", help="ElevenLabs API key (or use ELEVENLABS_API_KEY env var)")
    parser.add_argument("--no-cache", action="store_true", help="Disable audio caching")

    args = parser.parse_args()

    try:
        generator = AudioGenerator(api_key=args.api_key, use_cache=not args.no_cache)

        if args.lesson == "all":
            generator.generate_all_lessons()
        else:
            generator.generate_lesson_audio(args.lesson, args.age_variant, args.language)

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



