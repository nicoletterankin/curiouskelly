#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Audio Files for All DNA Lessons
Batch processes all DNA lesson files to generate audio for all age variants, languages, and phases
"""

import json
import os
import sys
import requests
from pathlib import Path
import time
import argparse

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ElevenLabs Configuration
API_KEY = os.environ.get("ELEVENLABS_API_KEY", "sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568")
VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly25 voice
BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Age buckets and languages
AGE_BUCKETS = ["2-5", "6-12", "13-17", "18-35", "36-60", "61-102"]
LANGUAGES = ["en", "es", "fr"]
PHASES = ["welcome", "mainContent", "wisdomMoment"]

# Base directories
LESSONS_DIR = Path(__file__).parent.parent / "lessons"
AUDIO_BASE_DIR = Path(__file__).parent.parent / "lessons" / "audio"


def generate_speech(text, output_path, voice_id=VOICE_ID, api_key=API_KEY, language="en"):
    """Generate speech using ElevenLabs API"""
    
    if not text or not text.strip():
        print(f"  [SKIP] Skipping empty text for {output_path.name}")
        return False
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    print(f"  [GENERATING] Generating: {output_path.name}")
    print(f"     Language: {language.upper()}, Text length: {len(text)} chars")
    
    try:
        response = requests.post(
            f"{BASE_URL}/{voice_id}",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"  [SUCCESS] Saved: {output_path.name}")
            return True
        else:
            print(f"  [ERROR] Error {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return False


def load_lesson_dna(lesson_file):
    """Load lesson DNA from JSON file"""
    lesson_path = Path(lesson_file)
    
    if not lesson_path.exists():
        print(f"❌ Lesson file not found: {lesson_path}")
        return None
    
    try:
        with open(lesson_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading lesson: {e}")
        return None


def get_text_for_phase(lesson_data, age_bucket, language, phase):
    """Extract text for a specific phase from lesson DNA"""
    try:
        age_variant = lesson_data["ageVariants"].get(age_bucket)
        if not age_variant:
            return None
        
        # Try to get from language structure
        lang_data = age_variant.get("language", {}).get(language)
        if lang_data and phase in lang_data:
            text = lang_data[phase]
            if text and text.strip():
                return text.strip()
        
        # Fallback: try direct phase field
        if phase in age_variant:
            text = age_variant[phase]
            if text and text.strip():
                return text.strip()
        
        # Fallback: use script for mainContent
        if phase == "mainContent" and "script" in age_variant:
            text = age_variant["script"]
            if text and text.strip():
                return text.strip()
        
        return None
    except Exception as e:
        print(f"    [WARN] Error extracting text: {e}")
        return None


def generate_lesson_audio(lesson_id, lesson_data, output_dir, dry_run=False):
    """Generate all audio files for a single lesson"""
    
    print(f"\n{'='*60}")
    print(f"Processing Lesson: {lesson_id}")
    print(f"{'='*60}")
    
    lesson_output_dir = output_dir / lesson_id
    audio_metadata = {
        "lesson_id": lesson_id,
        "title": lesson_data.get("title", ""),
        "audio_files": {}
    }
    
    successful = 0
    total = 0
    skipped = 0
    
    # Generate audio for each combination
    for age_bucket in AGE_BUCKETS:
        if age_bucket not in lesson_data.get("ageVariants", {}):
            print(f"  [WARN] Age bucket {age_bucket} not found, skipping")
            continue
        
        audio_metadata["audio_files"][age_bucket] = {}
        
        for language in LANGUAGES:
            audio_metadata["audio_files"][age_bucket][language] = {}
            
            for phase in PHASES:
                total += 1
                
                # Get text for this phase
                text = get_text_for_phase(lesson_data, age_bucket, language, phase)
                
                if not text:
                    print(f"  [SKIP] No text found for {age_bucket}/{language}/{phase}, skipping")
                    skipped += 1
                    continue
                
                # Create output file path
                output_file = lesson_output_dir / f"{age_bucket}-{language}-{phase}.mp3"
                
                # Check if file already exists
                if output_file.exists():
                    print(f"  [CACHED] Already exists: {output_file.name}")
                    successful += 1
                    audio_metadata["audio_files"][age_bucket][language][phase] = {
                        "file": output_file.name,
                        "path": str(output_file.relative_to(output_dir)),
                        "status": "cached"
                    }
                    continue
                
                if dry_run:
                    print(f"  [DRY RUN] Would generate: {output_file.name}")
                    successful += 1
                    continue
                
                # Generate speech
                if generate_speech(text, output_file, language=language):
                    successful += 1
                    audio_metadata["audio_files"][age_bucket][language][phase] = {
                        "file": output_file.name,
                        "path": str(output_file.relative_to(output_dir)),
                        "text_length": len(text),
                        "status": "generated"
                    }
                else:
                    audio_metadata["audio_files"][age_bucket][language][phase] = {
                        "file": output_file.name,
                        "status": "failed"
                    }
                
                # Rate limiting - be nice to API
                time.sleep(1)
    
    # Save metadata
    metadata_file = lesson_output_dir / "metadata.json"
    if not dry_run:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(audio_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Summary: {successful}/{total} generated ({skipped} skipped)")
    return successful, total, skipped


def find_dna_files(lessons_dir):
    """Find all DNA lesson files"""
    dna_files = []
    
    # Look for files ending in -dna.json
    for file_path in lessons_dir.glob("*-dna.json"):
        # Extract lesson ID from filename
        lesson_id = file_path.stem.replace("-dna", "")
        dna_files.append((lesson_id, file_path))
    
    return sorted(dna_files)


def main():
    parser = argparse.ArgumentParser(description="Generate audio files for all DNA lessons")
    parser.add_argument("--lesson", help="Process only a specific lesson ID")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't generate files)")
    parser.add_argument("--lessons-dir", default=LESSONS_DIR, help="Directory containing DNA files")
    parser.add_argument("--output-dir", default=AUDIO_BASE_DIR, help="Output directory for audio files")
    
    args = parser.parse_args()
    
    lessons_dir = Path(args.lessons_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("Batch Audio Generation for DNA Lessons")
    print("="*60)
    print(f"Lessons directory: {lessons_dir}")
    print(f"Output directory: {output_dir}")
    print(f"API Key: {'*' * 20}...{API_KEY[-10:] if len(API_KEY) > 10 else 'NOT SET'}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    # Find DNA files
    if args.lesson:
        # Process single lesson
        lesson_file = lessons_dir / f"{args.lesson}-dna.json"
        if not lesson_file.exists():
            print(f"[ERROR] Lesson file not found: {lesson_file}")
            return 1
        
        dna_files = [(args.lesson, lesson_file)]
    else:
        # Find all DNA files
        dna_files = find_dna_files(lessons_dir)
        print(f"\nFound {len(dna_files)} DNA lesson files")
    
    if not dna_files:
        print("[ERROR] No DNA files found!")
        return 1
    
    # Process each lesson
    total_successful = 0
    total_files = 0
    total_skipped = 0
    
    for lesson_id, lesson_file in dna_files:
        lesson_data = load_lesson_dna(lesson_file)
        if not lesson_data:
            continue
        
        successful, total, skipped = generate_lesson_audio(
            lesson_id, 
            lesson_data, 
            output_dir,
            dry_run=args.dry_run
        )
        
        total_successful += successful
        total_files += total
        total_skipped += skipped
    
    # Final summary
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)
    print(f"Lessons processed: {len(dna_files)}")
    print(f"Audio files generated: {total_successful}/{total_files}")
    print(f"Skipped (no text): {total_skipped}")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)
    
    return 0 if total_successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

