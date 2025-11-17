#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Lesson Manifests
Creates JSON manifest files for each lesson that link audio files, images, and lesson structure
"""

import json
import sys
from pathlib import Path
import argparse

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Base directories
LESSONS_DIR = Path(__file__).parent.parent / "lessons"
AUDIO_DIR = LESSONS_DIR / "audio"
IMAGES_DIR = LESSONS_DIR / "images"
MANIFESTS_DIR = LESSONS_DIR / "manifests"

# Age buckets, languages, and phases
AGE_BUCKETS = ["2-5", "6-12", "13-17", "18-35", "36-60", "61-102"]
LANGUAGES = ["en", "es", "fr"]
PHASES = ["welcome", "mainContent", "wisdomMoment"]

# Expression images
EXPRESSION_IMAGES = {
    "curious": "kelly-directors-chair-curious.png",
    "explaining": "kelly-directors-chair-explaining.png",
    "celebrating": "kelly-directors-chair-celebrating.png",
    "listening": "kelly-directors-chair-listening.png",
    "wisdom": "kelly-directors-chair-wisdom.png"
}


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


def check_audio_files(lesson_id, audio_dir):
    """Check which audio files exist for a lesson"""
    lesson_audio_dir = audio_dir / lesson_id
    audio_files = {}
    
    if not lesson_audio_dir.exists():
        return audio_files
    
    for age_bucket in AGE_BUCKETS:
        audio_files[age_bucket] = {}
        for language in LANGUAGES:
            audio_files[age_bucket][language] = {}
            for phase in PHASES:
                audio_file = lesson_audio_dir / f"{age_bucket}-{language}-{phase}.mp3"
                if audio_file.exists():
                    # Use relative path from lessons directory
                    audio_files[age_bucket][language][phase] = f"audio/{lesson_id}/{audio_file.name}"
                else:
                    audio_files[age_bucket][language][phase] = None
    
    return audio_files


def check_image_files(images_dir):
    """Check which expression images exist"""
    images = {}
    
    for expression, filename in EXPRESSION_IMAGES.items():
        image_path = images_dir / filename
        if image_path.exists():
            images[expression] = f"images/{filename}"
        else:
            images[expression] = None
    
    return images


def create_image_selection_rules():
    """Create image selection rules based on phases and interactions"""
    return {
        "phaseMapping": {
            "welcome": "curious",
            "teaching": "explaining",
            "mainContent": "explaining",
            "practice": "listening",
            "wisdom": "wisdom",
            "wisdomMoment": "wisdom",
            "reflection": "wisdom"
        },
        "interactionMapping": {
            "question": "curious",
            "explanation": "explaining",
            "celebration": "celebrating",
            "response": "listening",
            "feedback": "listening",
            "wisdom": "wisdom"
        },
        "sentimentMapping": {
            "positive": "celebrating",
            "correct": "celebrating",
            "encouraging": "celebrating",
            "neutral": "listening",
            "thoughtful": "listening",
            "reflective": "wisdom"
        }
    }


def generate_manifest(lesson_id, lesson_data, audio_dir, images_dir, output_dir):
    """Generate manifest JSON for a single lesson"""
    
    print(f"  Generating manifest for: {lesson_id}")
    
    # Check audio files
    audio_files = check_audio_files(lesson_id, audio_dir)
    
    # Check image files
    image_files = check_image_files(images_dir)
    
    # Create manifest structure
    manifest = {
        "version": "1.0.0",
        "lesson_id": lesson_id,
        "title": lesson_data.get("title", ""),
        "description": lesson_data.get("description", ""),
        "metadata": {
            "calendar": lesson_data.get("calendar", {}),
            "category": lesson_data.get("metadata", {}).get("category", ""),
            "difficulty": lesson_data.get("metadata", {}).get("difficulty", ""),
            "duration": lesson_data.get("metadata", {}).get("duration", {})
        },
        "audio": audio_files,
        "images": image_files,
        "imageSelection": create_image_selection_rules(),
        "ageVariants": list(lesson_data.get("ageVariants", {}).keys()),
        "languages": LANGUAGES,
        "phases": PHASES,
        "interactions": []
    }
    
    # Add interaction structure if available
    if "interactions" in lesson_data:
        for interaction in lesson_data["interactions"]:
            manifest["interactions"].append({
                "step": interaction.get("step", ""),
                "phase": interaction.get("phase", interaction.get("step", "")),
                "question": interaction.get("question", ""),
                "choices": interaction.get("choices", [])
            })
    
    # Save manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = output_dir / f"{lesson_id}-manifest.json"
    
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    # Count available audio files
    audio_count = sum(
        1 for age_data in audio_files.values()
        for lang_data in age_data.values()
        for phase_file in lang_data.values()
        if phase_file is not None
    )
    
    total_expected = len(AGE_BUCKETS) * len(LANGUAGES) * len(PHASES)
    
    print(f"    [OK] Manifest saved: {manifest_file.name}")
    print(f"    Audio files: {audio_count}/{total_expected}")
    print(f"    Images: {sum(1 for img in image_files.values() if img is not None)}/{len(image_files)}")
    
    return manifest_file


def find_dna_files(lessons_dir):
    """Find all DNA lesson files"""
    dna_files = []
    
    for file_path in lessons_dir.glob("*-dna.json"):
        lesson_id = file_path.stem.replace("-dna", "")
        dna_files.append((lesson_id, file_path))
    
    return sorted(dna_files)


def main():
    parser = argparse.ArgumentParser(description="Generate manifest files for DNA lessons")
    parser.add_argument("--lesson", help="Process only a specific lesson ID")
    parser.add_argument("--lessons-dir", default=LESSONS_DIR, help="Directory containing DNA files")
    parser.add_argument("--audio-dir", default=AUDIO_DIR, help="Directory containing audio files")
    parser.add_argument("--images-dir", default=IMAGES_DIR, help="Directory containing image files")
    parser.add_argument("--output-dir", default=MANIFESTS_DIR, help="Output directory for manifests")
    
    args = parser.parse_args()
    
    lessons_dir = Path(args.lessons_dir)
    audio_dir = Path(args.audio_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("Lesson Manifest Generator")
    print("="*60)
    print(f"Lessons directory: {lessons_dir}")
    print(f"Audio directory: {audio_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Find DNA files
    if args.lesson:
        lesson_file = lessons_dir / f"{args.lesson}-dna.json"
        if not lesson_file.exists():
            print(f"[ERROR] Lesson file not found: {lesson_file}")
            return 1
        dna_files = [(args.lesson, lesson_file)]
    else:
        dna_files = find_dna_files(lessons_dir)
        print(f"\nFound {len(dna_files)} DNA lesson files")
    
    if not dna_files:
        print("[ERROR] No DNA files found!")
        return 1
    
    # Generate manifests
    generated = 0
    for lesson_id, lesson_file in dna_files:
        lesson_data = load_lesson_dna(lesson_file)
        if not lesson_data:
            continue
        
        try:
            manifest_file = generate_manifest(
                lesson_id,
                lesson_data,
                audio_dir,
                images_dir,
                output_dir
            )
            generated += 1
        except Exception as e:
            print(f"  [ERROR] Error generating manifest: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Manifests generated: {generated}/{len(dna_files)}")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)
    
    return 0 if generated > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

