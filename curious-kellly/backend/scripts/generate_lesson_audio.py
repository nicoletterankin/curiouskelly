#!/usr/bin/env python3
"""
Curious Kellly - Audio Generation Script
Generates multilingual audio files for all lesson variants using ElevenLabs API
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
API_KEY = os.getenv('ELEVENLABS_API_KEY')
VOICE_ID = "wAdymQH5YucAkXwmrdL0"
MODEL_ID = "eleven_multilingual_v2"
BASE_DIR = Path(__file__).parent.parent
LESSONS_DIR = BASE_DIR / "config" / "lessons"
AUDIO_DIR = BASE_DIR / "config" / "audio"

AGE_BUCKETS = ["2-5", "6-12", "13-17", "18-35", "36-60", "61-102"]
LANGUAGES = ["en", "es", "fr"]
SECTIONS = ["welcome", "mainContent", "wisdomMoment"]

# Voice settings
VOICE_SETTINGS = {
    "stability": 0.6,
    "similarity_boost": 0.8,
    "style": 0.0,
    "use_speaker_boost": True
}

def generate_audio(text, output_path):
    """Generate audio using ElevenLabs API"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }
    data = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS
    }
    
    response = requests.post(url, json=data, headers=headers, timeout=30)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        print(f"Error {response.status_code}: {response.text[:100]}")
        return False

def generate_lesson_audio(lesson_id):
    """Generate all audio files for a lesson"""
    
    if not API_KEY:
        print("❌ ELEVENLABS_API_KEY not found in .env")
        return False
        
    # Load lesson
    lesson_path = LESSONS_DIR / f"{lesson_id}.json"
    if not lesson_path.exists():
        print(f"❌ Lesson not found: {lesson_path}")
        return False
        
    with open(lesson_path) as f:
        lesson = json.load(f)
    
    print(f"\n🎬 Generating audio for: {lesson.get('title')}")
    print(f"📍 Output: {AUDIO_DIR / lesson_id}")
    
    # Create output directory
    output_dir = AUDIO_DIR / lesson_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = 0
    generated = 0
    skipped = 0
    
    for age in AGE_BUCKETS:
        variant = lesson.get("ageVariants", {}).get(age, {})
        languages = variant.get("language", {})
        
        for lang in LANGUAGES:
            lang_content = languages.get(lang, {})
            
            for section in SECTIONS:
                text = lang_content.get(section, "")
                if not text:
                    continue
                    
                total += 1
                filename = f"{age}-{section}-{lang}.mp3"
                output_path = output_dir / filename
                
                if output_path.exists():
                    print(f"  ⏭️  {filename} (exists)")
                    skipped += 1
                    continue
                
                print(f"  🎤 {filename}")
                if generate_audio(text, output_path):
                    print(f"  ✅ Generated ({output_path.stat().st_size:,} bytes)")
                    generated += 1
                else:
                    print(f"  ❌ Failed")
                
                # Rate limiting
                time.sleep(1)
    
    print(f"\n📊 Summary: {generated} generated, {skipped} skipped, {total} total")
    return True

if __name__ == "__main__":
    lesson_id = sys.argv[1] if len(sys.argv) > 1 else "the-sun"
    generate_lesson_audio(lesson_id)
