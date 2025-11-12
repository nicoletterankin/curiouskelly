#!/usr/bin/env python3
"""
Generate Audio Files for Lesson Prototype
Creates audio files for each age variant using ElevenLabs API
"""

import json
import os
import requests
from pathlib import Path
import time

# ElevenLabs Configuration
API_KEY = "sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568"
VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly25 voice
BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# File paths
LESSON_FILE = "../lessons/leaves-change-color.json"
OUTPUT_DIR = Path("videos/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_speech(text, output_path, voice_id=VOICE_ID, api_key=API_KEY):
    """Generate speech using ElevenLabs API"""
    
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
    
    print(f"🎤 Generating audio for: {output_path.name}")
    print(f"   Text length: {len(text)} characters")
    
    response = requests.post(
        f"{BASE_URL}/{voice_id}",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved: {output_path}")
        return True
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return False

def load_lesson_data():
    """Load lesson DNA from JSON file"""
    lesson_path = Path(LESSON_FILE)
    
    if not lesson_path.exists():
        print(f"❌ Lesson file not found: {lesson_path}")
        return None
    
    with open(lesson_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_all_audio():
    """Generate audio files for all 6 age variants"""
    
    print("=" * 60)
    print("Audio Generation for Lesson Prototype")
    print("=" * 60)
    
    # Load lesson data
    lesson_data = load_lesson_data()
    if not lesson_data:
        return False
    
    age_buckets = ["2-5", "6-12", "13-17", "18-35", "36-60", "61-102"]
    audio_metadata = {}
    successful = 0
    
    # Generate audio for each age bucket
    for age_bucket in age_buckets:
        variant = lesson_data["ageVariants"][age_bucket]
        script_text = variant["script"]
        
        # Create output file path
        output_file = OUTPUT_DIR / f"kelly_leaves_{age_bucket}.mp3"
        
        # Generate speech
        if generate_speech(script_text, output_file):
            successful += 1
            
            # Store metadata
            audio_metadata[age_bucket] = {
                "file": output_file.name,
                "title": variant["title"],
                "script_length": len(script_text),
                "objectives": variant["objectives"]
            }
        
        # Rate limiting - be nice to API
        time.sleep(1)
    
    # Save metadata
    metadata_file = OUTPUT_DIR / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(audio_metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Complete: {successful}/6 audio files generated")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print(f"📄 Metadata saved: {metadata_file}")
    print("=" * 60)
    
    return successful == 6

if __name__ == "__main__":
    generate_all_audio()

