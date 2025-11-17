#!/usr/bin/env python3
"""
ElevenLabs → iClone TTS Integration
Automatically generate Kelly's voice from lesson DNA and prepare for AccuLips
"""

import requests
import json
from pathlib import Path
import sys

# ElevenLabs Configuration
API_KEY = "sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568"
VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly25 voice

def load_lesson_scripts():
    """Load lesson scripts from the lesson DNA file"""
    lesson_file = Path("lessons/leaves-change-color.json")
    
    if not lesson_file.exists():
        print(f"❌ Lesson file not found: {lesson_file}")
        return None
    
    with open(lesson_file, 'r', encoding='utf-8') as f:
        lesson_data = json.load(f)
    
    return lesson_data

def generate_audio_from_text(text, output_filename):
    """Generate audio using ElevenLabs API"""
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }
    
    # Voice settings optimized for educational content
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.6,  # Consistent delivery
            "similarity_boost": 0.8,  # High fidelity
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    print(f"\n🎤 Generating audio for: {output_filename}")
    print(f"📝 Text: {text[:100]}...")
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Save directly as WAV-compatible MP3
            output_path = Path(f"projects/Kelly/Audio/{output_filename}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Audio saved: {output_path}")
            print(f"📊 Size: {len(response.content):,} bytes")
            
            return str(output_path.absolute())
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def generate_all_age_variants():
    """Generate audio for all 6 age variants from lesson DNA"""
    
    print("=" * 80)
    print("🎬 ELEVENLABS → iCLONE TTS INTEGRATION")
    print("=" * 80)
    
    # Load lesson data
    lesson_data = load_lesson_scripts()
    if not lesson_data:
        return
    
    age_buckets = ["2-5", "6-12", "13-17", "18-35", "36-60", "61-102"]
    generated_files = []
    
    print(f"\n📚 Lesson: {lesson_data['title']}")
    print(f"🎯 Generating audio for {len(age_buckets)} age variants...\n")
    
    for age_bucket in age_buckets:
        variant = lesson_data["ageVariants"][age_bucket]
        script_text = variant["script"]
        
        output_filename = f"kelly_leaves_{age_bucket}.wav"
        
        audio_file = generate_audio_from_text(script_text, output_filename)
        
        if audio_file:
            generated_files.append({
                "age": age_bucket,
                "file": audio_file,
                "title": variant["title"]
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"✅ Generated {len(generated_files)}/{len(age_buckets)} audio files")
    print("=" * 80)
    
    print("\n📋 FILES READY FOR iCLONE:\n")
    for item in generated_files:
        print(f"  Age {item['age']:8} → {Path(item['file']).name}")
        print(f"             Title: {item['title']}")
        print()
    
    print("=" * 80)
    print("🎬 NEXT STEPS IN iCLONE:")
    print("=" * 80)
    print("""
1. Right-click on timeline audio track
2. Import Audio File → Select one of the generated WAV files
3. Select Kelly character in viewport
4. Animation → Facial Animation → AccuLips
5. Settings:
   - Audio: Select imported track
   - Language: English
   - Quality: High
6. Click "Apply" and wait 1-3 minutes
7. Press SPACEBAR to preview
8. Repeat for each age variant

💡 TIP: Generate lipsync for all 6 ages, save as separate animation files
    """)

def generate_custom_text(text, filename):
    """Generate audio from custom text"""
    
    print("=" * 80)
    print("🎤 CUSTOM TEXT → iCLONE AUDIO")
    print("=" * 80)
    
    output_filename = filename if filename.endswith('.wav') else filename + '.wav'
    
    audio_file = generate_audio_from_text(text, output_filename)
    
    if audio_file:
        print("\n✅ Audio ready for iClone!")
        print(f"📁 Location: {audio_file}")
        print("\n🎬 Import this file in iClone and run AccuLips")

def main():
    """Main menu"""
    
    print("\n" + "=" * 80)
    print("🎬 ELEVENLABS + iCLONE TTS INTEGRATION")
    print("=" * 80)
    print("""
Choose an option:

1. Generate audio for ALL 6 age variants (from lesson DNA)
2. Generate audio from CUSTOM text
3. Quick test (short greeting)

""")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        generate_all_age_variants()
    
    elif choice == "2":
        print("\n📝 Enter your text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        
        text = " ".join(lines)
        filename = input("\n💾 Output filename (without extension): ").strip()
        
        generate_custom_text(text, filename)
    
    elif choice == "3":
        test_text = "Hi! I'm Kelly. Welcome to today's lesson. Let's learn something amazing together!"
        generate_custom_text(test_text, "kelly_test_greeting")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()




















