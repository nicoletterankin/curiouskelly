#!/usr/bin/env python3
"""
Generate Kelly's voice for lipsync in iClone
Quick script to create lesson audio using ElevenLabs API
"""

import requests
import json
from pathlib import Path
import sys

def generate_kelly_lipsync(text, output_filename="kelly_lipsync.wav"):
    """Generate Kelly's voice audio for lipsync"""
    
    # ElevenLabs Configuration
    API_KEY = "sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568"
    VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly25 voice
    
    # API endpoint
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    
    # Headers
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }
    
    # Request data - optimized for natural speech
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.6,  # Slightly higher for consistent delivery
            "similarity_boost": 0.8,  # High fidelity to Kelly's voice
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    print(f"🎙️ Generating Kelly's voice...")
    print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"🔑 Voice ID: {VOICE_ID}")
    
    try:
        # Make API request
        print("\n⏳ Calling ElevenLabs API...")
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            print("✅ Audio generated successfully!")
            
            # Save as MP3 first
            mp3_file = Path("temp_kelly.mp3")
            with open(mp3_file, 'wb') as f:
                f.write(response.content)
            
            print(f"📊 Audio size: {len(response.content):,} bytes")
            
            # Convert to WAV for iClone compatibility
            try:
                import librosa
                import soundfile as sf
                
                print("\n🔄 Converting to WAV format for iClone...")
                
                # Load MP3 and convert to WAV
                audio, sr = librosa.load(str(mp3_file), sr=22050, mono=True)
                
                # Normalize audio
                audio = librosa.util.normalize(audio)
                
                # Determine output path
                if not output_filename.endswith('.wav'):
                    output_filename = output_filename + '.wav'
                
                # Save to iLearnStudio Kelly Audio folder
                output_path = Path(f"../iLearnStudio/projects/Kelly/Audio/{output_filename}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                sf.write(str(output_path), audio, 22050)
                
                print(f"✅ WAV file saved: {output_path.absolute()}")
                print(f"📊 Sample rate: 22,050 Hz (iClone compatible)")
                print(f"📊 Duration: {len(audio) / sr:.2f} seconds")
                print(f"📊 Format: Mono WAV")
                
                # Remove temporary MP3
                mp3_file.unlink()
                
                return str(output_path.absolute())
                
            except ImportError:
                print("⚠️ librosa not installed. Saving as MP3...")
                print("💡 For WAV conversion, install: pip install librosa soundfile")
                
                output_path = Path(f"../iLearnStudio/projects/Kelly/Audio/{output_filename.replace('.wav', '.mp3')}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"💾 MP3 saved: {output_path.absolute()}")
                return str(output_path.absolute())
                
        else:
            print(f"❌ API request failed!")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function with sample lesson texts"""
    
    print("=" * 70)
    print("🎬 KELLY LIPSYNC AUDIO GENERATOR")
    print("=" * 70)
    
    # Sample lesson texts for different ages
    lesson_texts = {
        "welcome": """Hello everyone! I'm Kelly, and I'm so excited to be here with you today. 
        Welcome to our lesson about the amazing world around us. 
        Are you ready to learn something incredible? Let's get started!""",
        
        "leaves_intro": """Have you ever wondered why leaves change color in the fall? 
        It's one of nature's most beautiful mysteries! 
        Today, we're going to discover the science behind those gorgeous autumn colors. 
        From bright reds to golden yellows, each color has a special story to tell.""",
        
        "science_lesson": """Today we're exploring photosynthesis and how plants use sunlight to make their food. 
        Inside every leaf are tiny structures called chloroplasts. 
        These amazing little factories capture sunlight and turn it into energy. 
        The green color we see comes from chlorophyll, the chemical that makes this magic happen. 
        Let's dive deeper into this fascinating process!""",
        
        "short_greeting": """Hi! I'm Kelly. Welcome to today's lesson. Let's learn something amazing together!""",
        
        "test_lipsync": """Testing, one, two, three. 
        This is Kelly speaking clearly for lipsync testing. 
        Notice how my mouth moves with each word. 
        Perfect synchronization is the key to realistic animation."""
    }
    
    print("\n📚 Available lesson texts:\n")
    for i, (name, text) in enumerate(lesson_texts.items(), 1):
        preview = text[:80].replace('\n', ' ').strip()
        print(f"{i}. {name}: {preview}...")
    
    print(f"\n{len(lesson_texts) + 1}. Custom text (enter your own)")
    
    # Get user choice
    try:
        choice = input("\n👉 Select option (1-6): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(lesson_texts):
            selected_key = list(lesson_texts.keys())[int(choice) - 1]
            text = lesson_texts[selected_key]
            filename = f"kelly_{selected_key}_lipsync.wav"
        elif choice == str(len(lesson_texts) + 1):
            text = input("\n📝 Enter your text: ").strip()
            filename = "kelly_custom_lipsync.wav"
        else:
            print("❌ Invalid choice. Using default welcome message.")
            text = lesson_texts["welcome"]
            filename = "kelly_welcome_lipsync.wav"
            
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
        return
    
    # Generate the audio
    print("\n" + "=" * 70)
    result = generate_kelly_lipsync(text, filename)
    
    if result:
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Kelly's voice is ready for lipsync!")
        print("=" * 70)
        print(f"\n📁 Audio file location: {result}")
        print("\n📋 NEXT STEPS:")
        print("1. Open iClone 8")
        print("2. Import your Kelly character")
        print("3. Import this audio file")
        print("4. Run AccuLips (select audio track, set to English, High quality)")
        print("5. Preview the lipsync")
        print("6. Render your video!")
        print("\n🎬 You're ready to make Kelly talk!")
    else:
        print("\n❌ Failed to generate audio. Please check the error above.")

if __name__ == "__main__":
    main()

































