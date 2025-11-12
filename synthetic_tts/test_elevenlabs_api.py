#!/usr/bin/env python3
"""
Test ElevenLabs API with Kelly voice
Generate a single WAV file to test the API connection
"""

import requests
import json
from pathlib import Path
import time

def test_elevenlabs_api():
    """Test ElevenLabs API with Kelly voice"""
    
    # Configuration
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
    
    # Test text
    test_text = "Hello! I'm Kelly, your friendly learning companion. This is a test of our ElevenLabs API integration."
    
    # Request data
    data = {
        "text": test_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    print(f"🎵 Testing ElevenLabs API with Kelly voice")
    print(f"📝 Text: {test_text}")
    print(f"🔑 Voice ID: {VOICE_ID}")
    print(f"🌐 URL: {url}")
    
    try:
        # Make API request
        print("\n⏳ Sending request to ElevenLabs API...")
        response = requests.post(url, json=data, headers=headers)
        
        # Check response
        if response.status_code == 200:
            print("✅ API request successful!")
            
            # Save audio file
            output_file = Path("kelly_test_audio.mp3")
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"💾 Audio saved to: {output_file.absolute()}")
            print(f"📊 File size: {len(response.content)} bytes")
            
            # Convert to WAV using librosa if available
            try:
                import librosa
                import soundfile as sf
                
                print("\n🔄 Converting MP3 to WAV...")
                
                # Load MP3 and convert to WAV
                audio, sr = librosa.load(str(output_file), sr=22050, mono=True)
                
                # Normalize audio
                audio = librosa.util.normalize(audio)
                
                # Save as WAV
                wav_file = Path("kelly_test_audio.wav")
                sf.write(str(wav_file), audio, 22050)
                
                print(f"✅ WAV file created: {wav_file.absolute()}")
                print(f"📊 Sample rate: 22050 Hz")
                print(f"📊 Duration: {len(audio) / sr:.2f} seconds")
                
                # Remove MP3 file
                output_file.unlink()
                print("🗑️ Temporary MP3 file removed")
                
            except ImportError:
                print("⚠️ librosa not available, keeping MP3 format")
                
        else:
            print(f"❌ API request failed!")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_elevenlabs_api()




































