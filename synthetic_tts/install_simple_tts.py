#!/usr/bin/env python3
"""
Install Simple TTS Libraries
Installs easy-to-use TTS libraries for generating real speech audio
"""

import subprocess
import sys
import os

def install_tts_libraries():
    """Install TTS libraries"""
    
    print("🎵 Installing TTS Libraries")
    print("=" * 40)
    
    # List of TTS libraries to install
    libraries = [
        "pyttsx3",           # Cross-platform TTS
        "gTTS",              # Google Text-to-Speech
        "edge-tts",          # Microsoft Edge TTS
        "azure-cognitiveservices-speech",  # Azure Speech
    ]
    
    for lib in libraries:
        print(f"📦 Installing {lib}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"✅ {lib} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {lib}: {e}")
    
    print("\n🎉 TTS libraries installation complete!")

def test_tts_libraries():
    """Test installed TTS libraries"""
    
    print("\n🧪 Testing TTS Libraries...")
    
    # Test pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print(f"✅ pyttsx3 working - {len(voices)} voices available")
    except Exception as e:
        print(f"❌ pyttsx3 test failed: {e}")
    
    # Test gTTS
    try:
        from gtts import gTTS
        print("✅ gTTS working")
    except Exception as e:
        print(f"❌ gTTS test failed: {e}")
    
    # Test edge-tts
    try:
        import edge_tts
        print("✅ edge-tts working")
    except Exception as e:
        print(f"❌ edge-tts test failed: {e}")

if __name__ == "__main__":
    install_tts_libraries()
    test_tts_libraries()
