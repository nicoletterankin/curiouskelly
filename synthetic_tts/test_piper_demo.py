#!/usr/bin/env python3
"""
Piper TTS Demo Script
Demonstrates Kelly and Ken voice synthesis using Piper TTS
"""

from piper_tts_wrapper import PiperTTS
import os

def demo_kelly_ken_voices():
    """Demo script for Kelly and Ken voices"""
    
    print("🎭 Piper TTS Demo - Kelly and Ken Voices")
    print("=" * 50)
    
    # Initialize TTS
    tts = PiperTTS()
    print(f"Available voices: {tts.list_voices()}")
    
    # Demo texts for Kelly and Ken
    kelly_texts = [
        "Hello! I'm Kelly, your friendly learning companion.",
        "Let's explore something exciting together today!",
        "I love helping students discover new things.",
        "What would you like to learn about?"
    ]
    
    ken_texts = [
        "Greetings! I'm Ken, your knowledgeable guide.",
        "I'm here to help you understand complex topics.",
        "Let's dive deep into the subject matter.",
        "Together, we can master any challenge."
    ]
    
    # Create output directory
    os.makedirs("demo_output", exist_ok=True)
    
    print("\n🎤 Generating Kelly's voice samples...")
    for i, text in enumerate(kelly_texts, 1):
        output_file = f"demo_output/kelly_sample_{i}.wav"
        try:
            result = tts.synthesize_kelly(text, output_file)
            print(f"✅ Kelly sample {i}: {result}")
        except Exception as e:
            print(f"❌ Kelly sample {i} failed: {e}")
    
    print("\n🎤 Generating Ken's voice samples...")
    for i, text in enumerate(ken_texts, 1):
        output_file = f"demo_output/ken_sample_{i}.wav"
        try:
            result = tts.synthesize_ken(text, output_file)
            print(f"✅ Ken sample {i}: {result}")
        except Exception as e:
            print(f"❌ Ken sample {i} failed: {e}")
    
    print("\n🎉 Demo complete! Check the 'demo_output' folder for audio files.")
    print("📁 Generated files:")
    
    # List generated files
    if os.path.exists("demo_output"):
        for file in os.listdir("demo_output"):
            if file.endswith('.wav'):
                file_path = os.path.join("demo_output", file)
                file_size = os.path.getsize(file_path)
                print(f"  - {file} ({file_size} bytes)")

def test_voice_parameters():
    """Test different voice parameters"""
    
    print("\n🔧 Testing Voice Parameters")
    print("=" * 30)
    
    tts = PiperTTS()
    
    # Test different speaker IDs (if supported)
    test_text = "This is a test of different voice parameters."
    
    for speaker_id in [0, 1, 2]:
        try:
            output_file = f"demo_output/voice_test_speaker_{speaker_id}.wav"
            result = tts.synthesize(test_text, output_file=output_file, speaker_id=speaker_id)
            print(f"✅ Speaker ID {speaker_id}: {result}")
        except Exception as e:
            print(f"❌ Speaker ID {speaker_id} failed: {e}")

if __name__ == "__main__":
    demo_kelly_ken_voices()
    test_voice_parameters()
    
    print("\n🎯 Next Steps:")
    print("1. Listen to the generated audio files")
    print("2. Customize voice parameters for Kelly and Ken")
    print("3. Integrate with your lesson system")
    print("4. Add emotional expression controls")






































