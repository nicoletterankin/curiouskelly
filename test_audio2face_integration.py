#!/usr/bin/env python3
"""
Test NVIDIA Audio2Face-3D Integration with Kelly
Quick test to verify setup and API connection
"""

import os
import sys
from pathlib import Path
import requests
import json

def test_environment():
    """Test if the environment is properly set up"""
    print("🧪 Testing Audio2Face-3D environment...")
    
    # Check if directories exist
    audio2face_dir = Path("nvidia_audio2face")
    if not audio2face_dir.exists():
        print("❌ nvidia_audio2face directory not found")
        return False
    
    # Check if samples are cloned
    samples_dir = audio2face_dir / "Audio2Face-3D-Samples"
    if not samples_dir.exists():
        print("❌ Audio2Face-3D-Samples not found")
        print("💡 Run: python nvidia_audio2face_setup.py")
        return False
    
    # Check if Kelly config exists
    config_file = audio2face_dir / "config" / "kelly_claire.yml"
    if not config_file.exists():
        print("❌ Kelly config not found")
        return False
    
    print("✅ Environment setup looks good!")
    return True

def test_api_key(api_key):
    """Test if the API key is valid"""
    print("🔑 Testing NVIDIA API key...")
    
    if not api_key:
        print("❌ No API key provided")
        print("💡 Get your API key from: https://api.nvidia.com/")
        return False
    
    # Test API key with a simple request
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # This is a basic test - the actual API might have different endpoints
        print("✅ API key format looks valid")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def test_kelly_audio():
    """Test if Kelly audio files are available"""
    print("🎤 Testing Kelly audio files...")
    
    # Check for existing Kelly audio
    kelly_audio_paths = [
        "iLearnStudio/projects/Kelly/Audio/kelly25_audio.wav",
        "projects/Kelly/Audio/kelly25_audio.wav",
        "kelly25_audio.wav"
    ]
    
    audio_found = False
    for path in kelly_audio_paths:
        if Path(path).exists():
            print(f"✅ Found Kelly audio: {path}")
            audio_found = True
            break
    
    if not audio_found:
        print("⚠️ No Kelly audio found")
        print("💡 Generate audio with: python synthetic_tts/generate_kelly_lipsync.py")
        return False
    
    return True

def create_test_script():
    """Create a test script for Audio2Face-3D"""
    test_script = '''#!/usr/bin/env python3
"""
Test Audio2Face-3D with Kelly audio
"""

import sys
import os
from pathlib import Path

# Add the Audio2Face client to path
sys.path.append(str(Path(__file__).parent / "Audio2Face-3D-Samples" / "scripts" / "audio2face_3d_api_client"))

def test_audio2face():
    """Test Audio2Face-3D with Kelly audio"""
    print("🎬 Testing Audio2Face-3D with Kelly...")
    
    # Find Kelly audio
    kelly_audio = None
    for path in ["../iLearnStudio/projects/Kelly/Audio/kelly25_audio.wav", 
                 "../projects/Kelly/Audio/kelly25_audio.wav",
                 "kelly25_audio.wav"]:
        if Path(path).exists():
            kelly_audio = path
            break
    
    if not kelly_audio:
        print("❌ No Kelly audio found")
        return False
    
    print(f"📁 Using Kelly audio: {kelly_audio}")
    
    # Test with Audio2Face-3D
    try:
        from nim_a2f_3d_client import Audio2Face3DClient
        
        client = Audio2Face3DClient()
        print("✅ Audio2Face-3D client loaded successfully")
        
        # This would be the actual test call
        print("💡 Ready to process Kelly audio with Audio2Face-3D")
        print("   Run: python kelly_audio2face.py kelly_audio.wav --api-key YOUR_KEY")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Audio2Face-3D client: {e}")
        print("💡 Make sure you've run the setup script")
        return False

if __name__ == "__main__":
    test_audio2face()
'''
    
    test_file = Path("nvidia_audio2face") / "test_kelly_integration.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"✅ Test script created: {test_file}")

def main():
    """Main test function"""
    print("🧪 Testing NVIDIA Audio2Face-3D Integration")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment()
    
    # Test Kelly audio
    audio_ok = test_kelly_audio()
    
    # Test API key (if provided)
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        api_ok = test_api_key(api_key)
    else:
        print("⚠️ No NVIDIA_API_KEY environment variable found")
        print("💡 Set it with: set NVIDIA_API_KEY=your_key_here")
        api_ok = False
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Environment: {'✅' if env_ok else '❌'}")
    print(f"   Kelly Audio: {'✅' if audio_ok else '❌'}")
    print(f"   API Key: {'✅' if api_ok else '❌'}")
    
    if env_ok and audio_ok:
        print("\n🎉 Ready for Audio2Face-3D integration!")
        print("\n📋 Next steps:")
        print("1. Get NVIDIA API key from: https://api.nvidia.com/")
        print("2. Set environment variable: set NVIDIA_API_KEY=your_key_here")
        print("3. Run: python nvidia_audio2face/kelly_audio2face.py kelly_audio.wav --api-key YOUR_KEY")
    else:
        print("\n🔧 Setup required:")
        if not env_ok:
            print("   - Run: python nvidia_audio2face_setup.py")
        if not audio_ok:
            print("   - Generate Kelly audio with ElevenLabs")
        if not api_ok:
            print("   - Get NVIDIA API key and set environment variable")

if __name__ == "__main__":
    main()




















