#!/usr/bin/env python3
"""
Kelly Audio2Face-3D Test Script
Test the integration and setup
"""

import os
import sys
from pathlib import Path
import subprocess

def test_environment():
    """Test if environment is properly set up"""
    print("🧪 Testing Kelly Audio2Face-3D environment...")
    
    # Check directories
    kelly_dir = Path("kelly_audio2face")
    if not kelly_dir.exists():
        print("❌ Kelly Audio2Face directory not found")
        return False
    
    # Check config
    config_file = kelly_dir / "config" / "kelly_config.yml"
    if not config_file.exists():
        print("❌ Kelly config not found")
        return False
    
    # Check client script
    client_script = kelly_dir / "scripts" / "kelly_audio2face_client.py"
    if not client_script.exists():
        print("❌ Kelly client script not found")
        return False
    
    print("✅ Environment setup looks good!")
    return True

def test_dependencies():
    """Test if dependencies are installed"""
    print("🔍 Testing dependencies...")
    
    try:
        import numpy
        import scipy
        import grpcio
        import protobuf
        import yaml
        import pandas
        print("✅ Core dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r kelly_audio2face/requirements.txt")
        return False

def test_audio2face_modules():
    """Test Audio2Face-3D module imports"""
    print("🔍 Testing Audio2Face-3D modules...")
    
    try:
        sys.path.append("Audio2Face-3D-Samples/scripts/audio2face_3d_api_client")
        import a2f_3d.client.auth
        import a2f_3d.client.service
        from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
        print("✅ Audio2Face-3D modules available")
        return True
    except ImportError as e:
        print(f"❌ Audio2Face-3D modules not available: {e}")
        print("💡 Install NVIDIA ACE wheel:")
        print("   pip install Audio2Face-3D-Samples/proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl")
        return False

def test_kelly_audio():
    """Test Kelly audio files"""
    print("🎤 Testing Kelly audio files...")
    
    audio_paths = [
        "projects/Kelly/Audio/kelly25_audio.wav",
        "kelly25_audio.wav"
    ]
    
    for path in audio_paths:
        if Path(path).exists():
            print(f"✅ Found Kelly audio: {path}")
            return True
    
    print("⚠️ No Kelly audio found")
    print("💡 Generate audio with ElevenLabs or place in projects/Kelly/Audio/")
    return False

def test_api_credentials():
    """Test API credentials"""
    print("🔑 Testing API credentials...")
    
    api_key = os.getenv("NVIDIA_API_KEY")
    function_id = os.getenv("AUDIO2FACE_FUNCTION_ID")
    
    if not api_key:
        print("❌ NVIDIA_API_KEY not set")
        print("💡 Get API key from: https://api.nvidia.com/")
        return False
    
    if not function_id:
        print("❌ AUDIO2FACE_FUNCTION_ID not set")
        print("💡 Get Function ID from NVIDIA Cloud Functions")
        return False
    
    print("✅ API credentials configured")
    return True

def main():
    """Main test function"""
    print("🧪 Kelly Audio2Face-3D Integration Test")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Dependencies", test_dependencies),
        ("Audio2Face Modules", test_audio2face_modules),
        ("Kelly Audio", test_kelly_audio),
        ("API Credentials", test_api_credentials)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Kelly Audio2Face-3D is ready!")
        print("\n📋 Next steps:")
        print("1. Run Kelly client: python kelly_audio2face/scripts/kelly_audio2face_client.py")
        print("2. Check workflow guide: kelly_audio2face/KELLY_WORKFLOW_GUIDE.md")
    else:
        print("\n🔧 Some tests failed. Please fix the issues above.")
        print("\n📚 See kelly_audio2face/KELLY_WORKFLOW_GUIDE.md for setup instructions")

if __name__ == "__main__":
    main()
