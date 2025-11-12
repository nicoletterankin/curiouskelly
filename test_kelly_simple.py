#!/usr/bin/env python3
"""
Simple Kelly Audio2Face-3D Test
Test basic functionality without API credentials
"""

import sys
from pathlib import Path

def test_imports():
    """Test if we can import the required modules"""
    print("🔍 Testing imports...")
    
    try:
        import numpy
        print("✅ numpy imported")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import scipy
        print("✅ scipy imported")
    except ImportError as e:
        print(f"❌ scipy import failed: {e}")
        return False
    
    try:
        import grpc
        print("✅ grpc imported")
    except ImportError as e:
        print(f"❌ grpc import failed: {e}")
        return False
    
    try:
        import yaml
        print("✅ yaml imported")
    except ImportError as e:
        print(f"❌ yaml import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ pandas imported")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    return True

def test_audio2face_modules():
    """Test Audio2Face-3D specific modules"""
    print("🔍 Testing Audio2Face-3D modules...")
    
    # Add the Audio2Face-3D client to path
    sys.path.append("Audio2Face-3D-Samples/scripts/audio2face_3d_api_client")
    
    try:
        import a2f_3d.client.auth
        print("✅ a2f_3d.client.auth imported")
    except ImportError as e:
        print(f"❌ a2f_3d.client.auth import failed: {e}")
        return False
    
    try:
        import a2f_3d.client.service
        print("✅ a2f_3d.client.service imported")
    except ImportError as e:
        print(f"❌ a2f_3d.client.service import failed: {e}")
        return False
    
    try:
        from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
        print("✅ A2FControllerServiceStub imported")
    except ImportError as e:
        print(f"❌ A2FControllerServiceStub import failed: {e}")
        return False
    
    return True

def test_kelly_files():
    """Test Kelly-specific files"""
    print("🔍 Testing Kelly files...")
    
    files_to_check = [
        "kelly_audio2face/config/kelly_config.yml",
        "kelly_audio2face/scripts/kelly_audio2face_client.py",
        "kelly_audio2face/KELLY_WORKFLOW_GUIDE.md",
        "projects/Kelly/Audio/kelly25_audio.wav"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Main test function"""
    print("🧪 Kelly Audio2Face-3D Simple Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Audio2Face Modules", test_audio2face_modules),
        ("Kelly Files", test_kelly_files)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All basic tests passed!")
        print("\n📋 Next steps:")
        print("1. Get NVIDIA API key from: https://api.nvidia.com/")
        print("2. Get Audio2Face-3D Function ID from NVIDIA Cloud Functions")
        print("3. Set environment variables:")
        print("   set NVIDIA_API_KEY=your_key_here")
        print("   set AUDIO2FACE_FUNCTION_ID=your_function_id_here")
        print("4. Run Kelly client:")
        print("   python kelly_audio2face/scripts/kelly_audio2face_client.py projects/Kelly/Audio/kelly25_audio.wav --apikey YOUR_KEY --function-id YOUR_ID")
    else:
        print("\n🔧 Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
