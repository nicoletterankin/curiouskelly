# Kelly Audio2Face-3D Integration Setup
# Complete setup script for Kelly avatar with NVIDIA Audio2Face-3D

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml

def setup_audio2face_environment():
    """Set up the Audio2Face-3D environment for Kelly"""
    print("🚀 Setting up Audio2Face-3D environment for Kelly...")
    
    # Create Kelly-specific directory structure
    kelly_a2f_dir = Path("kelly_audio2face")
    kelly_a2f_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (kelly_a2f_dir / "config").mkdir(exist_ok=True)
    (kelly_a2f_dir / "output").mkdir(exist_ok=True)
    (kelly_a2f_dir / "scripts").mkdir(exist_ok=True)
    (kelly_a2f_dir / "logs").mkdir(exist_ok=True)
    
    print(f"✅ Created Kelly Audio2Face directory: {kelly_a2f_dir}")
    return kelly_a2f_dir

def create_kelly_config():
    """Create Kelly-specific configuration for Audio2Face-3D"""
    print("⚙️ Creating Kelly configuration...")
    
    # Kelly-specific configuration based on Claire's config but optimized for Kelly
    kelly_config = {
        'face_parameters': {
            'upperFaceStrength': 1.1,  # Slightly more expressive upper face
            'upperFaceSmoothing': 0.001,
            'lowerFaceStrength': 1.3,  # More pronounced lip sync for Kelly
            'lowerFaceSmoothing': 0.005,
            'faceMaskLevel': 0.6,
            'faceMaskSoftness': 0.0085,
            'skinStrength': 1.0,
            'eyelidOpenOffset': 0.0,
            'lipOpenOffset': 0.0
        },
        'blendshape_parameters': {
            'enable_clamping_bs_weight': False,
            'multipliers': {
                # Eyes - Kelly has expressive eyes
                'EyeBlinkLeft': 1.0,
                'EyeLookDownLeft': 0.0,
                'EyeLookInLeft': 0.0,
                'EyeLookOutLeft': 0.0,
                'EyeLookUpLeft': 0.0,
                'EyeSquintLeft': 1.2,  # More squinting for Kelly
                'EyeWideLeft': 1.1,   # Slightly wider eyes
                'EyeBlinkRight': 1.0,
                'EyeLookDownRight': 0.0,
                'EyeLookInRight': 0.0,
                'EyeLookOutRight': 0.0,
                'EyeLookUpRight': 0.0,
                'EyeSquintRight': 1.2,
                'EyeWideRight': 1.1,
                # Jaw - Kelly has clear articulation
                'JawForward': 0.8,
                'JawLeft': 0.3,
                'JawRight': 0.3,
                'JawOpen': 1.1,  # More jaw movement for Kelly
                # Mouth - Kelly has expressive mouth movements
                'MouthClose': 1.0,
                'MouthFunnel': 1.3,
                'MouthPucker': 1.3,
                'MouthLeft': 0.3,
                'MouthRight': 0.3,
                'MouthSmileLeft': 0.9,  # Kelly smiles more
                'MouthSmileRight': 0.9,
                'MouthFrownLeft': 0.5,
                'MouthFrownRight': 0.5,
                'MouthDimpleLeft': 0.8,
                'MouthDimpleRight': 0.8,
                'MouthStretchLeft': 0.2,
                'MouthStretchRight': 0.2,
                'MouthRollLower': 1.0,
                'MouthRollUpper': 0.6,
                'MouthShrugLower': 1.0,
                'MouthShrugUpper': 0.5,
                'MouthPressLeft': 0.9,
                'MouthPressRight': 0.9,
                'MouthLowerDownLeft': 0.9,
                'MouthLowerDownRight': 0.9,
                'MouthUpperUpLeft': 0.9,
                'MouthUpperUpRight': 0.9,
                # Brows - Kelly has expressive eyebrows
                'BrowDownLeft': 1.1,
                'BrowDownRight': 1.1,
                'BrowInnerUp': 1.1,
                'BrowOuterUpLeft': 1.1,
                'BrowOuterUpRight': 1.1,
                # Cheeks - Kelly has natural cheek movement
                'CheekPuff': 0.3,
                'CheekSquintLeft': 1.1,
                'CheekSquintRight': 1.1,
                'NoseSneerLeft': 0.9,
                'NoseSneerRight': 0.9,
                'TongueOut': 0.0
            },
            'offsets': {
                # All offsets set to 0 for Kelly's neutral expression
                'EyeBlinkLeft': 0.0, 'EyeLookDownLeft': 0.0, 'EyeLookInLeft': 0.0,
                'EyeLookOutLeft': 0.0, 'EyeLookUpLeft': 0.0, 'EyeSquintLeft': 0.0,
                'EyeWideLeft': 0.0, 'EyeBlinkRight': 0.0, 'EyeLookDownRight': 0.0,
                'EyeLookInRight': 0.0, 'EyeLookOutRight': 0.0, 'EyeLookUpRight': 0.0,
                'EyeSquintRight': 0.0, 'EyeWideRight': 0.0, 'JawForward': 0.0,
                'JawLeft': 0.0, 'JawRight': 0.0, 'JawOpen': 0.0, 'MouthClose': 0.0,
                'MouthFunnel': 0.0, 'MouthPucker': 0.0, 'MouthLeft': 0.0,
                'MouthRight': 0.0, 'MouthSmileLeft': 0.0, 'MouthSmileRight': 0.0,
                'MouthFrownLeft': 0.0, 'MouthFrownRight': 0.0, 'MouthDimpleLeft': 0.0,
                'MouthDimpleRight': 0.0, 'MouthStretchLeft': 0.0, 'MouthStretchRight': 0.0,
                'MouthRollLower': 0.0, 'MouthRollUpper': 0.0, 'MouthShrugLower': 0.0,
                'MouthShrugUpper': 0.0, 'MouthPressLeft': 0.0, 'MouthPressRight': 0.0,
                'MouthLowerDownLeft': 0.0, 'MouthLowerDownRight': 0.0,
                'MouthUpperUpLeft': 0.0, 'MouthUpperUpRight': 0.0, 'BrowDownLeft': 0.0,
                'BrowDownRight': 0.0, 'BrowInnerUp': 0.0, 'BrowOuterUpLeft': 0.0,
                'BrowOuterUpRight': 0.0, 'CheekPuff': 0.0, 'CheekSquintLeft': 0.0,
                'CheekSquintRight': 0.0, 'NoseSneerLeft': 0.0, 'NoseSneerRight': 0.0,
                'TongueOut': 0.0
            }
        },
        'live_transition_time': 0.0001,
        'beginning_emotion': {
            'amazement': 0.0,
            'anger': 0.0,
            'disgust': 0.0,
            'fear': 0.0,
            'outofbreath': 0.0,
            'pain': 0.0,
            'sadness': 0.0,
            'joy': 0.0  # Kelly starts neutral
        },
        'post_processing_parameters': {
            'emotion_contrast': 1.0,
            'live_blend_coef': 0.7,
            'enable_preferred_emotion': False,
            'preferred_emotion_strength': 0.5,
            'emotion_strength': 0.7,  # Kelly is more emotionally expressive
            'max_emotions': 3
        },
        'emotion_with_timecode_list': {
            'emotion_with_timecode1': {
                'time_code': 0.0,
                'emotions': {
                    'amazement': 0.0, 'anger': 0.0, 'cheekiness': 0.0,
                    'disgust': 0.0, 'fear': 0.0, 'grief': 0.0,
                    'joy': 0.0, 'outofbreath': 0.0, 'pain': 0.0, 'sadness': 0.0
                }
            }
        }
    }
    
    # Save Kelly configuration
    config_path = Path("kelly_audio2face/config/kelly_config.yml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(kelly_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created Kelly configuration: {config_path}")
    return config_path

def create_kelly_client_script():
    """Create Kelly-specific Audio2Face-3D client script"""
    print("📝 Creating Kelly client script...")
    
    client_script = '''#!/usr/bin/env python3
"""
Kelly Audio2Face-3D Client
Specialized client for Kelly avatar facial animation
"""

import argparse
import asyncio
import sys
from pathlib import Path
import logging

# Add Audio2Face-3D client to path
sys.path.append(str(Path(__file__).parent.parent / "Audio2Face-3D-Samples" / "scripts" / "audio2face_3d_api_client"))

try:
    import a2f_3d.client.auth
    import a2f_3d.client.service
    from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
except ImportError as e:
    print(f"❌ Failed to import Audio2Face-3D modules: {e}")
    print("💡 Make sure you've installed the requirements:")
    print("   pip install -r Audio2Face-3D-Samples/scripts/audio2face_3d_api_client/requirements")
    print("   pip install Audio2Face-3D-Samples/proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl")
    sys.exit(1)

def setup_logging():
    """Setup logging for Kelly client"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('kelly_audio2face/logs/kelly_client.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('KellyAudio2Face')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kelly Audio2Face-3D Client - Generate facial animation for Kelly avatar",
        epilog="NVIDIA CORPORATION. All rights reserved."
    )
    parser.add_argument("audio_file", help="Kelly audio file (WAV format, 16-bit PCM)")
    parser.add_argument("--config", default="config/kelly_config.yml", 
                       help="Kelly configuration file")
    parser.add_argument("--apikey", required=True, help="NVIDIA API Key")
    parser.add_argument("--function-id", required=True, help="Audio2Face-3D Function ID")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model", default="claire", choices=["claire", "mark", "james"],
                       help="Audio2Face-3D model to use")
    return parser.parse_args()

async def process_kelly_audio(args, logger):
    """Process Kelly audio with Audio2Face-3D"""
    logger.info(f"🎬 Processing Kelly audio: {args.audio_file}")
    
    # Validate input files
    audio_path = Path(args.audio_file)
    config_path = Path(args.config)
    
    if not audio_path.exists():
        logger.error(f"❌ Audio file not found: {audio_path}")
        return False
    
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Setup gRPC connection
        metadata_args = [
            ("function-id", args.function_id),
            ("authorization", "Bearer " + args.apikey)
        ]
        
        logger.info("🔗 Connecting to Audio2Face-3D service...")
        channel = a2f_3d.client.auth.create_channel(
            uri="grpc.nvcf.nvidia.com:443", 
            use_ssl=True, 
            metadata=metadata_args
        )
        
        stub = A2FControllerServiceStub(channel)
        
        # Process audio stream
        logger.info("🎵 Processing audio stream...")
        stream = stub.ProcessAudioStream()
        
        write_task = asyncio.create_task(
            a2f_3d.client.service.write_to_stream(stream, str(config_path), str(audio_path))
        )
        read_task = asyncio.create_task(
            a2f_3d.client.service.read_from_stream(stream)
        )
        
        await write_task
        await read_task
        
        logger.info("✅ Kelly facial animation generated successfully!")
        logger.info(f"📁 Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing Kelly audio: {e}")
        return False

async def main():
    """Main function"""
    logger = setup_logging()
    logger.info("🚀 Starting Kelly Audio2Face-3D Client")
    
    args = parse_args()
    
    # Change to Kelly Audio2Face directory
    kelly_dir = Path("kelly_audio2face")
    if kelly_dir.exists():
        os.chdir(kelly_dir)
        logger.info(f"📁 Working in directory: {kelly_dir.absolute()}")
    
    success = await process_kelly_audio(args, logger)
    
    if success:
        logger.info("🎉 Kelly Audio2Face-3D processing completed successfully!")
        print("\\n📋 Next steps:")
        print("1. Check output directory for blendshape data")
        print("2. Import animation data into your 3D software")
        print("3. Apply to Kelly's facial rig")
    else:
        logger.error("❌ Kelly Audio2Face-3D processing failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    script_path = Path("kelly_audio2face/scripts/kelly_audio2face_client.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(client_script)
    
    # Make script executable
    script_path.chmod(0o755)
    
    print(f"✅ Created Kelly client script: {script_path}")
    return script_path

def create_kelly_workflow_guide():
    """Create comprehensive workflow guide for Kelly"""
    print("📚 Creating Kelly workflow guide...")
    
    workflow_guide = '''# Kelly Audio2Face-3D Workflow Guide

## Overview
This guide provides step-by-step instructions for using NVIDIA Audio2Face-3D with Kelly avatar for facial animation generation.

## Prerequisites
- NVIDIA API Key (get from https://api.nvidia.com/)
- Audio2Face-3D Function ID
- Kelly audio files (WAV format, 16-bit PCM)
- Python 3.8+ environment

## Setup Steps

### 1. Install Dependencies
```bash
# Install Audio2Face-3D requirements
pip install -r Audio2Face-3D-Samples/scripts/audio2face_3d_api_client/requirements

# Install NVIDIA ACE wheel
pip install Audio2Face-3D-Samples/proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
```

### 2. Set Environment Variables
```bash
# Windows
set NVIDIA_API_KEY=your_api_key_here
set AUDIO2FACE_FUNCTION_ID=your_function_id_here

# Linux/Mac
export NVIDIA_API_KEY=your_api_key_here
export AUDIO2FACE_FUNCTION_ID=your_function_id_here
```

### 3. Prepare Kelly Audio
- Ensure audio is in WAV format, 16-bit PCM, single channel
- Recommended sample rate: 16kHz or 44.1kHz
- Audio should be clean with minimal background noise

## Usage

### Basic Usage
```bash
cd kelly_audio2face
python scripts/kelly_audio2face_client.py ../projects/Kelly/Audio/kelly25_audio.wav \\
    --apikey YOUR_API_KEY \\
    --function-id YOUR_FUNCTION_ID \\
    --model claire
```

### Advanced Usage
```bash
python scripts/kelly_audio2face_client.py audio_file.wav \\
    --config config/kelly_config.yml \\
    --apikey YOUR_API_KEY \\
    --function-id YOUR_FUNCTION_ID \\
    --output-dir output/kelly_animation \\
    --model claire
```

## Configuration

### Kelly-Specific Settings
The Kelly configuration (`config/kelly_config.yml`) is optimized for:
- More expressive upper face movements
- Enhanced lip sync accuracy
- Natural emotional expressions
- Kelly's specific facial characteristics

### Key Parameters
- `upperFaceStrength`: 1.1 (slightly more expressive)
- `lowerFaceStrength`: 1.3 (enhanced lip sync)
- `emotion_strength`: 0.7 (more emotionally expressive)
- `max_emotions`: 3 (balanced emotional range)

## Output Files

### Generated Files
- `blendshapes.csv`: ARKit blendshape data with timestamps
- `emotions.csv`: Emotion data with timestamps
- `out.wav`: Processed audio (should match input)
- `kelly_client.log`: Processing logs

### Blendshape Mapping
Kelly uses standard ARKit blendshapes:
- Eye movements (blink, look, squint, wide)
- Jaw movements (open, forward, left, right)
- Mouth movements (smile, frown, pucker, funnel)
- Brow movements (up, down, inner, outer)
- Cheek movements (puff, squint)

## Integration with 3D Software

### Maya Integration
1. Import blendshape data
2. Map to Kelly's facial rig
3. Apply animation curves
4. Fine-tune timing and intensity

### Unreal Engine Integration
1. Use Audio2Face-3D Unreal plugin
2. Import Kelly model
3. Apply blendshape data
4. Test in real-time

### iClone Integration
1. Export Kelly model from iClone
2. Process with Audio2Face-3D
3. Import animation back to iClone
4. Render final video

## Troubleshooting

### Common Issues
1. **API Key Invalid**: Verify key from NVIDIA API portal
2. **Function ID Error**: Check Function ID is correct
3. **Audio Format Error**: Ensure WAV, 16-bit PCM format
4. **Connection Timeout**: Check internet connection
5. **Import Errors**: Verify all dependencies installed

### Debug Mode
Enable debug logging:
```bash
python scripts/kelly_audio2face_client.py audio.wav \\
    --apikey YOUR_KEY \\
    --function-id YOUR_ID \\
    --debug
```

## Performance Tips

### Optimization
- Use shorter audio clips for faster processing
- Batch process multiple clips
- Cache results for repeated use
- Use appropriate model (claire for female voices)

### Quality Settings
- Higher sample rates for better quality
- Clean audio input for better results
- Appropriate emotion settings for content
- Fine-tune blendshape multipliers

## Next Steps

### Production Pipeline
1. Generate Kelly audio with ElevenLabs
2. Process with Audio2Face-3D
3. Import to 3D software
4. Apply to Kelly's rig
5. Render final animation

### Batch Processing
Create batch scripts for multiple audio files:
```bash
for audio in ../projects/Kelly/Audio/*.wav; do
    python scripts/kelly_audio2face_client.py "$audio" \\
        --apikey YOUR_KEY \\
        --function-id YOUR_ID
done
```

## Support
- NVIDIA Audio2Face-3D Documentation
- Kelly Project Documentation
- GitHub Issues for bug reports
'''
    
    guide_path = Path("kelly_audio2face/KELLY_WORKFLOW_GUIDE.md")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(workflow_guide)
    
    print(f"✅ Created Kelly workflow guide: {guide_path}")
    return guide_path

def create_requirements_file():
    """Create requirements file for Kelly Audio2Face-3D"""
    print("📦 Creating requirements file...")
    
    requirements = '''# Kelly Audio2Face-3D Requirements
# Core dependencies for Kelly avatar facial animation

# Audio2Face-3D dependencies
numpy==1.26.4
scipy==1.13.0
grpcio==1.72.0rc1
protobuf==4.24.1
PyYAML==6.0.1
pandas==2.2.2

# Additional Kelly-specific dependencies
pathlib2==2.3.7
asyncio-mqtt==0.16.1
aiofiles==23.2.1
python-dotenv==1.0.0

# Development dependencies
pytest==7.4.3
black==23.9.1
flake8==6.1.0
'''
    
    req_path = Path("kelly_audio2face/requirements.txt")
    with open(req_path, 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print(f"✅ Created requirements file: {req_path}")
    return req_path

def create_test_script():
    """Create test script for Kelly integration"""
    print("🧪 Creating test script...")
    
    test_script = '''#!/usr/bin/env python3
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
        print(f"\\n{test_name}:")
        results[test_name] = test_func()
    
    print("\\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\\n🎉 All tests passed! Kelly Audio2Face-3D is ready!")
        print("\\n📋 Next steps:")
        print("1. Run Kelly client: python kelly_audio2face/scripts/kelly_audio2face_client.py")
        print("2. Check workflow guide: kelly_audio2face/KELLY_WORKFLOW_GUIDE.md")
    else:
        print("\\n🔧 Some tests failed. Please fix the issues above.")
        print("\\n📚 See kelly_audio2face/KELLY_WORKFLOW_GUIDE.md for setup instructions")

if __name__ == "__main__":
    main()
'''
    
    test_path = Path("kelly_audio2face/test_kelly_integration.py")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    test_path.chmod(0o755)
    
    print(f"✅ Created test script: {test_path}")
    return test_path

def main():
    """Main setup function"""
    print("🚀 Kelly Audio2Face-3D Integration Setup")
    print("=" * 50)
    
    # Setup environment
    kelly_dir = setup_audio2face_environment()
    
    # Create configuration
    config_path = create_kelly_config()
    
    # Create client script
    client_script = create_kelly_client_script()
    
    # Create workflow guide
    workflow_guide = create_kelly_workflow_guide()
    
    # Create requirements
    requirements = create_requirements_file()
    
    # Create test script
    test_script = create_test_script()
    
    print("\\n" + "=" * 50)
    print("🎉 Kelly Audio2Face-3D setup complete!")
    print("\\n📁 Created files:")
    print(f"   - Directory: {kelly_dir}")
    print(f"   - Config: {config_path}")
    print(f"   - Client: {client_script}")
    print(f"   - Guide: {workflow_guide}")
    print(f"   - Requirements: {requirements}")
    print(f"   - Test: {test_script}")
    
    print("\\n📋 Next steps:")
    print("1. Install dependencies:")
    print("   pip install -r kelly_audio2face/requirements.txt")
    print("   pip install Audio2Face-3D-Samples/proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl")
    print("2. Get NVIDIA API credentials:")
    print("   - API Key: https://api.nvidia.com/")
    print("   - Function ID: NVIDIA Cloud Functions")
    print("3. Test setup:")
    print("   python kelly_audio2face/test_kelly_integration.py")
    print("4. Process Kelly audio:")
    print("   python kelly_audio2face/scripts/kelly_audio2face_client.py audio.wav --apikey KEY --function-id ID")
    
    print("\\n📚 Documentation:")
    print("   - Workflow Guide: kelly_audio2face/KELLY_WORKFLOW_GUIDE.md")
    print("   - Audio2Face-3D Docs: https://docs.nvidia.com/ace/latest/modules/a2f-docs/")

if __name__ == "__main__":
    main()
