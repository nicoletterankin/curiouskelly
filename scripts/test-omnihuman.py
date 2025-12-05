"""
Test script for ElevenLabs Omnihuman 1.5 lip-sync video generation
Uses the official Python SDK

Usage:
    pip install elevenlabs
    python scripts/test-omnihuman.py
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env.local and .env
from dotenv import load_dotenv
load_dotenv('.env.local')
load_dotenv('.env')

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0'

# Test text for Kelly to speak
TEST_TEXT = "Hello! I'm Kelly, and I'm so excited to learn with you today! Let's discover something amazing together."

# Output directory
OUTPUT_DIR = Path('test-output')


def test_with_sdk():
    """Test using the official ElevenLabs Python SDK"""
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import play, save
    except ImportError:
        print("❌ ElevenLabs SDK not installed. Run: pip install elevenlabs")
        return False
    
    print("\n🔌 Using ElevenLabs Python SDK...")
    
    # Initialize client
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    
    # Check what attributes are available
    print("\n📋 Available SDK methods:")
    for attr in dir(client):
        if not attr.startswith('_'):
            print(f"   - {attr}")
    
    # Check if image_to_video is available
    if hasattr(client, 'image_to_video'):
        print("\n✅ client.image_to_video is available!")
        print("   Methods:", dir(client.image_to_video))
    else:
        print("\n❌ client.image_to_video NOT available in SDK")
        return False
    
    # Load image
    image_path = Path('public/kelly/poses/kelly_welcome.png')
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    print(f"✅ Image loaded: {len(image_data)} bytes")
    
    # First generate TTS audio
    print("\n📢 Generating TTS audio...")
    try:
        audio = client.text_to_speech.convert(
            voice_id=KELLY_VOICE_ID,
            text=TEST_TEXT,
            model_id="eleven_multilingual_v2"
        )
        
        # Save audio
        audio_path = OUTPUT_DIR / 'kelly-audio.mp3'
        with open(audio_path, 'wb') as f:
            for chunk in audio:
                f.write(chunk)
        
        print(f"✅ Audio saved: {audio_path}")
        
        # Read back for video generation
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False
    
    # Try to generate video
    print("\n🎬 Attempting video generation...")
    
    # Try different method names
    methods_to_try = ['create', 'generate', 'convert']
    
    for method_name in methods_to_try:
        if hasattr(client.image_to_video, method_name):
            print(f"\n🔄 Trying client.image_to_video.{method_name}()...")
            method = getattr(client.image_to_video, method_name)
            
            try:
                result = method(
                    image=image_data,
                    audio=audio_data
                )
                
                print(f"✅ Success! Result type: {type(result)}")
                
                # Save the video
                video_path = OUTPUT_DIR / 'kelly-talking.mp4'
                
                if hasattr(result, 'content'):
                    with open(video_path, 'wb') as f:
                        f.write(result.content)
                elif hasattr(result, 'read'):
                    with open(video_path, 'wb') as f:
                        f.write(result.read())
                elif isinstance(result, bytes):
                    with open(video_path, 'wb') as f:
                        f.write(result)
                else:
                    # Might be a generator
                    with open(video_path, 'wb') as f:
                        for chunk in result:
                            f.write(chunk)
                
                print(f"✅ Video saved: {video_path}")
                return True
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ⚠️ Method {method_name} not available")
    
    return False


def test_direct_api():
    """Test using direct API calls with requests"""
    import requests
    
    print("\n🌐 Testing direct API calls...")
    
    # Test user info first
    headers = {'xi-api-key': ELEVENLABS_API_KEY}
    
    resp = requests.get('https://api.elevenlabs.io/v1/user', headers=headers)
    if resp.ok:
        user = resp.json()
        print(f"✅ User: {user.get('subscription', {}).get('tier')}")
        print(f"   Characters: {user.get('subscription', {}).get('character_count')}/{user.get('subscription', {}).get('character_limit')}")
    
    # Check models
    resp = requests.get('https://api.elevenlabs.io/v1/models', headers=headers)
    if resp.ok:
        models = resp.json()
        print(f"\n📋 Available models ({len(models)}):")
        for model in models:
            name = model.get('name', model.get('model_id', 'Unknown'))
            print(f"   - {name}")
            if 'video' in name.lower() or 'omnihuman' in name.lower():
                print(f"     ⭐ Potential video model: {model}")
    
    # Try various video endpoints
    endpoints = [
        'https://api.elevenlabs.io/v1/image-to-video',
        'https://api.elevenlabs.io/v1/video/generate',
        'https://api.elevenlabs.io/v1/videos',
        'https://api.elevenlabs.io/v1/studio/video',
        'https://api.elevenlabs.io/v1/talking-head',
        'https://api.elevenlabs.io/v1/avatar',
    ]
    
    print("\n🔍 Probing video endpoints...")
    for endpoint in endpoints:
        # Try OPTIONS to see if endpoint exists
        resp = requests.options(endpoint, headers=headers)
        get_resp = requests.get(endpoint, headers=headers)
        
        status = get_resp.status_code
        status_str = "✅ Available" if status == 200 else "❓ POST-only" if status == 405 else "❌ Not found" if status == 404 else f"? {status}"
        print(f"   {endpoint.replace('https://api.elevenlabs.io/v1/', '')}: {status_str}")


def list_sdk_structure():
    """List the full structure of the ElevenLabs SDK"""
    try:
        import elevenlabs
        print("\n📦 ElevenLabs SDK structure:")
        print(f"   Version: {elevenlabs.__version__ if hasattr(elevenlabs, '__version__') else 'Unknown'}")
        
        for attr in sorted(dir(elevenlabs)):
            if not attr.startswith('_'):
                obj = getattr(elevenlabs, attr)
                print(f"   {attr}: {type(obj).__name__}")
                
    except ImportError:
        print("❌ SDK not installed")


def main():
    print("═" * 60)
    print("   🎬 ELEVENLABS OMNIHUMAN TEST (Python)")  
    print("═" * 60)
    
    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY not set")
        sys.exit(1)
    
    print(f"✅ API key: {ELEVENLABS_API_KEY[:10]}...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 Output: {OUTPUT_DIR.absolute()}")
    
    # List SDK structure
    list_sdk_structure()
    
    # Test direct API
    test_direct_api()
    
    # Test with SDK
    test_with_sdk()
    
    print("\n" + "═" * 60)
    print("   Test complete")
    print("═" * 60)


if __name__ == '__main__':
    main()


