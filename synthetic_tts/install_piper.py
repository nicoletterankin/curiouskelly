#!/usr/bin/env python3
"""
Install and Setup Piper TTS
Downloads and configures Piper TTS for the voice library system
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import json
from pathlib import Path

def install_piper():
    """Install Piper TTS system"""
    
    print("🎵 Installing Piper TTS System")
    print("=" * 50)
    
    # Check if we're on Windows
    if platform.system() != "Windows":
        print("❌ This script is designed for Windows. Please install Piper manually.")
        return False
    
    # Create piper directory
    piper_dir = Path("piper")
    piper_dir.mkdir(exist_ok=True)
    
    try:
        # Download Piper executable for Windows
        print("📥 Downloading Piper TTS...")
        piper_url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_windows_amd64.zip"
        piper_zip = piper_dir / "piper.zip"
        
        urllib.request.urlretrieve(piper_url, piper_zip)
        
        # Extract Piper
        print("📦 Extracting Piper...")
        with zipfile.ZipFile(piper_zip, 'r') as zip_ref:
            zip_ref.extractall(piper_dir)
        
        # Remove zip file
        piper_zip.unlink()
        
        # Download a sample voice model
        print("🎤 Downloading sample voice model...")
        voice_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        voice_file = piper_dir / "en_US-lessac-medium.onnx"
        
        urllib.request.urlretrieve(voice_url, voice_file)
        
        # Download voice config
        config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        config_file = piper_dir / "en_US-lessac-medium.onnx.json"
        
        urllib.request.urlretrieve(config_url, config_file)
        
        print("✅ Piper TTS installed successfully!")
        print(f"📁 Piper directory: {piper_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error installing Piper: {e}")
        return False

def test_piper():
    """Test Piper TTS installation"""
    
    print("\n🧪 Testing Piper TTS...")
    
    piper_dir = Path("piper")
    piper_exe = piper_dir / "piper.exe"
    
    if not piper_exe.exists():
        print("❌ Piper executable not found!")
        return False
    
    try:
        # Test Piper with a simple command
        result = subprocess.run([
            str(piper_exe), 
            "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Piper TTS is working correctly!")
            return True
        else:
            print(f"❌ Piper test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Piper: {e}")
        return False

def create_piper_wrapper():
    """Create a Python wrapper for Piper TTS"""
    
    wrapper_code = '''#!/usr/bin/env python3
"""
Piper TTS Python Wrapper
Provides easy-to-use interface for Piper TTS
"""

import subprocess
import json
import tempfile
from pathlib import Path
import os

class PiperTTS:
    """Piper TTS wrapper class"""
    
    def __init__(self, piper_dir="piper"):
        self.piper_dir = Path(piper_dir)
        self.piper_exe = self.piper_dir / "piper.exe"
        self.voice_models = {}
        self.load_voice_models()
    
    def load_voice_models(self):
        """Load available voice models"""
        for config_file in self.piper_dir.glob("*.onnx.json"):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    model_name = config_file.stem
                    self.voice_models[model_name] = {
                        'config': config,
                        'model_file': config_file.with_suffix('.onnx'),
                        'config_file': config_file
                    }
            except Exception as e:
                print(f"Warning: Could not load voice model {config_file}: {e}")
    
    def list_voices(self):
        """List available voices"""
        return list(self.voice_models.keys())
    
    def synthesize(self, text, voice=None, output_file=None):
        """Synthesize speech from text"""
        
        if not self.voice_models:
            raise RuntimeError("No voice models available")
        
        # Use first available voice if none specified
        if voice is None:
            voice = list(self.voice_models.keys())[0]
        
        if voice not in self.voice_models:
            raise ValueError(f"Voice '{voice}' not available. Available voices: {list(self.voice_models.keys())}")
        
        voice_info = self.voice_models[voice]
        
        # Create temporary output file if none specified
        if output_file is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_file = temp_file.name
            temp_file.close()
        
        try:
            # Run Piper TTS
            cmd = [
                str(self.piper_exe),
                '--model', str(voice_info['model_file']),
                '--config_file', str(voice_info['config_file']),
                '--output_file', str(output_file)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                raise RuntimeError(f"Piper TTS failed: {stderr}")
            
            return output_file
            
        except Exception as e:
            raise RuntimeError(f"Error synthesizing speech: {e}")
    
    def synthesize_to_file(self, text, output_file, voice=None):
        """Synthesize speech and save to file"""
        return self.synthesize(text, voice, output_file)

# Example usage
if __name__ == "__main__":
    tts = PiperTTS()
    print("Available voices:", tts.list_voices())
    
    # Test synthesis
    output_file = "test_output.wav"
    tts.synthesize_to_file("Hello, this is a test of Piper TTS.", output_file)
    print(f"Generated audio: {output_file}")
'''
    
    wrapper_file = Path("piper_tts.py")
    with open(wrapper_file, 'w') as f:
        f.write(wrapper_code)
    
    print(f"✅ Created Piper wrapper: {wrapper_file}")

def main():
    """Main installation function"""
    
    if install_piper():
        test_piper()
        create_piper_wrapper()
        print("\n🎉 Piper TTS setup complete!")
        print("📝 Next steps:")
        print("1. Run: python piper_tts.py")
        print("2. Use PiperTTS class in your voice generation scripts")
    else:
        print("❌ Piper TTS installation failed!")

if __name__ == "__main__":
    main()
