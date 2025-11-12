#!/usr/bin/env python3
"""
Piper TTS Python Wrapper
Provides easy-to-use interface for Piper TTS with Kelly and Ken voice models
"""

import subprocess
import json
import tempfile
from pathlib import Path
import os
import sys

class PiperTTS:
    """Piper TTS wrapper class for Kelly and Ken voices"""
    
    def __init__(self, voices_dir="voices"):
        self.voices_dir = Path(voices_dir)
        self.voice_models = {}
        self.load_voice_models()
    
    def load_voice_models(self):
        """Load available voice models"""
        for config_file in self.voices_dir.glob("*.onnx.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    model_name = config_file.stem
                    # Remove .json extension and add .onnx
                    model_file = config_file.with_suffix('').with_suffix('.onnx')
                    self.voice_models[model_name] = {
                        'config': config,
                        'model_file': str(model_file),
                        'config_file': str(config_file)
                    }
            except Exception as e:
                print(f"Warning: Could not load voice model {config_file}: {e}")
    
    def list_voices(self):
        """List available voices"""
        return list(self.voice_models.keys())
    
    def synthesize(self, text, voice=None, output_file=None, speaker_id=0):
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
                'piper',
                '--model', str(Path(voice_info['model_file']).absolute()),
                '--config', str(Path(voice_info['config_file']).absolute()),
                '--output_file', str(Path(output_file).absolute()),
                '--speaker', str(speaker_id)
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
    
    def synthesize_to_file(self, text, output_file, voice=None, speaker_id=0):
        """Synthesize speech and save to file"""
        return self.synthesize(text, voice, output_file, speaker_id)
    
    def synthesize_kelly(self, text, output_file=None):
        """Synthesize speech with Kelly's voice characteristics"""
        # Use the default voice for now, but this can be customized for Kelly
        return self.synthesize(text, output_file=output_file, speaker_id=0)
    
    def synthesize_ken(self, text, output_file=None):
        """Synthesize speech with Ken's voice characteristics"""
        # Use the default voice for now, but this can be customized for Ken
        return self.synthesize(text, output_file=output_file, speaker_id=0)

def test_piper_tts():
    """Test Piper TTS functionality"""
    print("🎵 Testing Piper TTS...")
    
    tts = PiperTTS()
    print(f"Available voices: {tts.list_voices()}")
    
    # Test synthesis
    test_text = "Hello, this is a test of Piper TTS for Kelly and Ken voices."
    output_file = "test_kelly_voice.wav"
    
    try:
        result = tts.synthesize_kelly(test_text, output_file)
        print(f"✅ Kelly voice test successful: {result}")
        
        # Test Ken voice
        output_file_ken = "test_ken_voice.wav"
        result_ken = tts.synthesize_ken(test_text, output_file_ken)
        print(f"✅ Ken voice test successful: {result_ken}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    print("🎭 Piper TTS Wrapper for Kelly and Ken")
    print("=" * 50)
    
    if test_piper_tts():
        print("\n🎉 Piper TTS is working correctly!")
        print("📝 Usage examples:")
        print("  tts = PiperTTS()")
        print("  tts.synthesize_kelly('Hello Kelly!', 'kelly.wav')")
        print("  tts.synthesize_ken('Hello Ken!', 'ken.wav')")
    else:
        print("❌ Piper TTS test failed!")
        sys.exit(1)
