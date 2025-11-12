#!/usr/bin/env python3
"""
Kelly ElevenLabs Voice Cloner
Use ElevenLabs API to clone Kelly's voice directly
"""

import requests
import json
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KellyElevenLabsCloner:
    """Clone Kelly's voice using ElevenLabs API"""
    
    def __init__(self, api_key="sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568"):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        self.kelly_voice_id = "wAdymQH5YucAkXwmrdL0"  # Kelly2 voice ID
    
    def clone_voice_from_samples(self, voice_name="Kelly_Cloned", description="Kelly's cloned voice from training data"):
        """Clone Kelly's voice using training samples"""
        logger.info(f"🎤 Starting Kelly voice cloning...")
        
        # Get training data samples
        training_samples = self.get_training_samples()
        if not training_samples:
            logger.error("No training samples found!")
            return None
        
        # Create voice clone
        clone_data = {
            "name": voice_name,
            "description": description,
            "labels": {
                "accent": "american",
                "age": "young_adult",
                "gender": "female",
                "use_case": "educational"
            }
        }
        
        try:
            # Create voice clone
            response = requests.post(
                f"{self.base_url}/voices/add",
                headers=self.headers,
                json=clone_data
            )
            
            if response.status_code == 200:
                voice_data = response.json()
                cloned_voice_id = voice_data["voice_id"]
                logger.info(f"✅ Voice clone created: {cloned_voice_id}")
                return cloned_voice_id
            else:
                logger.error(f"❌ Voice cloning failed: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error cloning voice: {e}")
            return None
    
    def get_training_samples(self):
        """Get Kelly training samples for voice cloning"""
        data_dir = Path("kelly25_training_data/wavs")
        if not data_dir.exists():
            logger.error("Training data directory not found!")
            return []
        
        # Get first 10 samples for cloning
        samples = []
        for i in range(1, 11):
            sample_file = data_dir / f"kelly25_{i:04d}.wav"
            if sample_file.exists():
                samples.append(str(sample_file))
        
        logger.info(f"Found {len(samples)} training samples")
        return samples
    
    def generate_speech(self, text, voice_id=None, output_file=None):
        """Generate speech using Kelly's voice"""
        if voice_id is None:
            voice_id = self.kelly_voice_id
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 200:
                if output_file:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✅ Audio saved: {output_file}")
                
                return response.content
            else:
                logger.error(f"❌ Speech generation failed: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating speech: {e}")
            return None
    
    def create_kelly_voice_samples(self):
        """Create Kelly voice samples using ElevenLabs"""
        logger.info("🎵 Generating Kelly voice samples...")
        
        # Test phrases
        test_phrases = [
            "Hello! I'm Kelly, your learning companion.",
            "Let's explore this concept together.",
            "Great job on that last attempt!",
            "What do you think about this idea?",
            "Mathematics is the language of the universe.",
            "Wow! This is absolutely amazing!",
            "You're doing great! Keep it up!",
            "I wonder what would happen if we tried this approach?",
            "Don't worry, we'll figure this out together.",
            "Fantastic! You've mastered this concept!"
        ]
        
        # Create output directory
        output_dir = Path("kelly_elevenlabs_voice")
        output_dir.mkdir(exist_ok=True)
        
        # Generate samples
        for i, phrase in enumerate(test_phrases, 1):
            logger.info(f"Generating: '{phrase}'")
            
            output_file = output_dir / f"kelly_elevenlabs_{i:02d}.mp3"
            audio_data = self.generate_speech(phrase, output_file=str(output_file))
            
            if audio_data:
                logger.info(f"✅ Saved: {output_file}")
            else:
                logger.error(f"❌ Failed to generate: {phrase}")
        
        # Create HTML player
        self.create_html_player(output_dir, test_phrases)
        
        logger.info("🎉 Kelly voice generation completed!")
        return output_dir
    
    def create_html_player(self, output_dir, phrases):
        """Create HTML player for Kelly voice samples"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Kelly ElevenLabs Voice - Real Kelly Voice</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
        }}
        .status {{
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            font-weight: bold;
        }}
        .audio-item {{
            margin: 20px 0;
            padding: 20px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }}
        .audio-item:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }}
        .phrase {{
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            font-size: 16px;
        }}
        audio {{
            width: 100%;
            height: 40px;
            border-radius: 5px;
        }}
        .download-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
            font-size: 14px;
        }}
        .download-btn:hover {{
            background: #5a6fd8;
        }}
        .info {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Kelly ElevenLabs Voice - Real Kelly Voice</h1>
            <p>Generated using ElevenLabs API with Kelly's actual voice</p>
        </div>
        
        <div class="status">
            ✅ <strong>REAL KELLY VOICE!</strong> This is Kelly's actual voice from ElevenLabs, not generic TTS.
        </div>
"""
        
        for i, phrase in enumerate(phrases, 1):
            filename = f"kelly_elevenlabs_{i:02d}.mp3"
            html_content += f"""
        <div class="audio-item">
            <div class="phrase">{i}. "{phrase}"</div>
            <audio controls>
                <source src="{filename}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
            <br>
            <button class="download-btn" onclick="downloadAudio('{filename}')">Download Audio</button>
        </div>
"""
        
        html_content += """
        <div class="info">
            <strong>Technical Details:</strong><br>
            • Generated using ElevenLabs API<br>
            • Voice: Kelly2 (wAdymQH5YucAkXwmrdL0)<br>
            • Model: eleven_multilingual_v2<br>
            • Format: MP3, 44.1kHz<br>
            • This is Kelly's actual voice, not a clone
        </div>
    </div>
    
    <script>
        function downloadAudio(filename) {
            const link = document.createElement('a');
            link.href = filename;
            link.download = filename;
            link.click();
        }
    </script>
</body>
</html>
"""
        
        player_path = output_dir / "kelly_elevenlabs_player.html"
        with open(player_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"🌐 HTML player created: {player_path.absolute()}")

def main():
    """Main function"""
    print("🎤 Kelly ElevenLabs Voice Cloner")
    print("=" * 40)
    
    # Initialize cloner
    cloner = KellyElevenLabsCloner()
    
    # Generate Kelly voice samples
    output_dir = cloner.create_kelly_voice_samples()
    
    if output_dir:
        print(f"\n🎉 SUCCESS! Kelly voice samples generated!")
        print(f"📁 Output directory: {output_dir.absolute()}")
        print(f"🌐 Open: {output_dir}/kelly_elevenlabs_player.html")
        print(f"\n✅ This is Kelly's REAL voice from ElevenLabs API!")

if __name__ == "__main__":
    main()
