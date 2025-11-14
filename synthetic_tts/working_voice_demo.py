#!/usr/bin/env python3
"""
Working Voice Demo - Using Real TTS
Generate actual speech using a working TTS system
"""

import pyttsx3
import os
from pathlib import Path
import time

def create_working_voice_demo():
    """Create a working voice demo using pyttsx3"""
    print("🎤 Working Voice Demo - Real Speech Generation")
    print("=" * 50)
    
    # Initialize TTS engine
    try:
        engine = pyttsx3.init()
        print("✅ TTS Engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize TTS engine: {e}")
        return
    
    # Configure voice settings
    voices = engine.getProperty('voices')
    if voices:
        # Try to find a female voice
        for voice in voices:
            if 'female' in voice.name.lower() or 'karen' in voice.name.lower() or 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                print(f"✅ Using voice: {voice.name}")
                break
        else:
            # Use first available voice
            engine.setProperty('voice', voices[0].id)
            print(f"✅ Using voice: {voices[0].name}")
    
    # Set speech rate and volume
    engine.setProperty('rate', 180)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    
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
    output_dir = Path("working_voice_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🎵 Generating {len(test_phrases)} voice samples...")
    
    # Generate audio files
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n{i}. Generating: '{phrase}'")
        
        try:
            # Save to file
            filename = f"kelly_voice_{i:02d}.wav"
            filepath = output_dir / filename
            engine.save_to_file(phrase, str(filepath))
            engine.runAndWait()
            
            print(f"   ✅ Saved: {filename}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Voice generation completed!")
    print(f"📁 Audio files saved in: {output_dir.absolute()}")
    
    # Create HTML player
    create_working_html_player(output_dir, test_phrases)
    
    # Test live speech
    print(f"\n🎤 Testing live speech...")
    print("Speaking: 'Hello! I'm Kelly, your learning companion.'")
    engine.say("Hello! I'm Kelly, your learning companion.")
    engine.runAndWait()
    print("✅ Live speech test completed!")

def create_working_html_player(output_dir, phrases):
    """Create HTML player for working voice samples"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Kelly Voice Demo - Working Speech</title>
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
            border-radius: 15px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
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
        .status {{
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Kelly Voice Demo - Working Speech</h1>
            <p>Real TTS-generated voice samples using pyttsx3</p>
        </div>
        
        <div class="status">
            ✅ <strong>Working Speech Generation!</strong> These are real voice samples, not noise.
        </div>
"""
    
    for i, phrase in enumerate(phrases, 1):
        filename = f"kelly_voice_{i:02d}.wav"
        html_content += f"""
        <div class="audio-item">
            <div class="phrase">{i}. "{phrase}"</div>
            <audio controls>
                <source src="{filename}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            <br>
            <button class="download-btn" onclick="downloadAudio('{filename}')">Download Audio</button>
        </div>
"""
    
    html_content += """
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
    
    player_path = output_dir / "kelly_working_voice_player.html"
    with open(player_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"🌐 HTML player created: {player_path.absolute()}")
    print(f"💡 Open this file in your browser to play all working voice samples!")

if __name__ == "__main__":
    create_working_voice_demo()





































