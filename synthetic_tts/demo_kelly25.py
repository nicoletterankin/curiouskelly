#!/usr/bin/env python3
"""
Kelly25 Voice Demo - Direct Generation
Generate Kelly25 voice samples directly without API server
"""

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from pathlib import Path
import json

class SimpleTTSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Embedding(256, 128)
        self.text_lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(256, 128)
        self.generator = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 110250)
        )
        # Add discriminator to match saved model
        self.discriminator = nn.Sequential(
            nn.Linear(110250, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
    
    def forward(self, text_ids):
        embedded = self.text_embedding(text_ids)
        encoded, _ = self.text_lstm(embedded)
        projected = self.text_proj(encoded)
        pooled = torch.mean(projected, dim=1)
        audio = self.generator(pooled)
        return audio

def text_to_ids(text, max_length=50):
    text_ids = [ord(c) for c in text[:max_length]]
    text_ids = text_ids + [0] * (max_length - len(text_ids))
    return torch.LongTensor(text_ids).unsqueeze(0)

def load_model():
    """Load the trained Kelly25 model"""
    model = SimpleTTSModel()
    device = torch.device('cpu')
    
    try:
        if Path("kelly25_model_output/best_model.pth").exists():
            checkpoint = torch.load("kelly25_model_output/best_model.pth", map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Trained model loaded successfully")
        else:
            print("⚠️ No trained model found, using random weights")
        
        model.eval()
        return model, device
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def generate_audio(model, text, device):
    """Generate audio from text"""
    with torch.no_grad():
        text_ids = text_to_ids(text).to(device)
        generated_audio = model(text_ids)
        audio_np = generated_audio.squeeze().cpu().numpy()
        return audio_np

def main():
    print("🎤 Kelly25 Voice Demo - Direct Generation")
    print("=" * 50)
    
    # Load model
    model, device = load_model()
    if model is None:
        print("❌ Failed to load model")
        return
    
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
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🎵 Generating {len(test_phrases)} voice samples...")
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n{i}. Generating: '{phrase}'")
        
        try:
            # Generate audio
            audio = generate_audio(model, phrase, device)
            
            # Save audio
            filename = f"kelly25_demo_{i:02d}.wav"
            filepath = output_dir / filename
            sf.write(filepath, audio, 22050)
            
            # Analyze audio
            duration = len(audio) / 22050
            rms = np.sqrt(np.mean(audio**2))
            max_amp = np.max(np.abs(audio))
            
            print(f"   ✅ Saved: {filename}")
            print(f"   Duration: {duration:.2f}s, RMS: {rms:.4f}, Max: {max_amp:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Demo completed!")
    print(f"📁 Audio files saved in: {output_dir.absolute()}")
    print(f"🎵 You can now play the generated WAV files!")
    
    # Create a simple HTML player
    create_html_player(output_dir, test_phrases)

def create_html_player(output_dir, phrases):
    """Create a simple HTML player for the generated audio"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Kelly25 Voice Demo Player</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        .audio-item {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .phrase {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
        audio {{ width: 100%; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Kelly25 Voice Demo Player</h1>
        <p>Generated voice samples from the trained Kelly25 model:</p>
"""
    
    for i, phrase in enumerate(phrases, 1):
        filename = f"kelly25_demo_{i:02d}.wav"
        html_content += f"""
        <div class="audio-item">
            <div class="phrase">{i}. "{phrase}"</div>
            <audio controls>
                <source src="{filename}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    player_path = output_dir / "kelly25_demo_player.html"
    with open(player_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"🌐 HTML player created: {player_path.absolute()}")
    print(f"💡 Open this file in your browser to play all samples!")

if __name__ == "__main__":
    main()
