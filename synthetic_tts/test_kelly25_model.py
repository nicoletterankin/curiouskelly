#!/usr/bin/env python3
"""
Kelly25 Voice Model Testing Suite
Comprehensive testing of the trained Kelly25 voice model
"""

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicTTSModel(nn.Module):
    """Basic TTS model architecture (matching training)"""
    def __init__(self, vocab_size=256, hidden_dim=128, audio_length=110250):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.audio_length = audio_length
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, audio_length)
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(audio_length, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
    
    def forward(self, text_ids):
        # Encode text
        embedded = self.text_embedding(text_ids)
        encoded, _ = self.text_lstm(embedded)
        projected = self.text_proj(encoded)
        
        # Pool over sequence
        pooled = torch.mean(projected, dim=1)
        
        # Generate audio
        audio = self.generator(pooled)
        
        return audio

class Kelly25ModelTester:
    """Comprehensive testing suite for Kelly25 voice model"""
    
    def __init__(self, model_path, config_path):
        self.device = torch.device('cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.model = BasicTTSModel(
            vocab_size=self.config.get('vocab_size', 256),
            hidden_dim=self.config.get('hidden_dim', 128),
            audio_length=self.config.get('audio_length', 110250)
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def text_to_ids(self, text, max_length=50):
        """Convert text to character IDs"""
        text_ids = [ord(c) for c in text[:max_length]]
        text_ids = text_ids + [0] * (max_length - len(text_ids))
        return torch.LongTensor(text_ids).unsqueeze(0)
    
    def generate_audio(self, text):
        """Generate audio from text"""
        with torch.no_grad():
            text_ids = self.text_to_ids(text)
            generated_audio = self.model(text_ids)
            audio_np = generated_audio.squeeze().numpy()
            return audio_np
    
    def test_basic_generation(self):
        """Test basic audio generation"""
        logger.info("Testing basic audio generation...")
        
        test_phrases = [
            "Hello! I'm Kelly, your learning companion.",
            "Let's explore this concept together.",
            "Great job on that last attempt!",
            "What do you think about this idea?",
            "Mathematics is the language of the universe."
        ]
        
        results = []
        for i, phrase in enumerate(test_phrases):
            logger.info(f"Generating: '{phrase}'")
            audio = self.generate_audio(phrase)
            
            # Save audio
            output_file = f"test_output/test_phrase_{i+1}.wav"
            Path("test_output").mkdir(exist_ok=True)
            sf.write(output_file, audio, 22050)
            
            # Analyze audio
            duration = float(len(audio) / 22050)
            rms = float(np.sqrt(np.mean(audio**2)))
            max_amp = float(np.max(np.abs(audio)))
            
            results.append({
                'phrase': phrase,
                'file': output_file,
                'duration': duration,
                'rms': rms,
                'max_amplitude': max_amp
            })
            
            logger.info(f"  Duration: {duration:.2f}s, RMS: {rms:.4f}, Max: {max_amp:.4f}")
        
        return results
    
    def test_emotional_range(self):
        """Test emotional range of the voice"""
        logger.info("Testing emotional range...")
        
        emotional_phrases = {
            'excitement': "Wow! This is absolutely amazing!",
            'encouragement': "You're doing great! Keep it up!",
            'curiosity': "I wonder what would happen if we tried this approach?",
            'reassurance': "Don't worry, we'll figure this out together.",
            'celebration': "Fantastic! You've mastered this concept!"
        }
        
        results = {}
        for emotion, phrase in emotional_phrases.items():
            logger.info(f"Generating {emotion}: '{phrase}'")
            audio = self.generate_audio(phrase)
            
            output_file = f"test_output/emotion_{emotion}.wav"
            sf.write(output_file, audio, 22050)
            
            # Analyze prosodic features
            duration = float(len(audio) / 22050)
            rms = float(np.sqrt(np.mean(audio**2)))
            
            results[emotion] = {
                'phrase': phrase,
                'file': output_file,
                'duration': duration,
                'rms': rms
            }
        
        return results
    
    def create_test_report(self, basic_results, emotional_results):
        """Create comprehensive test report"""
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'model_info': {
                'config': self.config,
                'device': str(self.device)
            },
            'basic_generation': basic_results,
            'emotional_range': emotional_results,
            'summary': {
                'total_tests': len(basic_results) + len(emotional_results),
                'average_duration': np.mean([r['duration'] for r in basic_results]),
                'average_rms': np.mean([r['rms'] for r in basic_results])
            }
        }
        
        # Save report
        with open('test_output/kelly25_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Test report saved to test_output/kelly25_test_report.json")
        return report

def main():
    """Run comprehensive Kelly25 model tests"""
    print("🧪 Kelly25 Voice Model Testing Suite")
    print("=" * 50)
    
    # Initialize tester
    model_path = "kelly25_model_output/best_model.pth"
    config_path = "kelly25_model_output/config.json"
    
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    tester = Kelly25ModelTester(model_path, config_path)
    
    # Run tests
    logger.info("Starting comprehensive testing...")
    
    # Basic generation test
    basic_results = tester.test_basic_generation()
    
    # Emotional range test
    emotional_results = tester.test_emotional_range()
    
    # Create comprehensive report
    report = tester.create_test_report(basic_results, emotional_results)
    
    logger.info("🎉 Testing completed successfully!")
    logger.info(f"Generated {report['summary']['total_tests']} test samples")
    logger.info("Check test_output/ directory for all generated files")

if __name__ == "__main__":
    main()
