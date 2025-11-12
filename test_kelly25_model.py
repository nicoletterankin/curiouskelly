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
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicTTSModel(nn.Module):
    """Basic TTS model architecture (matching training)"""
    def __init__(self, vocab_size=256, hidden_dim=256, audio_length=110250):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.audio_length = audio_length
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, audio_length)
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(audio_length, hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 1)
        )
    
    def forward(self, text_ids):
        # Encode text
        embedded = self.text_embedding(text_ids)
        encoded, _ = self.text_encoder(embedded)
        
        # Pool over sequence
        pooled = torch.mean(encoded, dim=1)
        
        # Generate audio
        audio = self.generator(pooled)
        
        return audio

class Kelly25ModelTester:
    """Comprehensive testing suite for Kelly25 voice model"""
    
    def __init__(self, model_path, config_path):
        self.device = torch.device('cpu')  # Force CPU for compatibility
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.model = BasicTTSModel(
            vocab_size=self.config.get('vocab_size', 256),
            hidden_dim=self.config.get('hidden_dim', 256),
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
            duration = len(audio) / 22050
            rms = np.sqrt(np.mean(audio**2))
            max_amp = np.max(np.abs(audio))
            
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
            duration = len(audio) / 22050
            rms = np.sqrt(np.mean(audio**2))
            
            # Calculate pitch variation (simplified)
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/22050)
            dominant_freq = freqs[np.argmax(np.abs(fft))]
            
            results[emotion] = {
                'phrase': phrase,
                'file': output_file,
                'duration': duration,
                'rms': rms,
                'dominant_frequency': abs(dominant_freq)
            }
        
        return results
    
    def test_conversation_types(self):
        """Test different conversation types"""
        logger.info("Testing conversation types...")
        
        conversation_types = {
            'question': "Can you explain how this works?",
            'instruction': "First, let's break this down into smaller steps.",
            'explanation': "The key concept here is understanding the relationship between these variables.",
            'encouragement': "You're making excellent progress!",
            'reflection': "What do you think we learned from this exercise?"
        }
        
        results = {}
        for conv_type, phrase in conversation_types.items():
            logger.info(f"Generating {conv_type}: '{phrase}'")
            audio = self.generate_audio(phrase)
            
            output_file = f"test_output/conversation_{conv_type}.wav"
            sf.write(output_file, audio, 22050)
            
            results[conv_type] = {
                'phrase': phrase,
                'file': output_file,
                'duration': len(audio) / 22050
            }
        
        return results
    
    def analyze_audio_quality(self, audio):
        """Analyze audio quality metrics"""
        # Basic metrics
        duration = len(audio) / 22050
        rms = np.sqrt(np.mean(audio**2))
        max_amp = np.max(np.abs(audio))
        
        # Frequency analysis
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/22050)
        power_spectrum = np.abs(fft)**2
        
        # Find dominant frequency
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power_spectrum[:len(power_spectrum)//2]
        dominant_freq = positive_freqs[np.argmax(positive_power)]
        
        # Calculate spectral centroid
        spectral_centroid = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
        
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        zcr = zero_crossings / len(audio)
        
        return {
            'duration': duration,
            'rms': rms,
            'max_amplitude': max_amp,
            'dominant_frequency': abs(dominant_freq),
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zcr
        }
    
    def create_test_report(self, basic_results, emotional_results, conversation_results):
        """Create comprehensive test report"""
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'model_info': {
                'config': self.config,
                'device': str(self.device)
            },
            'basic_generation': basic_results,
            'emotional_range': emotional_results,
            'conversation_types': conversation_results,
            'summary': {
                'total_tests': len(basic_results) + len(emotional_results) + len(conversation_results),
                'average_duration': np.mean([r['duration'] for r in basic_results]),
                'average_rms': np.mean([r['rms'] for r in basic_results])
            }
        }
        
        # Save report
        with open('test_output/kelly25_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kelly25 Voice Model Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .test-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .audio-sample {{ margin: 10px 0; }}
                audio {{ width: 100%; }}
                .metrics {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Kelly25 Voice Model Test Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="test-section">
                <h2>Basic Generation Tests</h2>
                {self._generate_html_audio_section(basic_results)}
            </div>
            
            <div class="test-section">
                <h2>Emotional Range Tests</h2>
                {self._generate_html_audio_section(emotional_results)}
            </div>
            
            <div class="test-section">
                <h2>Conversation Type Tests</h2>
                {self._generate_html_audio_section(conversation_results)}
            </div>
            
            <div class="metrics">
                <h3>Summary Statistics</h3>
                <p>Total Tests: {report['summary']['total_tests']}</p>
                <p>Average Duration: {report['summary']['average_duration']:.2f} seconds</p>
                <p>Average RMS: {report['summary']['average_rms']:.4f}</p>
            </div>
        </body>
        </html>
        """
        
        with open('test_output/kelly25_test_report.html', 'w') as f:
            f.write(html_content)
        
        logger.info("Test report saved to test_output/kelly25_test_report.html")
        return report
    
    def _generate_html_audio_section(self, results):
        """Generate HTML for audio section"""
        html = ""
        for key, result in results.items():
            if isinstance(result, dict) and 'file' in result:
                html += f"""
                <div class="audio-sample">
                    <h4>{key.replace('_', ' ').title()}</h4>
                    <p><strong>Text:</strong> "{result.get('phrase', 'N/A')}"</p>
                    <audio controls>
                        <source src="{result['file']}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <div class="metrics">
                        <p>Duration: {result.get('duration', 0):.2f}s</p>
                        <p>RMS: {result.get('rms', 0):.4f}</p>
                    </div>
                </div>
                """
        return html

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
    
    # Conversation types test
    conversation_results = tester.test_conversation_types()
    
    # Create comprehensive report
    report = tester.create_test_report(basic_results, emotional_results, conversation_results)
    
    logger.info("🎉 Testing completed successfully!")
    logger.info(f"Generated {report['summary']['total_tests']} test samples")
    logger.info("Check test_output/ directory for all generated files")
    logger.info("Open test_output/kelly25_test_report.html to view results")

if __name__ == "__main__":
    main()




































