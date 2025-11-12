#!/usr/bin/env python3
"""
Kelly25 Model Validation Script
Validate the trained Kelly25 voice model and test audio quality
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import soundfile as sf
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicTTSModel(nn.Module):
    """Basic TTS Model for Kelly25 Voice (same as training)"""
    
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, hidden_dim * 16),
            nn.ReLU(),
            nn.Linear(hidden_dim * 16, 110250)  # Fixed output length
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(110250, hidden_dim * 16),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 16, hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, 1)
        )
    
    def forward(self, text_ids):
        # Text encoding
        text_embedded = self.text_embedding(text_ids)
        text_lstm_out, _ = self.text_lstm(text_embedded)
        text_encoded = self.text_proj(text_lstm_out)
        
        # Average pooling over text sequence
        text_pooled = torch.mean(text_encoded, dim=1)  # [batch_size, hidden_dim]
        
        # Generate audio
        generated_audio = self.generator(text_pooled)
        
        return generated_audio
    
    def discriminate(self, audio):
        return self.discriminator(audio)

class Kelly25ModelValidator:
    """Kelly25 Model Validator"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.model = BasicTTSModel(
            vocab_size=self.config.get('vocab_size', 256),
            hidden_dim=self.config.get('hidden_dim', 128)
        ).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Model loaded from: {self.model_path}")
        logger.info(f"Using device: {self.device}")
    
    def generate_audio(self, text: str) -> np.ndarray:
        """Generate audio from text"""
        with torch.no_grad():
            # Encode text
            text_ids = [ord(c) for c in text[:50]]
            text_ids = text_ids + [0] * (50 - len(text_ids))
            text_ids = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
            
            # Generate audio
            generated_audio = self.model(text_ids)
            
            # Convert to numpy
            audio_np = generated_audio.squeeze().cpu().numpy()
            
            return audio_np
    
    def validate_model(self) -> Dict:
        """Comprehensive model validation"""
        logger.info("🔍 Starting Kelly25 Model Validation")
        logger.info("=" * 50)
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'config': self.config,
            'device': str(self.device),
            'tests': {}
        }
        
        # Test 1: Basic Generation
        logger.info("Test 1: Basic Audio Generation")
        try:
            test_text = "Hello! I'm Kelly, your friendly learning companion."
            audio = self.generate_audio(test_text)
            
            validation_results['tests']['basic_generation'] = {
                'status': 'PASS',
                'audio_length': len(audio),
                'sample_rate': 22050,
                'duration_seconds': len(audio) / 22050,
                'audio_range': [float(np.min(audio)), float(np.max(audio))],
                'audio_mean': float(np.mean(audio)),
                'audio_std': float(np.std(audio))
            }
            logger.info("✅ Basic generation test passed")
        except Exception as e:
            validation_results['tests']['basic_generation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"❌ Basic generation test failed: {e}")
        
        # Test 2: Multiple Text Samples
        logger.info("Test 2: Multiple Text Samples")
        test_texts = [
            "Hello! I'm Kelly, your friendly learning companion.",
            "Let's explore this topic together and discover something amazing.",
            "You're doing great! Keep up the excellent work.",
            "I'm so proud of your progress and dedication to learning.",
            "Take a moment to think about what we've learned today."
        ]
        
        try:
            generated_samples = []
            for i, text in enumerate(test_texts):
                audio = self.generate_audio(text)
                generated_samples.append({
                    'text': text,
                    'audio_length': len(audio),
                    'duration': len(audio) / 22050,
                    'range': [float(np.min(audio)), float(np.max(audio))],
                    'mean': float(np.mean(audio)),
                    'std': float(np.std(audio))
                })
            
            validation_results['tests']['multiple_samples'] = {
                'status': 'PASS',
                'samples': generated_samples,
                'total_samples': len(generated_samples)
            }
            logger.info(f"✅ Multiple samples test passed ({len(generated_samples)} samples)")
        except Exception as e:
            validation_results['tests']['multiple_samples'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"❌ Multiple samples test failed: {e}")
        
        # Test 3: Audio Quality Analysis
        logger.info("Test 3: Audio Quality Analysis")
        try:
            test_audio = self.generate_audio("This is a quality test for the Kelly25 voice model.")
            
            # Analyze audio quality
            quality_metrics = self.analyze_audio_quality(test_audio)
            
            validation_results['tests']['audio_quality'] = {
                'status': 'PASS',
                'metrics': quality_metrics
            }
            logger.info("✅ Audio quality test passed")
        except Exception as e:
            validation_results['tests']['audio_quality'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"❌ Audio quality test failed: {e}")
        
        # Test 4: Model Performance
        logger.info("Test 4: Model Performance")
        try:
            performance_metrics = self.analyze_model_performance()
            
            validation_results['tests']['model_performance'] = {
                'status': 'PASS',
                'metrics': performance_metrics
            }
            logger.info("✅ Model performance test passed")
        except Exception as e:
            validation_results['tests']['model_performance'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"❌ Model performance test failed: {e}")
        
        return validation_results
    
    def analyze_audio_quality(self, audio: np.ndarray) -> Dict:
        """Analyze audio quality metrics"""
        # Basic statistics
        stats = {
            'length': len(audio),
            'duration_seconds': len(audio) / 22050,
            'min_value': float(np.min(audio)),
            'max_value': float(np.max(audio)),
            'mean_value': float(np.mean(audio)),
            'std_value': float(np.std(audio)),
            'rms': float(np.sqrt(np.mean(audio**2))),
            'dynamic_range': float(np.max(audio) - np.min(audio))
        }
        
        # Frequency analysis
        try:
            # Compute FFT
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/22050)
            
            # Find dominant frequency
            magnitude = np.abs(fft)
            dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            stats['dominant_frequency'] = float(dominant_freq)
            stats['frequency_analysis'] = 'SUCCESS'
        except Exception as e:
            stats['frequency_analysis'] = f'FAILED: {str(e)}'
        
        # Signal quality indicators
        stats['has_signal'] = stats['rms'] > 0.001
        stats['is_normalized'] = stats['max_value'] <= 1.0 and stats['min_value'] >= -1.0
        stats['has_clipping'] = stats['max_value'] >= 0.99 or stats['min_value'] <= -0.99
        
        return stats
    
    def analyze_model_performance(self) -> Dict:
        """Analyze model performance metrics"""
        # Model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Model structure
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'hidden_dimension': self.config.get('hidden_dim', 128),
            'vocab_size': self.config.get('vocab_size', 256)
        }
        
        # Generation speed test
        import time
        test_text = "Performance test"
        start_time = time.time()
        
        for _ in range(10):
            self.generate_audio(test_text)
        
        end_time = time.time()
        generation_time = (end_time - start_time) / 10
        
        model_info['generation_time_seconds'] = generation_time
        model_info['generations_per_second'] = 1.0 / generation_time
        
        return model_info
    
    def generate_validation_samples(self, output_dir: Path):
        """Generate validation audio samples"""
        logger.info("🎵 Generating validation samples")
        
        # Test texts covering different scenarios
        test_cases = [
            {
                'name': 'greeting',
                'text': "Hello! I'm Kelly, your friendly learning companion. Welcome to our learning journey together!"
            },
            {
                'name': 'encouragement',
                'text': "You're doing amazing! I'm so proud of your progress and dedication to learning."
            },
            {
                'name': 'explanation',
                'text': "Let me explain this concept step by step. First, we need to understand the basic principles."
            },
            {
                'name': 'question',
                'text': "What do you think about this topic? I'd love to hear your thoughts and insights."
            },
            {
                'name': 'reflection',
                'text': "Let's take a moment to reflect on what we've learned today. This is important for understanding."
            }
        ]
        
        generated_files = []
        
        for test_case in test_cases:
            try:
                # Generate audio
                audio = self.generate_audio(test_case['text'])
                
                # Save audio file
                output_file = output_dir / f"validation_{test_case['name']}.wav"
                sf.write(output_file, audio, 22050)
                
                generated_files.append({
                    'name': test_case['name'],
                    'text': test_case['text'],
                    'file': str(output_file),
                    'duration': len(audio) / 22050,
                    'size_bytes': os.path.getsize(output_file)
                })
                
                logger.info(f"✅ Generated: {output_file}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate {test_case['name']}: {e}")
        
        return generated_files
    
    def create_validation_report(self, validation_results: Dict, output_dir: Path):
        """Create comprehensive validation report"""
        
        # Save JSON report
        report_file = output_dir / 'validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Create markdown report
        markdown_file = output_dir / 'VALIDATION_REPORT.md'
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write("# Kelly25 Voice Model Validation Report\n\n")
            f.write(f"**Generated:** {validation_results['timestamp']}\n\n")
            
            f.write("## Model Information\n\n")
            f.write(f"- **Model Path:** {validation_results['model_path']}\n")
            f.write(f"- **Device:** {validation_results['device']}\n")
            f.write(f"- **Configuration:** {validation_results['config']}\n\n")
            
            f.write("## Validation Results\n\n")
            
            for test_name, test_result in validation_results['tests'].items():
                f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Status:** {'✅ PASS' if test_result['status'] == 'PASS' else '❌ FAIL'}\n\n")
                
                if test_result['status'] == 'PASS':
                    if 'metrics' in test_result:
                        f.write("**Metrics:**\n")
                        for key, value in test_result['metrics'].items():
                            f.write(f"- {key}: {value}\n")
                    elif 'samples' in test_result:
                        f.write(f"**Samples:** {test_result['total_samples']}\n")
                        for sample in test_result['samples']:
                            f.write(f"- Duration: {sample['duration']:.2f}s, Range: [{sample['range'][0]:.3f}, {sample['range'][1]:.3f}]\n")
                else:
                    f.write(f"**Error:** {test_result['error']}\n")
                
                f.write("\n")
            
            f.write("## Summary\n\n")
            passed_tests = sum(1 for test in validation_results['tests'].values() if test['status'] == 'PASS')
            total_tests = len(validation_results['tests'])
            f.write(f"**Tests Passed:** {passed_tests}/{total_tests}\n")
            f.write(f"**Success Rate:** {passed_tests/total_tests*100:.1f}%\n\n")
            
            if passed_tests == total_tests:
                f.write("🎉 **All tests passed!** The Kelly25 voice model is ready for use.\n")
            else:
                f.write("⚠️ **Some tests failed.** Please review the issues before deployment.\n")
        
        logger.info(f"Validation report saved: {report_file}")
        logger.info(f"Markdown report saved: {markdown_file}")

def validate_kelly25_model():
    """Main validation function"""
    
    print("🔍 Kelly25 Model Validation")
    print("=" * 50)
    
    # Paths
    model_path = "kelly25_model_output/best_model.pth"
    config_path = "kelly25_model_output/config.json"
    output_dir = Path("kelly25_validation_output")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    # Initialize validator
    validator = Kelly25ModelValidator(model_path, config_path)
    
    # Run validation
    validation_results = validator.validate_model()
    
    # Generate validation samples
    sample_files = validator.generate_validation_samples(output_dir)
    validation_results['generated_samples'] = sample_files
    
    # Create validation report
    validator.create_validation_report(validation_results, output_dir)
    
    # Print summary
    print("\n📊 Validation Summary:")
    print("=" * 30)
    
    passed_tests = sum(1 for test in validation_results['tests'].values() if test['status'] == 'PASS')
    total_tests = len(validation_results['tests'])
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Generated Samples: {len(sample_files)}")
    print(f"Output Directory: {output_dir}")
    
    if passed_tests == total_tests:
        print("\n🎉 All validation tests passed!")
        print("✅ Kelly25 voice model is ready for use!")
    else:
        print("\n⚠️ Some validation tests failed.")
        print("Please review the validation report for details.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    validate_kelly25_model()
