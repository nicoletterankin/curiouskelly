#!/usr/bin/env python3
"""
Setup script for the hybrid TTS system.

This script initializes the hybrid TTS system, creates voice databases,
and sets up all necessary components for voice interpolation and morphing.
"""

import sys
import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
import json
import time
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from synthesis.enhanced_synthesizer import EnhancedSynthesizer
from models.model_factory import ModelFactory
from voice.voice_interpolator import VoiceInterpolator
from voice.voice_analyzer import VoiceAnalyzer
from data.hybrid_data_generator import HybridDataGenerator
from utils.voice_utils import VoiceUtils


def create_voice_database(num_voices: int = 20) -> Dict[str, np.ndarray]:
    """Create a comprehensive voice database."""
    print(f"Creating voice database with {num_voices} voices...")
    
    voice_database = {}
    
    # Create diverse voice types
    voice_types = [
        # High pitch voices
        ('high_pitch_bright', [0.8, 0.3, 0.7, 0.6, 0.2, 0.5, 0.4, 0.3]),
        ('high_pitch_warm', [0.7, 0.4, 0.5, 0.4, 0.3, 0.6, 0.5, 0.4]),
        ('high_pitch_energetic', [0.9, 0.6, 0.8, 0.7, 0.2, 0.4, 0.7, 0.6]),
        
        # Medium pitch voices
        ('medium_balanced', [0.5, 0.5, 0.5, 0.5, 0.4, 0.6, 0.5, 0.4]),
        ('medium_warm', [0.4, 0.3, 0.4, 0.3, 0.5, 0.7, 0.6, 0.5]),
        ('medium_bright', [0.6, 0.4, 0.6, 0.6, 0.3, 0.5, 0.6, 0.5]),
        
        # Low pitch voices
        ('low_pitch_warm', [0.3, 0.4, 0.4, 0.3, 0.6, 0.7, 0.6, 0.5]),
        ('low_pitch_deep', [0.2, 0.3, 0.3, 0.2, 0.7, 0.8, 0.7, 0.6]),
        ('low_pitch_rich', [0.4, 0.5, 0.4, 0.4, 0.5, 0.6, 0.5, 0.4]),
        
        # Specialized voices
        ('whisper', [0.3, 0.2, 0.2, 0.1, 0.8, 0.9, 0.2, 0.1]),
        ('shout', [0.8, 0.7, 0.9, 0.8, 0.1, 0.3, 0.8, 0.7]),
        ('singing', [0.6, 0.5, 0.6, 0.6, 0.4, 0.5, 0.6, 0.5]),
        ('narrator', [0.5, 0.3, 0.5, 0.4, 0.4, 0.8, 0.5, 0.3]),
        ('child', [0.7, 0.6, 0.7, 0.6, 0.3, 0.4, 0.6, 0.5]),
        ('elderly', [0.3, 0.4, 0.3, 0.2, 0.6, 0.8, 0.4, 0.3]),
    ]
    
    # Create base voices
    for name, characteristics in voice_types:
        embedding = np.zeros(64)
        embedding[:8] = characteristics
        # Fill remaining dimensions with random values
        embedding[8:] = np.random.normal(0, 0.1, 56)
        voice_database[name] = embedding
    
    # Create additional random voices
    for i in range(num_voices - len(voice_types)):
        voice_name = f"voice_{i+1}"
        embedding = np.random.normal(0, 0.5, 64)
        # Ensure reasonable voice characteristics
        embedding[0] = np.clip(embedding[0], 0.2, 0.8)  # Pitch
        embedding[1] = np.clip(embedding[1], 0.1, 0.6)  # Pitch variation
        embedding[2] = np.clip(embedding[2], 0.2, 0.8)  # Spectral centroid
        embedding[3] = np.clip(embedding[3], 0.2, 0.8)  # Spectral rolloff
        embedding[4] = np.clip(embedding[4], 0.1, 0.7)  # Zero crossing rate
        embedding[5] = np.clip(embedding[5], 0.3, 0.8)  # Duration factor
        embedding[6] = np.clip(embedding[6], 0.2, 0.8)  # MFCC mean
        embedding[7] = np.clip(embedding[7], 0.1, 0.5)  # MFCC std
        voice_database[voice_name] = embedding
    
    print(f"Created {len(voice_database)} voices")
    return voice_database


def setup_enhanced_synthesizer(voice_database: Dict[str, np.ndarray]) -> EnhancedSynthesizer:
    """Setup the enhanced synthesizer."""
    print("Setting up enhanced synthesizer...")
    
    config = {
        'sample_rate': 22050,
        'n_mels': 80,
        'hop_length': 256,
        'win_length': 1024,
        'embedding_dim': 64,
    }
    
    # Create synthesizer
    synthesizer = EnhancedSynthesizer(config)
    
    # Initialize models
    synthesizer.initialize_models(
        acoustic_architecture="fastpitch",
        vocoder_architecture="hifigan",
    )
    
    # Load voice database
    synthesizer.load_voice_database(voice_database)
    
    print("Enhanced synthesizer setup complete")
    return synthesizer


def create_voice_families(synthesizer: EnhancedSynthesizer, num_families: int = 5) -> Dict[str, List[str]]:
    """Create voice families from existing voices."""
    print(f"Creating {num_families} voice families...")
    
    families = {}
    voice_ids = list(synthesizer.voice_database.keys())
    
    for i in range(num_families):
        parent_voice = voice_ids[i % len(voice_ids)]
        family_ids = synthesizer.create_voice_family(
            parent_voice, family_size=3, variation_scale=0.2
        )
        families[parent_voice] = family_ids
    
    print(f"Created {len(families)} voice families")
    return families


def analyze_voice_database(voice_database: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze the voice database."""
    print("Analyzing voice database...")
    
    analyzer = VoiceAnalyzer(embedding_dim=64)
    analysis = analyzer.analyze_voice_database(voice_database)
    
    return analysis


def create_voice_space_visualization(voice_database: Dict[str, np.ndarray]) -> None:
    """Create voice space visualization."""
    print("Creating voice space visualization...")
    
    interpolator = VoiceInterpolator(embedding_dim=64)
    
    # Create visualization
    interpolator.visualize_voice_space(
        voice_database,
        method="tsne",
        save_path="output/voice_space_visualization.png"
    )
    
    print("Voice space visualization saved")


def test_voice_interpolation(synthesizer: EnhancedSynthesizer) -> None:
    """Test voice interpolation functionality."""
    print("Testing voice interpolation...")
    
    voice_ids = list(synthesizer.voice_database.keys())
    
    if len(voice_ids) >= 2:
        # Test basic interpolation
        voice1_id = voice_ids[0]
        voice2_id = voice_ids[1]
        
        # Create continuum
        continuum_results = synthesizer.create_voice_continuum(
            voice1_id, voice2_id, "Hello, this is a voice interpolation test.", num_steps=5
        )
        
        print(f"  Created voice continuum with {len(continuum_results)} steps")
        
        # Test voice morphing
        if len(voice_ids) >= 3:
            source_voices = [voice_ids[0], voice_ids[1]]
            target_voice = voice_ids[2]
            
            morphing_results = synthesizer.morph_voice(
                source_voices, target_voice, "Hello, this is a voice morphing test.", morphing_steps=3
            )
            
            print(f"  Created voice morphing with {len(morphing_results)} steps")
    
    print("Voice interpolation testing complete")


def test_voice_quality(synthesizer: EnhancedSynthesizer) -> None:
    """Test voice quality assessment."""
    print("Testing voice quality assessment...")
    
    # Test voice similarity
    voice_ids = list(synthesizer.voice_database.keys())
    
    if len(voice_ids) >= 2:
        similarity = synthesizer.get_voice_similarity(voice_ids[0], voice_ids[1])
        print(f"  Voice similarity: {similarity:.4f}")
        
        # Find similar voices
        similar_voices = synthesizer.find_similar_voices(voice_ids[0], top_k=3)
        print(f"  Found {len(similar_voices)} similar voices")
    
    print("Voice quality testing complete")


def save_setup_results(voice_database: Dict[str, np.ndarray], 
                      families: Dict[str, List[str]], 
                      analysis: Dict[str, Any]) -> None:
    """Save setup results to files."""
    print("Saving setup results...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save voice database
    voice_database_path = output_dir / "voice_database.json"
    voice_database_serializable = {
        k: v.tolist() for k, v in voice_database.items()
    }
    with open(voice_database_path, 'w') as f:
        json.dump(voice_database_serializable, f, indent=2)
    
    # Save voice families
    families_path = output_dir / "voice_families.json"
    with open(families_path, 'w') as f:
        json.dump(families, f, indent=2)
    
    # Save analysis results
    analysis_path = output_dir / "voice_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Setup results saved to {output_dir}")


def main():
    """Main setup function."""
    print("🚀 Hybrid TTS System Setup")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create voice database
    voice_database = create_voice_database(num_voices=20)
    
    # Setup enhanced synthesizer
    synthesizer = setup_enhanced_synthesizer(voice_database)
    
    # Create voice families
    families = create_voice_families(synthesizer, num_families=5)
    
    # Analyze voice database
    analysis = analyze_voice_database(voice_database)
    
    # Create voice space visualization
    create_voice_space_visualization(voice_database)
    
    # Test functionality
    test_voice_interpolation(synthesizer)
    test_voice_quality(synthesizer)
    
    # Save results
    save_setup_results(voice_database, families, analysis)
    
    setup_time = time.time() - start_time
    
    print(f"\n✅ Hybrid TTS system setup complete!")
    print(f"⏱️ Setup time: {setup_time:.2f} seconds")
    print(f"🎭 Voice database: {len(voice_database)} voices")
    print(f"👨‍👩‍👧‍👦 Voice families: {len(families)} families")
    print(f"📊 Quality score: {analysis.get('quality_scores', {}).get('overall_quality', 0):.4f}")
    
    print("\n🎯 System is ready for use!")
    print("Run the following commands to test the system:")
    print("  python test_hybrid_system.py")
    print("  python scripts/voice_interpolation_demo.py")
    print("  python scripts/model_comparison.py")
    print("  python scripts/voice_quality_test.py")
    print("  python scripts/voice_space_explorer.py")


if __name__ == "__main__":
    main()








































