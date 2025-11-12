#!/usr/bin/env python3
"""
Voice interpolation demonstration script.

This script demonstrates the voice interpolation capabilities of the
hybrid TTS system, showing voice morphing and continuum creation.
"""

import sys
import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.hybrid_data_generator import HybridDataGenerator
from voice.voice_interpolator import VoiceInterpolator
from synthesis.enhanced_synthesizer import EnhancedSynthesizer
from utils.voice_utils import VoiceUtils


def main():
    """Main demonstration function."""
    print("🎤 Voice Interpolation Demonstration")
    print("=" * 50)
    
    # Configuration
    config = {
        'sample_rate': 22050,
        'n_mels': 80,
        'hop_length': 256,
        'win_length': 1024,
        'embedding_dim': 64,
    }
    
    # Initialize components
    print("Initializing components...")
    hybrid_generator = HybridDataGenerator(config)
    voice_interpolator = VoiceInterpolator(embedding_dim=64)
    voice_utils = VoiceUtils()
    
    # Create enhanced synthesizer
    synthesizer = EnhancedSynthesizer(config)
    
    # Initialize models (mock for demonstration)
    print("Initializing TTS models...")
    synthesizer.initialize_models()
    
    # Generate or load voice database
    print("Setting up voice database...")
    voice_database = create_demo_voice_database()
    synthesizer.load_voice_database(voice_database)
    
    # Demonstration text
    demo_text = "Hello, this is a voice interpolation demonstration. Listen to how smoothly the voice changes."
    
    print(f"\n📝 Demo text: '{demo_text}'")
    print(f"🎭 Available voices: {list(voice_database.keys())}")
    
    # 1. Basic voice interpolation
    print("\n1️⃣ Basic Voice Interpolation")
    print("-" * 30)
    
    voice1_id = "voice_1"
    voice2_id = "voice_2"
    
    if voice1_id in voice_database and voice2_id in voice_database:
        print(f"Interpolating between {voice1_id} and {voice2_id}...")
        
        # Create voice continuum
        continuum_results = synthesizer.create_voice_continuum(
            voice1_id, voice2_id, demo_text, num_steps=5
        )
        
        print(f"Created voice continuum with {len(continuum_results)} steps")
        
        # Save continuum samples
        save_continuum_samples(continuum_results, "voice_continuum")
        
        # Analyze interpolation quality
        analyze_interpolation_quality(voice_database[voice1_id], voice_database[voice2_id], continuum_results)
    
    # 2. Voice morphing demonstration
    print("\n2️⃣ Voice Morphing Demonstration")
    print("-" * 30)
    
    source_voices = ["voice_1", "voice_2"]
    target_voice = "voice_3"
    
    if all(voice in voice_database for voice in source_voices + [target_voice]):
        print(f"Morphing from {source_voices} to {target_voice}...")
        
        morphing_results = synthesizer.morph_voice(
            source_voices, target_voice, demo_text, morphing_steps=5
        )
        
        print(f"Created voice morphing with {len(morphing_results)} steps")
        
        # Save morphing samples
        save_continuum_samples(morphing_results, "voice_morphing")
    
    # 3. Voice similarity analysis
    print("\n3️⃣ Voice Similarity Analysis")
    print("-" * 30)
    
    analyze_voice_similarities(voice_database, synthesizer)
    
    # 4. Voice family creation
    print("\n4️⃣ Voice Family Creation")
    print("-" * 30)
    
    parent_voice = "voice_1"
    if parent_voice in voice_database:
        print(f"Creating voice family from {parent_voice}...")
        
        family_ids = synthesizer.create_voice_family(
            parent_voice, family_size=5, variation_scale=0.2
        )
        
        print(f"Created voice family with {len(family_ids)} members")
        
        # Synthesize samples from family
        family_results = []
        for family_id in family_ids:
            result = synthesizer.synthesize_speech(
                demo_text, voice_id=family_id
            )
            family_results.append(result)
        
        # Save family samples
        save_continuum_samples(family_results, "voice_family")
    
    # 5. Real-time voice switching
    print("\n5️⃣ Real-time Voice Switching")
    print("-" * 30)
    
    demonstrate_voice_switching(synthesizer, demo_text)
    
    # 6. Voice space visualization
    print("\n6️⃣ Voice Space Visualization")
    print("-" * 30)
    
    visualize_voice_space(voice_database, voice_interpolator)
    
    # 7. Performance metrics
    print("\n7️⃣ Performance Metrics")
    print("-" * 30)
    
    metrics = synthesizer.get_performance_metrics()
    print_performance_metrics(metrics)
    
    print("\n✅ Voice interpolation demonstration complete!")
    print("Check the 'output' directory for generated audio samples.")


def create_demo_voice_database() -> dict:
    """Create a demo voice database for testing."""
    print("Creating demo voice database...")
    
    # Create synthetic voice embeddings
    voice_database = {}
    
    # Voice 1: High pitch, bright
    voice1 = np.array([
        0.8,  # High pitch
        0.3,  # Moderate pitch variation
        0.7,  # High spectral centroid (bright)
        0.6,  # High spectral rolloff
        0.2,  # Low zero crossing rate
        0.5,  # Medium duration
        0.4,  # MFCC mean
        0.3,  # MFCC std
    ] + [np.random.normal(0, 0.1) for _ in range(56)])  # Fill remaining dimensions
    
    # Voice 2: Low pitch, warm
    voice2 = np.array([
        0.3,  # Low pitch
        0.4,  # High pitch variation
        0.4,  # Low spectral centroid (warm)
        0.3,  # Low spectral rolloff
        0.6,  # High zero crossing rate
        0.7,  # Long duration
        0.6,  # MFCC mean
        0.5,  # MFCC std
    ] + [np.random.normal(0, 0.1) for _ in range(56)])
    
    # Voice 3: Medium pitch, balanced
    voice3 = np.array([
        0.5,  # Medium pitch
        0.5,  # Balanced pitch variation
        0.5,  # Balanced spectral centroid
        0.5,  # Balanced spectral rolloff
        0.4,  # Balanced zero crossing rate
        0.6,  # Medium duration
        0.5,  # MFCC mean
        0.4,  # MFCC std
    ] + [np.random.normal(0, 0.1) for _ in range(56)])
    
    voice_database = {
        "voice_1": voice1,
        "voice_2": voice2,
        "voice_3": voice3,
    }
    
    print(f"Created {len(voice_database)} demo voices")
    return voice_database


def save_continuum_samples(continuum_results: list, prefix: str) -> None:
    """Save continuum samples to files."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    for i, result in enumerate(continuum_results):
        # Save audio
        audio_path = output_dir / f"{prefix}_step_{i:02d}.wav"
        torchaudio.save(
            str(audio_path),
            torch.tensor(result['audio'], dtype=torch.float32).unsqueeze(0),
            22050
        )
        
        # Save metadata
        metadata_path = output_dir / f"{prefix}_step_{i:02d}_metadata.json"
        metadata = {
            'step': i,
            'weight': result.get('weight', i / (len(continuum_results) - 1)),
            'morphing_progress': result.get('morphing_progress', 0),
            'synthesis_time': result.get('synthesis_time', 0),
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(continuum_results)} samples with prefix '{prefix}'")


def analyze_interpolation_quality(voice1: np.ndarray, voice2: np.ndarray, continuum_results: list) -> None:
    """Analyze the quality of voice interpolation."""
    print("Analyzing interpolation quality...")
    
    # Calculate interpolation quality for each step
    qualities = []
    for i, result in enumerate(continuum_results):
        weight = i / (len(continuum_results) - 1)
        interpolated_voice = result.get('voice_embedding', np.zeros(64))
        
        # Calculate quality metrics
        distance_to_voice1 = np.linalg.norm(interpolated_voice - voice1)
        distance_to_voice2 = np.linalg.norm(interpolated_voice - voice2)
        
        # Expected distances
        expected_dist1 = (1 - weight) * np.linalg.norm(voice2 - voice1)
        expected_dist2 = weight * np.linalg.norm(voice2 - voice1)
        
        # Quality score
        quality = 1.0 - abs(distance_to_voice1 - expected_dist1) / expected_dist1
        qualities.append(quality)
    
    avg_quality = np.mean(qualities)
    print(f"Average interpolation quality: {avg_quality:.4f}")
    print(f"Quality range: {np.min(qualities):.4f} - {np.max(qualities):.4f}")


def analyze_voice_similarities(voice_database: dict, synthesizer: EnhancedSynthesizer) -> None:
    """Analyze similarities between voices."""
    print("Analyzing voice similarities...")
    
    voice_ids = list(voice_database.keys())
    similarities = []
    
    for i, voice1_id in enumerate(voice_ids):
        for j, voice2_id in enumerate(voice_ids[i+1:], i+1):
            similarity = synthesizer.get_voice_similarity(voice1_id, voice2_id)
            similarities.append((voice1_id, voice2_id, similarity))
            print(f"Similarity between {voice1_id} and {voice2_id}: {similarity:.4f}")
    
    if similarities:
        avg_similarity = np.mean([s[2] for s in similarities])
        print(f"Average voice similarity: {avg_similarity:.4f}")


def demonstrate_voice_switching(synthesizer: EnhancedSynthesizer, text: str) -> None:
    """Demonstrate real-time voice switching."""
    print("Demonstrating voice switching...")
    
    voice_ids = list(synthesizer.voice_database.keys())
    
    for voice_id in voice_ids:
        print(f"Switching to {voice_id}...")
        synthesizer.switch_voice(voice_id)
        
        # Synthesize with current voice
        result = synthesizer.synthesize_speech(text)
        
        # Save sample
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        audio_path = output_dir / f"voice_switch_{voice_id}.wav"
        torchaudio.save(
            str(audio_path),
            torch.tensor(result['audio'], dtype=torch.float32).unsqueeze(0),
            22050
        )
        
        print(f"Saved voice switch sample: {audio_path}")


def visualize_voice_space(voice_database: dict, voice_interpolator: VoiceInterpolator) -> None:
    """Visualize voice embeddings in 2D space."""
    print("Creating voice space visualization...")
    
    # Create visualization
    voice_interpolator.visualize_voice_space(
        voice_database,
        method="tsne",
        save_path="output/voice_space_visualization.png"
    )
    
    print("Voice space visualization saved to: output/voice_space_visualization.png")


def print_performance_metrics(metrics: dict) -> None:
    """Print performance metrics."""
    print("Performance Metrics:")
    print(f"  Voice switches: {metrics.get('voice_switches', 0)}")
    
    if 'avg_synthesis_time' in metrics:
        print(f"  Average synthesis time: {metrics['avg_synthesis_time']:.4f}s")
    
    if 'avg_interpolation_time' in metrics:
        print(f"  Average interpolation time: {metrics['avg_interpolation_time']:.4f}s")
    
    if 'synthesis_times' in metrics:
        print(f"  Total synthesis operations: {len(metrics['synthesis_times'])}")


if __name__ == "__main__":
    main()








































