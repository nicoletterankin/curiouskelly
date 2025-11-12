#!/usr/bin/env python3
"""
Enhanced synthesis script for the Hybrid TTS System.

This script provides a comprehensive command-line interface for synthesizing speech
with voice interpolation, morphing, and advanced features.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
import torchaudio
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from synthesis.enhanced_synthesizer import EnhancedSynthesizer
from models.model_factory import ModelFactory
from voice.voice_interpolator import VoiceInterpolator
from voice.voice_analyzer import VoiceAnalyzer
from utils.voice_utils import VoiceUtils


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced speech synthesis with voice interpolation and morphing"
    )
    
    # Required arguments
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    
    # Basic synthesis arguments
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        choices=["neutral", "happy", "sad", "angry", "excited", "calm", "question"],
        help="Target emotion for synthesis"
    )
    
    parser.add_argument(
        "--voice-id",
        type=str,
        help="Voice ID from voice database"
    )
    
    parser.add_argument(
        "--custom-voice",
        action="store_true",
        help="Use custom voice characteristics"
    )
    
    # Voice interpolation arguments
    parser.add_argument(
        "--interpolation-target",
        type=str,
        help="Target voice for interpolation"
    )
    
    parser.add_argument(
        "--interpolation-weight",
        type=float,
        default=0.5,
        help="Interpolation weight (0-1)"
    )
    
    parser.add_argument(
        "--interpolation-method",
        type=str,
        default="linear",
        choices=["linear", "spherical", "weighted", "pca_based", "gaussian", "spline"],
        help="Interpolation method"
    )
    
    # Voice morphing arguments
    parser.add_argument(
        "--morph-voices",
        nargs="+",
        help="Source voices for morphing"
    )
    
    parser.add_argument(
        "--morph-target",
        type=str,
        help="Target voice for morphing"
    )
    
    parser.add_argument(
        "--morph-steps",
        type=int,
        default=5,
        help="Number of morphing steps"
    )
    
    # Model architecture arguments
    parser.add_argument(
        "--acoustic-model",
        type=str,
        default="fastpitch",
        choices=["fastpitch", "tacotron2"],
        help="Acoustic model architecture"
    )
    
    parser.add_argument(
        "--vocoder",
        type=str,
        default="hifigan",
        choices=["hifigan", "waveglow"],
        help="Vocoder architecture"
    )
    
    # Voice characteristics
    parser.add_argument(
        "--pitch-mean",
        type=float,
        default=0.0,
        help="Mean pitch offset (-1 to 1)"
    )
    
    parser.add_argument(
        "--pitch-variability",
        type=float,
        default=0.0,
        help="Pitch variability (-1 to 1)"
    )
    
    parser.add_argument(
        "--timbre-brightness",
        type=float,
        default=0.0,
        help="Timbre brightness (-1 to 1)"
    )
    
    parser.add_argument(
        "--timbre-warmth",
        type=float,
        default=0.0,
        help="Timbre warmth (-1 to 1)"
    )
    
    parser.add_argument(
        "--energy-level",
        type=float,
        default=0.0,
        help="Energy level (-1 to 1)"
    )
    
    # Prosody control
    parser.add_argument(
        "--pitch-shift",
        type=float,
        default=0.0,
        help="Pitch shift (-1 to 1)"
    )
    
    parser.add_argument(
        "--rate-scale",
        type=float,
        default=1.0,
        help="Speech rate scale (0.5 to 2.0)"
    )
    
    parser.add_argument(
        "--energy-scale",
        type=float,
        default=1.0,
        help="Energy scale (0.5 to 2.0)"
    )
    
    # Voice continuum arguments
    parser.add_argument(
        "--create-continuum",
        action="store_true",
        help="Create voice continuum between two voices"
    )
    
    parser.add_argument(
        "--continuum-voice1",
        type=str,
        help="First voice for continuum"
    )
    
    parser.add_argument(
        "--continuum-voice2",
        type=str,
        help="Second voice for continuum"
    )
    
    parser.add_argument(
        "--continuum-steps",
        type=int,
        default=10,
        help="Number of continuum steps"
    )
    
    # Voice family arguments
    parser.add_argument(
        "--create-family",
        action="store_true",
        help="Create voice family from parent voice"
    )
    
    parser.add_argument(
        "--family-parent",
        type=str,
        help="Parent voice for family creation"
    )
    
    parser.add_argument(
        "--family-size",
        type=int,
        default=5,
        help="Number of family members to create"
    )
    
    parser.add_argument(
        "--variation-scale",
        type=float,
        default=0.2,
        help="Variation scale for family creation"
    )
    
    # Analysis arguments
    parser.add_argument(
        "--analyze-voice",
        action="store_true",
        help="Analyze voice characteristics"
    )
    
    parser.add_argument(
        "--find-similar",
        type=str,
        help="Find similar voices to specified voice"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for voice search"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for multiple files"
    )
    
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save synthesis metadata"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def create_enhanced_synthesizer(args) -> EnhancedSynthesizer:
    """Create and configure the enhanced synthesizer."""
    print("Initializing enhanced synthesizer...")
    
    # Configuration
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
        acoustic_architecture=args.acoustic_model,
        vocoder_architecture=args.vocoder,
    )
    
    # Load voice database if available
    voice_database_path = "voice_database.json"
    if os.path.exists(voice_database_path):
        synthesizer.load_voice_database(voice_database_path)
        print(f"Loaded voice database with {len(synthesizer.voice_database)} voices")
    else:
        print("No voice database found, using default voices")
        # Create default voice database
        default_voices = create_default_voice_database()
        synthesizer.load_voice_database(default_voices)
    
    # Set interpolation method
    synthesizer.set_interpolation_method(args.interpolation_method)
    
    return synthesizer


def create_default_voice_database() -> Dict[str, np.ndarray]:
    """Create a default voice database."""
    voice_database = {}
    
    # Create diverse default voices
    voice_types = [
        ('high_pitch_bright', [0.8, 0.3, 0.7, 0.6, 0.2, 0.5, 0.4, 0.3]),
        ('low_pitch_warm', [0.3, 0.4, 0.4, 0.3, 0.6, 0.7, 0.6, 0.5]),
        ('medium_balanced', [0.5, 0.5, 0.5, 0.5, 0.4, 0.6, 0.5, 0.4]),
        ('energetic', [0.7, 0.6, 0.8, 0.7, 0.3, 0.4, 0.7, 0.6]),
        ('calm', [0.4, 0.2, 0.3, 0.2, 0.5, 0.8, 0.3, 0.2]),
    ]
    
    for name, characteristics in voice_types:
        embedding = np.zeros(64)
        embedding[:8] = characteristics
        # Fill remaining dimensions with random values
        embedding[8:] = np.random.normal(0, 0.1, 56)
        voice_database[name] = embedding
    
    return voice_database


def create_custom_voice_embedding(args) -> np.ndarray:
    """Create custom voice embedding from arguments."""
    embedding = np.zeros(64)
    
    # Set custom characteristics
    embedding[0] = args.pitch_mean
    embedding[1] = args.pitch_variability
    embedding[2] = args.timbre_brightness
    embedding[3] = args.timbre_warmth
    embedding[4] = args.energy_level
    embedding[5] = 0.5  # Default duration factor
    embedding[6] = 0.5  # Default MFCC mean
    embedding[7] = 0.3  # Default MFCC std
    
    # Fill remaining dimensions with random values
    embedding[8:] = np.random.normal(0, 0.1, 56)
    
    return embedding


def synthesize_speech(synthesizer: EnhancedSynthesizer, args) -> Dict[str, Any]:
    """Synthesize speech with the given arguments."""
    print(f"Synthesizing: '{args.text}'")
    
    # Determine voice embedding
    voice_embedding = None
    voice_id = None
    
    if args.voice_id and args.voice_id in synthesizer.voice_database:
        voice_id = args.voice_id
        print(f"Using voice: {voice_id}")
    elif args.custom_voice:
        voice_embedding = create_custom_voice_embedding(args)
        print("Using custom voice characteristics")
    else:
        # Use default voice
        voice_id = list(synthesizer.voice_database.keys())[0]
        print(f"Using default voice: {voice_id}")
    
    # Create prosody control
    prosody_control = {
        'pitch_shift': args.pitch_shift,
        'rate_scale': args.rate_scale,
        'energy_scale': args.energy_scale,
    }
    
    # Synthesize speech
    result = synthesizer.synthesize_speech(
        text=args.text,
        voice_id=voice_id,
        voice_embedding=voice_embedding,
        emotion=args.emotion,
        prosody_control=prosody_control,
        interpolation_target=args.interpolation_target,
        interpolation_weight=args.interpolation_weight,
    )
    
    return result


def create_voice_continuum(synthesizer: EnhancedSynthesizer, args) -> List[Dict[str, Any]]:
    """Create voice continuum between two voices."""
    print(f"Creating voice continuum: {args.continuum_voice1} -> {args.continuum_voice2}")
    
    continuum_results = synthesizer.create_voice_continuum(
        args.continuum_voice1,
        args.continuum_voice2,
        args.text,
        num_steps=args.continuum_steps
    )
    
    return continuum_results


def create_voice_morphing(synthesizer: EnhancedSynthesizer, args) -> List[Dict[str, Any]]:
    """Create voice morphing from source voices to target."""
    print(f"Creating voice morphing: {args.morph_voices} -> {args.morph_target}")
    
    morphing_results = synthesizer.morph_voice(
        args.morph_voices,
        args.morph_target,
        args.text,
        morphing_steps=args.morph_steps
    )
    
    return morphing_results


def create_voice_family(synthesizer: EnhancedSynthesizer, args) -> List[str]:
    """Create voice family from parent voice."""
    print(f"Creating voice family from: {args.family_parent}")
    
    family_ids = synthesizer.create_voice_family(
        args.family_parent,
        family_size=args.family_size,
        variation_scale=args.variation_scale
    )
    
    return family_ids


def analyze_voice(synthesizer: EnhancedSynthesizer, voice_id: str) -> Dict[str, Any]:
    """Analyze voice characteristics."""
    print(f"Analyzing voice: {voice_id}")
    
    if voice_id not in synthesizer.voice_database:
        raise ValueError(f"Voice ID not found: {voice_id}")
    
    voice_embedding = synthesizer.voice_database[voice_id]
    
    # Create voice analyzer
    voice_analyzer = VoiceAnalyzer()
    
    # Analyze voice
    analysis = voice_analyzer.analyze_voice_characteristics(voice_embedding)
    
    return analysis


def find_similar_voices(synthesizer: EnhancedSynthesizer, voice_id: str, threshold: float) -> List[Tuple[str, float]]:
    """Find similar voices."""
    print(f"Finding voices similar to: {voice_id}")
    
    similar_voices = synthesizer.find_similar_voices(
        voice_id, top_k=5, similarity_threshold=threshold
    )
    
    return similar_voices


def save_results(results: List[Dict[str, Any]], output_dir: str, prefix: str = "synthesis") -> None:
    """Save synthesis results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for i, result in enumerate(results):
        # Save audio
        audio_path = output_path / f"{prefix}_{i:03d}.wav"
        torchaudio.save(
            str(audio_path),
            torch.tensor(result['audio'], dtype=torch.float32).unsqueeze(0),
            22050
        )
        
        # Save metadata
        metadata_path = output_path / f"{prefix}_{i:03d}_metadata.json"
        metadata = {
            'text': result.get('text', ''),
            'voice_id': result.get('voice_id', ''),
            'emotion': result.get('emotion', ''),
            'synthesis_time': result.get('synthesis_time', 0),
            'interpolation_target': result.get('interpolation_target', ''),
            'interpolation_weight': result.get('interpolation_weight', 0),
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved: {audio_path}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create enhanced synthesizer
    synthesizer = create_enhanced_synthesizer(args)
    
    # Handle different synthesis modes
    if args.create_continuum:
        # Create voice continuum
        results = create_voice_continuum(synthesizer, args)
        
        if args.output_dir:
            save_results(results, args.output_dir, "continuum")
        else:
            # Save first result as main output
            torchaudio.save(
                args.output,
                torch.tensor(results[0]['audio'], dtype=torch.float32).unsqueeze(0),
                22050
            )
            print(f"Continuum created. Main output saved to: {args.output}")
    
    elif args.morph_voices and args.morph_target:
        # Create voice morphing
        results = create_voice_morphing(synthesizer, args)
        
        if args.output_dir:
            save_results(results, args.output_dir, "morphing")
        else:
            # Save first result as main output
            torchaudio.save(
                args.output,
                torch.tensor(results[0]['audio'], dtype=torch.float32).unsqueeze(0),
                22050
            )
            print(f"Morphing created. Main output saved to: {args.output}")
    
    elif args.create_family:
        # Create voice family
        family_ids = create_voice_family(synthesizer, args)
        
        # Synthesize samples from family
        results = []
        for family_id in family_ids:
            result = synthesizer.synthesize_speech(args.text, voice_id=family_id)
            results.append(result)
        
        if args.output_dir:
            save_results(results, args.output_dir, "family")
        else:
            # Save first result as main output
            torchaudio.save(
                args.output,
                torch.tensor(results[0]['audio'], dtype=torch.float32).unsqueeze(0),
                22050
            )
            print(f"Voice family created. Main output saved to: {args.output}")
    
    elif args.analyze_voice:
        # Analyze voice
        if not args.voice_id:
            print("Error: --voice-id required for voice analysis")
            sys.exit(1)
        
        analysis = analyze_voice(synthesizer, args.voice_id)
        
        # Save analysis
        analysis_path = args.output.replace('.wav', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Voice analysis saved to: {analysis_path}")
    
    elif args.find_similar:
        # Find similar voices
        similar_voices = find_similar_voices(synthesizer, args.find_similar, args.similarity_threshold)
        
        print("Similar voices:")
        for voice_id, similarity in similar_voices:
            print(f"  {voice_id}: {similarity:.4f}")
    
    else:
        # Regular synthesis
        result = synthesize_speech(synthesizer, args)
        
        # Save audio
        torchaudio.save(
            args.output,
            torch.tensor(result['audio'], dtype=torch.float32).unsqueeze(0),
            22050
        )
        
        print(f"Synthesis complete. Output saved to: {args.output}")
        
        # Save metadata if requested
        if args.save_metadata:
            metadata_path = args.output.replace('.wav', '_metadata.json')
            metadata = {
                'text': result.get('text', ''),
                'voice_id': result.get('voice_id', ''),
                'emotion': result.get('emotion', ''),
                'synthesis_time': result.get('synthesis_time', 0),
                'interpolation_target': result.get('interpolation_target', ''),
                'interpolation_weight': result.get('interpolation_weight', 0),
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to: {metadata_path}")
        
        # Display verbose info
        if args.verbose:
            print(f"Text length: {len(args.text)} characters")
            print(f"Emotion: {args.emotion}")
            print(f"Synthesis time: {result.get('synthesis_time', 0):.4f}s")
            
            if result.get('interpolation_target'):
                print(f"Interpolation: {result['interpolation_target']} (weight: {result['interpolation_weight']})")


if __name__ == "__main__":
    main()








































