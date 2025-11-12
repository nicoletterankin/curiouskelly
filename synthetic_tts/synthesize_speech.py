#!/usr/bin/env python3
"""
Main synthesis script for the Synthetic Digital TTS System.

This script provides a command-line interface for synthesizing speech
from text using the trained TTS models.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from synthesis.enhanced_synthesizer import EnhancedSynthesizer
from models.model_factory import ModelFactory
from voice.voice_interpolator import VoiceInterpolator
from voice.voice_analyzer import VoiceAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthesize speech using the Synthetic Digital TTS System"
    )
    
    # Required arguments
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    
    # Optional arguments
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        choices=["neutral", "happy", "sad", "angry", "excited", "calm", "question"],
        help="Target emotion for synthesis"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/character_voice.json",
        help="Path to character voice configuration"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained models"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--speaker-archetype",
        type=str,
        choices=[
            "young_female", "young_male", "mature_female", 
            "mature_male", "child", "elderly"
        ],
        help="Use a predefined speaker archetype"
    )
    
    parser.add_argument(
        "--custom-voice",
        action="store_true",
        help="Use custom voice characteristics"
    )
    
    parser.add_argument(
        "--pitch-mean",
        type=float,
        default=0.0,
        help="Mean pitch level (-1 to 1)"
    )
    
    parser.add_argument(
        "--pitch-range",
        type=float,
        default=0.5,
        help="Pitch variability (-1 to 1)"
    )
    
    parser.add_argument(
        "--timbre-brightness",
        type=float,
        default=0.0,
        help="Voice brightness (-1 to 1)"
    )
    
    parser.add_argument(
        "--timbre-warmth",
        type=float,
        default=0.0,
        help="Voice warmth (-1 to 1)"
    )
    
    parser.add_argument(
        "--energy-level",
        type=float,
        default=0.0,
        help="Overall energy level (-1 to 1)"
    )
    
    parser.add_argument(
        "--breathiness",
        type=float,
        default=0.0,
        help="Voice breathiness (-1 to 1)"
    )
    
    parser.add_argument(
        "--emphasis",
        type=str,
        help="Emphasis markers in format 'start:end:type,start:end:type'"
    )
    
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to text file for batch synthesis"
    )
    
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of voice variants to generate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch synthesis or variants"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def parse_emphasis_markers(emphasis_str: str) -> List[Tuple[int, int, str]]:
    """Parse emphasis markers from string format."""
    if not emphasis_str:
        return []
    
    markers = []
    for marker in emphasis_str.split(','):
        parts = marker.split(':')
        if len(parts) == 3:
            start, end, marker_type = parts
            markers.append((int(start), int(end), marker_type))
    
    return markers


def load_batch_texts(batch_file: str) -> List[str]:
    """Load texts from batch file."""
    with open(batch_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def create_custom_speaker_embedding(args) -> torch.Tensor:
    """Create custom speaker embedding from command line arguments."""
    voice_generator = SyntheticVoiceGenerator()
    
    return voice_generator.create_custom_voice(
        pitch_mean=args.pitch_mean,
        pitch_range=args.pitch_range,
        timbre_brightness=args.timbre_brightness,
        timbre_warmth=args.timbre_warmth,
        energy_level=args.energy_level,
        breathiness=args.breathiness,
    )


def create_archetype_speaker_embedding(archetype: str) -> torch.Tensor:
    """Create speaker embedding from archetype."""
    voice_generator = SyntheticVoiceGenerator()
    return voice_generator.create_voice_from_archetype(archetype)


def main():
    """Main synthesis function."""
    args = parse_arguments()
    
    # Validate arguments
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Initialize synthesizer
    try:
        synthesizer = Synthesizer(
            model_dir=args.model_dir,
            config_path=args.config,
            device=args.device,
        )
    except Exception as e:
        print(f"Error initializing synthesizer: {e}")
        sys.exit(1)
    
    if args.verbose:
        print("Synthesizer initialized successfully")
        print(f"Device: {synthesizer.device}")
        print(f"Sample rate: {synthesizer.sample_rate}")
        print(f"Available emotions: {synthesizer.get_available_emotions()}")
    
    # Determine speaker embedding
    speaker_embedding = None
    if args.speaker_archetype:
        speaker_embedding = create_archetype_speaker_embedding(args.speaker_archetype)
        if args.verbose:
            print(f"Using speaker archetype: {args.speaker_archetype}")
    elif args.custom_voice:
        speaker_embedding = create_custom_speaker_embedding(args)
        if args.verbose:
            print("Using custom voice characteristics")
    
    # Parse emphasis markers
    emphasis_markers = parse_emphasis_markers(args.emphasis)
    
    # Handle batch synthesis
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"Error: Batch file not found: {args.batch}")
            sys.exit(1)
        
        texts = load_batch_texts(args.batch)
        emotions = [args.emotion] * len(texts)
        speaker_embeddings = [speaker_embedding] * len(texts) if speaker_embedding else None
        
        # Create output directory
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Synthesize batch
        results = synthesizer.synthesize_batch(
            texts=texts,
            emotions=emotions,
            speaker_embeddings=speaker_embeddings,
            output_dir=args.output_dir,
        )
        
        print(f"Batch synthesis complete. Generated {len(results)} audio files.")
        if args.output_dir:
            print(f"Output directory: {args.output_dir}")
        
        return
    
    # Handle voice variants
    if args.variants > 1:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        variants = synthesizer.create_voice_variants(
            text=args.text,
            n_variants=args.variants,
            emotion=args.emotion,
            output_dir=args.output_dir,
        )
        
        print(f"Generated {len(variants)} voice variants.")
        if args.output_dir:
            print(f"Output directory: {args.output_dir}")
        
        return
    
    # Single synthesis
    try:
        result = synthesizer.synthesize(
            text=args.text,
            emotion=args.emotion,
            speaker_embedding=speaker_embedding,
            emphasis=emphasis_markers,
            output_path=args.output,
        )
        
        print(f"Synthesis complete. Output saved to: {args.output}")
        
        # Display synthesis info
        if args.verbose:
            info = synthesizer.get_synthesis_info()
            print(f"Text length: {len(args.text)} characters")
            print(f"Emotion: {args.emotion}")
            print(f"Output format: WAV, {info['sample_rate']} Hz")
            
            if emphasis_markers:
                print(f"Emphasis markers: {len(emphasis_markers)}")
    
    except Exception as e:
        print(f"Error during synthesis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
