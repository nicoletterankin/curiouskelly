#!/usr/bin/env python3
"""
Demo script for the Synthetic Digital TTS System.

This script demonstrates the capabilities of the TTS system by generating
various voice samples and comparisons.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from synthesis.inference import InferenceEngine


def main():
    """Run the TTS demo."""
    parser = argparse.ArgumentParser(description="TTS System Demo")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--config", default="config/character_voice.json", help="Config file")
    parser.add_argument("--output-dir", default="demo_output", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    print("Initializing TTS system...")
    engine = InferenceEngine(
        model_dir=args.model_dir,
        config_path=args.config,
        device=args.device,
    )
    
    # Demo texts
    demo_texts = [
        "Hello, I'm a synthetic digital voice created entirely by artificial intelligence.",
        "I can express different emotions like happiness, sadness, anger, and excitement.",
        "The future of voice synthesis is here, and it's completely vendorless and offline.",
        "What would you like me to say next?",
    ]
    
    # Voice archetypes to demonstrate
    voice_archetypes = [
        "young_female",
        "young_male", 
        "mature_female",
        "mature_male",
        "child",
        "elderly",
    ]
    
    # Emotions to demonstrate
    emotions = [
        "neutral",
        "happy",
        "sad",
        "angry",
        "excited",
        "calm",
        "question",
    ]
    
    print(f"Creating demo in: {args.output_dir}")
    
    # Create voice comparison
    print("\n1. Creating voice comparison...")
    engine.compare_voices(
        text="This is a demonstration of different synthetic voice archetypes.",
        voice_archetypes=voice_archetypes,
        emotion="neutral",
        output_dir=os.path.join(args.output_dir, "voice_comparison"),
    )
    
    # Create emotional speech
    print("\n2. Creating emotional speech...")
    engine.generate_emotional_speech(
        text="I can express many different emotions through my voice.",
        voice_archetype="young_female",
        output_dir=os.path.join(args.output_dir, "emotional_speech"),
    )
    
    # Create voice variants
    print("\n3. Creating voice variants...")
    engine.create_voice_variants(
        text="Each voice variant has slightly different characteristics.",
        n_variants=5,
        emotion="neutral",
        output_dir=os.path.join(args.output_dir, "voice_variants"),
    )
    
    # Create comprehensive demo
    print("\n4. Creating comprehensive demo...")
    engine.create_voice_demo(
        demo_texts=demo_texts,
        voice_archetypes=voice_archetypes[:3],  # Use first 3 for demo
        emotions=emotions[:4],  # Use first 4 for demo
        output_dir=os.path.join(args.output_dir, "comprehensive_demo"),
    )
    
    # Run quality test
    print("\n5. Running quality test...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "Peter Piper picked a peck of pickled peppers.",
    ]
    
    quality_results = engine.test_synthesis_quality(
        test_texts=test_texts,
        output_dir=os.path.join(args.output_dir, "quality_test"),
    )
    
    # Display results
    print(f"\nDemo complete! Generated files in: {args.output_dir}")
    print(f"Voice comparison: {len(voice_archetypes)} files")
    print(f"Emotional speech: {len(emotions)} files")
    print(f"Voice variants: 5 files")
    print(f"Comprehensive demo: {len(demo_texts) * 3 * 4} files")
    print(f"Quality test: {len(quality_results['generated_files'])} files")
    
    if quality_results['errors']:
        print(f"Quality test errors: {len(quality_results['errors'])}")
    
    # Display voice info
    print("\nVoice system information:")
    voice_info = engine.get_voice_info()
    print(f"Available emotions: {voice_info['available_emotions']}")
    print(f"Sample rate: {voice_info['synthesis_info']['sample_rate']} Hz")
    print(f"Device: {voice_info['synthesis_info']['device']}")


if __name__ == "__main__":
    main()









































