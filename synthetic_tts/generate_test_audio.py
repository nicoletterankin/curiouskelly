#!/usr/bin/env python3
"""
Test Audio Generation Script for Hybrid TTS System
Generates comprehensive test audio samples to demonstrate all features
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_audio_samples():
    """Generate comprehensive test audio samples"""
    
    print("🎵 Generating Test Audio Samples for Hybrid TTS System")
    print("=" * 60)
    
    # Create output directories
    output_dir = Path("test_audio_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different test types
    test_dirs = {
        'basic': output_dir / "basic_synthesis",
        'interpolation': output_dir / "voice_interpolation", 
        'morphing': output_dir / "voice_morphing",
        'continuum': output_dir / "voice_continuum",
        'family': output_dir / "voice_family",
        'analysis': output_dir / "voice_analysis",
        'quality': output_dir / "quality_tests"
    }
    
    for test_dir in test_dirs.values():
        test_dir.mkdir(exist_ok=True)
    
    # Test texts for different scenarios
    test_texts = {
        'basic': [
            "Hello, this is a basic synthesis test.",
            "The quick brown fox jumps over the lazy dog.",
            "Welcome to the hybrid TTS system demonstration."
        ],
        'interpolation': [
            "This is voice interpolation between two different voices.",
            "Smooth transitions create natural voice morphing.",
            "Real-time voice switching is now possible."
        ],
        'morphing': [
            "Voice morphing transforms one voice into another.",
            "Advanced voice transformation capabilities.",
            "Seamless voice character changes."
        ],
        'continuum': [
            "Voice continuum creates smooth voice transitions.",
            "Navigate through the voice space effortlessly.",
            "Discover new voice characteristics."
        ],
        'family': [
            "Voice family members share similar characteristics.",
            "Generate related voice variations automatically.",
            "Create voice clusters with common traits."
        ],
        'analysis': [
            "Voice analysis provides detailed characteristics.",
            "Quality metrics assess audio performance.",
            "Similarity measurement finds related voices."
        ]
    }
    
    # Generate basic synthesis samples
    print("\n📝 Generating Basic Synthesis Samples...")
    generate_basic_samples(test_dirs['basic'], test_texts['basic'])
    
    # Generate voice interpolation samples
    print("\n🔄 Generating Voice Interpolation Samples...")
    generate_interpolation_samples(test_dirs['interpolation'], test_texts['interpolation'])
    
    # Generate voice morphing samples
    print("\n🎭 Generating Voice Morphing Samples...")
    generate_morphing_samples(test_dirs['morphing'], test_texts['morphing'])
    
    # Generate voice continuum samples
    print("\n🗺️ Generating Voice Continuum Samples...")
    generate_continuum_samples(test_dirs['continuum'], test_texts['continuum'])
    
    # Generate voice family samples
    print("\n👨‍👩‍👧‍👦 Generating Voice Family Samples...")
    generate_family_samples(test_dirs['family'], test_texts['family'])
    
    # Generate voice analysis samples
    print("\n📊 Generating Voice Analysis Samples...")
    generate_analysis_samples(test_dirs['analysis'], test_texts['analysis'])
    
    # Generate quality test samples
    print("\n🔍 Generating Quality Test Samples...")
    generate_quality_samples(test_dirs['quality'])
    
    # Create summary report
    create_summary_report(output_dir, test_dirs)
    
    print(f"\n✅ Test audio generation complete!")
    print(f"📁 All samples saved to: {output_dir.absolute()}")
    print(f"📊 Check the summary report for details.")

def generate_basic_samples(output_dir, texts):
    """Generate basic synthesis samples"""
    for i, text in enumerate(texts):
        filename = f"basic_sample_{i+1}.wav"
        filepath = output_dir / filename
        
        # Create a simple audio file (placeholder for actual synthesis)
        create_placeholder_audio(filepath, text, duration=3.0)
        print(f"  ✓ Created: {filename}")

def generate_interpolation_samples(output_dir, texts):
    """Generate voice interpolation samples"""
    interpolation_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for i, text in enumerate(texts):
        for j, weight in enumerate(interpolation_weights):
            filename = f"interpolation_{i+1}_weight_{weight:.2f}.wav"
            filepath = output_dir / filename
            
            # Create placeholder audio with interpolation metadata
            create_placeholder_audio(filepath, text, duration=3.0, 
                                  metadata={'interpolation_weight': weight})
            print(f"  ✓ Created: {filename}")

def generate_morphing_samples(output_dir, texts):
    """Generate voice morphing samples"""
    morph_steps = [0, 1, 2, 3, 4, 5]
    
    for i, text in enumerate(texts):
        for step in morph_steps:
            filename = f"morphing_{i+1}_step_{step}.wav"
            filepath = output_dir / filename
            
            # Create placeholder audio with morphing metadata
            create_placeholder_audio(filepath, text, duration=3.0,
                                  metadata={'morph_step': step, 'total_steps': 5})
            print(f"  ✓ Created: {filename}")

def generate_continuum_samples(output_dir, texts):
    """Generate voice continuum samples"""
    continuum_steps = 10
    
    for i, text in enumerate(texts):
        for step in range(continuum_steps):
            filename = f"continuum_{i+1}_step_{step:02d}.wav"
            filepath = output_dir / filename
            
            # Create placeholder audio with continuum metadata
            create_placeholder_audio(filepath, text, duration=3.0,
                                  metadata={'continuum_step': step, 'total_steps': continuum_steps})
            print(f"  ✓ Created: {filename}")

def generate_family_samples(output_dir, texts):
    """Generate voice family samples"""
    family_sizes = [3, 5, 8]
    
    for i, text in enumerate(texts):
        for size in family_sizes:
            for member in range(size):
                filename = f"family_{i+1}_size_{size}_member_{member+1}.wav"
                filepath = output_dir / filename
                
                # Create placeholder audio with family metadata
                create_placeholder_audio(filepath, text, duration=3.0,
                                      metadata={'family_size': size, 'member': member+1})
                print(f"  ✓ Created: {filename}")

def generate_analysis_samples(output_dir, texts):
    """Generate voice analysis samples"""
    for i, text in enumerate(texts):
        filename = f"analysis_{i+1}.wav"
        filepath = output_dir / filename
        
        # Create placeholder audio with analysis metadata
        create_placeholder_audio(filepath, text, duration=3.0,
                              metadata={'analysis_type': 'comprehensive'})
        print(f"  ✓ Created: {filename}")

def generate_quality_samples(output_dir):
    """Generate quality test samples"""
    quality_tests = [
        "Low quality test sample",
        "Medium quality test sample", 
        "High quality test sample",
        "Noise reduction test",
        "Echo cancellation test",
        "Dynamic range test"
    ]
    
    for i, test in enumerate(quality_tests):
        filename = f"quality_test_{i+1}.wav"
        filepath = output_dir / filename
        
        # Create placeholder audio with quality metadata
        create_placeholder_audio(filepath, test, duration=2.0,
                              metadata={'quality_test': test})
        print(f"  ✓ Created: {filename}")

def create_placeholder_audio(filepath, text, duration=3.0, metadata=None):
    """Create a placeholder audio file with metadata"""
    try:
        import wave
        import struct
        
        # Create a simple sine wave as placeholder
        sample_rate = 22050
        samples = int(sample_rate * duration)
        
        # Generate a simple tone with some variation
        frequency = 440 + hash(text) % 200  # Vary frequency based on text
        t = np.linspace(0, duration, samples)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Add some envelope to make it more natural
        envelope = np.exp(-t * 2)  # Decay envelope
        audio_data *= envelope
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(str(filepath), 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Create metadata file
        if metadata:
            metadata_file = filepath.with_suffix('.json')
            metadata['text'] = text
            metadata['duration'] = duration
            metadata['sample_rate'] = sample_rate
            metadata['frequency'] = frequency
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    except ImportError:
        # Fallback: create empty file if wave module not available
        with open(filepath, 'w') as f:
            f.write(f"Placeholder audio file for: {text}")

def create_summary_report(output_dir, test_dirs):
    """Create a summary report of generated samples"""
    report_file = output_dir / "test_audio_summary.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Test Audio Generation Summary\n\n")
        f.write("This report summarizes all generated test audio samples.\n\n")
        
        f.write("## Generated Samples by Category\n\n")
        
        for category, test_dir in test_dirs.items():
            f.write(f"### {category.replace('_', ' ').title()}\n\n")
            
            # Count files in directory
            audio_files = list(test_dir.glob("*.wav"))
            metadata_files = list(test_dir.glob("*.json"))
            
            f.write(f"- **Directory**: `{test_dir.name}/`\n")
            f.write(f"- **Audio Files**: {len(audio_files)}\n")
            f.write(f"- **Metadata Files**: {len(metadata_files)}\n\n")
            
            if audio_files:
                f.write("**Sample Files:**\n")
                for audio_file in sorted(audio_files):
                    f.write(f"- `{audio_file.name}`\n")
                f.write("\n")
        
        f.write("## Usage Instructions\n\n")
        f.write("1. **Basic Synthesis**: Use `synthesize_speech.py` with different voice IDs\n")
        f.write("2. **Voice Interpolation**: Use `synthesize_speech_enhanced.py` with interpolation parameters\n")
        f.write("3. **Voice Morphing**: Use the morphing features in the enhanced synthesizer\n")
        f.write("4. **Voice Continuum**: Explore the voice space with continuum generation\n")
        f.write("5. **Voice Family**: Generate related voice variations\n")
        f.write("6. **Voice Analysis**: Analyze voice characteristics and quality\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Install required dependencies: `pip install -r requirements.txt`\n")
        f.write("2. Run the setup script: `python setup_hybrid_system.py`\n")
        f.write("3. Train models with: `python train_models.py`\n")
        f.write("4. Generate real audio with: `python synthesize_speech_enhanced.py`\n")
        f.write("5. Explore voice features with the demonstration scripts\n\n")
        
        f.write("## File Structure\n\n")
        f.write("```\n")
        f.write(f"{output_dir.name}/\n")
        for category, test_dir in test_dirs.items():
            f.write(f"├── {test_dir.name}/\n")
            audio_files = list(test_dir.glob("*.wav"))
            for audio_file in sorted(audio_files)[:3]:  # Show first 3 files
                f.write(f"│   ├── {audio_file.name}\n")
            if len(audio_files) > 3:
                f.write(f"│   └── ... ({len(audio_files)-3} more files)\n")
        f.write("└── test_audio_summary.md\n")
        f.write("```\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate test audio samples for Hybrid TTS System")
    parser.add_argument("--output-dir", default="test_audio_output", 
                       help="Output directory for test audio files")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose mode enabled")
    
    create_test_audio_samples()

if __name__ == "__main__":
    main()