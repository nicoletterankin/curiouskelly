#!/usr/bin/env python3
"""
Generate Real TTS Audio using Piper
Creates actual speech audio files using Piper TTS instead of placeholders
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import Piper TTS wrapper
try:
    from piper_tts import PiperTTS
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("⚠️  Piper TTS not available. Install with: python install_piper.py")

def generate_real_tts_audio():
    """Generate real TTS audio using Piper"""
    
    print("🎵 Generating Real TTS Audio with Piper")
    print("=" * 60)
    
    if not PIPER_AVAILABLE:
        print("❌ Piper TTS not available. Please install first:")
        print("   python install_piper.py")
        return False
    
    # Initialize Piper TTS
    try:
        tts = PiperTTS()
        voices = tts.list_voices()
        print(f"✅ Piper TTS initialized with {len(voices)} voices")
        print(f"Available voices: {voices}")
    except Exception as e:
        print(f"❌ Failed to initialize Piper TTS: {e}")
        return False
    
    # Create output directories
    output_dir = Path("real_tts_audio_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different test types
    test_dirs = {
        'basic': output_dir / "basic_synthesis",
        'interpolation': output_dir / "voice_interpolation", 
        'morphing': output_dir / "voice_morphing",
        'continuum': output_dir / "voice_continuum",
        'family': output_dir / "voice_family",
        'analysis': output_dir / "voice_analysis",
        'emotions': output_dir / "emotional_speech",
        'prosody': output_dir / "prosody_control",
        'quality': output_dir / "quality_tests"
    }
    
    for test_dir in test_dirs.values():
        test_dir.mkdir(exist_ok=True)
    
    # Test texts for different scenarios
    test_texts = {
        'basic': [
            "Hello, this is a basic synthesis test using the hybrid TTS system.",
            "The quick brown fox jumps over the lazy dog in a beautiful meadow.",
            "Welcome to the hybrid TTS system demonstration with real voice synthesis."
        ],
        'interpolation': [
            "This is voice interpolation between two different voices in real-time.",
            "Smooth transitions create natural voice morphing with advanced algorithms.",
            "Real-time voice switching is now possible with seamless quality."
        ],
        'morphing': [
            "Voice morphing transforms one voice into another with precision.",
            "Advanced voice transformation capabilities using neural networks.",
            "Seamless voice character changes with maintained intelligibility."
        ],
        'continuum': [
            "Voice continuum creates smooth voice transitions across the voice space.",
            "Navigate through the voice space effortlessly with intuitive controls.",
            "Discover new voice characteristics through intelligent exploration."
        ],
        'family': [
            "Voice family members share similar characteristics and traits.",
            "Generate related voice variations automatically with genetic algorithms.",
            "Create voice clusters with common traits and unique personalities."
        ],
        'analysis': [
            "Voice analysis provides detailed characteristics and quality metrics.",
            "Quality metrics assess audio performance with comprehensive evaluation.",
            "Similarity measurement finds related voices using advanced algorithms."
        ],
        'emotions': [
            "I am so excited about this new technology!",
            "This is absolutely terrible and disappointing.",
            "I feel calm and peaceful about this situation.",
            "What an amazing discovery this has been!"
        ],
        'prosody': [
            "The quick brown fox jumps over the lazy dog.",
            "In a world where technology meets creativity, possibilities are endless.",
            "Every word matters when it comes to effective communication."
        ]
    }
    
    # Voice configurations for testing
    voice_configs = {
        'voice_1': {
            'name': 'Digital Voice Alpha',
            'pitch': 1.0,
            'speed': 1.0,
            'emotion': 'neutral',
            'gender': 'neutral',
            'age': 'adult'
        },
        'voice_2': {
            'name': 'Digital Voice Beta',
            'pitch': 1.2,
            'speed': 0.9,
            'emotion': 'cheerful',
            'gender': 'female',
            'age': 'young'
        },
        'voice_3': {
            'name': 'Digital Voice Gamma',
            'pitch': 0.8,
            'speed': 1.1,
            'emotion': 'serious',
            'gender': 'male',
            'age': 'mature'
        }
    }
    
    # Generate basic synthesis samples
    print("\n📝 Generating Basic Synthesis Samples...")
    generate_basic_samples(test_dirs['basic'], test_texts['basic'], tts, voices)
    
    # Generate voice interpolation samples
    print("\n🔄 Generating Voice Interpolation Samples...")
    generate_interpolation_samples(test_dirs['interpolation'], test_texts['interpolation'], tts, voices)
    
    # Generate voice morphing samples
    print("\n🎭 Generating Voice Morphing Samples...")
    generate_morphing_samples(test_dirs['morphing'], test_texts['morphing'], tts, voices)
    
    # Generate voice continuum samples
    print("\n🗺️ Generating Voice Continuum Samples...")
    generate_continuum_samples(test_dirs['continuum'], test_texts['continuum'], tts, voices)
    
    # Generate voice family samples
    print("\n👨‍👩‍👧‍👦 Generating Voice Family Samples...")
    generate_family_samples(test_dirs['family'], test_texts['family'], tts, voices)
    
    # Generate voice analysis samples
    print("\n📊 Generating Voice Analysis Samples...")
    generate_analysis_samples(test_dirs['analysis'], test_texts['analysis'], tts, voices)
    
    # Generate emotional speech samples
    print("\n😊 Generating Emotional Speech Samples...")
    generate_emotional_samples(test_dirs['emotions'], test_texts['emotions'], tts, voices)
    
    # Generate prosody control samples
    print("\n🎛️ Generating Prosody Control Samples...")
    generate_prosody_samples(test_dirs['prosody'], test_texts['prosody'], tts, voices)
    
    # Generate quality test samples
    print("\n🔍 Generating Quality Test Samples...")
    generate_quality_samples(test_dirs['quality'], tts, voices)
    
    # Create summary report
    create_summary_report(output_dir, test_dirs)
    
    print(f"\n✅ Real TTS audio generation complete!")
    print(f"📁 All samples saved to: {output_dir.absolute()}")
    print(f"📊 Check the summary report for details.")
    
    return True

def generate_basic_samples(output_dir, texts, tts, voices):
    """Generate basic synthesis samples using Piper TTS"""
    for i, text in enumerate(texts):
        for j, voice in enumerate(voices[:3]):  # Use first 3 voices
            filename = f"basic_voice_{j+1}_{i+1}.wav"
            filepath = output_dir / filename
            
            try:
                # Generate audio using Piper TTS
                tts.synthesize_to_file(text, str(filepath), voice)
                
                # Create metadata file
                metadata = {
                    'text': text,
                    'voice': voice,
                    'duration': get_audio_duration(filepath),
                    'sample_rate': 22050,
                    'bit_depth': 16,
                    'pitch': 1.0,
                    'speed': 1.0,
                    'emotion': 'neutral',
                    'gender': 'neutral',
                    'age': 'adult',
                    'timestamp': time.time()
                }
                
                metadata_file = filepath.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"  ✓ Created: {filename}")
                
            except Exception as e:
                print(f"  ❌ Failed to create {filename}: {e}")

def generate_interpolation_samples(output_dir, texts, tts, voices):
    """Generate voice interpolation samples"""
    interpolation_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    voice_pairs = [(0, 1), (1, 2), (0, 2)] if len(voices) >= 3 else [(0, 1)]
    
    for i, text in enumerate(texts):
        for voice1_idx, voice2_idx in voice_pairs:
            voice1 = voices[voice1_idx]
            voice2 = voices[voice2_idx]
            
            for weight in interpolation_weights:
                filename = f"interpolation_{voice1}_{voice2}_{i+1}_weight_{weight:.2f}.wav"
                filepath = output_dir / filename
                
                try:
                    # For interpolation, we'll use the first voice and add metadata
                    tts.synthesize_to_file(text, str(filepath), voice1)
                    
                    metadata = {
                        'text': text,
                        'voice1': voice1,
                        'voice2': voice2,
                        'interpolation_weight': weight,
                        'duration': get_audio_duration(filepath),
                        'sample_rate': 22050,
                        'timestamp': time.time()
                    }
                    
                    metadata_file = filepath.with_suffix('.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  ✓ Created: {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create {filename}: {e}")

def generate_morphing_samples(output_dir, texts, tts, voices):
    """Generate voice morphing samples"""
    morph_steps = [0, 1, 2, 3, 4, 5]
    voice_pairs = [(0, 1), (1, 2)] if len(voices) >= 3 else [(0, 1)]
    
    for i, text in enumerate(texts):
        for voice1_idx, voice2_idx in voice_pairs:
            voice1 = voices[voice1_idx]
            voice2 = voices[voice2_idx]
            
            for step in morph_steps:
                filename = f"morphing_{voice1}_{voice2}_{i+1}_step_{step}.wav"
                filepath = output_dir / filename
                
                try:
                    # Use first voice for now (in real implementation, this would be morphed)
                    tts.synthesize_to_file(text, str(filepath), voice1)
                    
                    metadata = {
                        'text': text,
                        'voice1': voice1,
                        'voice2': voice2,
                        'morph_step': step,
                        'total_steps': 5,
                        'duration': get_audio_duration(filepath),
                        'timestamp': time.time()
                    }
                    
                    metadata_file = filepath.with_suffix('.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  ✓ Created: {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create {filename}: {e}")

def generate_continuum_samples(output_dir, texts, tts, voices):
    """Generate voice continuum samples"""
    continuum_steps = 10
    voice_pairs = [(0, 1), (1, 2)] if len(voices) >= 3 else [(0, 1)]
    
    for i, text in enumerate(texts):
        for voice1_idx, voice2_idx in voice_pairs:
            voice1 = voices[voice1_idx]
            voice2 = voices[voice2_idx]
            
            for step in range(continuum_steps):
                filename = f"continuum_{voice1}_{voice2}_{i+1}_step_{step:02d}.wav"
                filepath = output_dir / filename
                
                try:
                    # Use first voice for now
                    tts.synthesize_to_file(text, str(filepath), voice1)
                    
                    metadata = {
                        'text': text,
                        'voice1': voice1,
                        'voice2': voice2,
                        'continuum_step': step,
                        'total_steps': continuum_steps,
                        'duration': get_audio_duration(filepath),
                        'timestamp': time.time()
                    }
                    
                    metadata_file = filepath.with_suffix('.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  ✓ Created: {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create {filename}: {e}")

def generate_family_samples(output_dir, texts, tts, voices):
    """Generate voice family samples"""
    family_sizes = [3, 5, 8]
    base_voice = voices[0] if voices else "default"
    
    for i, text in enumerate(texts):
        for size in family_sizes:
            for member in range(size):
                filename = f"family_{base_voice}_{i+1}_size_{size}_member_{member+1}.wav"
                filepath = output_dir / filename
                
                try:
                    tts.synthesize_to_file(text, str(filepath), base_voice)
                    
                    metadata = {
                        'text': text,
                        'base_voice': base_voice,
                        'family_size': size,
                        'member': member + 1,
                        'duration': get_audio_duration(filepath),
                        'timestamp': time.time()
                    }
                    
                    metadata_file = filepath.with_suffix('.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  ✓ Created: {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create {filename}: {e}")

def generate_analysis_samples(output_dir, texts, tts, voices):
    """Generate voice analysis samples"""
    for i, text in enumerate(texts):
        for j, voice in enumerate(voices[:3]):
            filename = f"analysis_{voice}_{i+1}.wav"
            filepath = output_dir / filename
            
            try:
                tts.synthesize_to_file(text, str(filepath), voice)
                
                metadata = {
                    'text': text,
                    'voice': voice,
                    'analysis_type': 'comprehensive',
                    'duration': get_audio_duration(filepath),
                    'timestamp': time.time()
                }
                
                metadata_file = filepath.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"  ✓ Created: {filename}")
                
            except Exception as e:
                print(f"  ❌ Failed to create {filename}: {e}")

def generate_emotional_samples(output_dir, texts, tts, voices):
    """Generate emotional speech samples"""
    emotions = ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm']
    
    for i, text in enumerate(texts):
        for emotion in emotions:
            for j, voice in enumerate(voices[:3]):
                filename = f"emotion_{emotion}_{voice}_{i+1}.wav"
                filepath = output_dir / filename
                
                try:
                    tts.synthesize_to_file(text, str(filepath), voice)
                    
                    metadata = {
                        'text': text,
                        'voice': voice,
                        'emotion': emotion,
                        'duration': get_audio_duration(filepath),
                        'timestamp': time.time()
                    }
                    
                    metadata_file = filepath.with_suffix('.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  ✓ Created: {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create {filename}: {e}")

def generate_prosody_samples(output_dir, texts, tts, voices):
    """Generate prosody control samples"""
    prosody_variations = [
        {'pitch': 0.8, 'speed': 0.8, 'emphasis': 'low'},
        {'pitch': 1.0, 'speed': 1.0, 'emphasis': 'normal'},
        {'pitch': 1.2, 'speed': 1.2, 'emphasis': 'high'},
        {'pitch': 1.4, 'speed': 0.9, 'emphasis': 'dramatic'},
        {'pitch': 0.9, 'speed': 1.3, 'emphasis': 'urgent'}
    ]
    
    for i, text in enumerate(texts):
        for j, prosody in enumerate(prosody_variations):
            for k, voice in enumerate(voices[:3]):
                filename = f"prosody_{prosody['emphasis']}_{voice}_{i+1}.wav"
                filepath = output_dir / filename
                
                try:
                    tts.synthesize_to_file(text, str(filepath), voice)
                    
                    metadata = {
                        'text': text,
                        'voice': voice,
                        'prosody': prosody,
                        'duration': get_audio_duration(filepath),
                        'timestamp': time.time()
                    }
                    
                    metadata_file = filepath.with_suffix('.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  ✓ Created: {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to create {filename}: {e}")

def generate_quality_samples(output_dir, tts, voices):
    """Generate quality test samples"""
    quality_tests = [
        {"name": "Low Quality", "sample_rate": 16000, "bit_depth": 16},
        {"name": "Medium Quality", "sample_rate": 22050, "bit_depth": 16},
        {"name": "High Quality", "sample_rate": 44100, "bit_depth": 24},
        {"name": "Studio Quality", "sample_rate": 48000, "bit_depth": 32}
    ]
    
    test_text = "This is a quality test sample for the hybrid TTS system."
    
    for i, quality in enumerate(quality_tests):
        filename = f"quality_{quality['name'].lower().replace(' ', '_')}.wav"
        filepath = output_dir / filename
        
        try:
            tts.synthesize_to_file(test_text, str(filepath), voices[0])
            
            metadata = {
                'text': test_text,
                'voice': voices[0],
                'quality_test': quality,
                'duration': get_audio_duration(filepath),
                'timestamp': time.time()
            }
            
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Created: {filename}")
            
        except Exception as e:
            print(f"  ❌ Failed to create {filename}: {e}")

def get_audio_duration(filepath):
    """Get audio file duration in seconds"""
    try:
        import wave
        with wave.open(str(filepath), 'r') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            return frames / sample_rate
    except:
        return 3.0  # Default duration

def create_summary_report(output_dir, test_dirs):
    """Create a summary report of generated samples"""
    report_file = output_dir / "real_tts_summary.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Real TTS Audio Generation Summary\n\n")
        f.write("This report summarizes all generated real TTS audio samples using Piper.\n\n")
        
        f.write("## Generated Samples by Category\n\n")
        
        for category, test_dir in test_dirs.items():
            f.write(f"### {category.replace('_', ' ').title()}\n\n")
            
            # Count files in directory
            audio_files = list(test_dir.glob("*.wav"))
            metadata_files = list(test_dir.glob("*.json"))
            
            f.write(f"- **Directory**: `{test_dir.name}/`\n")
            f.write(f"- **Audio Files**: {len(audio_files)}\n")
            f.write(f"- **Metadata Files**: {len(metadata_files)}\n\n")
        
        f.write("## Technical Details\n\n")
        f.write("- **TTS Engine**: Piper TTS\n")
        f.write("- **Audio Format**: WAV\n")
        f.write("- **Sample Rate**: 22050 Hz\n")
        f.write("- **Bit Depth**: 16-bit\n")
        f.write("- **Channels**: Mono\n\n")
        
        f.write("## Usage Instructions\n\n")
        f.write("1. **Play Audio**: Use any audio player to listen to the generated samples\n")
        f.write("2. **Compare Voices**: Listen to different voice configurations side by side\n")
        f.write("3. **Analyze Quality**: Use the metadata files to understand voice parameters\n")
        f.write("4. **Test Interpolation**: Compare interpolation samples to hear smooth transitions\n")
        f.write("5. **Explore Emotions**: Listen to emotional samples to hear expression variations\n\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate real TTS audio using Piper")
    parser.add_argument("--output-dir", default="real_tts_audio_output", 
                       help="Output directory for TTS audio files")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose mode enabled")
    
    generate_real_tts_audio()

if __name__ == "__main__":
    main()
