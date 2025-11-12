#!/usr/bin/env python3
"""
Generate Voice Library Data for HTML Interface
Creates JSON data file with all audio files and metadata for the voice library manager
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time
import hashlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def generate_voice_library_data():
    """Generate comprehensive voice library data for the HTML interface"""
    
    print("🎵 Generating Voice Library Data for HTML Interface")
    print("=" * 60)
    
    # Define output paths
    test_audio_dir = Path("test_audio_output")
    real_audio_dir = Path("real_audio_output")
    output_file = Path("voice_library_data.json")
    
    voice_library = []
    voice_id = 1
    
    # Process test audio files
    if test_audio_dir.exists():
        print("\n📁 Processing test audio files...")
        voice_library.extend(process_audio_directory(test_audio_dir, "Test", voice_id))
        voice_id += len([f for f in test_audio_dir.rglob("*.wav")])
    
    # Process real audio files
    if real_audio_dir.exists():
        print("\n📁 Processing real audio files...")
        voice_library.extend(process_audio_directory(real_audio_dir, "Real", voice_id))
    
    # Generate comprehensive statistics
    statistics = generate_statistics(voice_library)
    
    # Create final data structure
    voice_library_data = {
        "metadata": {
            "generated_at": time.time(),
            "generated_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_voices": len(voice_library),
            "version": "1.0.0"
        },
        "statistics": statistics,
        "voices": voice_library,
        "categories": list(set(voice["category"] for voice in voice_library)),
        "quality_criteria": {
            "Naturalness": {"min": 1, "max": 10, "description": "How natural the voice sounds"},
            "Clarity": {"min": 1, "max": 10, "description": "How clear and intelligible the speech is"},
            "Emotion": {"min": 1, "max": 10, "description": "How well emotions are expressed"},
            "Pitch": {"min": 1, "max": 10, "description": "Pitch accuracy and variation"},
            "Speed": {"min": 1, "max": 10, "description": "Speaking rate appropriateness"},
            "Overall": {"min": 1, "max": 10, "description": "Overall quality assessment"}
        }
    }
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(voice_library_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Voice library data generated successfully!")
    print(f"📁 Output file: {output_file.absolute()}")
    print(f"📊 Total voices: {len(voice_library)}")
    print(f"📂 Categories: {len(voice_library_data['categories'])}")
    
    return voice_library_data

def process_audio_directory(directory, prefix, start_id):
    """Process all audio files in a directory and return voice data"""
    voices = []
    voice_id = start_id
    
    for category_dir in directory.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name.replace('_', ' ').title()
        print(f"  Processing {category_name}...")
        
        for audio_file in category_dir.glob("*.wav"):
            # Get corresponding metadata file
            metadata_file = audio_file.with_suffix('.json')
            metadata = {}
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"    Warning: Could not load metadata for {audio_file.name}: {e}")
            
            # Generate voice data
            voice_data = create_voice_data(
                voice_id=voice_id,
                category=f"{prefix} {category_name}",
                audio_file=audio_file,
                metadata=metadata,
                base_dir=directory
            )
            
            voices.append(voice_data)
            voice_id += 1
    
    return voices

def create_voice_data(voice_id, category, audio_file, metadata, base_dir):
    """Create individual voice data entry"""
    
    # Extract voice characteristics from filename and metadata
    filename = audio_file.name
    relative_path = audio_file.relative_to(base_dir)
    
    # Determine voice name from filename
    voice_name = generate_voice_name(filename, category)
    
    # Extract transcript from metadata or generate default
    transcript = metadata.get('text', generate_default_transcript(category, filename))
    
    # Extract voice characteristics
    voice_config = metadata.get('voice_config', {})
    
    # Generate file hash for unique identification
    file_hash = generate_file_hash(audio_file)
    
    voice_data = {
        "id": voice_id,
        "category": category,
        "name": voice_name,
        "transcript": transcript,
        "filename": filename,
        "filepath": str(relative_path).replace('\\', '/'),
        "file_hash": file_hash,
        "metadata": {
            "duration": metadata.get('duration', 3.0),
            "sample_rate": metadata.get('sample_rate', 22050),
            "bit_depth": metadata.get('bit_depth', 16),
            "pitch": voice_config.get('pitch', 1.0),
            "speed": voice_config.get('speed', 1.0),
            "emotion": voice_config.get('emotion', 'neutral'),
            "gender": voice_config.get('gender', 'neutral'),
            "age": voice_config.get('age', 'adult'),
            "interpolation_weight": metadata.get('interpolation_weight'),
            "morph_step": metadata.get('morph_step'),
            "continuum_step": metadata.get('continuum_step'),
            "family_member": metadata.get('member'),
            "quality_test": metadata.get('quality_test'),
            "prosody_type": metadata.get('prosody', {}).get('emphasis') if metadata.get('prosody') else None
        },
        "scores": {},
        "average_score": 0.0,
        "is_scored": False,
        "created_at": time.time(),
        "file_size": audio_file.stat().st_size if audio_file.exists() else 0
    }
    
    return voice_data

def generate_voice_name(filename, category):
    """Generate a descriptive voice name from filename and category"""
    
    # Remove file extension
    name = filename.replace('.wav', '')
    
    # Handle different naming patterns
    if 'interpolation' in name:
        parts = name.split('_')
        if len(parts) >= 4:
            voice1 = parts[1]
            voice2 = parts[2]
            weight = parts[-1]
            return f"Interpolation {voice1}→{voice2} ({weight})"
    
    elif 'morphing' in name:
        parts = name.split('_')
        if len(parts) >= 4:
            voice1 = parts[1]
            voice2 = parts[2]
            step = parts[-1]
            return f"Morphing {voice1}→{voice2} (Step {step})"
    
    elif 'continuum' in name:
        parts = name.split('_')
        if len(parts) >= 4:
            voice1 = parts[1]
            voice2 = parts[2]
            step = parts[-1]
            return f"Continuum {voice1}→{voice2} (Step {step})"
    
    elif 'family' in name:
        parts = name.split('_')
        if len(parts) >= 4:
            size = parts[-3]
            member = parts[-1]
            return f"Family Member {member}/{size}"
    
    elif 'emotion' in name:
        parts = name.split('_')
        if len(parts) >= 2:
            emotion = parts[1]
            return f"{emotion.title()} Voice"
    
    elif 'prosody' in name:
        parts = name.split('_')
        if len(parts) >= 2:
            prosody = parts[1]
            return f"{prosody.title()} Prosody"
    
    elif 'quality' in name:
        parts = name.split('_')
        if len(parts) >= 2:
            quality = parts[1]
            return f"{quality.title()} Quality"
    
    elif 'analysis' in name:
        parts = name.split('_')
        if len(parts) >= 2:
            voice = parts[1]
            return f"Analysis {voice}"
    
    else:
        # Basic synthesis or other
        return name.replace('_', ' ').title()

def generate_default_transcript(category, filename):
    """Generate default transcript based on category and filename"""
    
    transcripts = {
        "Basic Synthesis": [
            "Hello, this is a basic synthesis test using the hybrid TTS system.",
            "The quick brown fox jumps over the lazy dog in a beautiful meadow.",
            "Welcome to the hybrid TTS system demonstration with real voice synthesis."
        ],
        "Voice Interpolation": [
            "This is voice interpolation between two different voices in real-time.",
            "Smooth transitions create natural voice morphing with advanced algorithms.",
            "Real-time voice switching is now possible with seamless quality."
        ],
        "Voice Morphing": [
            "Voice morphing transforms one voice into another with precision.",
            "Advanced voice transformation capabilities using neural networks.",
            "Seamless voice character changes with maintained intelligibility."
        ],
        "Voice Continuum": [
            "Voice continuum creates smooth voice transitions across the voice space.",
            "Navigate through the voice space effortlessly with intuitive controls.",
            "Discover new voice characteristics through intelligent exploration."
        ],
        "Voice Family": [
            "Voice family members share similar characteristics and traits.",
            "Generate related voice variations automatically with genetic algorithms.",
            "Create voice clusters with common traits and unique personalities."
        ],
        "Emotional Speech": [
            "I am so excited about this new technology!",
            "This is absolutely terrible and disappointing.",
            "I feel calm and peaceful about this situation.",
            "What an amazing discovery this has been!"
        ],
        "Prosody Control": [
            "The quick brown fox jumps over the lazy dog.",
            "In a world where technology meets creativity, possibilities are endless.",
            "Every word matters when it comes to effective communication."
        ],
        "Quality Tests": [
            "This is a quality test sample for the hybrid TTS system.",
            "Quality assessment helps improve voice synthesis performance.",
            "Different quality levels demonstrate system capabilities."
        ],
        "Voice Analysis": [
            "Voice analysis provides detailed characteristics and quality metrics.",
            "Quality metrics assess audio performance with comprehensive evaluation.",
            "Similarity measurement finds related voices using advanced algorithms."
        ]
    }
    
    # Get category without prefix
    clean_category = category.replace("Test ", "").replace("Real ", "")
    
    if clean_category in transcripts:
        # Use hash of filename to consistently select transcript
        hash_value = hash(filename) % len(transcripts[clean_category])
        return transcripts[clean_category][hash_value]
    
    return "This is a voice synthesis sample for testing and evaluation."

def generate_file_hash(file_path):
    """Generate a hash for the file for unique identification"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            return hashlib.md5(file_content).hexdigest()[:16]
    except:
        return hashlib.md5(str(file_path).encode()).hexdigest()[:16]

def generate_statistics(voice_library):
    """Generate comprehensive statistics for the voice library"""
    
    if not voice_library:
        return {
            "total_voices": 0,
            "scored_voices": 0,
            "average_quality": 0.0,
            "categories": 0,
            "total_duration": 0.0,
            "file_size_mb": 0.0
        }
    
    # Basic counts
    total_voices = len(voice_library)
    scored_voices = len([v for v in voice_library if v.get('is_scored', False)])
    
    # Quality statistics
    scored_voice_scores = [v.get('average_score', 0) for v in voice_library if v.get('is_scored', False)]
    average_quality = sum(scored_voice_scores) / len(scored_voice_scores) if scored_voice_scores else 0.0
    
    # Categories
    categories = len(set(v['category'] for v in voice_library))
    
    # Duration and file size
    total_duration = sum(v['metadata'].get('duration', 0) for v in voice_library)
    total_file_size = sum(v.get('file_size', 0) for v in voice_library)
    file_size_mb = total_file_size / (1024 * 1024)
    
    # Category breakdown
    category_breakdown = {}
    for voice in voice_library:
        category = voice['category']
        if category not in category_breakdown:
            category_breakdown[category] = 0
        category_breakdown[category] += 1
    
    # Quality distribution
    quality_distribution = {
        "excellent": len([v for v in voice_library if v.get('average_score', 0) >= 9]),
        "good": len([v for v in voice_library if 7 <= v.get('average_score', 0) < 9]),
        "fair": len([v for v in voice_library if 5 <= v.get('average_score', 0) < 7]),
        "poor": len([v for v in voice_library if 1 <= v.get('average_score', 0) < 5]),
        "unscored": len([v for v in voice_library if not v.get('is_scored', False)])
    }
    
    return {
        "total_voices": total_voices,
        "scored_voices": scored_voices,
        "average_quality": round(average_quality, 2),
        "categories": categories,
        "total_duration": round(total_duration, 2),
        "file_size_mb": round(file_size_mb, 2),
        "category_breakdown": category_breakdown,
        "quality_distribution": quality_distribution
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate voice library data for HTML interface")
    parser.add_argument("--test-audio-dir", default="test_audio_output", 
                       help="Directory containing test audio files")
    parser.add_argument("--real-audio-dir", default="real_audio_output", 
                       help="Directory containing real audio files")
    parser.add_argument("--output-file", default="voice_library_data.json", 
                       help="Output JSON file for voice library data")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose mode enabled")
    
    # Update global paths
    global test_audio_dir, real_audio_dir, output_file
    test_audio_dir = Path(args.test_audio_dir)
    real_audio_dir = Path(args.real_audio_dir)
    output_file = Path(args.output_file)
    
    generate_voice_library_data()

if __name__ == "__main__":
    main()
