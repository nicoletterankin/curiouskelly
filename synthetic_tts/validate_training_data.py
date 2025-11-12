#!/usr/bin/env python3
"""
Validate Kelly25 Training Data
Analyze the generated training dataset for quality and completeness
"""

import os
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
import numpy as np
import json
from datetime import datetime

def validate_training_data():
    """Validate the generated Kelly25 training dataset"""
    
    print("🔍 Validating Kelly25 Training Dataset")
    print("=" * 50)
    
    # Paths
    data_dir = Path("kelly25_training_data")
    wavs_dir = data_dir / "wavs"
    metadata_file = data_dir / "metadata.csv"
    
    # Check if directories exist
    if not data_dir.exists():
        print("❌ Training data directory not found!")
        return False
    
    if not wavs_dir.exists():
        print("❌ WAVs directory not found!")
        return False
    
    if not metadata_file.exists():
        print("❌ Metadata file not found!")
        return False
    
    print("✅ All required directories and files found")
    
    # Load metadata
    print("\n📊 Loading metadata...")
    metadata = pd.read_csv(metadata_file, sep='|', comment='#', names=['id', 'normalized_text', 'raw_text'])
    print(f"✅ Loaded {len(metadata)} metadata entries")
    
    # Get audio files
    audio_files = list(wavs_dir.glob("*.wav"))
    print(f"✅ Found {len(audio_files)} WAV files")
    
    # Validate file count matches
    if len(audio_files) != len(metadata):
        print(f"⚠️ Warning: Audio files ({len(audio_files)}) don't match metadata ({len(metadata)})")
    
    # Analyze audio files
    print("\n🎵 Analyzing audio files...")
    
    durations = []
    sample_rates = []
    file_sizes = []
    valid_files = 0
    
    for i, audio_file in enumerate(audio_files):
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=None)
            
            # Calculate duration
            duration = len(audio) / sr
            durations.append(duration)
            sample_rates.append(sr)
            
            # Get file size
            file_size = audio_file.stat().st_size
            file_sizes.append(file_size)
            
            valid_files += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files...")
                
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
    
    print(f"✅ Successfully analyzed {valid_files}/{len(audio_files)} audio files")
    
    # Calculate statistics
    total_duration = sum(durations)
    avg_duration = np.mean(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    total_size = sum(file_sizes)
    avg_size = np.mean(file_sizes)
    
    unique_sample_rates = list(set(sample_rates))
    
    # Print results
    print("\n📈 Training Data Statistics:")
    print(f"  Total Files: {valid_files}")
    print(f"  Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"  Average Duration: {avg_duration:.2f} seconds")
    print(f"  Min Duration: {min_duration:.2f} seconds")
    print(f"  Max Duration: {max_duration:.2f} seconds")
    print(f"  Total Size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"  Average Size: {avg_size:,.0f} bytes")
    print(f"  Sample Rates: {unique_sample_rates}")
    
    # Check duration requirements
    print("\n🎯 Duration Analysis:")
    if total_duration >= 3600:  # 60 minutes
        print(f"✅ Meets minimum requirement: {total_duration/60:.2f} minutes >= 60 minutes")
    else:
        print(f"⚠️ Below minimum requirement: {total_duration/60:.2f} minutes < 60 minutes")
    
    # Check sample rate consistency
    if len(unique_sample_rates) == 1 and unique_sample_rates[0] == 22050:
        print("✅ Sample rate consistent: 22050 Hz")
    else:
        print(f"⚠️ Sample rate inconsistency: {unique_sample_rates}")
    
    # Check duration distribution
    short_files = [d for d in durations if d < 3.0]
    medium_files = [d for d in durations if 3.0 <= d < 10.0]
    long_files = [d for d in durations if d >= 10.0]
    
    print(f"\n📊 Duration Distribution:")
    print(f"  Short (< 3s): {len(short_files)} files ({len(short_files)/len(durations)*100:.1f}%)")
    print(f"  Medium (3-10s): {len(medium_files)} files ({len(medium_files)/len(durations)*100:.1f}%)")
    print(f"  Long (> 10s): {len(long_files)} files ({len(long_files)/len(durations)*100:.1f}%)")
    
    # Analyze text content
    print("\n📝 Text Content Analysis:")
    
    # Character counts
    char_counts = [len(text) for text in metadata['raw_text']]
    avg_chars = np.mean(char_counts)
    min_chars = min(char_counts)
    max_chars = max(char_counts)
    
    print(f"  Average characters per text: {avg_chars:.1f}")
    print(f"  Min characters: {min_chars}")
    print(f"  Max characters: {max_chars}")
    
    # Word counts
    word_counts = [len(text.split()) for text in metadata['raw_text']]
    avg_words = np.mean(word_counts)
    min_words = min(word_counts)
    max_words = max(word_counts)
    
    print(f"  Average words per text: {avg_words:.1f}")
    print(f"  Min words: {min_words}")
    print(f"  Max words: {max_words}")
    
    # Category distribution
    if 'category' in metadata.columns:
        category_counts = metadata['category'].value_counts()
        print(f"\n📂 Category Distribution:")
        for category, count in category_counts.head(10).items():
            print(f"  {category}: {count} files ({count/len(metadata)*100:.1f}%)")
    
    # Create validation report
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "total_files": valid_files,
        "total_duration_seconds": total_duration,
        "total_duration_minutes": total_duration / 60,
        "average_duration_seconds": avg_duration,
        "min_duration_seconds": min_duration,
        "max_duration_seconds": max_duration,
        "total_size_bytes": total_size,
        "total_size_mb": total_size / 1024 / 1024,
        "average_size_bytes": avg_size,
        "sample_rates": unique_sample_rates,
        "meets_duration_requirement": total_duration >= 3600,
        "sample_rate_consistent": len(unique_sample_rates) == 1 and unique_sample_rates[0] == 22050,
        "duration_distribution": {
            "short_files": len(short_files),
            "medium_files": len(medium_files),
            "long_files": len(long_files)
        },
        "text_statistics": {
            "average_characters": avg_chars,
            "min_characters": min_chars,
            "max_characters": max_chars,
            "average_words": avg_words,
            "min_words": min_words,
            "max_words": max_words
        }
    }
    
    # Save validation report
    report_file = data_dir / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Validation report saved to: {report_file}")
    
    # Final assessment
    print("\n🎯 Final Assessment:")
    
    all_good = True
    
    if total_duration < 3600:
        print("❌ Duration requirement not met")
        all_good = False
    else:
        print("✅ Duration requirement met")
    
    if len(unique_sample_rates) != 1 or unique_sample_rates[0] != 22050:
        print("❌ Sample rate not consistent")
        all_good = False
    else:
        print("✅ Sample rate consistent")
    
    if valid_files < len(audio_files):
        print("❌ Some audio files failed validation")
        all_good = False
    else:
        print("✅ All audio files valid")
    
    if all_good:
        print("\n🎉 VALIDATION PASSED! Training data is ready for use.")
    else:
        print("\n⚠️ VALIDATION ISSUES FOUND! Please review the issues above.")
    
    return all_good

if __name__ == "__main__":
    validate_training_data()
