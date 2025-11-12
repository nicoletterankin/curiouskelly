#!/usr/bin/env python3
"""
Audio Quality Optimization for Kelly25 Training Data
Enhance audio consistency and quality for optimal training
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_audio_quality():
    """Optimize audio quality for consistent training"""
    
    print("🔧 Optimizing Kelly25 Audio Quality")
    print("=" * 50)
    
    # Paths
    data_dir = Path("kelly25_training_data")
    wavs_dir = data_dir / "wavs"
    backup_dir = data_dir / "wavs_backup"
    
    # Create backup
    if not backup_dir.exists():
        backup_dir.mkdir(exist_ok=True)
        logger.info("Created backup directory")
    
    # Get all audio files
    audio_files = list(wavs_dir.glob("*.wav"))
    logger.info(f"Found {len(audio_files)} audio files to optimize")
    
    # Quality metrics
    quality_metrics = {
        "total_files": len(audio_files),
        "processed_files": 0,
        "skipped_files": 0,
        "error_files": 0,
        "rms_levels": [],
        "duration_stats": [],
        "sample_rate_issues": [],
        "quality_issues": []
    }
    
    # Process each file
    for audio_file in tqdm(audio_files, desc="Optimizing audio"):
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=22050, mono=True)
            
            # Check sample rate
            if sr != 22050:
                quality_metrics["sample_rate_issues"].append(str(audio_file))
                logger.warning(f"Sample rate issue: {audio_file.name}")
            
            # Calculate RMS level
            rms = float(np.sqrt(np.mean(audio**2)))
            quality_metrics["rms_levels"].append(rms)
            
            # Calculate duration
            duration = float(len(audio) / sr)
            quality_metrics["duration_stats"].append(duration)
            
            # Quality checks
            if rms < 0.01:  # Too quiet
                quality_metrics["quality_issues"].append(f"{audio_file.name}: Too quiet (RMS: {rms:.4f})")
            elif rms > 0.5:  # Too loud
                quality_metrics["quality_issues"].append(f"{audio_file.name}: Too loud (RMS: {rms:.4f})")
            
            # Normalize audio
            if rms > 0:
                # Target RMS level
                target_rms = 0.1
                normalized_audio = audio * (target_rms / rms)
                
                # Clip to prevent distortion
                normalized_audio = np.clip(normalized_audio, -0.95, 0.95)
                
                # Save optimized audio
                sf.write(str(audio_file), normalized_audio, 22050)
                quality_metrics["processed_files"] += 1
            else:
                quality_metrics["skipped_files"] += 1
                logger.warning(f"Skipped silent file: {audio_file.name}")
                
        except Exception as e:
            quality_metrics["error_files"] += 1
            logger.error(f"Error processing {audio_file.name}: {e}")
    
    # Calculate statistics
    if quality_metrics["rms_levels"]:
        quality_metrics["rms_stats"] = {
            "mean": float(np.mean(quality_metrics["rms_levels"])),
            "std": float(np.std(quality_metrics["rms_levels"])),
            "min": float(np.min(quality_metrics["rms_levels"])),
            "max": float(np.max(quality_metrics["rms_levels"]))
        }
    
    if quality_metrics["duration_stats"]:
        quality_metrics["duration_stats_summary"] = {
            "mean": float(np.mean(quality_metrics["duration_stats"])),
            "std": float(np.std(quality_metrics["duration_stats"])),
            "min": float(np.min(quality_metrics["duration_stats"])),
            "max": float(np.max(quality_metrics["duration_stats"]))
        }
    
    # Save quality report
    quality_report = data_dir / "audio_quality_report.json"
    with open(quality_report, 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    
    # Print summary
    print("\n📊 Audio Quality Optimization Summary:")
    print(f"✅ Processed: {quality_metrics['processed_files']} files")
    print(f"⚠️  Skipped: {quality_metrics['skipped_files']} files")
    print(f"❌ Errors: {quality_metrics['error_files']} files")
    
    if quality_metrics["rms_stats"]:
        print(f"📈 RMS Levels: {quality_metrics['rms_stats']['mean']:.4f} ± {quality_metrics['rms_stats']['std']:.4f}")
    
    if quality_metrics["duration_stats_summary"]:
        print(f"⏱️  Duration: {quality_metrics['duration_stats_summary']['mean']:.2f}s ± {quality_metrics['duration_stats_summary']['std']:.2f}s")
    
    if quality_metrics["sample_rate_issues"]:
        print(f"⚠️  Sample Rate Issues: {len(quality_metrics['sample_rate_issues'])} files")
    
    if quality_metrics["quality_issues"]:
        print(f"⚠️  Quality Issues: {len(quality_metrics['quality_issues'])} files")
        for issue in quality_metrics["quality_issues"][:5]:  # Show first 5
            print(f"   - {issue}")
        if len(quality_metrics["quality_issues"]) > 5:
            print(f"   ... and {len(quality_metrics['quality_issues']) - 5} more")
    
    print(f"\n💾 Quality report saved to: {quality_report}")
    
    return quality_metrics

if __name__ == "__main__":
    optimize_audio_quality()
