#!/usr/bin/env python3
"""
Prepare Training Splits for Kelly25 Voice Training
Create optimized training, validation, and test splits
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_training_splits():
    """Prepare optimized training splits for Kelly25 voice training"""
    
    print("🔧 Preparing Kelly25 Training Splits")
    print("=" * 50)
    
    # Paths
    data_dir = Path("kelly25_training_data")
    metadata_file = data_dir / "metadata.csv"
    splits_dir = data_dir / "training_splits"
    
    # Create splits directory
    splits_dir.mkdir(exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_file, sep='|', comment='#', names=['id', 'normalized_text', 'raw_text'])
    logger.info(f"Loaded {len(metadata)} metadata entries")
    
    # Add text characteristics for stratified splitting
    metadata['text_length'] = metadata['raw_text'].str.len()
    metadata['word_count'] = metadata['raw_text'].str.split().str.len()
    metadata['has_question'] = metadata['raw_text'].str.contains(r'\?', na=False)
    metadata['has_exclamation'] = metadata['raw_text'].str.contains(r'!', na=False)
    metadata['is_complex'] = metadata['raw_text'].str.contains(r'[,;]', na=False)
    
    # Create length bins for stratified splitting
    metadata['length_bin'] = pd.cut(metadata['text_length'], bins=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    
    # Create complexity bins
    metadata['complexity_bin'] = 'simple'
    metadata.loc[metadata['is_complex'], 'complexity_bin'] = 'complex'
    
    # Stratified split to ensure balanced representation
    # First split: 80% train, 20% temp (for val/test)
    train_data, temp_data = train_test_split(
        metadata, 
        test_size=0.2, 
        random_state=42,
        stratify=metadata['length_bin']
    )
    
    # Second split: 50% val, 50% test from temp data
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42,
        stratify=temp_data['length_bin']
    )
    
    # Create splits summary
    splits_summary = {
        "total_samples": len(metadata),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "train_percentage": (len(train_data) / len(metadata)) * 100,
        "val_percentage": (len(val_data) / len(metadata)) * 100,
        "test_percentage": (len(test_data) / len(metadata)) * 100
    }
    
    # Analyze splits quality
    def analyze_split_quality(split_data, split_name):
        """Analyze quality of a data split"""
        return {
            "name": split_name,
            "samples": len(split_data),
            "text_length": {
                "mean": float(split_data['text_length'].mean()),
                "std": float(split_data['text_length'].std()),
                "min": int(split_data['text_length'].min()),
                "max": int(split_data['text_length'].max())
            },
            "word_count": {
                "mean": float(split_data['word_count'].mean()),
                "std": float(split_data['word_count'].std()),
                "min": int(split_data['word_count'].min()),
                "max": int(split_data['word_count'].max())
            },
            "questions_ratio": float((split_data['has_question'].sum() / len(split_data)) * 100),
            "exclamations_ratio": float((split_data['has_exclamation'].sum() / len(split_data)) * 100),
            "complexity_ratio": float((split_data['is_complex'].sum() / len(split_data)) * 100)
        }
    
    splits_analysis = {
        "train": analyze_split_quality(train_data, "train"),
        "val": analyze_split_quality(val_data, "val"),
        "test": analyze_split_quality(test_data, "test")
    }
    
    # Save splits
    train_file = splits_dir / "train_metadata.csv"
    val_file = splits_dir / "val_metadata.csv"
    test_file = splits_dir / "test_metadata.csv"
    
    # Save metadata files
    train_data[['id', 'normalized_text', 'raw_text']].to_csv(train_file, sep='|', index=False, header=False)
    val_data[['id', 'normalized_text', 'raw_text']].to_csv(val_file, sep='|', index=False, header=False)
    test_data[['id', 'normalized_text', 'raw_text']].to_csv(test_file, sep='|', index=False, header=False)
    
    # Create file lists for easy access
    train_files = splits_dir / "train_files.txt"
    val_files = splits_dir / "val_files.txt"
    test_files = splits_dir / "test_files.txt"
    
    with open(train_files, 'w') as f:
        for file_id in train_data['id']:
            f.write(f"{file_id}.wav\n")
    
    with open(val_files, 'w') as f:
        for file_id in val_data['id']:
            f.write(f"{file_id}.wav\n")
    
    with open(test_files, 'w') as f:
        for file_id in test_data['id']:
            f.write(f"{file_id}.wav\n")
    
    # Create comprehensive summary
    summary = {
        "splits_summary": splits_summary,
        "splits_analysis": splits_analysis,
        "files_created": {
            "train_metadata": str(train_file),
            "val_metadata": str(val_file),
            "test_metadata": str(test_file),
            "train_files": str(train_files),
            "val_files": str(val_files),
            "test_files": str(test_files)
        }
    }
    
    # Save summary
    summary_file = splits_dir / "splits_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n📊 Training Splits Summary:")
    print(f"📝 Total Samples: {splits_summary['total_samples']:,}")
    print(f"🚂 Train: {splits_summary['train_samples']:,} ({splits_summary['train_percentage']:.1f}%)")
    print(f"✅ Validation: {splits_summary['val_samples']:,} ({splits_summary['val_percentage']:.1f}%)")
    print(f"🧪 Test: {splits_summary['test_samples']:,} ({splits_summary['test_percentage']:.1f}%)")
    
    print("\n📈 Split Quality Analysis:")
    for split_name, analysis in splits_analysis.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  📊 Text Length: {analysis['text_length']['mean']:.1f} ± {analysis['text_length']['std']:.1f} chars")
        print(f"  📊 Word Count: {analysis['word_count']['mean']:.1f} ± {analysis['word_count']['std']:.1f} words")
        print(f"  ❓ Questions: {analysis['questions_ratio']:.1f}%")
        print(f"  ❗ Exclamations: {analysis['exclamations_ratio']:.1f}%")
        print(f"  🔗 Complexity: {analysis['complexity_ratio']:.1f}%")
    
    print(f"\n💾 Files created in: {splits_dir}")
    print(f"📋 Summary saved to: {summary_file}")
    
    # Create training configuration
    config = {
        "data": {
            "train_metadata": str(train_file),
            "val_metadata": str(val_file),
            "test_metadata": str(test_file),
            "wavs_dir": str(data_dir / "wavs")
        },
        "model": {
            "name": "kelly25",
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0,
            "mel_fmax": 8000
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "epochs": 1000,
            "early_stopping_patience": 50,
            "gradient_clip_val": 1.0
        }
    }
    
    config_file = splits_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️  Training config saved to: {config_file}")
    
    return summary

if __name__ == "__main__":
    prepare_training_splits()
