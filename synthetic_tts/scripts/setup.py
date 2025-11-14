#!/usr/bin/env python3
"""
Setup script for the Synthetic Digital TTS System.

This script sets up the environment and generates initial training data.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.synthetic_data_generator import SyntheticDataGenerator


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "models",
        "data",
        "logs",
        "demo_output",
        "config",
        "scripts",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("✓ All directories created")


def generate_initial_data(data_size: int = 1000):
    """Generate initial synthetic training data."""
    print(f"Generating {data_size} synthetic training samples...")
    
    # Load configuration
    import json
    with open("config/character_voice.json", "r") as f:
        config = json.load(f)
    
    # Generate data
    data_generator = SyntheticDataGenerator(config)
    data_generator.generate_dataset(
        output_dir="data",
        num_samples=data_size,
        max_text_length=100,
    )
    
    print("✓ Initial training data generated")


def create_sample_configs():
    """Create sample configuration files."""
    print("Creating sample configuration files...")
    
    # Character voice config (already exists)
    if not os.path.exists("config/character_voice.json"):
        print("✓ Character voice config already exists")
    else:
        print("✓ Character voice config created")
    
    # Training config
    training_config = {
        "model": {
            "fastpitch": {
                "n_mel_channels": 80,
                "n_symbols": 256,
                "encoder_embedding_dim": 384,
                "decoder_rnn_dim": 1024,
                "speaker_embedding_dim": 64,
            },
            "hifigan": {
                "n_mel_channels": 80,
                "upsample_rates": [8, 8, 2, 2],
                "upsample_initial_channel": 512,
            }
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "epochs": 100,
            "save_interval": 10,
            "validate_interval": 5,
        },
        "data": {
            "sample_rate": 22050,
            "n_mels": 80,
            "hop_length": 256,
            "win_length": 1024,
        }
    }
    
    with open("config/training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    print("✓ Training config created")


def run_initial_training():
    """Run initial training with a small dataset."""
    print("Running initial training...")
    
    try:
        subprocess.run([
            sys.executable, "train_models.py",
            "--data-dir", "data",
            "--output-dir", "models",
            "--config", "config/character_voice.json",
            "--epochs", "10",
            "--batch-size", "16",
            "--generate-data",
            "--data-size", "500",
        ], check=True)
        print("✓ Initial training completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during training: {e}")
        return False
    
    return True


def test_synthesis():
    """Test the synthesis pipeline."""
    print("Testing synthesis pipeline...")
    
    try:
        subprocess.run([
            sys.executable, "synthesize_speech.py",
            "--text", "Hello, this is a test of the synthetic TTS system.",
            "--emotion", "neutral",
            "--output", "test_output.wav",
            "--config", "config/character_voice.json",
            "--model-dir", "models",
        ], check=True)
        
        if os.path.exists("test_output.wav"):
            print("✓ Synthesis test successful")
            os.remove("test_output.wav")  # Clean up
        else:
            print("✗ Synthesis test failed - no output file")
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during synthesis test: {e}")
        return False
    
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup TTS System")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-training", action="store_true", help="Skip initial training")
    parser.add_argument("--data-size", type=int, default=1000, help="Number of training samples to generate")
    
    args = parser.parse_args()
    
    print("Setting up Synthetic Digital TTS System...")
    print("=" * 50)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("Setup failed at dependency installation")
            return
    else:
        print("Skipping dependency installation")
    
    # Create directories
    create_directories()
    
    # Create sample configs
    create_sample_configs()
    
    # Generate initial data
    generate_initial_data(args.data_size)
    
    # Run initial training
    if not args.skip_training:
        if not run_initial_training():
            print("Setup failed at training")
            return
    else:
        print("Skipping initial training")
    
    # Test synthesis
    if not test_synthesis():
        print("Setup failed at synthesis test")
        return
    
    print("=" * 50)
    print("Setup complete! The TTS system is ready to use.")
    print("\nNext steps:")
    print("1. Run 'python synthesize_speech.py --text \"Your text here\"' to synthesize speech")
    print("2. Run 'python scripts/demo.py' to see a comprehensive demonstration")
    print("3. Run 'python train_models.py --help' to see training options")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()









































