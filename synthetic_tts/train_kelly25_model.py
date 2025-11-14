#!/usr/bin/env python3
"""
Kelly25 Voice Training Script for Piper TTS
Train a high-quality Kelly25 voice model using optimized dataset
"""

import os
import json
import subprocess
import logging
from pathlib import Path
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_kelly25_model():
    """Train Kelly25 voice model using Piper TTS"""
    
    print("🚀 Starting Kelly25 Voice Training")
    print("=" * 60)
    
    # Paths
    config_file = Path("kelly25_training_data/training_splits/training_config.json")
    train_metadata = Path("kelly25_training_data/training_splits/train_metadata.csv")
    val_metadata = Path("kelly25_training_data/training_splits/val_metadata.csv")
    wavs_dir = Path("kelly25_training_data/wavs")
    output_dir = Path("kelly25_model_output")
    
    # Check if required files exist
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return False
    
    if not train_metadata.exists():
        logger.error(f"Training metadata not found: {train_metadata}")
        return False
    
    if not val_metadata.exists():
        logger.error(f"Validation metadata not found: {val_metadata}")
        return False
    
    if not wavs_dir.exists():
        logger.error(f"WAVs directory not found: {wavs_dir}")
        return False
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded training configuration from: {config_file}")
    logger.info(f"Training samples: {len(open(train_metadata).readlines())}")
    logger.info(f"Validation samples: {len(open(val_metadata).readlines())}")
    
    # Prepare Piper TTS training command
    # Piper TTS uses a different command structure
    piper_cmd = [
        "piper-train",
        "--dataset-format", "ljspeech",
        "--dataset-path", str(train_metadata.parent),
        "--checkpoint-interval", "100",
        "--epochs", str(config["training"]["epochs"]),
        "--batch-size", str(config["training"]["batch_size"]),
        "--learning-rate", str(config["training"]["learning_rate"]),
        "--output-dir", str(output_dir),
        "--model-name", config["model"]["name"]
    ]
    
    # Add validation if available
    if val_metadata.exists():
        piper_cmd.extend(["--val-dataset-path", str(val_metadata.parent)])
    
    logger.info("Starting Piper TTS training...")
    logger.info(f"Command: {' '.join(piper_cmd)}")
    
    # Start training
    start_time = time.time()
    
    try:
        # Run training process
        process = subprocess.Popen(
            piper_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=os.getcwd()
        )
        
        # Monitor training progress
        training_log = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(line)
                training_log.append(line)
                
                # Log important milestones
                if "epoch" in line.lower() and "loss" in line.lower():
                    logger.info(f"Training progress: {line}")
                elif "validation" in line.lower():
                    logger.info(f"Validation: {line}")
                elif "checkpoint" in line.lower():
                    logger.info(f"Checkpoint: {line}")
        
        # Wait for process to complete
        return_code = process.poll()
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        if return_code == 0:
            logger.info("✅ Training completed successfully!")
            logger.info(f"Training duration: {training_duration/3600:.2f} hours")
            
            # Save training log
            log_file = output_dir / "training_log.txt"
            with open(log_file, 'w') as f:
                f.write(f"Kelly25 Training Log - {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Training duration: {training_duration/3600:.2f} hours\n")
                f.write(f"Return code: {return_code}\n")
                f.write("=" * 60 + "\n")
                for line in training_log:
                    f.write(line + "\n")
            
            logger.info(f"Training log saved to: {log_file}")
            
            # Check for model files
            model_files = list(output_dir.glob("*.onnx"))
            if model_files:
                logger.info(f"✅ Model files generated: {len(model_files)}")
                for model_file in model_files:
                    logger.info(f"  - {model_file}")
            else:
                logger.warning("⚠️ No .onnx model files found in output directory")
            
            return True
            
        else:
            logger.error(f"❌ Training failed with return code: {return_code}")
            logger.error("Check the training log for details")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return False

def validate_trained_model():
    """Validate the trained Kelly25 model"""
    
    print("\n🔍 Validating Trained Kelly25 Model")
    print("=" * 40)
    
    output_dir = Path("kelly25_model_output")
    model_files = list(output_dir.glob("*.onnx"))
    
    if not model_files:
        logger.error("No trained model files found")
        return False
    
    # Find the best model (usually the last one)
    best_model = max(model_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using model: {best_model}")
    
    # Test the model with sample texts
    test_texts = [
        "Hello! I'm Kelly, your friendly learning companion.",
        "Let's explore this topic together and discover something amazing.",
        "You're doing great! Keep up the excellent work.",
        "I'm so proud of your progress and dedication to learning.",
        "Take a moment to think about what we've learned today."
    ]
    
    for i, text in enumerate(test_texts, 1):
        output_file = output_dir / f"test_sample_{i}.wav"
        
        try:
            # Use piper-tts to generate sample
            cmd = [
                "piper-tts",
                "--model", str(best_model),
                "--text", text,
                "--output_file", str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Test sample {i} generated: {output_file}")
            else:
                logger.error(f"❌ Failed to generate test sample {i}: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ Error generating test sample {i}: {e}")
    
    return True

if __name__ == "__main__":
    # Start training
    success = train_kelly25_model()
    
    if success:
        # Validate the trained model
        validate_trained_model()
        
        print("\n🎉 Kelly25 Training Complete!")
        print("=" * 40)
        print("📁 Model files: kelly25_model_output/")
        print("📋 Training log: kelly25_model_output/training_log.txt")
        print("🧪 Test samples: kelly25_model_output/test_sample_*.wav")
        print("\n🚀 Your Kelly25 voice model is ready for use!")
    else:
        print("\n❌ Training failed. Check the logs for details.")





































