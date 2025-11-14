#!/usr/bin/env python3
"""
Training script for the Synthetic Digital TTS System.

This script trains the FastPitch acoustic model and HiFi-GAN vocoder
using synthetic data generated from multi-speaker datasets.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models import FastPitch, HiFiGAN, SpeakerEmbedding
from data import TextProcessor, SyntheticDataGenerator, TTSDataset
from synthesis.prosody_controller import ProsodyController


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Synthetic Digital TTS System"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing training data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/character_voice.json",
        help="Path to character voice configuration"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run training on"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save model every N epochs"
    )
    
    parser.add_argument(
        "--validate-interval",
        type=int,
        default=5,
        help="Run validation every N epochs"
    )
    
    # Model arguments
    parser.add_argument(
        "--fastpitch-only",
        action="store_true",
        help="Train only FastPitch model"
    )
    
    parser.add_argument(
        "--hifigan-only",
        action="store_true",
        help="Train only HiFi-GAN model"
    )
    
    parser.add_argument(
        "--pretrained-fastpitch",
        type=str,
        help="Path to pretrained FastPitch model"
    )
    
    parser.add_argument(
        "--pretrained-hifigan",
        type=str,
        help="Path to pretrained HiFi-GAN model"
    )
    
    # Data generation arguments
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic training data"
    )
    
    parser.add_argument(
        "--data-size",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate"
    )
    
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=100,
        help="Maximum text length for training"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for training logs"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def get_device(device: str) -> torch.device:
    """Determine the appropriate device for training."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def create_directories(output_dir: str, log_dir: str):
    """Create necessary directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)


def generate_synthetic_data(
    data_dir: str,
    data_size: int,
    max_text_length: int,
    config: Dict,
) -> None:
    """Generate synthetic training data."""
    print("Generating synthetic training data...")
    
    # Initialize data generator
    data_generator = SyntheticDataGenerator(config)
    
    # Generate data
    data_generator.generate_dataset(
        output_dir=data_dir,
        num_samples=data_size,
        max_text_length=max_text_length,
    )
    
    print(f"Synthetic data generation complete. Saved to: {data_dir}")


def create_data_loaders(
    data_dir: str,
    batch_size: int,
    config: Dict,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    # Load dataset
    dataset = TTSDataset(data_dir, config)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def train_fastpitch(
    model: FastPitch,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: str,
    log_dir: str,
    save_interval: int,
    validate_interval: int,
    verbose: bool = False,
) -> None:
    """Train the FastPitch acoustic model."""
    print("Training FastPitch model...")
    
    # Move model to device
    model.to(device)
    model.train()
    
    # Initialize optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mel = nn.L1Loss()
    criterion_duration = nn.MSELoss()
    criterion_pitch = nn.MSELoss()
    criterion_energy = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Move batch to device
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            mel_target = batch['mel_target'].to(device)
            duration_target = batch['duration'].to(device)
            pitch_target = batch['pitch'].to(device)
            energy_target = batch['energy'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(text, text_lengths, speaker_embedding)
            
            # Calculate losses
            mel_loss = criterion_mel(outputs['mel_outputs'], mel_target)
            duration_loss = criterion_duration(outputs['duration'], duration_target)
            pitch_loss = criterion_pitch(outputs['pitch'], pitch_target)
            energy_loss = criterion_energy(outputs['energy'], energy_target)
            
            total_loss = mel_loss + duration_loss + pitch_loss + energy_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if verbose and batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Loss = {total_loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if (epoch + 1) % validate_interval == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    text = batch['text'].to(device)
                    text_lengths = batch['text_lengths'].to(device)
                    speaker_embedding = batch['speaker_embedding'].to(device)
                    mel_target = batch['mel_target'].to(device)
                    duration_target = batch['duration'].to(device)
                    pitch_target = batch['pitch'].to(device)
                    energy_target = batch['energy'].to(device)
                    
                    outputs = model(text, text_lengths, speaker_embedding)
                    
                    mel_loss = criterion_mel(outputs['mel_outputs'], mel_target)
                    duration_loss = criterion_duration(outputs['duration'], duration_target)
                    pitch_loss = criterion_pitch(outputs['pitch'], pitch_target)
                    energy_loss = criterion_energy(outputs['energy'], energy_target)
                    
                    total_loss = mel_loss + duration_loss + pitch_loss + energy_loss
                    val_loss += total_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save model
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if (epoch + 1) % validate_interval == 0 else None,
            }
            
            torch.save(checkpoint, os.path.join(output_dir, f"fastpitch_epoch_{epoch+1}.pt"))
            print(f"Saved FastPitch checkpoint: epoch {epoch+1}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(output_dir, "fastpitch.pt"))
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, log_dir, "fastpitch")
    
    print("FastPitch training complete!")


def train_hifigan(
    model: HiFiGAN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: str,
    log_dir: str,
    save_interval: int,
    validate_interval: int,
    verbose: bool = False,
) -> None:
    """Train the HiFi-GAN vocoder."""
    print("Training HiFi-GAN model...")
    
    # Move model to device
    model.to(device)
    model.train()
    
    # Initialize optimizers and loss functions
    optimizer_g = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mel = nn.L1Loss()
    criterion_adv = nn.BCEWithLogitsLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Move batch to device
            mel = batch['mel_target'].to(device)
            audio_target = batch['audio_target'].to(device)
            
            # Forward pass
            optimizer_g.zero_grad()
            audio_generated = model(mel)
            
            # Calculate losses
            mel_loss = criterion_mel(audio_generated, audio_target)
            
            # For simplicity, we'll use a basic reconstruction loss
            # In practice, you'd also train discriminators
            total_loss = mel_loss
            
            # Backward pass
            total_loss.backward()
            optimizer_g.step()
            
            train_loss += total_loss.item()
            
            if verbose and batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Loss = {total_loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if (epoch + 1) % validate_interval == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    mel = batch['mel_target'].to(device)
                    audio_target = batch['audio_target'].to(device)
                    
                    audio_generated = model(mel)
                    mel_loss = criterion_mel(audio_generated, audio_target)
                    val_loss += mel_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save model
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if (epoch + 1) % validate_interval == 0 else None,
            }
            
            torch.save(checkpoint, os.path.join(output_dir, f"hifigan_epoch_{epoch+1}.pt"))
            print(f"Saved HiFi-GAN checkpoint: epoch {epoch+1}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_g.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(output_dir, "hifigan.pt"))
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, log_dir, "hifigan")
    
    print("HiFi-GAN training complete!")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    log_dir: str,
    model_name: str,
) -> None:
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name.title()} Training Curves')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(log_dir, "plots", f"{model_name}_training_curves.png"))
    plt.close()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create directories
    create_directories(args.output_dir, args.log_dir)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Generate synthetic data if requested
    if args.generate_data:
        generate_synthetic_data(
            args.data_dir,
            args.data_size,
            args.max_text_length,
            config,
        )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        args.batch_size,
        config,
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Train FastPitch
    if not args.hifigan_only:
        fastpitch = FastPitch(
            n_mel_channels=config.get('audio_settings', {}).get('n_mels', 80),
            n_symbols=256,
            speaker_embedding_dim=64,
        )
        
        if args.pretrained_fastpitch:
            checkpoint = torch.load(args.pretrained_fastpitch, map_location=device)
            fastpitch.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded pretrained FastPitch model")
        
        train_fastpitch(
            fastpitch,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.learning_rate,
            args.output_dir,
            args.log_dir,
            args.save_interval,
            args.validate_interval,
            args.verbose,
        )
    
    # Train HiFi-GAN
    if not args.fastpitch_only:
        hifigan = HiFiGAN(
            n_mel_channels=config.get('audio_settings', {}).get('n_mels', 80),
        )
        
        if args.pretrained_hifigan:
            checkpoint = torch.load(args.pretrained_hifigan, map_location=device)
            hifigan.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded pretrained HiFi-GAN model")
        
        train_hifigan(
            hifigan,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.learning_rate,
            args.output_dir,
            args.log_dir,
            args.save_interval,
            args.validate_interval,
            args.verbose,
        )
    
    print("Training complete!")


if __name__ == "__main__":
    main()









































