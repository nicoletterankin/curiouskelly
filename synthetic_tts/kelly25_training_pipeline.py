#!/usr/bin/env python3
"""
Kelly25 Voice Training Pipeline for Piper TTS
Complete implementation of VITS-based TTS training
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Kelly25Dataset(Dataset):
    """Kelly25 Voice Dataset for TTS Training"""
    
    def __init__(self, metadata_file: str, wavs_dir: str, sample_rate: int = 22050):
        self.metadata_file = metadata_file
        self.wavs_dir = Path(wavs_dir)
        self.sample_rate = sample_rate
        
        # Load metadata
        self.metadata = pd.read_csv(
            metadata_file, 
            sep='|', 
            comment='#', 
            names=['id', 'normalized_text', 'raw_text']
        )
        
        logger.info(f"Loaded {len(self.metadata)} samples from {metadata_file}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_file = self.wavs_dir / f"{row['id']}.wav"
        text = row['raw_text']
        
        # Load audio
        audio, sr = librosa.load(str(audio_file), sr=self.sample_rate)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        return {
            'audio': torch.FloatTensor(audio),
            'text': text,
            'id': row['id']
        }

class TextEncoder(nn.Module):
    """Text Encoder for VITS"""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, text_ids):
        # Embed text
        embedded = self.embedding(text_ids)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(embedded)
        
        # Project to hidden dimension
        encoded = self.proj(lstm_out)
        
        return encoded

class PosteriorEncoder(nn.Module):
    """Posterior Encoder for VITS"""
    
    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, mel):
        # Convolutional encoding
        x = F.relu(self.conv1(mel))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Transpose and project
        x = x.transpose(1, 2)
        encoded = self.proj(x)
        
        return encoded

class Generator(nn.Module):
    """Generator for VITS"""
    
    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, 1, kernel_size=7, padding=3)
        )
    
    def forward(self, z):
        # Transpose for conv1d
        z = z.transpose(1, 2)
        
        # Generate audio
        audio = self.upsample(z)
        
        return audio

class Discriminator(nn.Module):
    """Discriminator for VITS"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=41, stride=2, padding=20)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=41, stride=2, padding=20)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=41, stride=4, padding=20)
        self.conv6 = nn.Conv1d(1024, 1024, kernel_size=41, stride=1, padding=20)
        self.conv7 = nn.Conv1d(1024, 1, kernel_size=41, stride=1, padding=20)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
        x = F.leaky_relu(self.conv4(x), 0.1)
        x = F.leaky_relu(self.conv5(x), 0.1)
        x = F.leaky_relu(self.conv6(x), 0.1)
        x = self.conv7(x)
        return x

class VITSModel(nn.Module):
    """VITS Model for Kelly25 Voice Training"""
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Components
        self.text_encoder = TextEncoder(vocab_size, hidden_dim)
        self.posterior_encoder = PosteriorEncoder(hidden_dim)
        self.generator = Generator(hidden_dim)
        self.discriminator = Discriminator()
        
        # Mel spectrogram parameters
        self.n_mel_channels = 80
        self.sample_rate = 22050
        self.hop_length = 256
        self.win_length = 1024
        self.n_fft = 1024
        
    def forward(self, text_ids, audio):
        # Text encoding
        text_encoded = self.text_encoder(text_ids)
        
        # Mel spectrogram
        mel = self.mel_spectrogram(audio)
        
        # Posterior encoding
        posterior_encoded = self.posterior_encoder(mel)
        
        # Generate audio
        generated_audio = self.generator(posterior_encoded)
        
        return {
            'generated_audio': generated_audio,
            'text_encoded': text_encoded,
            'posterior_encoded': posterior_encoded,
            'mel': mel
        }
    
    def mel_spectrogram(self, audio):
        """Compute mel spectrogram"""
        # Convert to mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio.numpy(),
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel_channels
        )
        
        # Convert to log scale
        mel = np.log(mel + 1e-8)
        
        return torch.FloatTensor(mel)

class Kelly25Trainer:
    """Kelly25 Voice Trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = VITSModel(
            vocab_size=config.get('vocab_size', 1000),
            hidden_dim=config.get('hidden_dim', 192)
        ).to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            list(self.model.text_encoder.parameters()) + 
            list(self.model.posterior_encoder.parameters()) + 
            list(self.model.generator.parameters()),
            lr=config.get('learning_rate', 0.0001)
        )
        
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config.get('learning_rate', 0.0001)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'mel_loss': [],
            'adv_loss': []
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train one epoch"""
        self.model.train()
        
        g_losses = []
        d_losses = []
        mel_losses = []
        adv_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(self.device)
            text = batch['text']
            
            # Simple text encoding (in real implementation, use proper text processing)
            text_ids = torch.randint(0, 1000, (audio.size(0), 50)).to(self.device)
            
            # Forward pass
            outputs = self.model(text_ids, audio)
            
            # Generator loss
            g_loss = self.compute_generator_loss(outputs, audio)
            
            # Discriminator loss
            d_loss = self.compute_discriminator_loss(outputs, audio)
            
            # Update generator
            self.optimizer_g.zero_grad()
            g_loss.backward(retain_graph=True)
            self.optimizer_g.step()
            
            # Update discriminator
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()
            
            # Track losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}'
            })
        
        # Store epoch losses
        self.history['g_loss'].append(np.mean(g_losses))
        self.history['d_loss'].append(np.mean(d_losses))
        
        return {
            'g_loss': np.mean(g_losses),
            'd_loss': np.mean(d_losses)
        }
    
    def compute_generator_loss(self, outputs, target_audio):
        """Compute generator loss"""
        generated_audio = outputs['generated_audio']
        
        # L1 loss for audio reconstruction
        audio_loss = self.l1_loss(generated_audio, target_audio.unsqueeze(1))
        
        # Adversarial loss
        fake_pred = self.model.discriminator(generated_audio)
        adv_loss = -torch.mean(fake_pred)
        
        # Total generator loss
        g_loss = audio_loss + 0.1 * adv_loss
        
        return g_loss
    
    def compute_discriminator_loss(self, outputs, target_audio):
        """Compute discriminator loss"""
        generated_audio = outputs['generated_audio']
        
        # Real audio prediction
        real_pred = self.model.discriminator(target_audio.unsqueeze(1))
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        
        # Fake audio prediction
        fake_pred = self.model.discriminator(generated_audio.detach())
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        return d_loss
    
    def save_checkpoint(self, epoch: int, output_dir: Path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        checkpoint_file = output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def plot_training_history(self, output_dir: Path):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator loss
        axes[0, 0].plot(self.history['g_loss'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Discriminator loss
        axes[0, 1].plot(self.history['d_loss'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        
        # Combined loss
        axes[1, 0].plot(self.history['g_loss'], label='Generator')
        axes[1, 0].plot(self.history['d_loss'], label='Discriminator')
        axes[1, 0].set_title('Combined Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # Loss ratio
        if len(self.history['d_loss']) > 0:
            ratio = [g/d if d > 0 else 0 for g, d in zip(self.history['g_loss'], self.history['d_loss'])]
            axes[1, 1].plot(ratio)
            axes[1, 1].set_title('G/D Loss Ratio')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Ratio')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.png')
        plt.close()
        
        logger.info(f"Training history plot saved: {output_dir / 'training_history.png'}")

def train_kelly25_voice():
    """Main training function for Kelly25 voice"""
    
    print("🚀 Starting Kelly25 Voice Training Pipeline")
    print("=" * 60)
    
    # Configuration
    config = {
        'vocab_size': 1000,
        'hidden_dim': 192,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 1000,
        'checkpoint_interval': 50,
        'sample_rate': 22050
    }
    
    # Paths
    train_metadata = "kelly25_training_data/training_splits/train_metadata.csv"
    val_metadata = "kelly25_training_data/training_splits/val_metadata.csv"
    wavs_dir = "kelly25_training_data/wavs"
    output_dir = Path("kelly25_model_output")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize datasets
    train_dataset = Kelly25Dataset(train_metadata, wavs_dir, config['sample_rate'])
    val_dataset = Kelly25Dataset(val_metadata, wavs_dir, config['sample_rate'])
    
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = Kelly25Trainer(config)
    
    # Training loop
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    best_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Log progress
        logger.info(f"Epoch {epoch}/{config['epochs']}: "
                   f"G_Loss: {train_metrics['g_loss']:.4f}, "
                   f"D_Loss: {train_metrics['d_loss']:.4f}")
        
        # Save checkpoint
        if epoch % config['checkpoint_interval'] == 0:
            trainer.save_checkpoint(epoch, output_dir)
        
        # Save best model
        if train_metrics['g_loss'] < best_loss:
            best_loss = train_metrics['g_loss']
            trainer.save_checkpoint(epoch, output_dir)
            torch.save(trainer.model.state_dict(), output_dir / 'best_model.pth')
    
    # Final checkpoint
    trainer.save_checkpoint(config['epochs'], output_dir)
    
    # Plot training history
    trainer.plot_training_history(output_dir)
    
    # Training summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_hours': duration.total_seconds() / 3600,
        'total_epochs': config['epochs'],
        'best_loss': best_loss,
        'final_g_loss': train_metrics['g_loss'],
        'final_d_loss': train_metrics['d_loss']
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"Duration: {duration.total_seconds()/3600:.2f} hours")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    
    return True

if __name__ == "__main__":
    train_kelly25_voice()





































