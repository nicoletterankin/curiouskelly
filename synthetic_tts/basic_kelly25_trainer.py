#!/usr/bin/env python3
"""
Basic Kelly25 Voice Trainer
A basic TTS training implementation for Kelly25 voice
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
from typing import Dict, List, Tuple
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicTTSDataset(Dataset):
    """Basic TTS Dataset for Kelly25 Voice"""
    
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
        
        # Pad or truncate audio to fixed length (5 seconds = 110250 samples)
        target_length = 110250  # 5 seconds at 22050 Hz
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Create simple text encoding (character-level)
        text_ids = [ord(c) for c in text[:50]]  # Limit to 50 chars
        text_ids = text_ids + [0] * (50 - len(text_ids))  # Pad to 50
        
        return {
            'audio': torch.FloatTensor(audio),
            'text_ids': torch.LongTensor(text_ids),
            'text': text,
            'id': row['id']
        }

class BasicTTSModel(nn.Module):
    """Basic TTS Model for Kelly25 Voice"""
    
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, hidden_dim * 16),
            nn.ReLU(),
            nn.Linear(hidden_dim * 16, 110250)  # Fixed output length
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(110250, hidden_dim * 16),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 16, hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, 1)
        )
    
    def forward(self, text_ids):
        # Text encoding
        text_embedded = self.text_embedding(text_ids)
        text_lstm_out, _ = self.text_lstm(text_embedded)
        text_encoded = self.text_proj(text_lstm_out)
        
        # Average pooling over text sequence
        text_pooled = torch.mean(text_encoded, dim=1)  # [batch_size, hidden_dim]
        
        # Generate audio
        generated_audio = self.generator(text_pooled)
        
        return generated_audio
    
    def discriminate(self, audio):
        return self.discriminator(audio)

class BasicKelly25Trainer:
    """Basic Kelly25 Voice Trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cpu')  # Force CPU due to RTX 5090 compatibility
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = BasicTTSModel(
            vocab_size=config.get('vocab_size', 256),
            hidden_dim=config.get('hidden_dim', 128)
        ).to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            [p for name, p in self.model.named_parameters() if 'discriminator' not in name],
            lr=config.get('learning_rate', 0.0001)
        )
        
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config.get('learning_rate', 0.0001)
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'recon_loss': [],
            'adv_loss': []
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train one epoch"""
        self.model.train()
        
        g_losses = []
        d_losses = []
        recon_losses = []
        adv_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(self.device)
            text_ids = batch['text_ids'].to(self.device)
            
            # Generate audio
            generated_audio = self.model(text_ids)
            
            # Generator loss
            g_loss = self.compute_generator_loss(generated_audio, audio)
            
            # Discriminator loss
            d_loss = self.compute_discriminator_loss(generated_audio, audio)
            
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
    
    def compute_generator_loss(self, generated_audio, target_audio):
        """Compute generator loss"""
        # Reconstruction loss
        recon_loss = self.l1_loss(generated_audio, target_audio)
        
        # Adversarial loss
        fake_pred = self.model.discriminate(generated_audio)
        adv_loss = -torch.mean(fake_pred)
        
        # Total generator loss
        g_loss = recon_loss + 0.1 * adv_loss
        
        return g_loss
    
    def compute_discriminator_loss(self, generated_audio, target_audio):
        """Compute discriminator loss"""
        # Real audio prediction
        real_pred = self.model.discriminate(target_audio)
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        
        # Fake audio prediction
        fake_pred = self.model.discriminate(generated_audio.detach())
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
    
    def generate_sample(self, text: str, output_file: str):
        """Generate sample audio"""
        self.model.eval()
        
        with torch.no_grad():
            # Encode text
            text_ids = [ord(c) for c in text[:50]]
            text_ids = text_ids + [0] * (50 - len(text_ids))
            text_ids = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
            
            # Generate audio
            generated_audio = self.model(text_ids)
            
            # Convert to numpy and save
            audio_np = generated_audio.squeeze().cpu().numpy()
            sf.write(output_file, audio_np, 22050)
            
            logger.info(f"Sample generated: {output_file}")

def train_kelly25_voice():
    """Main training function for Kelly25 voice"""
    
    print("🚀 Starting Basic Kelly25 Voice Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'vocab_size': 256,
        'hidden_dim': 128,
        'learning_rate': 0.0001,
        'batch_size': 8,  # Smaller batch size for memory
        'epochs': 50,
        'checkpoint_interval': 5,
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
    train_dataset = BasicTTSDataset(train_metadata, wavs_dir, config['sample_rate'])
    val_dataset = BasicTTSDataset(val_metadata, wavs_dir, config['sample_rate'])
    
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize trainer
    trainer = BasicKelly25Trainer(config)
    
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
            
            # Generate sample
            sample_text = "Hello! I'm Kelly, your friendly learning companion."
            sample_file = output_dir / f'sample_epoch_{epoch}.wav'
            trainer.generate_sample(sample_text, str(sample_file))
        
        # Save best model
        if train_metrics['g_loss'] < best_loss:
            best_loss = train_metrics['g_loss']
            trainer.save_checkpoint(epoch, output_dir)
            torch.save(trainer.model.state_dict(), output_dir / 'best_model.pth')
    
    # Final checkpoint
    trainer.save_checkpoint(config['epochs'], output_dir)
    
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
