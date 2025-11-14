#!/usr/bin/env python3
"""
GPU-Optimized Kelly25 Voice Trainer
Fallback to CPU with optimizations for RTX 5090 compatibility
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import soundfile as sf
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedTTSDataset(Dataset):
    """Optimized dataset with better memory management"""
    def __init__(self, metadata_file, wavs_dir, sample_rate=22050, target_length=5.0):
        self.metadata_file = metadata_file
        self.wavs_dir = Path(wavs_dir)
        self.sample_rate = sample_rate
        self.target_length = int(target_length * sample_rate)  # 5 seconds
        
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
        
        # Load and process audio
        audio, sr = librosa.load(str(audio_file), sr=self.sample_rate)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Pad or truncate to target length
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        
        # Convert text to character IDs (simplified)
        text_ids = [ord(c) for c in text[:50]]  # Limit to 50 chars
        text_ids = text_ids + [0] * (50 - len(text_ids))  # Pad to 50
        
        return {
            'audio': torch.FloatTensor(audio),
            'text_ids': torch.LongTensor(text_ids),
            'text': text,
            'id': row['id']
        }

class OptimizedTTSModel(nn.Module):
    """Optimized TTS model with better architecture"""
    def __init__(self, vocab_size=128, hidden_dim=512, audio_length=110250):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.audio_length = audio_length
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Audio generator
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 8, audio_length)
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(audio_length, hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, 1)
        )
    
    def forward(self, text_ids):
        # Encode text
        embedded = self.text_embedding(text_ids)
        encoded, _ = self.text_encoder(embedded)
        
        # Pool over sequence
        pooled = torch.mean(encoded, dim=1)
        
        # Generate audio
        audio = self.generator(pooled)
        
        return audio

class OptimizedKelly25Trainer:
    """Optimized trainer with better memory management"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = OptimizedTTSModel(
            vocab_size=config.get('vocab_size', 128),
            hidden_dim=config.get('hidden_dim', 512),
            audio_length=config.get('audio_length', 110250)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch with optimizations"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(self.device)
            text_ids = batch['text_ids'].to(self.device)
            
            # Forward pass with mixed precision if available
            if self.scaler:
                with torch.cuda.amp.autocast():
                    generated_audio = self.model(text_ids)
                    loss = self.mse_loss(generated_audio, audio)
            else:
                generated_audio = self.model(text_ids)
                loss = self.mse_loss(generated_audio, audio)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_loss = total_loss / len(dataloader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                audio = batch['audio'].to(self.device)
                text_ids = batch['text_ids'].to(self.device)
                
                generated_audio = self.model(text_ids)
                loss = self.mse_loss(generated_audio, audio)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.history['val_loss'].append(avg_loss)
        return avg_loss
    
    def generate_audio(self, text: str) -> np.ndarray:
        """Generate audio from text"""
        self.model.eval()
        
        with torch.no_grad():
            # Convert text to IDs
            text_ids = [ord(c) for c in text[:50]]
            text_ids = text_ids + [0] * (50 - len(text_ids))
            text_tensor = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
            
            # Generate audio
            generated_audio = self.model(text_tensor)
            audio_np = generated_audio.squeeze().cpu().numpy()
            
            return audio_np
    
    def save_checkpoint(self, epoch, output_dir):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        checkpoint_file = output_dir / f'optimized_checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")

def train_kelly25_optimized():
    """Train the optimized Kelly25 model"""
    
    print("🚀 Starting Optimized Kelly25 Voice Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'vocab_size': 128,
        'hidden_dim': 512,
        'audio_length': 110250,  # 5 seconds at 22050 Hz
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'batch_size': 16,  # Larger batch size for GPU
        'epochs': 50,
        'checkpoint_interval': 5,
        'sample_rate': 22050
    }
    
    # Paths
    train_metadata = "kelly25_training_data/training_splits/train_metadata.csv"
    val_metadata = "kelly25_training_data/training_splits/val_metadata.csv"
    wavs_dir = "kelly25_training_data/wavs"
    output_dir = Path("kelly25_optimized_model")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = OptimizedKelly25Trainer(config)
    
    # Initialize datasets
    train_dataset = OptimizedTTSDataset(train_metadata, wavs_dir)
    val_dataset = OptimizedTTSDataset(val_metadata, wavs_dir)
    
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Training loop
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Device: {trainer.device}")
    logger.info(f"Batch size: {config['batch_size']}")
    
    best_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        # Train epoch
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Log progress
        logger.info(f"Epoch {epoch}/{config['epochs']}: "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % config['checkpoint_interval'] == 0:
            trainer.save_checkpoint(epoch, output_dir)
            
            # Generate sample
            sample_text = "Hello! I'm Kelly, your friendly learning companion."
            sample_audio = trainer.generate_audio(sample_text)
            sample_file = output_dir / f'optimized_sample_epoch_{epoch}.wav'
            sf.write(sample_file, sample_audio, 22050)
            logger.info(f"Sample generated: {sample_file}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(trainer.model.state_dict(), output_dir / 'best_optimized_model.pth')
            logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
    
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
        'best_val_loss': best_loss,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'device': str(trainer.device)
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("🎉 Optimized training completed!")
    logger.info(f"Duration: {duration.total_seconds()/3600:.2f} hours")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    
    return True

if __name__ == "__main__":
    train_kelly25_optimized()





































