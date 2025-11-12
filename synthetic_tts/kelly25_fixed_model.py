#!/usr/bin/env python3
"""
Kelly25 Fixed Voice Model
Proper TTS implementation with correct architecture and training
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
import matplotlib.pyplot as plt
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessor:
    """Proper text processing for TTS"""
    
    def __init__(self):
        # Create character vocabulary
        self.chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-"
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        logger.info(f"Text processor initialized with {self.vocab_size} characters")
    
    def text_to_ids(self, text: str, max_length: int = 100) -> torch.Tensor:
        """Convert text to character IDs"""
        # Clean and normalize text
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text.lower())
        
        # Convert to IDs
        ids = [self.char_to_id.get(char, 0) for char in text]
        
        # Pad or truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [0] * (max_length - len(ids))
        
        return torch.LongTensor(ids)
    
    def ids_to_text(self, ids: torch.Tensor) -> str:
        """Convert character IDs back to text"""
        text = ''.join([self.id_to_char.get(id.item(), '') for id in ids])
        return text.strip()

class MelSpectrogram:
    """Mel spectrogram processing"""
    
    def __init__(self, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mel spectrogram"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=8000
        )
        mel = np.log(mel + 1e-8)
        return mel
    
    def mel_to_audio(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram back to audio"""
        mel_exp = np.exp(mel)
        audio = librosa.feature.inverse.mel_to_audio(
            mel_exp,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return audio

class Kelly25FixedDataset(Dataset):
    """Fixed dataset for Kelly25 training"""
    
    def __init__(self, metadata_file: str, wavs_dir: str, text_processor: TextProcessor, mel_processor: MelSpectrogram):
        self.metadata_file = metadata_file
        self.wavs_dir = Path(wavs_dir)
        self.text_processor = text_processor
        self.mel_processor = mel_processor
        
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
        audio, sr = librosa.load(str(audio_file), sr=22050)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Convert to mel spectrogram
        mel = self.mel_processor.audio_to_mel(audio)
        
        # Process text
        text_ids = self.text_processor.text_to_ids(text)
        
        return {
            'audio': torch.FloatTensor(audio),
            'mel': torch.FloatTensor(mel),
            'text_ids': text_ids,
            'text': text,
            'id': row['id']
        }

class TextEncoder(nn.Module):
    """Proper text encoder for TTS"""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Output projection
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, text_ids):
        # Embed characters
        embedded = self.embedding(text_ids)
        
        # Transformer encoding
        encoded = self.transformer(embedded)
        
        # Project to hidden dimension
        output = self.proj(encoded)
        
        return output

class MelDecoder(nn.Module):
    """Mel spectrogram decoder"""
    
    def __init__(self, hidden_dim: int = 256, n_mels: int = 80):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, n_mels * 256),  # Output mel frames
        )
        
    def forward(self, text_encoded):
        batch_size = text_encoded.size(0)
        
        # Average pool over text sequence
        pooled = torch.mean(text_encoded, dim=1)
        
        # Decode to mel spectrogram
        mel_flat = self.decoder(pooled)
        mel = mel_flat.view(batch_size, self.n_mels, -1)
        
        return mel

class Vocoder(nn.Module):
    """Neural vocoder to convert mel to audio"""
    
    def __init__(self, n_mels: int = 80, sample_rate: int = 22050):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(n_mels, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, padding=3)
        )
        
    def forward(self, mel):
        # Convert mel to audio
        audio = self.upsample(mel)
        return audio

class Kelly25FixedModel(nn.Module):
    """Fixed Kelly25 TTS model"""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 256, n_mels: int = 80):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        
        # Components
        self.text_encoder = TextEncoder(vocab_size, hidden_dim)
        self.mel_decoder = MelDecoder(hidden_dim, n_mels)
        self.vocoder = Vocoder(n_mels)
        
    def forward(self, text_ids):
        # Encode text
        text_encoded = self.text_encoder(text_ids)
        
        # Decode to mel spectrogram
        mel = self.mel_decoder(text_encoded)
        
        # Convert mel to audio
        audio = self.vocoder(mel)
        
        return audio, mel

class Kelly25FixedTrainer:
    """Fixed trainer for Kelly25 model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.mel_processor = MelSpectrogram()
        
        # Initialize model
        self.model = Kelly25FixedModel(
            vocab_size=self.text_processor.vocab_size,
            hidden_dim=config.get('hidden_dim', 256),
            n_mels=config.get('n_mels', 80)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.0001)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Training history
        self.history = {
            'total_loss': [],
            'mel_loss': [],
            'audio_loss': []
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train one epoch"""
        self.model.train()
        
        total_losses = []
        mel_losses = []
        audio_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(self.device)
            mel = batch['mel'].to(self.device)
            text_ids = batch['text_ids'].to(self.device)
            
            # Forward pass
            generated_audio, generated_mel = self.model(text_ids)
            
            # Calculate losses
            mel_loss = self.mse_loss(generated_mel, mel)
            audio_loss = self.l1_loss(generated_audio.squeeze(1), audio)
            total_loss = mel_loss + audio_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_losses.append(total_loss.item())
            mel_losses.append(mel_loss.item())
            audio_losses.append(audio_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Mel': f'{mel_loss.item():.4f}',
                'Audio': f'{audio_loss.item():.4f}'
            })
        
        # Store epoch losses
        self.history['total_loss'].append(np.mean(total_losses))
        self.history['mel_loss'].append(np.mean(mel_losses))
        self.history['audio_loss'].append(np.mean(audio_losses))
        
        return {
            'total_loss': np.mean(total_losses),
            'mel_loss': np.mean(mel_losses),
            'audio_loss': np.mean(audio_losses)
        }
    
    def generate_audio(self, text: str) -> np.ndarray:
        """Generate audio from text"""
        self.model.eval()
        
        with torch.no_grad():
            # Process text
            text_ids = self.text_processor.text_to_ids(text).unsqueeze(0).to(self.device)
            
            # Generate audio
            generated_audio, _ = self.model(text_ids)
            
            # Convert to numpy
            audio_np = generated_audio.squeeze().cpu().numpy()
            
            return audio_np
    
    def save_checkpoint(self, epoch: int, output_dir: Path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config,
            'text_processor': {
                'chars': self.text_processor.chars,
                'char_to_id': self.text_processor.char_to_id
            }
        }
        
        checkpoint_file = output_dir / f'fixed_checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")

def train_kelly25_fixed():
    """Train the fixed Kelly25 model"""
    
    print("🔧 Training Fixed Kelly25 Voice Model")
    print("=" * 60)
    
    # Configuration
    config = {
        'hidden_dim': 256,
        'n_mels': 80,
        'learning_rate': 0.0001,
        'batch_size': 4,  # Smaller batch size
        'epochs': 20,
        'checkpoint_interval': 5,
        'sample_rate': 22050
    }
    
    # Paths
    train_metadata = "kelly25_training_data/training_splits/train_metadata.csv"
    val_metadata = "kelly25_training_data/training_splits/val_metadata.csv"
    wavs_dir = "kelly25_training_data/wavs"
    output_dir = Path("kelly25_fixed_model")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = Kelly25FixedTrainer(config)
    
    # Initialize datasets
    train_dataset = Kelly25FixedDataset(train_metadata, wavs_dir, trainer.text_processor, trainer.mel_processor)
    val_dataset = Kelly25FixedDataset(val_metadata, wavs_dir, trainer.text_processor, trainer.mel_processor)
    
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
                   f"Total: {train_metrics['total_loss']:.4f}, "
                   f"Mel: {train_metrics['mel_loss']:.4f}, "
                   f"Audio: {train_metrics['audio_loss']:.4f}")
        
        # Save checkpoint
        if epoch % config['checkpoint_interval'] == 0:
            trainer.save_checkpoint(epoch, output_dir)
            
            # Generate sample
            sample_text = "Hello! I'm Kelly, your friendly learning companion."
            sample_audio = trainer.generate_audio(sample_text)
            sample_file = output_dir / f'fixed_sample_epoch_{epoch}.wav'
            sf.write(sample_file, sample_audio, 22050)
            logger.info(f"Sample generated: {sample_file}")
        
        # Save best model
        if train_metrics['total_loss'] < best_loss:
            best_loss = train_metrics['total_loss']
            trainer.save_checkpoint(epoch, output_dir)
            torch.save(trainer.model.state_dict(), output_dir / 'best_fixed_model.pth')
    
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
        'final_total_loss': train_metrics['total_loss'],
        'final_mel_loss': train_metrics['mel_loss'],
        'final_audio_loss': train_metrics['audio_loss']
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("🎉 Fixed training completed!")
    logger.info(f"Duration: {duration.total_seconds()/3600:.2f} hours")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    
    return True

if __name__ == "__main__":
    train_kelly25_fixed()




































