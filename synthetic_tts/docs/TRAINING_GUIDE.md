# Synthetic Digital TTS System - Training Guide

This guide provides comprehensive instructions for training the Synthetic Digital TTS System models.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Evaluation and Testing](#evaluation-and-testing)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)

## Overview

The Synthetic Digital TTS System consists of three main components:

1. **FastPitch Acoustic Model**: Generates mel-spectrograms from text and speaker embeddings
2. **HiFi-GAN Vocoder**: Converts mel-spectrograms to high-quality audio waveforms
3. **Speaker Embedding Generator**: Creates synthetic speaker embeddings from voice characteristics

### Training Pipeline

The training process involves:

1. **Data Generation**: Creating synthetic training data from multi-speaker datasets
2. **Model Training**: Training each component with appropriate loss functions
3. **Fine-tuning**: Optimizing models for specific voice characteristics
4. **Evaluation**: Testing model performance and quality

## Data Preparation

### Synthetic Data Generation

The system generates its own training data without requiring human voice samples:

```bash
# Generate synthetic training data
python train_models.py --generate-data --data-size 10000
```

This creates:
- `data/audio/`: Synthetic audio files
- `data/mel/`: Mel-spectrograms
- `data/metadata/`: Text and acoustic feature metadata

### Data Structure

Each training sample contains:

```json
{
  "text": "Hello, world!",
  "text_metadata": {
    "phonemes": ["HH", "EH", "L", "OW", "SIL", "W", "ER", "L", "D"],
    "tokens": [1, 2, 3, 4, 0, 5, 6, 3, 7],
    "emotion": "neutral"
  },
  "speaker_embedding": [0.12, -0.45, 0.78, ...],
  "voice_archetype": "young_female",
  "mel_spectrogram": [[0.1, 0.2, ...], ...],
  "duration": [5.0, 3.0, 4.0, ...],
  "pitch": [180.0, 185.0, 190.0, ...],
  "energy": [0.7, 0.8, 0.6, ...]
}
```

### Data Augmentation

The system includes several data augmentation techniques:

1. **Voice Variation**: Random variations in voice characteristics
2. **Prosodic Variation**: Different emotional expressions
3. **Text Variation**: Diverse text samples and lengths
4. **Acoustic Variation**: Synthetic acoustic feature generation

## Model Architecture

### FastPitch Acoustic Model

The FastPitch model consists of:

- **Text Encoder**: Converts phonemes to hidden representations
- **Speaker Embedding Projection**: Incorporates speaker information
- **Duration Predictor**: Predicts phoneme durations
- **Pitch Predictor**: Predicts fundamental frequency
- **Energy Predictor**: Predicts speech energy
- **Decoder**: Generates mel-spectrograms

```python
# Model configuration
fastpitch = FastPitch(
    n_mel_channels=80,
    n_symbols=256,
    encoder_embedding_dim=384,
    decoder_rnn_dim=1024,
    speaker_embedding_dim=64,
)
```

### HiFi-GAN Vocoder

The HiFi-GAN vocoder includes:

- **Upsampling Layers**: Convert mel-spectrograms to audio
- **Residual Blocks**: Refine audio quality
- **Multi-Period Discriminator**: Adversarial training
- **Multi-Scale Discriminator**: Multi-resolution quality control

```python
# Vocoder configuration
hifigan = HiFiGAN(
    n_mel_channels=80,
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=512,
)
```

### Speaker Embedding Generator

The speaker embedding system:

- **Mapping Network**: Converts voice descriptors to embeddings
- **Basis Vectors**: Learnable voice characteristic bases
- **Characteristic Encoders**: Individual voice feature encoders

```python
# Speaker embedding configuration
speaker_embedding = SpeakerEmbedding(
    embedding_dim=64,
    descriptor_dim=8,
    hidden_dim=128,
)
```

## Training Process

### Basic Training

Train all models with default parameters:

```bash
python train_models.py \
    --data-dir data \
    --output-dir models \
    --config config/character_voice.json \
    --epochs 100 \
    --batch-size 32
```

### Individual Model Training

Train specific models:

```bash
# Train only FastPitch
python train_models.py --fastpitch-only --epochs 50

# Train only HiFi-GAN
python train_models.py --hifigan-only --epochs 50
```

### Resume Training

Resume from a checkpoint:

```bash
python train_models.py \
    --resume models/fastpitch_epoch_50.pt \
    --epochs 100
```

### Training Configuration

Create a custom training configuration:

```json
{
  "model": {
    "fastpitch": {
      "n_mel_channels": 80,
      "n_symbols": 256,
      "encoder_embedding_dim": 384,
      "decoder_rnn_dim": 1024,
      "speaker_embedding_dim": 64
    },
    "hifigan": {
      "n_mel_channels": 80,
      "upsample_rates": [8, 8, 2, 2],
      "upsample_initial_channel": 512
    }
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 100,
    "save_interval": 10,
    "validate_interval": 5
  }
}
```

## Hyperparameter Tuning

### Learning Rate

Start with a learning rate of 1e-4 and adjust based on training progress:

```bash
# Lower learning rate for fine-tuning
python train_models.py --learning-rate 5e-5

# Higher learning rate for initial training
python train_models.py --learning-rate 2e-4
```

### Batch Size

Adjust batch size based on available memory:

```bash
# Small batch size for limited memory
python train_models.py --batch-size 16

# Large batch size for better convergence
python train_models.py --batch-size 64
```

### Model Architecture

Experiment with different model sizes:

```python
# Smaller model for faster training
fastpitch = FastPitch(
    encoder_embedding_dim=256,
    decoder_rnn_dim=512,
)

# Larger model for better quality
fastpitch = FastPitch(
    encoder_embedding_dim=512,
    decoder_rnn_dim=2048,
)
```

### Loss Function Weights

Balance different loss components:

```python
# FastPitch loss weights
mel_loss_weight = 1.0
duration_loss_weight = 0.1
pitch_loss_weight = 0.1
energy_loss_weight = 0.1

total_loss = (
    mel_loss_weight * mel_loss +
    duration_loss_weight * duration_loss +
    pitch_loss_weight * pitch_loss +
    energy_loss_weight * energy_loss
)
```

## Evaluation and Testing

### Training Metrics

Monitor these metrics during training:

1. **Mel-spectrogram Loss**: L1 loss between predicted and target mel-spectrograms
2. **Duration Loss**: MSE loss for phoneme durations
3. **Pitch Loss**: MSE loss for fundamental frequency
4. **Energy Loss**: MSE loss for speech energy
5. **Vocoder Loss**: L1 loss for audio reconstruction

### Validation

Run validation during training:

```bash
python train_models.py \
    --validate-interval 5 \
    --epochs 100
```

### Quality Assessment

Test synthesis quality:

```python
from src.synthesis.inference import InferenceEngine

engine = InferenceEngine("models", "config/character_voice.json")

# Test with various texts
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
]

results = engine.test_synthesis_quality(test_texts, "quality_test")
```

### Objective Metrics

Calculate objective quality metrics:

```python
import librosa
import numpy as np

def calculate_mel_cepstral_distortion(pred_mel, target_mel):
    """Calculate Mel-Cepstral Distortion (MCD)"""
    pred_mfcc = librosa.feature.mfcc(S=pred_mel, n_mfcc=13)
    target_mfcc = librosa.feature.mfcc(S=target_mel, n_mfcc=13)
    
    mcd = np.mean(np.sqrt(2 * np.sum((pred_mfcc - target_mfcc) ** 2, axis=0)))
    return mcd

def calculate_spectral_distance(pred_audio, target_audio):
    """Calculate spectral distance"""
    pred_stft = librosa.stft(pred_audio)
    target_stft = librosa.stft(target_audio)
    
    pred_mag = np.abs(pred_stft)
    target_mag = np.abs(target_stft)
    
    distance = np.mean(np.abs(pred_mag - target_mag))
    return distance
```

## Troubleshooting

### Common Training Issues

#### 1. Loss Not Decreasing

**Symptoms**: Loss remains constant or increases

**Solutions**:
- Reduce learning rate
- Check data quality
- Verify model architecture
- Ensure proper data preprocessing

```bash
# Lower learning rate
python train_models.py --learning-rate 1e-5

# Check data quality
python -c "from src.data.dataset import TTSDataset; dataset = TTSDataset('data', {}); print(dataset.get_dataset_stats())"
```

#### 2. Memory Issues

**Symptoms**: CUDA out of memory errors

**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

```bash
# Smaller batch size
python train_models.py --batch-size 16

# Use CPU if needed
python train_models.py --device cpu
```

#### 3. Poor Audio Quality

**Symptoms**: Generated audio sounds unnatural

**Solutions**:
- Train for more epochs
- Adjust model architecture
- Improve data quality
- Fine-tune hyperparameters

```bash
# Train longer
python train_models.py --epochs 200

# Use larger model
python train_models.py --config config/large_model.json
```

#### 4. Training Instability

**Symptoms**: Loss oscillates or diverges

**Solutions**:
- Use gradient clipping
- Adjust learning rate schedule
- Check data normalization
- Verify loss function weights

```python
# Add gradient clipping
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Debugging Tools

#### 1. Training Visualization

Plot training curves:

```python
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
```

#### 2. Model Inspection

Check model parameters:

```python
def inspect_model(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check for NaN or Inf values
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {name}")
        if torch.isinf(param).any():
            print(f"Inf found in {name}")
```

#### 3. Data Validation

Validate training data:

```python
def validate_data(dataset):
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        
        # Check for NaN or Inf values
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    print(f"NaN found in {key} at sample {i}")
                if torch.isinf(value).any():
                    print(f"Inf found in {key} at sample {i}")
```

## Advanced Topics

### Transfer Learning

Use pretrained models for faster training:

```bash
# Use pretrained FastPitch
python train_models.py --pretrained-fastpitch pretrained/fastpitch.pt

# Use pretrained HiFi-GAN
python train_models.py --pretrained-hifigan pretrained/hifigan.pt
```

### Multi-GPU Training

Train on multiple GPUs:

```python
import torch.nn as nn

# Wrap model for multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Mixed Precision Training

Use mixed precision for faster training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Custom Loss Functions

Implement custom loss functions:

```python
class CustomLoss(nn.Module):
    def __init__(self, mel_weight=1.0, duration_weight=0.1):
        super().__init__()
        self.mel_weight = mel_weight
        self.duration_weight = duration_weight
        self.mel_criterion = nn.L1Loss()
        self.duration_criterion = nn.MSELoss()
    
    def forward(self, pred, target):
        mel_loss = self.mel_criterion(pred['mel'], target['mel'])
        duration_loss = self.duration_criterion(pred['duration'], target['duration'])
        
        total_loss = (
            self.mel_weight * mel_loss +
            self.duration_weight * duration_loss
        )
        
        return total_loss, {
            'mel_loss': mel_loss.item(),
            'duration_loss': duration_loss.item(),
            'total_loss': total_loss.item()
        }
```

### Model Ensemble

Combine multiple models for better quality:

```python
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average predictions
        return torch.mean(torch.stack(outputs), dim=0)
```

### Continuous Learning

Implement continuous learning for new voices:

```python
def fine_tune_for_new_voice(model, new_voice_data, learning_rate=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(10):  # Few epochs for fine-tuning
        for batch in new_voice_data:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch['target'])
            loss.backward()
            optimizer.step()
    
    return model
```

## Best Practices

### 1. Data Quality

- Ensure diverse and representative training data
- Validate data preprocessing pipeline
- Monitor data distribution and balance

### 2. Model Architecture

- Start with proven architectures
- Gradually increase model complexity
- Use appropriate regularization techniques

### 3. Training Strategy

- Use learning rate scheduling
- Implement early stopping
- Monitor multiple metrics
- Save checkpoints regularly

### 4. Evaluation

- Use both objective and subjective metrics
- Test on diverse text samples
- Compare with baseline models
- Gather user feedback

### 5. Deployment

- Optimize models for inference
- Test on target hardware
- Implement proper error handling
- Monitor performance in production

## Conclusion

Training the Synthetic Digital TTS System requires careful attention to data preparation, model architecture, and training strategy. By following this guide and experimenting with different approaches, you can achieve high-quality synthetic speech generation.

For more information, see the [Usage Guide](USAGE_GUIDE.md) and [API documentation](API.md).








































