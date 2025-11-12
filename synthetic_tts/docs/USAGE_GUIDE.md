# Synthetic Digital TTS System - Usage Guide

This guide provides comprehensive instructions for using the Synthetic Digital TTS System to generate high-quality synthetic speech.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Voice Customization](#voice-customization)
4. [Emotional Expression](#emotional-expression)
5. [Batch Processing](#batch-processing)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Quick Start

### 1. Setup

First, run the setup script to initialize the system:

```bash
python scripts/setup.py
```

This will:
- Install all required dependencies
- Create necessary directories
- Generate initial training data
- Train basic models
- Test the synthesis pipeline

### 2. Basic Synthesis

Generate speech from text:

```bash
python synthesize_speech.py --text "Hello, world!" --output hello.wav
```

### 3. Run Demo

See the system in action:

```bash
python scripts/demo.py
```

## Basic Usage

### Command Line Interface

The main synthesis script provides a comprehensive command-line interface:

```bash
python synthesize_speech.py [OPTIONS]
```

#### Required Arguments

- `--text TEXT`: Text to synthesize

#### Optional Arguments

- `--emotion {neutral,happy,sad,angry,excited,calm,question}`: Target emotion (default: neutral)
- `--output OUTPUT`: Output audio file path (default: output.wav)
- `--config CONFIG`: Path to character voice configuration (default: config/character_voice.json)
- `--model-dir MODEL_DIR`: Directory containing trained models (default: models)
- `--device {auto,cpu,cuda}`: Device to run inference on (default: auto)

#### Voice Selection

- `--speaker-archetype {young_female,young_male,mature_female,mature_male,child,elderly}`: Use predefined voice archetype
- `--custom-voice`: Use custom voice characteristics
- `--pitch-mean FLOAT`: Mean pitch level (-1 to 1)
- `--pitch-range FLOAT`: Pitch variability (-1 to 1)
- `--timbre-brightness FLOAT`: Voice brightness (-1 to 1)
- `--timbre-warmth FLOAT`: Voice warmth (-1 to 1)
- `--energy-level FLOAT`: Overall energy level (-1 to 1)
- `--breathiness FLOAT`: Voice breathiness (-1 to 1)

#### Batch Processing

- `--batch BATCH_FILE`: Path to text file for batch synthesis
- `--variants N`: Number of voice variants to generate
- `--output-dir OUTPUT_DIR`: Output directory for batch synthesis

#### Emphasis Control

- `--emphasis EMPHASIS`: Emphasis markers in format 'start:end:type,start:end:type'

### Examples

#### Basic Synthesis

```bash
# Simple text synthesis
python synthesize_speech.py --text "Hello, how are you today?"

# With specific emotion
python synthesize_speech.py --text "I'm so excited!" --emotion excited --output excited.wav

# With voice archetype
python synthesize_speech.py --text "Welcome to our presentation" --speaker-archetype mature_male --emotion calm
```

#### Custom Voice

```bash
# Create a custom voice
python synthesize_speech.py \
    --text "This is my custom voice" \
    --custom-voice \
    --pitch-mean 0.3 \
    --timbre-brightness 0.5 \
    --energy-level 0.7 \
    --output custom_voice.wav
```

#### Batch Processing

```bash
# Create a text file with multiple lines
echo -e "Hello world\nHow are you?\nThis is a test" > texts.txt

# Synthesize all texts
python synthesize_speech.py --batch texts.txt --output-dir batch_output

# Generate voice variants
python synthesize_speech.py --text "Hello" --variants 5 --output-dir variants
```

#### Emphasis Control

```bash
# Add emphasis markers
python synthesize_speech.py \
    --text "This is *very* important" \
    --emphasis "0:4:strong,8:17:strong" \
    --emotion excited
```

## Voice Customization

### Voice Archetypes

The system includes six predefined voice archetypes:

1. **young_female**: High pitch, bright timbre, energetic
2. **young_male**: Medium pitch, warm timbre, confident
3. **mature_female**: Medium-high pitch, warm timbre, professional
4. **mature_male**: Low pitch, warm timbre, authoritative
5. **child**: Very high pitch, bright timbre, playful
6. **elderly**: Medium pitch, warm timbre, calm

### Custom Voice Parameters

You can create custom voices by adjusting these parameters:

- **pitch_mean**: Overall pitch level (-1 = very low, +1 = very high)
- **pitch_range**: Pitch variability (-1 = monotone, +1 = very expressive)
- **timbre_brightness**: Voice brightness (-1 = dark, +1 = bright)
- **timbre_warmth**: Voice warmth (-1 = cold, +1 = warm)
- **rhythm_regularity**: Speech rhythm (-1 = irregular, +1 = very regular)
- **energy_level**: Overall energy (-1 = quiet, +1 = loud)
- **vocal_tract_length**: Vocal tract length (-1 = short, +1 = long)
- **breathiness**: Voice breathiness (-1 = clear, +1 = very breathy)

### Voice Configuration File

You can save custom voice configurations in JSON format:

```json
{
  "name": "My_Custom_Voice",
  "voice_parameters": {
    "pitch_mean": 0.2,
    "pitch_range": 0.6,
    "timbre_brightness": 0.3,
    "timbre_warmth": 0.4,
    "energy_level": 0.5,
    "breathiness": 0.1
  }
}
```

## Emotional Expression

### Available Emotions

The system supports seven emotional states:

1. **neutral**: Balanced, professional tone
2. **happy**: Higher pitch, faster rate, increased energy
3. **sad**: Lower pitch, slower rate, decreased energy
4. **angry**: Higher pitch, faster rate, increased energy and tension
5. **excited**: Very high pitch, fast rate, high energy
6. **calm**: Lower pitch, slower rate, relaxed energy
7. **question**: Rising intonation pattern

### Emotion Customization

You can customize emotional expressions by modifying the configuration file:

```json
{
  "emotional_presets": {
    "custom_emotion": {
      "pitch_shift": 0.3,
      "rate_scale": 1.1,
      "energy_scale": 1.2,
      "spectral_tilt_shift": 0.1
    }
  }
}
```

### Emotional Sequences

For complex emotional expressions, you can specify emotion sequences:

```python
from src.synthesis.inference import InferenceEngine

engine = InferenceEngine("models", "config/character_voice.json")

# Synthesize with changing emotions
emotion_sequence = [
    (0, 10, "neutral"),    # First 10 characters: neutral
    (10, 20, "excited"),   # Next 10 characters: excited
    (20, 30, "calm"),      # Last 10 characters: calm
]

result = engine.synthesize_with_emotion_sequence(
    text="This is a test of emotional sequences.",
    emotion_sequence=emotion_sequence
)
```

## Batch Processing

### Text File Format

Create a text file with one sentence per line:

```
Hello, how are you today?
This is a test of batch synthesis.
The system can process multiple texts efficiently.
```

### Batch Synthesis

```bash
python synthesize_speech.py --batch texts.txt --output-dir batch_output
```

### Voice Comparison

Generate the same text with different voices:

```python
from src.synthesis.inference import InferenceEngine

engine = InferenceEngine("models", "config/character_voice.json")

engine.compare_voices(
    text="This is a comparison of different voices.",
    voice_archetypes=["young_female", "mature_male", "child"],
    emotion="neutral",
    output_dir="voice_comparison"
)
```

### Emotional Speech Generation

Generate the same text with different emotions:

```python
engine.generate_emotional_speech(
    text="I can express many different emotions.",
    voice_archetype="young_female",
    output_dir="emotional_speech"
)
```

## Advanced Features

### Programmatic Usage

Use the system programmatically in Python:

```python
from src.synthesis.inference import InferenceEngine

# Initialize the engine
engine = InferenceEngine(
    model_dir="models",
    config_path="config/character_voice.json",
    device="cuda"
)

# Synthesize speech
audio = engine.synthesize_text(
    text="Hello, this is a programmatic synthesis.",
    emotion="happy",
    voice_archetype="young_female",
    output_path="output.wav"
)

# Create custom voice
custom_voice_params = {
    "pitch_mean": 0.3,
    "timbre_brightness": 0.5,
    "energy_level": 0.7
}

audio = engine.synthesize_text(
    text="This is my custom voice.",
    custom_voice_params=custom_voice_params,
    emotion="neutral"
)
```

### Voice Variants

Create multiple variants of the same voice:

```python
# Generate 5 voice variants
variants = engine.create_voice_variants(
    text="Each variant has slightly different characteristics.",
    n_variants=5,
    emotion="neutral",
    output_dir="variants"
)
```

### Quality Testing

Test the synthesis quality:

```python
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
]

results = engine.test_synthesis_quality(
    test_texts=test_texts,
    output_dir="quality_test"
)

print(f"Generated {len(results['generated_files'])} files")
print(f"Errors: {len(results['errors'])}")
```

### Custom Training

Train the models with your own data:

```bash
# Generate custom training data
python train_models.py --generate-data --data-size 5000

# Train with custom parameters
python train_models.py \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 2e-4 \
    --data-dir custom_data \
    --output-dir custom_models
```

## Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `Model directory not found`

**Solution**: Ensure the model directory exists and contains trained models:
```bash
python scripts/setup.py  # This will train initial models
```

#### 2. CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solution**: Use CPU or reduce batch size:
```bash
python synthesize_speech.py --text "Hello" --device cpu
```

#### 3. Audio Quality Issues

**Problem**: Poor audio quality

**Solutions**:
- Ensure models are properly trained
- Check audio settings in configuration
- Try different voice archetypes
- Adjust prosody parameters

#### 4. Text Processing Errors

**Error**: `Text processing failed`

**Solution**: Check text format and encoding:
```bash
# Use simple ASCII text
python synthesize_speech.py --text "Hello world"
```

### Performance Optimization

#### 1. GPU Acceleration

Ensure CUDA is available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name()}")
```

#### 2. Memory Management

For large batch processing:
```python
# Process in smaller batches
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    engine.batch_synthesize(batch, output_dir=f"batch_{i}")
```

#### 3. Model Optimization

Use optimized models for inference:
```python
# Load model in evaluation mode
model.eval()
with torch.no_grad():
    output = model(input)
```

## API Reference

### InferenceEngine Class

#### `__init__(model_dir, config_path, device="auto")`

Initialize the inference engine.

**Parameters**:
- `model_dir` (str): Directory containing trained models
- `config_path` (str): Path to character voice configuration
- `device` (str): Device to run inference on

#### `synthesize_text(text, emotion="neutral", voice_archetype=None, custom_voice_params=None, output_path=None, emphasis_markers=None)`

Synthesize speech from text.

**Parameters**:
- `text` (str): Input text to synthesize
- `emotion` (str): Target emotion
- `voice_archetype` (str): Predefined voice archetype
- `custom_voice_params` (dict): Custom voice parameters
- `output_path` (str): Path to save audio file
- `emphasis_markers` (list): Emphasis markers

**Returns**: Generated audio tensor or file path

#### `create_voice_demo(demo_texts, voice_archetypes, emotions, output_dir)`

Create a comprehensive voice demonstration.

**Parameters**:
- `demo_texts` (list): List of texts to synthesize
- `voice_archetypes` (list): List of voice archetypes
- `emotions` (list): List of emotions
- `output_dir` (str): Output directory

#### `compare_voices(text, voice_archetypes, emotion="neutral", output_dir="voice_comparison")`

Generate the same text with different voices.

**Parameters**:
- `text` (str): Text to synthesize
- `voice_archetypes` (list): List of voice archetypes to compare
- `emotion` (str): Target emotion
- `output_dir` (str): Output directory

#### `generate_emotional_speech(text, voice_archetype="young_female", output_dir="emotional_speech")`

Generate the same text with different emotions.

**Parameters**:
- `text` (str): Text to synthesize
- `voice_archetype` (str): Voice archetype to use
- `output_dir` (str): Output directory

#### `create_voice_variants(text, n_variants=5, emotion="neutral", output_dir="voice_variants")`

Create multiple variants of the same voice.

**Parameters**:
- `text` (str): Text to synthesize
- `n_variants` (int): Number of variants to create
- `emotion` (str): Target emotion
- `output_dir` (str): Output directory

#### `batch_synthesize(texts, voice_archetype="young_female", emotion="neutral", output_dir="batch_synthesis")`

Synthesize multiple texts in batch.

**Parameters**:
- `texts` (list): List of texts to synthesize
- `voice_archetype` (str): Voice archetype to use
- `emotion` (str): Target emotion
- `output_dir` (str): Output directory

#### `test_synthesis_quality(test_texts, output_dir="quality_test")`

Test synthesis quality with various texts.

**Parameters**:
- `test_texts` (list): List of test texts
- `output_dir` (str): Output directory

**Returns**: Dictionary with test results

#### `get_voice_info()`

Get information about available voices and capabilities.

**Returns**: Dictionary with voice information

### Configuration File Format

The character voice configuration file uses JSON format:

```json
{
  "name": "Character_Name",
  "speaker_embedding": [0.12, -0.45, 0.78, ...],
  "voice_parameters": {
    "base_pitch_hz": 180,
    "pitch_variability": 0.8,
    "default_speech_rate": 1.0,
    "spectral_tilt": 0.2,
    "vocal_tract_length": 1.0,
    "breathiness": 0.3,
    "brightness": 0.7,
    "warmth": 0.6
  },
  "emotional_presets": {
    "emotion_name": {
      "pitch_shift": 0.0,
      "rate_scale": 1.0,
      "energy_scale": 1.0,
      "spectral_tilt_shift": 0.0
    }
  },
  "audio_settings": {
    "sample_rate": 22050,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 80,
    "n_fft": 1024
  }
}
```

## Conclusion

The Synthetic Digital TTS System provides a comprehensive solution for generating high-quality synthetic speech. With its flexible voice customization, emotional expression capabilities, and batch processing features, it can be used for a wide range of applications.

For more information, see the [README.md](../README.md) and [API documentation](API.md).








































