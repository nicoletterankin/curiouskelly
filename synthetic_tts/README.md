# Synthetic Digital TTS System

A completely self-contained, vendorless Text-to-Speech (TTS) system for synthetic digital characters. This system generates voices entirely from synthetic data without relying on human voice samples or external APIs.

## 🎯 Key Features

- **🔒 Vendorless & Offline**: Runs entirely on local hardware with no external dependencies
- **🤖 Digitally-Sourced Voice**: Built from synthetically generated audio data, not human samples
- **📦 Self-Contained**: Includes all models, weights, and inference code
- **🎭 Configurable Personality**: Voice characteristics controllable via configuration
- **😊 Emotional Expression**: Support for 7+ emotional states and prosodic control
- **👥 Multi-Speaker Support**: Generate multiple synthetic voices with different characteristics
- **⚡ Real-time Synthesis**: Fast inference suitable for interactive applications
- **🎨 Voice Customization**: Create custom voices with specific timbre, pitch, and energy

### 🆕 Enhanced Features (Hybrid System)

- **🔄 Voice Interpolation**: Smooth transitions between different voices
- **🎭 Voice Morphing**: Advanced voice transformation capabilities
- **⚡ Real-time Voice Switching**: Dynamic voice changes during synthesis
- **🏗️ Multi-Architecture Support**: FastPitch, Tacotron2, HiFi-GAN, WaveGlow
- **🗺️ Voice Space Navigation**: Interactive exploration of voice characteristics
- **📊 Voice Quality Assessment**: Comprehensive audio quality metrics
- **🎯 Voice Clustering**: Automatic grouping of similar voices
- **👨‍👩‍👧‍👦 Voice Family Creation**: Generate related voice variations
- **🔀 Hybrid Data Generation**: Combines real and synthetic voice data
- **🎛️ Advanced Prosody Control**: Fine-grained speech parameter manipulation

## 🏗️ Architecture

The system consists of three main components:

- **🎵 FastPitch Acoustic Model**: Generates mel-spectrograms from text and speaker embeddings
- **🔊 HiFi-GAN Vocoder**: Converts mel-spectrograms to high-quality audio waveforms  
- **🎤 Speaker Embedding Generator**: Creates synthetic speaker vectors from voice characteristics
- **📝 Text Processing Pipeline**: Advanced G2P conversion and prosodic markup processing
- **🎭 Prosody Controller**: Rule-based and neural emotion control system

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/synthetic-tts/synthetic-tts.git
cd synthetic-tts

# Install dependencies
pip install -r requirements.txt

# Run setup (generates data, trains models, tests system)
python scripts/setup.py
```

### 2. Basic Usage

```bash
# Synthesize speech with default voice
python synthesize_speech.py --text "Hello, world!" --output hello.wav

# Synthesize with specific emotion and voice archetype
python synthesize_speech.py \
    --text "I'm so excited about this!" \
    --emotion excited \
    --speaker-archetype young_female \
    --output excited.wav

# Create custom voice
python synthesize_speech.py \
    --text "This is my custom voice" \
    --custom-voice \
    --pitch-mean 0.3 \
    --timbre-brightness 0.5 \
    --energy-level 0.7 \
    --output custom.wav
```

### 3. Enhanced Usage (Hybrid System)

```bash
# Voice interpolation between two voices
python synthesize_speech_enhanced.py \
    --text "Hello, this is voice interpolation" \
    --voice-id voice_1 \
    --interpolation-target voice_2 \
    --interpolation-weight 0.5 \
    --output interpolated.wav

# Create voice continuum
python synthesize_speech_enhanced.py \
    --text "Voice continuum demonstration" \
    --create-continuum \
    --continuum-voice1 voice_1 \
    --continuum-voice2 voice_2 \
    --continuum-steps 10 \
    --output-dir continuum_output

# Voice morphing
python synthesize_speech_enhanced.py \
    --text "Voice morphing demonstration" \
    --morph-voices voice_1 voice_2 \
    --morph-target voice_3 \
    --morph-steps 5 \
    --output-dir morphing_output

# Create voice family
python synthesize_speech_enhanced.py \
    --text "Voice family demonstration" \
    --create-family \
    --family-parent voice_1 \
    --family-size 5 \
    --output-dir family_output

# Analyze voice characteristics
python synthesize_speech_enhanced.py \
    --text "Voice analysis" \
    --analyze-voice \
    --voice-id voice_1 \
    --output analysis.wav

# Find similar voices
python synthesize_speech_enhanced.py \
    --text "Similar voices" \
    --find-similar voice_1 \
    --similarity-threshold 0.5
```

### 4. Run Demo

```bash
# See comprehensive demonstration
python scripts/demo.py

# Voice interpolation demonstration
python scripts/voice_interpolation_demo.py

# Model comparison
python scripts/model_comparison.py

# Voice quality testing
python scripts/voice_quality_test.py

# Voice space exploration
python scripts/voice_space_explorer.py
```

## 📁 Project Structure

```
synthetic_tts/
├── 📁 models/                 # Trained model weights
├── 📁 data/                   # Training and synthetic data
├── 📁 config/                 # Character voice configurations
├── 📁 src/                    # Source code
│   ├── 📁 models/            # Neural network models
│   │   ├── fastpitch.py      # FastPitch acoustic model
│   │   ├── hifigan.py        # HiFi-GAN vocoder
│   │   ├── speaker_embedding.py # Speaker embedding generator
│   │   └── model_factory.py  # Model factory for multiple architectures
│   ├── 📁 data/              # Data processing utilities
│   │   ├── text_processor.py # Text normalization and G2P
│   │   ├── synthetic_data_generator.py # Synthetic data creation
│   │   ├── hybrid_data_generator.py # Hybrid data generation
│   │   ├── real_dataset_loader.py # Real dataset loading
│   │   └── dataset.py        # Dataset classes
│   ├── 📁 synthesis/         # Inference pipeline
│   │   ├── synthesizer.py    # Main synthesizer class
│   │   ├── enhanced_synthesizer.py # Enhanced synthesizer with interpolation
│   │   ├── prosody_controller.py # Emotion and prosody control
│   │   └── inference.py      # High-level inference engine
│   ├── 📁 voice/             # Voice manipulation
│   │   ├── voice_interpolator.py # Voice interpolation engine
│   │   └── voice_analyzer.py # Voice analysis and quality assessment
│   └── 📁 utils/             # Utility functions
│       └── voice_utils.py    # Voice manipulation utilities
├── 📁 scripts/               # Training and utility scripts
│   ├── setup.py              # Environment setup
│   ├── demo.py               # Demonstration script
│   ├── voice_interpolation_demo.py # Voice interpolation demonstration
│   ├── model_comparison.py   # Model architecture comparison
│   ├── voice_quality_test.py # Voice quality testing
│   └── voice_space_explorer.py # Voice space exploration
├── 📁 docs/                  # Comprehensive documentation
│   ├── USAGE_GUIDE.md        # Detailed usage instructions
│   ├── TRAINING_GUIDE.md     # Model training guide
│   └── API.md                # Complete API reference
├── 📄 synthesize_speech.py   # Main synthesis script
├── 📄 synthesize_speech_enhanced.py # Enhanced synthesis with interpolation
├── 📄 train_models.py        # Model training script
├── 📄 requirements.txt       # Python dependencies
└── 📄 README.md              # This file
```

## 🎭 Voice Characteristics

### Predefined Voice Archetypes

The system includes six voice archetypes:

1. **👩 young_female**: High pitch, bright timbre, energetic
2. **👨 young_male**: Medium pitch, warm timbre, confident  
3. **👩‍💼 mature_female**: Medium-high pitch, warm timbre, professional
4. **👨‍💼 mature_male**: Low pitch, warm timbre, authoritative
5. **👶 child**: Very high pitch, bright timbre, playful
6. **👴 elderly**: Medium pitch, warm timbre, calm

### Custom Voice Parameters

Create custom voices by adjusting:

- **🎵 pitch_mean**: Overall pitch level (-1 to +1)
- **📈 pitch_range**: Pitch variability (-1 to +1)  
- **✨ timbre_brightness**: Voice brightness (-1 to +1)
- **🔥 timbre_warmth**: Voice warmth (-1 to +1)
- **⏱️ rhythm_regularity**: Speech rhythm (-1 to +1)
- **⚡ energy_level**: Overall energy (-1 to +1)
- **🗣️ vocal_tract_length**: Vocal tract length (-1 to +1)
- **💨 breathiness**: Voice breathiness (-1 to +1)

## 😊 Emotional Expression

### Supported Emotions

- **😐 neutral**: Balanced, professional tone
- **😊 happy**: Higher pitch, faster rate, increased energy
- **😢 sad**: Lower pitch, slower rate, decreased energy
- **😠 angry**: Higher pitch, faster rate, increased energy and tension
- **🤩 excited**: Very high pitch, fast rate, high energy
- **😌 calm**: Lower pitch, slower rate, relaxed energy
- **❓ question**: Rising intonation pattern

### Advanced Emotional Control

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

## 🔄 Batch Processing

### Text File Processing

```bash
# Create text file
echo -e "Hello world\nHow are you?\nThis is a test" > texts.txt

# Synthesize all texts
python synthesize_speech.py --batch texts.txt --output-dir batch_output
```

### Voice Comparison

```python
# Compare different voices
engine.compare_voices(
    text="This is a comparison of different voices.",
    voice_archetypes=["young_female", "mature_male", "child"],
    emotion="neutral",
    output_dir="voice_comparison"
)
```

### Voice Variants

```python
# Create multiple voice variants
engine.create_voice_variants(
    text="Each variant has slightly different characteristics.",
    n_variants=5,
    emotion="neutral",
    output_dir="voice_variants"
)
```

## 🧠 Model Training

### Generate Training Data

```bash
# Generate synthetic training data
python train_models.py --generate-data --data-size 10000
```

### Train Models

```bash
# Train all models
python train_models.py \
    --data-dir data \
    --output-dir models \
    --epochs 100 \
    --batch-size 32

# Train specific models
python train_models.py --fastpitch-only --epochs 50
python train_models.py --hifigan-only --epochs 50
```

### Custom Training

```bash
# Train with custom parameters
python train_models.py \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 2e-4 \
    --data-dir custom_data \
    --output-dir custom_models
```

## 🐍 Programmatic Usage

### Basic Synthesis

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
```

### Custom Voice Creation

```python
from src.models.speaker_embedding import SyntheticVoiceGenerator

# Create custom voice
generator = SyntheticVoiceGenerator()
custom_voice = generator.create_custom_voice(
    pitch_mean=0.3,
    timbre_brightness=0.5,
    energy_level=0.7
)

# Use custom voice
audio = engine.synthesize_text(
    text="This is my custom voice.",
    custom_voice_params=custom_voice,
    emotion="neutral"
)
```

### Advanced Features

```python
# Create voice demonstration
engine.create_voice_demo(
    demo_texts=["Hello", "How are you?", "Goodbye"],
    voice_archetypes=["young_female", "mature_male"],
    emotions=["happy", "sad", "excited"],
    output_dir="demo"
)

# Test synthesis quality
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
]

results = engine.test_synthesis_quality(test_texts, "quality_test")
```

## 📊 Performance

### System Requirements

- **Python**: 3.8+
- **PyTorch**: 1.12+
- **CUDA**: 11.6+ (for GPU training)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 10GB+ for models and data
- **GPU**: RTX 3060+ (for training), any GPU (for inference)

### Performance Metrics

- **Inference Speed**: ~10x real-time on RTX 3060
- **Model Size**: ~500MB total
- **Memory Usage**: ~2GB during inference
- **Audio Quality**: 22kHz, 16-bit WAV output

## 🔧 Configuration

### Character Voice Configuration

```json
{
  "name": "My_Custom_Voice",
  "voice_parameters": {
    "base_pitch_hz": 180,
    "pitch_variability": 0.8,
    "timbre_brightness": 0.3,
    "timbre_warmth": 0.4,
    "energy_level": 0.5
  },
  "emotional_presets": {
    "happy": {
      "pitch_shift": 0.5,
      "rate_scale": 1.1,
      "energy_scale": 1.2
    }
  }
}
```

## 📚 Documentation

- **[Usage Guide](docs/USAGE_GUIDE.md)**: Comprehensive usage instructions
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Model training and optimization
- **[API Documentation](docs/API.md)**: Complete API reference
- **[Examples](examples/)**: Code examples and tutorials

## 🛠️ Troubleshooting

### Common Issues

1. **Model Not Found**: Run `python scripts/setup.py` to train initial models
2. **CUDA Out of Memory**: Use `--device cpu` or reduce batch size
3. **Poor Audio Quality**: Ensure models are properly trained, try different voice archetypes
4. **Text Processing Errors**: Use simple ASCII text, check encoding

### Performance Optimization

- Use GPU acceleration when available
- Process in smaller batches for large datasets
- Use optimized models for inference
- Monitor memory usage during training

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastPitch architecture by NVIDIA
- HiFi-GAN vocoder by NVIDIA
- Phonemizer for G2P conversion
- PyTorch for deep learning framework
- Librosa for audio processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/synthetic-tts/synthetic-tts/issues)
- **Discussions**: [GitHub Discussions](https://github.com/synthetic-tts/synthetic-tts/discussions)
- **Documentation**: [Full Documentation](docs/)

---

**Ready to create synthetic voices?** Start with our [Quick Start Guide](docs/USAGE_GUIDE.md#quick-start) or run the [demo script](scripts/demo.py) to see the system in action!
