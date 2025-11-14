# Piper TTS Usage Guide

## 🎯 Overview

Piper TTS is now successfully installed and working locally! This guide shows you how to use it for Kelly and Ken voice generation.

## 🚀 Quick Start

### Basic Usage

```python
from piper_tts_wrapper import PiperTTS

# Initialize TTS
tts = PiperTTS()

# Generate Kelly's voice
tts.synthesize_kelly("Hello, I'm Kelly!", "kelly_output.wav")

# Generate Ken's voice  
tts.synthesize_ken("Hello, I'm Ken!", "ken_output.wav")
```

### Advanced Usage

```python
# Use specific voice model
tts.synthesize("Your text here", voice="en_US-lessac-medium.onnx", output_file="output.wav")

# Use different speaker ID
tts.synthesize("Your text here", output_file="output.wav", speaker_id=1)
```

## 📁 File Structure

```
synthetic_tts/
├── piper_tts_wrapper.py      # Main wrapper class
├── test_piper_demo.py        # Demo script
├── voices/                    # Voice models directory
│   ├── en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
└── demo_output/              # Generated audio files
```

## 🎤 Available Commands

### Command Line Usage

```bash
# Direct Piper usage
echo "Hello world" | piper --model voices/en_US-lessac-medium.onnx --config voices/en_US-lessac-medium.onnx.json --output_file output.wav

# Run demo
python test_piper_demo.py

# Test wrapper
python piper_tts_wrapper.py
```

### Python API

```python
# List available voices
voices = tts.list_voices()
print(f"Available voices: {voices}")

# Synthesize with custom parameters
result = tts.synthesize(
    text="Your text here",
    voice="en_US-lessac-medium.onnx",
    output_file="custom_output.wav",
    speaker_id=0
)
```

## 🎭 Voice Customization

### Current Setup
- **Voice Model**: en_US-lessac-medium.onnx
- **Language**: English (US)
- **Quality**: Medium
- **Sample Rate**: 22050 Hz

### Adding More Voices

1. Download voice models from [Piper Voice Samples](https://rhasspy.github.io/piper-samples/)
2. Place `.onnx` and `.onnx.json` files in the `voices/` directory
3. Restart the wrapper to load new voices

### Voice Parameters

```python
# Adjust synthesis parameters
tts.synthesize(
    text="Your text",
    output_file="output.wav",
    speaker_id=0,  # Try different speaker IDs
    # Additional parameters can be added to the wrapper
)
```

## 🔧 Troubleshooting

### Common Issues

1. **No voices available**: Check that voice files are in the `voices/` directory
2. **Encoding errors**: Ensure JSON files are UTF-8 encoded
3. **Path issues**: Use absolute paths for model files

### Testing Installation

```bash
# Test Piper directly
piper --help

# Test Python wrapper
python piper_tts_wrapper.py

# Run full demo
python test_piper_demo.py
```

## 📊 Performance

- **Inference Speed**: ~10x real-time on modern hardware
- **Audio Quality**: 22050 Hz, 16-bit WAV
- **Model Size**: ~60MB per voice model
- **Memory Usage**: ~500MB during synthesis

## 🎯 Integration with Kelly and Ken

### For Kelly's Voice
```python
# Kelly's friendly, energetic voice
kelly_text = "Hi there! I'm Kelly, and I'm so excited to help you learn!"
tts.synthesize_kelly(kelly_text, "kelly_greeting.wav")
```

### For Ken's Voice
```python
# Ken's knowledgeable, calm voice
ken_text = "Hello. I'm Ken, and I'm here to guide you through this lesson."
tts.synthesize_ken(ken_text, "ken_greeting.wav")
```

## 🚀 Next Steps

1. **Download More Voices**: Add more voice models for variety
2. **Custom Voice Training**: Train custom voices for Kelly and Ken
3. **Emotional Control**: Add emotion parameters to the wrapper
4. **Batch Processing**: Process multiple texts at once
5. **Integration**: Connect with your lesson system

## 📚 Additional Resources

- [Piper GitHub Repository](https://github.com/rhasspy/piper)
- [Piper Voice Samples](https://rhasspy.github.io/piper-samples/)
- [Piper Documentation](https://github.com/rhasspy/piper#usage)

---

**Piper TTS is now ready for use with Kelly and Ken voices!** 🎉







































