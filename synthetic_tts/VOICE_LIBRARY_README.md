# Voice Library Manager - HTML Interface

## 🎵 Overview

The Voice Library Manager is a comprehensive HTML-based interface for listening to, scoring, and managing your custom voice library. It provides an intuitive way to fine-tune and train your TTS voices with quality scoring and analysis tools.

## 🚀 Quick Start

### 1. Generate Voice Library Data
```bash
# Generate the voice library data from your audio files
python generate_voice_library_data.py --verbose
```

### 2. Start the Web Server
```bash
# Start the local web server
python serve_voice_library.py

# Or specify a custom port
python serve_voice_library.py --port 8080
```

### 3. Open in Browser
The interface will automatically open in your browser at `http://localhost:8000`

## 📁 Files Structure

```
synthetic_tts/
├── voice_library_manager_enhanced.html    # Main HTML interface
├── voice_library_data.json               # Generated voice library data
├── generate_voice_library_data.py        # Data generation script
├── serve_voice_library.py                # Local web server
├── test_audio_output/                    # Test audio files
├── real_audio_output/                    # Real audio files
└── VOICE_LIBRARY_README.md              # This file
```

## 🎯 Features

### Voice Management
- **451 Voice Samples** across 16 categories
- **Real-time Audio Playback** with progress tracking
- **Comprehensive Metadata** display
- **Advanced Filtering** and search capabilities

### Quality Scoring System
- **6 Quality Criteria**:
  - Naturalness (1-10)
  - Clarity (1-10)
  - Emotion (1-10)
  - Pitch (1-10)
  - Speed (1-10)
  - Overall (1-10)
- **Interactive Scoring** with click-to-score interface
- **Average Score Calculation** for each voice
- **Quality Distribution** analysis

### Training Data Export
- **High-Quality Voice Selection** (scores ≥ 8)
- **Training Recommendations** based on scores
- **Category-based Analysis** for targeted training
- **JSON Export** for integration with training pipelines

### Voice Categories
1. **Basic Synthesis** - Fundamental voice generation
2. **Voice Interpolation** - Smooth transitions between voices
3. **Voice Morphing** - Advanced voice transformation
4. **Voice Continuum** - Navigation through voice space
5. **Voice Family** - Related voice variations
6. **Emotional Speech** - Context-aware emotional expression
7. **Prosody Control** - Fine-grained speech parameters
8. **Quality Tests** - Different quality levels
9. **Voice Analysis** - Comprehensive voice characteristics

## 🎮 How to Use

### 1. Browse Voices
- Use the **search bar** to find specific voices
- **Filter by category** to focus on specific voice types
- **Filter by quality** to find high-scoring voices
- **Filter by scoring status** to see scored/unscored voices

### 2. Listen to Audio
- Click the **Play button** to listen to each voice
- **Progress bar** shows playback progress
- **Time display** shows current/total time
- Audio automatically stops when finished

### 3. Score Voices
- Click on any **quality criterion** to score (1-10)
- Scores are **automatically calculated** and displayed
- **Average score** is computed across all criteria
- **Reset scores** to start over

### 4. Export Data
- **Export Voice Library Data** - Complete dataset with scores
- **Export Training Data** - High-quality voices for training
- **JSON format** for easy integration

## 📊 Quality Scoring Guide

### Scoring Scale
- **9-10**: Excellent - Perfect for production use
- **7-8**: Good - High quality, minor improvements needed
- **5-6**: Fair - Decent quality, some issues present
- **1-4**: Poor - Significant quality issues

### Quality Criteria
- **Naturalness**: How natural and human-like the voice sounds
- **Clarity**: How clear and intelligible the speech is
- **Emotion**: How well emotions are expressed and conveyed
- **Pitch**: Accuracy and appropriate variation in pitch
- **Speed**: Appropriate speaking rate and rhythm
- **Overall**: Overall quality assessment considering all factors

## 🔧 Technical Details

### Audio File Support
- **Format**: WAV files
- **Sample Rates**: 16kHz, 22.05kHz, 44.1kHz, 48kHz
- **Bit Depths**: 16-bit, 24-bit, 32-bit
- **Channels**: Mono and Stereo

### Data Structure
Each voice entry contains:
```json
{
  "id": 1,
  "category": "Basic Synthesis",
  "name": "Voice 1",
  "transcript": "Hello, this is a test...",
  "filename": "voice_1.wav",
  "filepath": "real_audio_output/basic_synthesis/voice_1.wav",
  "metadata": {
    "duration": 3.5,
    "sample_rate": 22050,
    "pitch": 1.0,
    "speed": 1.0,
    "emotion": "neutral",
    "gender": "neutral"
  },
  "scores": {
    "Naturalness": 8,
    "Clarity": 7,
    "Emotion": 6,
    "Pitch": 8,
    "Speed": 7,
    "Overall": 7
  },
  "average_score": 7.0,
  "is_scored": true
}
```

### Browser Compatibility
- **Chrome** 80+ (Recommended)
- **Firefox** 75+
- **Safari** 13+
- **Edge** 80+

## 🎯 Training Workflow

### 1. Score Your Voices
- Listen to each voice sample
- Score based on quality criteria
- Focus on voices that will be used for training

### 2. Identify High-Quality Voices
- Filter by quality score (8+ recommended)
- Export training data for high-quality voices
- Use category filtering for specific voice types

### 3. Export Training Data
- Use "Export Training Data" for high-quality voices
- JSON format includes metadata and scores
- Integrate with your training pipeline

### 4. Iterate and Improve
- Re-score voices as you improve them
- Track quality improvements over time
- Build a curated library of best voices

## 🚨 Troubleshooting

### Common Issues

**Audio not playing:**
- Check that audio files exist in the correct directories
- Ensure file paths are correct in the JSON data
- Try refreshing the page

**Data not loading:**
- Run `python generate_voice_library_data.py` first
- Check that `voice_library_data.json` exists
- Verify file permissions

**Server won't start:**
- Port might be in use, try a different port
- Check firewall settings
- Ensure Python has network permissions

### Error Messages

**"Failed to load voice library data":**
- Run the data generation script first
- Check file paths and permissions

**"Port already in use":**
- Try a different port: `python serve_voice_library.py --port 8080`
- Stop other services using the port

## 🔮 Future Enhancements

- **Batch scoring** for multiple voices
- **Voice comparison** side-by-side
- **Advanced analytics** and reporting
- **Integration** with training pipelines
- **Voice cloning** capabilities
- **Real-time synthesis** testing

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console for error messages
3. Ensure all required files are present
4. Verify audio file formats and paths

## 🎉 Conclusion

The Voice Library Manager provides a comprehensive solution for managing and scoring your custom voice library. With its intuitive interface, quality scoring system, and training data export capabilities, it's the perfect tool for fine-tuning and training your TTS voices.

Start by generating your voice library data, then open the interface to begin scoring and managing your voices!
