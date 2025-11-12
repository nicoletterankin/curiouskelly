# Kelly Audio2Face-3D Workflow Guide

## Overview
This guide provides step-by-step instructions for using NVIDIA Audio2Face-3D with Kelly avatar for facial animation generation.

## Prerequisites
- NVIDIA API Key (get from https://api.nvidia.com/)
- Audio2Face-3D Function ID
- Kelly audio files (WAV format, 16-bit PCM)
- Python 3.8+ environment

## Setup Steps

### 1. Install Dependencies
```bash
# Install Audio2Face-3D requirements
pip install -r Audio2Face-3D-Samples/scripts/audio2face_3d_api_client/requirements

# Install NVIDIA ACE wheel
pip install Audio2Face-3D-Samples/proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
```

### 2. Set Environment Variables
```bash
# Windows
set NVIDIA_API_KEY=your_api_key_here
set AUDIO2FACE_FUNCTION_ID=your_function_id_here

# Linux/Mac
export NVIDIA_API_KEY=your_api_key_here
export AUDIO2FACE_FUNCTION_ID=your_function_id_here
```

### 3. Prepare Kelly Audio
- Ensure audio is in WAV format, 16-bit PCM, single channel
- Recommended sample rate: 16kHz or 44.1kHz
- Audio should be clean with minimal background noise

## Usage

### Basic Usage
```bash
cd kelly_audio2face
python scripts/kelly_audio2face_client.py ../projects/Kelly/Audio/kelly25_audio.wav \
    --apikey YOUR_API_KEY \
    --function-id YOUR_FUNCTION_ID \
    --model claire
```

### Advanced Usage
```bash
python scripts/kelly_audio2face_client.py audio_file.wav \
    --config config/kelly_config.yml \
    --apikey YOUR_API_KEY \
    --function-id YOUR_FUNCTION_ID \
    --output-dir output/kelly_animation \
    --model claire
```

## Configuration

### Kelly-Specific Settings
The Kelly configuration (`config/kelly_config.yml`) is optimized for:
- More expressive upper face movements
- Enhanced lip sync accuracy
- Natural emotional expressions
- Kelly's specific facial characteristics

### Key Parameters
- `upperFaceStrength`: 1.1 (slightly more expressive)
- `lowerFaceStrength`: 1.3 (enhanced lip sync)
- `emotion_strength`: 0.7 (more emotionally expressive)
- `max_emotions`: 3 (balanced emotional range)

## Output Files

### Generated Files
- `blendshapes.csv`: ARKit blendshape data with timestamps
- `emotions.csv`: Emotion data with timestamps
- `out.wav`: Processed audio (should match input)
- `kelly_client.log`: Processing logs

### Blendshape Mapping
Kelly uses standard ARKit blendshapes:
- Eye movements (blink, look, squint, wide)
- Jaw movements (open, forward, left, right)
- Mouth movements (smile, frown, pucker, funnel)
- Brow movements (up, down, inner, outer)
- Cheek movements (puff, squint)

## Integration with 3D Software

### Maya Integration
1. Import blendshape data
2. Map to Kelly's facial rig
3. Apply animation curves
4. Fine-tune timing and intensity

### Unreal Engine Integration
1. Use Audio2Face-3D Unreal plugin
2. Import Kelly model
3. Apply blendshape data
4. Test in real-time

### iClone Integration
1. Export Kelly model from iClone
2. Process with Audio2Face-3D
3. Import animation back to iClone
4. Render final video

## Troubleshooting

### Common Issues
1. **API Key Invalid**: Verify key from NVIDIA API portal
2. **Function ID Error**: Check Function ID is correct
3. **Audio Format Error**: Ensure WAV, 16-bit PCM format
4. **Connection Timeout**: Check internet connection
5. **Import Errors**: Verify all dependencies installed

### Debug Mode
Enable debug logging:
```bash
python scripts/kelly_audio2face_client.py audio.wav \
    --apikey YOUR_KEY \
    --function-id YOUR_ID \
    --debug
```

## Performance Tips

### Optimization
- Use shorter audio clips for faster processing
- Batch process multiple clips
- Cache results for repeated use
- Use appropriate model (claire for female voices)

### Quality Settings
- Higher sample rates for better quality
- Clean audio input for better results
- Appropriate emotion settings for content
- Fine-tune blendshape multipliers

## Next Steps

### Production Pipeline
1. Generate Kelly audio with ElevenLabs
2. Process with Audio2Face-3D
3. Import to 3D software
4. Apply to Kelly's rig
5. Render final animation

### Batch Processing
Create batch scripts for multiple audio files:
```bash
for audio in ../projects/Kelly/Audio/*.wav; do
    python scripts/kelly_audio2face_client.py "$audio" \
        --apikey YOUR_KEY \
        --function-id YOUR_ID
done
```

## Support
- NVIDIA Audio2Face-3D Documentation
- Kelly Project Documentation
- GitHub Issues for bug reports
