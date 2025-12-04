# Kelly Forced Alignment Service

Extract precise word-level and phoneme-level timing from ElevenLabs audio for perfect lip-sync.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install Montreal Forced Aligner (one-time setup)
conda install -c conda-forge montreal-forced-aligner

# Download English models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Run alignment
python align_audio.py --audio kelly_audio.wav --text "Hello everyone!"
```

## Output Format

```json
{
  "words": [
    { "word": "Hello", "start": 0.0, "end": 0.35, "confidence": 0.95 },
    { "word": "everyone", "start": 0.38, "end": 0.92, "confidence": 0.95 }
  ],
  "phones": [
    { "phone": "HH", "start": 0.0, "end": 0.05, "word": "Hello", "viseme": "A" },
    { "phone": "AH", "start": 0.05, "end": 0.15, "word": "Hello", "viseme": "A" },
    { "phone": "L", "start": 0.15, "end": 0.22, "word": "Hello", "viseme": "L" },
    { "phone": "OW", "start": 0.22, "end": 0.35, "word": "Hello", "viseme": "O" }
  ],
  "duration": 0.95,
  "method": "mfa",
  "confidence": 0.95
}
```

## Integration with Kelly Lip-Sync

This output feeds directly into `app/lipsync/phoneme-viseme-map.js`:

```javascript
import { generateBlendshapeTimeline } from './phoneme-viseme-map.js';

// Load alignment from Python service
const alignment = await fetch('/api/align', { ... }).then(r => r.json());

// Generate 30fps blendshape timeline
const timeline = generateBlendshapeTimeline(alignment.phones, 30);

// Apply to Kelly's face in Unity
timeline.forEach(frame => {
  unityInstance.SendMessage('Kelly', 'SetBlendshapes', JSON.stringify(frame.blendshapes));
});
```

## Alignment Methods

### 1. Montreal Forced Aligner (Default, Most Accurate)
```bash
python align_audio.py --audio kelly.wav --text "Hello!" --method mfa
```

### 2. Gentle Aligner (Requires Running Service)
```bash
# Start Gentle server first
docker run -p 8765:8765 lowerquality/gentle

# Then align
python align_audio.py --audio kelly.wav --text "Hello!" --method gentle
```

### 3. Simple Estimation (No External Dependencies)
```bash
python align_audio.py --audio kelly.wav --text "Hello!" --method simple
```

## Batch Processing

Process all audio files in a directory:

```bash
# Structure your directory:
# audio_files/
#   ├── lesson1.wav
#   ├── lesson1.txt  (or lesson1.json with "transcript" field)
#   ├── lesson2.wav
#   └── lesson2.txt

python align_audio.py --batch-dir ./audio_files --output ./alignments
```

## API Endpoint

See `api/align.ts` for the Vercel serverless function that wraps this script.

```typescript
// POST /api/align
// Body: { audio_url: string, transcript: string }
// Returns: { words: [...], phones: [...], duration: number }
```

## Troubleshooting

### MFA Not Found
```bash
# Install via conda (recommended)
conda install -c conda-forge montreal-forced-aligner

# Or via pip (may have issues)
pip install montreal-forced-aligner
```

### Model Download Fails
```bash
# Manual download
mfa model download acoustic english_us_arpa --ignore_cache
mfa model download dictionary english_us_arpa --ignore_cache
```

### Audio Format Issues
The service automatically converts to WAV 16kHz mono. If issues persist:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

