# 🎬 LOCAL VIDEO GENERATION PIPELINE

**100% Local Processing • Zero API Costs • RTX 5090 Optimized**

## Overview

This pipeline generates lip-synced Kelly videos entirely on local hardware:

```
Script → Tortoise TTS → SadTalker → Supabase Upload
         (voice)        (lip-sync)
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | **RTX 5090 (32GB)** |
| VRAM | 10GB | 24GB+ |
| RAM | 16GB | 32GB |
| Storage | 50GB SSD | 100GB NVMe |

## Quick Start

### 1. Setup Environment

```bash
cd kelly-sync

# Check dependencies
python scripts/setup_local_pipeline.py --check-only

# Install everything
python scripts/setup_local_pipeline.py --install-all

# Download model weights (manual step required)
python scripts/setup_local_pipeline.py --download-models
```

### 2. Test Pipeline

```bash
# Test with Day 1 (has content in database)
python scripts/local_video_pipeline.py --test

# Or specify day/phase
python scripts/local_video_pipeline.py --day 1 --phase hook --archetype "The Scientist"
```

### 3. Generate Day 51

```bash
# Generate Day 51 Hook phase
python scripts/local_video_pipeline.py --day 51 --phase hook

# All phases for a day
for phase in hook q1 q2 q3 wisdom; do
    python scripts/local_video_pipeline.py --day 51 --phase $phase
done
```

## Pipeline Components

### 1. TTS Engine (Voice Synthesis)

**Option A: Tortoise TTS** (Default)
- High quality, natural voice
- Slower (~30s for 10s audio)
- Best for production

```bash
python scripts/local_video_pipeline.py --day 51 --tts tortoise
```

**Option B: Piper TTS** (Fast)
- Lightweight, fast
- Good quality
- Best for testing

```bash
python scripts/local_video_pipeline.py --day 51 --tts piper
```

### 2. Lip Sync Engine (SadTalker)

- Generates realistic lip movements
- Works with single reference image
- Includes face enhancement (GFPGAN)

### 3. Supabase Integration

- Uploads to `kelly-videos` bucket
- Registers in `lesson_video_generation_status`
- Use `--no-upload` for local-only generation

## Directory Structure

```
kelly-sync/
├── models/
│   ├── SadTalker/          # Lip sync model
│   │   └── checkpoints/    # Model weights (download separately)
│   └── tortoise-tts/       # TTS model (auto-downloaded)
├── output/                 # Generated videos
├── scripts/
│   ├── local_video_pipeline.py    # Main pipeline
│   ├── setup_local_pipeline.py    # Setup script
│   ├── generate_video.py          # Alternative generator
│   └── quality_check.py           # Quality validation
├── config.yaml             # Pipeline configuration
├── requirements-local.txt  # Python dependencies
└── LOCAL_PIPELINE_README.md
```

## Model Downloads

SadTalker requires manual model downloads (~2GB):

1. Go to: https://github.com/OpenTalker/SadTalker#-2-download-models
2. Download:
   - `SadTalker_V0.0.2_256.safetensors`
   - `SadTalker_V0.0.2_512.safetensors`
   - `mapping_00109-model.pth.tar`
   - `mapping_00229-model.pth.tar`
3. Place in: `kelly-sync/models/SadTalker/checkpoints/`

## Cost Comparison

| Method | Cost per Video | Time per Video |
|--------|----------------|----------------|
| HeyGen API | $0.50-2.00 | 2-5 minutes |
| fal.ai SadTalker | $0.05-0.10 | 30-60 seconds |
| **Local Pipeline** | **$0.01** (electricity) | **30-90 seconds** |

For 1,775 videos (remaining gap):
- HeyGen: $887 - $3,550
- fal.ai: $88 - $177
- **Local: ~$18** + 15-44 hours processing

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller batch size
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Or use CPU fallback (slow)
python scripts/local_video_pipeline.py --day 51 --device cpu
```

### SadTalker Not Found
```bash
# Reinstall SadTalker
cd kelly-sync/models
git clone https://github.com/OpenTalker/SadTalker.git
pip install -r SadTalker/requirements.txt
```

### No Script Found for Day
Day 51 may not have `lesson_atoms` content yet. Options:
1. Use `--script "Your custom text here"`
2. Generate content first
3. Test with Day 1: `--test`

## Next Steps

1. ✅ Setup complete
2. ⬜ Download SadTalker models
3. ⬜ Test with Day 1
4. ⬜ Generate Day 51 Hook
5. ⬜ Batch generate remaining videos

## API Reference

```python
from local_video_pipeline import LocalVideoPipeline

pipeline = LocalVideoPipeline(
    device='cuda:0',
    tts_engine='tortoise',
)

result = pipeline.generate(
    day_number=51,
    phase='hook',
    archetype='The Scientist',
    upload=True,
)

print(result['public_url'])
```
