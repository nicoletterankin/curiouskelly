# 🎬 KELLY-SYNC: Production-Grade Local Video Pipeline

> **Target Quality:** 4K-8K photorealistic, zero blur, zero uncanny valley  
> **Processing:** Fully local on RTX 5090 (32GB VRAM)  
> **Cost:** $0 per video after setup  
> **Speed:** 30-90 seconds per 90-second video

---

## Architecture Overview

This is NOT a quick hack. This is a multi-stage production pipeline that chains state-of-the-art models specifically tuned for Kelly.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       KELLY-SYNC PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  AUDIO                    REFERENCE                  MOTION          │
│  (ElevenLabs)             (Kelly 4K Photo)          (HeyGen Archive) │
│       │                        │                         │           │
│       ▼                        ▼                         ▼           │
│  ┌──────────┐            ┌──────────┐            ┌──────────┐       │
│  │ Whisper  │            │ MediaPipe│            │ Optical  │       │
│  │ Phonemes │            │ Face Mesh│            │ Flow     │       │
│  └────┬─────┘            └────┬─────┘            └────┬─────┘       │
│       │                       │                       │              │
│       └───────────────────────┴───────────────────────┘              │
│                               │                                      │
│                               ▼                                      │
│                    ┌─────────────────────┐                          │
│                    │   VideoReTalking    │  ← Best current SOTA     │
│                    │   (Lip Synthesis)   │    for photorealistic    │
│                    └──────────┬──────────┘                          │
│                               │                                      │
│                               ▼                                      │
│                    ┌─────────────────────┐                          │
│                    │    CodeFormer       │  ← Face restoration      │
│                    │    (Enhancement)    │    (critical for detail) │
│                    └──────────┬──────────┘                          │
│                               │                                      │
│                               ▼                                      │
│                    ┌─────────────────────┐                          │
│                    │    Real-ESRGAN      │  ← 4K/8K upscaling      │
│                    │    (Super-Res)      │                          │
│                    └──────────┬──────────┘                          │
│                               │                                      │
│                               ▼                                      │
│                    ┌─────────────────────┐                          │
│                    │    Motion Blend     │  ← Apply HeyGen motion   │
│                    │    + Composite      │    templates             │
│                    └──────────┬──────────┘                          │
│                               │                                      │
│                               ▼                                      │
│                    ┌─────────────────────┐                          │
│                    │   4K/8K Kelly MP4   │                          │
│                    │   Production Ready  │                          │
│                    └─────────────────────┘                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why This Pipeline

### The Quality Killers (and how we solve them)

| Problem | Cause | Our Solution |
|---------|-------|--------------|
| **Blur** | Low-res mouth region upscaled | CodeFormer restores detail before upscale |
| **Uncanny Valley** | Inconsistent expressions | Motion templates from HeyGen reference |
| **Temporal Jitter** | Frame-by-frame detection | MediaPipe with Kalman smoothing |
| **Lighting Mismatch** | Synthesized mouth different lighting | Poisson blending with color matching |
| **Resolution Loss** | Native 256px output | Real-ESRGAN x4 → 1024px → x2 → 2048px |
| **Lip Sync Drift** | Audio/video desync | Whisper phoneme alignment |

---

## Model Selection (Why These Specific Models)

### 1. VideoReTalking (Lip Synthesis)
- **Why not Wav2Lip?** Wav2Lip works at 96x96 mouth region, creates blur
- **Why not SadTalker?** SadTalker is good for expressions but lower quality mouth detail
- **VideoReTalking advantages:**
  - Works at higher resolution (256x256 mouth region)
  - Better temporal consistency
  - Preserves more facial detail
  - Handles various head poses

### 2. CodeFormer (Face Restoration)
- **Why not GFPGAN?** GFPGAN over-smooths and creates "plastic" look
- **CodeFormer advantages:**
  - Controllable fidelity vs quality tradeoff
  - Better at preserving identity
  - Handles partial occlusion (like during speech)
  - Trained on more diverse faces

### 3. Real-ESRGAN (Super Resolution)
- **Specific model:** `RealESRGAN_x4plus_anime_6B` variant
- **Why anime variant?** Counter-intuitively, it better preserves:
  - Sharp edge definition around lips
  - Eye detail
  - Hair texture
- We then blend with photorealistic model for skin

### 4. First Order Motion Model (Motion Transfer)
- Extracts motion patterns from HeyGen reference videos
- Applies to new lip-sync outputs
- Maintains Kelly's signature gestures and expressions

---

## Directory Structure

```
kelly-sync/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Installation script
├── config.yaml                  # Pipeline configuration
├── models/                      # Downloaded model weights
│   ├── video_retalking/
│   ├── codeformer/
│   ├── real_esrgan/
│   └── fomm/
├── assets/
│   ├── kelly_reference_4k.png   # Primary Kelly image (4K)
│   ├── kelly_face_mesh.json     # Pre-computed face landmarks
│   ├── motion_templates/        # Extracted from HeyGen
│   └── viseme_sprites/          # Fallback sprites
├── src/
│   ├── __init__.py
│   ├── audio_processor.py       # Whisper phoneme extraction
│   ├── lip_synthesizer.py       # VideoReTalking wrapper
│   ├── face_restorer.py         # CodeFormer wrapper
│   ├── super_resolution.py      # Real-ESRGAN wrapper
│   ├── motion_transfer.py       # FOMM wrapper
│   ├── compositor.py            # Final blending
│   └── pipeline.py              # Full pipeline orchestration
├── scripts/
│   ├── download_models.py       # One-time model download
│   ├── extract_motion.py        # Extract motion from HeyGen videos
│   ├── generate_video.py        # Main generation script
│   └── quality_check.py         # Automated quality validation
└── output/                      # Generated videos
```

---

## Hardware Requirements

| Component | Minimum | Recommended (Current) |
|-----------|---------|----------------------|
| GPU | RTX 3080 (10GB) | **RTX 5090 (32GB)** ✅ |
| VRAM | 10GB | 32GB ✅ |
| RAM | 16GB | 32GB+ |
| Storage | 50GB SSD | 100GB NVMe |

With RTX 5090, we can run ALL stages in VRAM simultaneously without model swapping.

---

## Quality Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Resolution | 3840×2160 (4K) or 7680×4320 (8K) | Output dimensions |
| Lip Sync Accuracy | >95% | Syncnet confidence score |
| Face Identity Preservation | >0.85 | ArcFace cosine similarity |
| Temporal Consistency | <0.02 | Inter-frame LPIPS variance |
| Uncanny Valley Score | <0.1 | Custom trained classifier |
| Processing Time | <60s for 60s video | Benchmarked on RTX 5090 |

---

## Installation

See `setup.py` for automated installation.

Manual steps:
1. Create conda environment
2. Install PyTorch with CUDA 12.x
3. Download model weights (~15GB total)
4. Configure Kelly reference assets

---

## Usage

```bash
# Single video generation
python scripts/generate_video.py \
  --audio "path/to/audio.mp3" \
  --archetype scientist \
  --output "output/day-352-scientist.mp4" \
  --resolution 4k

# Batch generation
python scripts/generate_video.py \
  --day 352 \
  --all-archetypes \
  --resolution 4k
```

---

## Development Roadmap

### Phase 1: Core Pipeline (Current Sprint)
- [x] Architecture design
- [ ] Model download and verification
- [ ] VideoReTalking integration
- [ ] CodeFormer integration
- [ ] Real-ESRGAN integration
- [ ] Basic compositor

### Phase 2: Kelly Optimization
- [ ] Extract motion templates from HeyGen archives
- [ ] Fine-tune on Kelly face data
- [ ] Optimize inference for RTX 5090
- [ ] Kelly-specific quality validation

### Phase 3: Production Hardening
- [ ] Automated quality gates
- [ ] Error recovery and retry logic
- [ ] Progress reporting
- [ ] Integration with main pipeline

### Phase 4: Real-Time (Future)
- [ ] WebGL-based real-time rendering
- [ ] Live avatar mode
- [ ] Zero-latency playback

---

## Why Not Just Use APIs?

| Factor | External APIs | Kelly-Sync Local |
|--------|--------------|------------------|
| Cost per video | $0.50-5.00 | $0.01 (electricity) |
| Queue time | 2-60 minutes | 30-90 seconds |
| 4K/8K support | Limited | Full control |
| Kelly optimization | None | Fully customized |
| Dependency | External service | Self-contained |
| Privacy | Videos on their servers | Local only |

For 4,380 videos (365 days × 12 archetypes):
- **APIs: $10,000-20,000 + weeks of queue time**
- **Local: $44 in electricity + 4-5 days of processing**

---

## License

Internal use only. Kelly avatar and voice assets are proprietary.
