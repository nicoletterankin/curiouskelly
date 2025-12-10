# Kelly Video System Architecture

## Overview

The Kelly Video System generates character-consistent talking videos of Kelly for the Daily Lesson experience. It operates in two modes:

1. **Pre-computed** - High-quality videos generated ahead of time
2. **Real-time** - Fast approximation for interactive scenarios

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KELLY VIDEO PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌─────────┐ │
│  │   Kelly    │    │  Animate   │    │  Generate  │    │  Apply  │ │
│  │   LoRA     │───▶│   Image    │───▶│   Voice    │───▶│ Lipsync │ │
│  │  (Flux)    │    │   (SVD)    │    │ (11Labs)   │    │ (V2V)   │ │
│  └────────────┘    └────────────┘    └────────────┘    └─────────┘ │
│        │                 │                 │                │      │
│        ▼                 ▼                 ▼                ▼      │
│   Kelly Image      Kelly Animation     Kelly Audio      Kelly Video│
│   (Character       (Natural motion)    (Her voice)     (Full!)    │
│    Consistent)                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Kelly LoRA

The **Kelly LoRA** is a trained Low-Rank Adaptation model that ensures Kelly looks like Kelly across all generated images.

- **Location**: `huggingface.co/CuriousKellycom/curious-kelly-lora`
- **Base Model**: Flux Dev
- **Trigger Word**: `kelly`
- **Scale**: 0.85 (calibrated for best results)

### Character Specification

```
Hair:   Long wavy brown hair
Eyes:   Brown eyes
Outfit: Powder blue sweater
```

### Prompt Format

```
kelly, woman with long wavy brown hair and brown eyes, wearing powder blue sweater, [action], [environment], [emotion], professional photography, 4K
```

## Quality Tiers

| Tier | Resolution | Animation | Lipsync | Upscale | Time | Use Case |
|------|------------|-----------|---------|---------|------|----------|
| Preview | 512×512 | SVD 14f | Wav2Lip | No | ~90s | Testing |
| Standard | 1344×768 | SVD 25f | Wav2Lip | No | ~180s | Most content |
| Production | 1344×768 | SVD-XT 25f | SadTalker | 4K | ~300s | Hero content |

## Templates

Templates define Kelly's pose, environment, and emotional state.

| Template | Environment | Emotion | Action | Phase |
|----------|-------------|---------|--------|-------|
| welcome | Forest path | Warm | Arms open | Intro |
| explain | Studio | Engaged | Gesturing | Q2/Teaching |
| heartfelt | Golden light | Sincere | Hand on heart | Wisdom |
| curious | Natural | Curious | Head tilt | Q1/Q3 |
| excited | Bright | Excited | Hands up | Hook |
| thoughtful | Library | Pensive | Hand to chin | Reflection |

## Pre-computation Strategy

### Content Scope (30 Days)
- 30 days × 5 phases = 150 unique images
- 150 animations (one per image)
- 2,700 audio files (per age/language variant)
- 2,700 final videos

### Optimization: Reuse Animations

Since the audio varies by age group and language but Kelly's visual doesn't need to, we can:

1. **Generate 150 images** (once per day×phase)
2. **Generate 150 animations** (once per image)
3. **Apply lipsync 2,700 times** (once per variant)

This reduces compute by ~10x.

### Cost Estimate (Standard Quality)

| Component | Count | Unit Cost | Total |
|-----------|-------|-----------|-------|
| Images | 150 | $0.003 | $0.45 |
| Animations | 150 | $0.05 | $7.50 |
| Audio | 2,700 | $0.002 | $5.40 |
| Lipsync | 2,700 | $0.02 | $54.00 |
| **Total** | | | **~$68** |

### Time Estimate

Sequential: 2,700 × 180s = 135 hours
Parallel (5 jobs): 27 hours
Parallel (10 jobs): 13.5 hours

## Real-time Considerations

For interactive scenarios (conversations, Q&A), we need faster responses.

### Option A: Pre-rendered Base + Fast Lipsync

1. Pre-render "base" Kelly animations for each template
2. Apply fast lipsync at runtime (~5-10s)
3. Quality: Good, Latency: 5-10s

### Option B: 2D Avatar Fallback

1. Use 2D Kelly with blendshape-based lipsync
2. Real-time audio analysis
3. Quality: Lower, Latency: <100ms

### Option C: Hybrid

1. Show 2D immediately for responsiveness
2. Generate video in background
3. Transition to video when ready
4. Quality: Best of both, Latency: Instant visual

## Storage Architecture

```
supabase/
└── kelly-templates/
    ├── reference/
    │   └── kelly_primary_face.jpeg
    ├── production/
    │   ├── images/
    │   │   └── day_001_hook.png
    │   ├── animations/
    │   │   └── day_001_hook.mp4
    │   ├── audio/
    │   │   └── day_001_hook_age6-8_en.mp3
    │   └── videos/
    │       └── day_001_hook_age6-8_en.mp4
    └── lora/
        └── (LoRA test outputs)
```

## API Endpoints

### Pre-computed Content

```
GET /api/kelly-video?day=1&phase=hook&age=6-8&lang=en
→ Returns: { videoUrl, audioUrl, transcript }
```

### On-demand Generation

```
POST /api/kelly-generate
Body: { template, script, quality }
→ Returns: { jobId, status }

GET /api/kelly-generate/:jobId
→ Returns: { status, videoUrl, progress }
```

## Quality Assurance

### Face Audit

Every generated Kelly image should pass the face audit:

```bash
python kelly_face_audit.py <image.png>
→ MATCH (similarity > 0.85) or FAIL
```

### Automated Checks

1. **Character consistency**: Face audit on all images
2. **Sweater color**: Color histogram check (blue > pink)
3. **Animation quality**: Motion smoothness metric
4. **Lipsync quality**: Alignment score

## Next Steps

1. **Calibration**: Run LoRA scale tests to find optimal settings
2. **Batch Processing**: Build pipeline to generate all 30-day content
3. **Storage**: Set up Supabase buckets for permanent storage
4. **API**: Create endpoints for serving pre-computed content
5. **Real-time**: Prototype hybrid approach (2D + video)



