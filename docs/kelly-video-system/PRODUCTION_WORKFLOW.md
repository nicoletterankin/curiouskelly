# Kelly Video Production Workflow

## Overview

This document describes the complete workflow for generating production-quality Kelly videos at scale. The system is designed to produce **10,950+ unique videos** for a billion learners.

## Scale

| Dimension | Count | Notes |
|-----------|-------|-------|
| Days | 365 | Full year of lessons |
| Phases | 5 | hook, q1, q2, q3, wisdom |
| Age Groups | 6 | 4-5, 6-8, 9-11, 12-14, 15-17, 18+ |
| Languages | 3 | en, es, fr |
| **Total Videos** | **32,850** | Maximum scope |

## Optimization Strategy

### Key Insight: Reuse Base Assets

Kelly's visual appearance doesn't change by age group or language—only the audio does. This means:

1. **Images**: 365 days × 5 phases = **1,825 unique images**
2. **Animations**: Same as images = **1,825 unique animations**
3. **Audio**: Full matrix = **32,850 unique audio files**
4. **Final Videos**: Full matrix = **32,850 unique videos**

By reusing images/animations, we reduce compute by **~90%**.

## Production Pipeline

```
Phase 1: IMAGE GENERATION (1,825 images)
    ↓
    Flux Dev + Kelly LoRA (scale 0.85)
    ↓
    Quality Gate: Face Audit
    ↓
Phase 2: ANIMATION (1,825 videos)
    ↓
    Stable Video Diffusion
    ↓
    Quality Gate: Motion Smoothness
    ↓
Phase 3: AUDIO GENERATION (32,850 audio files)
    ↓
    ElevenLabs API (Kelly's voice)
    ↓
    Store in Supabase
    ↓
Phase 4: LIPSYNC (32,850 final videos)
    ↓
    Wav2Lip V2V (fast) or SadTalker HQ (quality)
    ↓
    Store in Supabase
    ↓
Phase 5: 4K UPSCALE (optional, hero content)
    ↓
    Real-ESRGAN
    ↓
    Final delivery
```

## Step-by-Step Workflow

### Step 1: Calibration (One-time)

Test LoRA scales to find optimal character consistency:

```bash
cd C:\Users\user\UI-TARS-desktop
node scripts/kelly-video-factory/systematic-calibration.cjs --lora
```

Review results at: `/lipsync/calibration.html`

Current optimal scale: **0.85**

### Step 2: Batch Image Generation

Generate all base images:

```bash
# Preview (cost estimate only)
node scripts/kelly-video-factory/batch-image-generator.cjs --days 30 --dry-run

# Generate (resumable)
node scripts/kelly-video-factory/batch-image-generator.cjs --days 30
```

Output: `template-forge/production-images/`

Manifest: `manifest.json` with all image paths and metadata

### Step 3: Quality Gate

Run quality checks on generated images:

```bash
node scripts/kelly-video-factory/quality-gate.cjs --production
```

Output: `template-forge/quality-reports/`

### Step 4: Animation Generation

*(Coming soon)*

For each image, generate animation:

```bash
node scripts/kelly-video-factory/batch-animation-generator.cjs
```

### Step 5: Audio Generation

Use existing lesson content from Supabase:

```bash
npx ts-node scripts/lipsync-pipeline/generate-lesson-audio.ts
```

### Step 6: Lipsync Application

*(Coming soon)*

Apply lipsync to each animation + audio pair:

```bash
node scripts/kelly-video-factory/batch-lipsync-generator.cjs
```

## File Organization

```
template-forge/
├── production-images/
│   ├── manifest.json
│   ├── day_001_hook.png
│   ├── day_001_q1.png
│   ├── day_001_q2.png
│   ├── day_001_q3.png
│   ├── day_001_wisdom.png
│   └── ... (1,825 files)
├── production-animations/
│   └── ... (1,825 files)
├── calibration/
│   └── lora_*/
│       └── comparison.html
└── quality-reports/
    └── quality_report_*.html
```

## Supabase Storage

```
kelly-templates/
├── reference/
│   └── kelly_primary_face.jpeg
├── production/
│   ├── images/
│   ├── animations/
│   ├── audio/
│   └── videos/
├── calibration/
│   └── lora/
├── factory/
└── lora/
```

## Cost Estimates

### 30-Day Pilot

| Component | Count | Unit Cost | Total |
|-----------|-------|-----------|-------|
| Images | 150 | $0.003 | $0.45 |
| Animations | 150 | $0.05 | $7.50 |
| Audio | 2,700 | $0.002 | $5.40 |
| Lipsync | 2,700 | $0.02 | $54.00 |
| **Total** | | | **~$68** |

### Full Year

| Component | Count | Unit Cost | Total |
|-----------|-------|-----------|-------|
| Images | 1,825 | $0.003 | $5.48 |
| Animations | 1,825 | $0.05 | $91.25 |
| Audio | 32,850 | $0.002 | $65.70 |
| Lipsync | 32,850 | $0.02 | $657.00 |
| **Total** | | | **~$820** |

## Time Estimates

### 30-Day Pilot (Standard Quality)

| Phase | Sequential | Parallel (5 jobs) |
|-------|------------|-------------------|
| Images | 75 min | 15 min |
| Animations | 7.5 hours | 1.5 hours |
| Audio | 22 min | 5 min |
| Lipsync | 7.5 hours | 1.5 hours |
| **Total** | ~16 hours | ~3.5 hours |

## Quality Settings

### Preview (Testing)
- Resolution: 512×512
- Animation: 14 frames
- Lipsync: Wav2Lip
- Time: ~90s/video

### Standard (Production)
- Resolution: 1344×768
- Animation: 25 frames
- Lipsync: Wav2Lip
- Time: ~140s/video

### Production (Hero)
- Resolution: 4K (upscaled)
- Animation: 25 frames
- Lipsync: SadTalker HQ
- Time: ~300s/video

## Monitoring Dashboard

Live at: `curiouskelly.com/lipsync/`

- Factory output: `/lipsync/factory.html`
- LoRA templates: `/lipsync/kelly-lora-audit.html`
- Calibration: `/lipsync/calibration.html`

## CLI Reference

```bash
# Factory CLI
node scripts/kelly-video-factory/cli.cjs generate <template> "<script>" [--quality preview|standard|production]
node scripts/kelly-video-factory/cli.cjs batch <manifest.json>
node scripts/kelly-video-factory/cli.cjs templates

# Calibration
node scripts/kelly-video-factory/systematic-calibration.cjs --lora
node scripts/kelly-video-factory/systematic-calibration.cjs --motion
node scripts/kelly-video-factory/systematic-calibration.cjs --full

# Batch Generation
node scripts/kelly-video-factory/batch-image-generator.cjs --days 30
node scripts/kelly-video-factory/quality-gate.cjs --production

# Pre-computation Planning
node scripts/kelly-video-factory/precompute-planner.cjs
```

## Next Steps

1. ✅ LoRA scale calibration complete
2. 🔄 Generate 30-day image batch
3. ⏳ Build animation cache system
4. ⏳ Integrate face audit automation
5. ⏳ Deploy batch lipsync generator
6. ⏳ Create progress monitoring dashboard

