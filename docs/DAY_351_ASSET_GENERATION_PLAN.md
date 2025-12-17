# 🚀 DAY 351 ASSET GENERATION MASTER PLAN

**Launch Day:** December 17, 2025  
**Topic:** "Practicing in Your Mind" (Visualization)  
**GROW Track:** "Learning Accountability - Staying on Track"  
**Status:** COMPREHENSIVE GENERATION

---

## 🎯 MISSION

Generate a COMPLETE, world-class Day 351 lesson using every available pipeline to serve every learner in every format across every platform.

---

## 📊 ASSET INVENTORY

### Already Complete ✅
| Asset | Location | Status |
|-------|----------|--------|
| Lesson JSON | `/public/lessons/day-351.json` | ✅ Complete |
| Data Pack JS | `/public/data/day-351-complete.js` | ✅ Complete |
| Phase Images (5) | `/public/kelly/phases/351/` | ✅ Placeholder (copied) |
| Email Template | `api/send-full-lesson-email.ts` | ✅ Ready |
| Content Strategy | `docs/content/` | ✅ Complete |

### To Generate 🔄
| Asset Type | Count | Pipeline | Priority |
|------------|-------|----------|----------|
| Kelly Phase Images (authentic) | 5 | Flux + LoRA | P0 |
| Kelly Response Videos | 5 | ElevenLabs + Sync Labs | P0 |
| Social Media Visuals | 10+ | Flux/Imagen | P1 |
| Infographics | 3-5 | Gemini/Imagen | P1 |
| Thumbnail | 1 | Flux + LoRA | P1 |
| Age Variant Audio | 36 | ElevenLabs | P2 |
| Full Lesson Video | 1 | iClone + A2F | P2 |
| Animated Shorts | 3-5 | MiniMax | P2 |

---

## 🏭 PIPELINE INVENTORY

### 1. IMAGE GENERATION

```
┌─────────────────────────────────────────────────────────────────┐
│  KELLY AUTHENTIC IMAGES                                         │
│  ──────────────────────                                          │
│  Pipeline: scripts/kelly-visual-identity/                        │
│  Model: Flux-dev-lora with CuriousKelly LoRA                     │
│  Quality Gate: Kelly Cop face audit (< 0.385 distance)           │
│                                                                  │
│  Assets to Generate:                                             │
│  • hook.png   - Kelly curious, leaning in                        │
│  • q1.png     - Kelly explaining, gesturing                      │
│  • q2.png     - Kelly thoughtful, hand on chin                   │
│  • q3.png     - Kelly excited, eyes wide                         │
│  • wisdom.png - Kelly warm smile, knowing look                   │
│                                                                  │
│  Command:                                                        │
│  npx tsx scripts/kelly-phase-visuals/phase-visual-generator.ts \ │
│    --day 351 --topic "visualization"                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  INFOGRAPHICS                                                    │
│  ────────────                                                    │
│  Pipeline: scripts/generate-day-infographics.ts                  │
│  Model: Gemini + Imagen 3                                        │
│                                                                  │
│  Assets to Generate:                                             │
│  • Brain scan comparison (doing vs imagining)                    │
│  • The Forgetting Curve graph                                    │
│  • Olympic athlete visualization stats                           │
│  • Piano study results infographic                               │
│  • Step-by-step visualization guide                              │
│                                                                  │
│  Command:                                                        │
│  npx tsx scripts/generate-day-infographics.ts --day 351          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SOCIAL MEDIA VISUALS                                            │
│  ────────────────────                                            │
│  Pipeline: Custom generation                                     │
│  Model: Flux Pro 1.1 (via Replicate)                            │
│                                                                  │
│  Assets to Generate:                                             │
│  • Instagram carousel backgrounds (10)                           │
│  • Twitter thread images (3)                                     │
│  • TikTok thumbnail                                              │
│  • YouTube Shorts thumbnail                                      │
│  • Quote card backgrounds (5)                                    │
│  • Story templates (5)                                           │
│                                                                  │
│  Style: Dark gradient, Kelly brand colors, minimal               │
└─────────────────────────────────────────────────────────────────┘
```

### 2. AUDIO GENERATION

```
┌─────────────────────────────────────────────────────────────────┐
│  KELLY VOICE AUDIO                                               │
│  ─────────────────                                               │
│  Pipeline: scripts/generate-day-audio-elevenlabs.ts              │
│  Model: ElevenLabs (Kelly Voice ID: wAdymQH5YucAkXwmrdL0)        │
│                                                                  │
│  Assets to Generate:                                             │
│                                                                  │
│  Per Age Group (6 groups × 6 phases = 36 files):                │
│  ┌──────────┬────────────────────────────────────┐               │
│  │ Age Group│ Persona                            │               │
│  ├──────────┼────────────────────────────────────┤               │
│  │ 2-5      │ Playful Friend                     │               │
│  │ 6-12     │ Cool Big Sister                    │               │
│  │ 13-17    │ Smart Mentor                       │               │
│  │ 18-35    │ Equal Partner                      │               │
│  │ 36-60    │ Respectful Guide                   │               │
│  │ 61-102   │ Warm Companion                     │               │
│  └──────────┴────────────────────────────────────┘               │
│                                                                  │
│  Phases per age group:                                           │
│  • hook.mp3, cliff.mp3, fact1.mp3                                │
│  • fact2.mp3, fact3.mp3, wisdom.mp3, outro.mp3                  │
│                                                                  │
│  Command:                                                        │
│  npx tsx scripts/generate-day-audio-elevenlabs.ts \              │
│    --day 351 --all                                               │
│                                                                  │
│  Pre-flight:                                                     │
│  npx tsx scripts/kelly-video-factory/kelly-voice-check.ts        │
└─────────────────────────────────────────────────────────────────┘
```

### 3. VIDEO GENERATION

```
┌─────────────────────────────────────────────────────────────────┐
│  LIPSYNC VIDEOS                                                  │
│  ─────────────                                                   │
│  Pipeline: scripts/lipsync-pipeline/                             │
│  Model: Sync Labs lipsync-2-pro                                  │
│                                                                  │
│  Process:                                                        │
│  1. Take Kelly source image (from phase images)                  │
│  2. Generate audio via ElevenLabs                                │
│  3. Apply lipsync via Sync Labs                                  │
│  4. Output: Kelly talking video                                  │
│                                                                  │
│  Command:                                                        │
│  npx tsx scripts/lipsync-pipeline/run-pipeline.ts \              │
│    --day 351 --age adult                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MOTION VIDEOS (Animated Kelly)                                  │
│  ──────────────────────────────                                  │
│  Pipeline: scripts/generate-kelly-animation.ts                   │
│  Model: MiniMax video-01                                         │
│                                                                  │
│  Process:                                                        │
│  1. Take Kelly source image                                      │
│  2. Generate motion (subtle breathing, blinking)                 │
│  3. Apply audio via lipsync                                      │
│                                                                  │
│  Output: 5-10 second animated Kelly clips                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  FULL LESSON VIDEO (Premium)                                     │
│  ───────────────────────────                                     │
│  Pipeline: iClone + Audio2Face (Manual)                          │
│  Owner: User handles iClone export                               │
│                                                                  │
│  Assets Needed:                                                  │
│  • Full audio track (all phases combined)                        │
│  • Lip sync data (Rhubarb or A2F)                               │
│  • Kelly CC5 model                                               │
│                                                                  │
│  Output: 5-7 minute full lesson video                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  HEYGEN VIDEOS (Alternative)                                     │
│  ───────────────────────────                                     │
│  Pipeline: scripts/generate-day-videos-heygen.ts                 │
│  Model: HeyGen Talking Photo API                                 │
│                                                                  │
│  Process:                                                        │
│  1. Upload Kelly talking photo                                   │
│  2. Provide audio or text                                        │
│  3. Generate video                                               │
│                                                                  │
│  Command:                                                        │
│  npx tsx scripts/generate-day-videos-heygen.ts --day 351         │
└─────────────────────────────────────────────────────────────────┘
```

### 4. ALIGNMENT & SYNC

```
┌─────────────────────────────────────────────────────────────────┐
│  FORCED ALIGNMENT                                                │
│  ────────────────                                                │
│  Pipeline: scripts/forced-alignment/                             │
│  Model: Rhubarb Lip Sync                                         │
│                                                                  │
│  Purpose: Generate word-level timing for:                        │
│  • Captions                                                      │
│  • Lipsync                                                       │
│  • Interactive highlighting                                      │
│                                                                  │
│  Command:                                                        │
│  python scripts/forced-alignment/align_audio.py \                │
│    --audio generated-audio/day-351/ \                            │
│    --transcript lessons/day-351.json                             │
│                                                                  │
│  Output: JSON with word timings                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5. QUALITY ASSURANCE

```
┌─────────────────────────────────────────────────────────────────┐
│  KELLY COP (Face Verification)                                   │
│  ─────────────────────────────                                   │
│  Pipeline: tools/kelly-cop/                                      │
│                                                                  │
│  Commands:                                                       │
│  # Face audit on new images                                      │
│  python tools/kelly-cop/kelly_face_audit.py --html               │
│                                                                  │
│  # Quarantine failed images                                      │
│  python tools/kelly-cop/quarantine_batch.py                      │
│                                                                  │
│  Acceptance Criteria:                                            │
│  • MATCH (distance < 0.385) → APPROVED                           │
│  • SUSPICIOUS (0.385-0.55) → MANUAL REVIEW                       │
│  • NO_MATCH (> 0.55) → REJECT & REGENERATE                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎬 GENERATION SEQUENCE

### Phase 1: Pre-Flight Checks (5 min)

```bash
# 1. Verify environment
npx tsx scripts/lesson-factory/preflight-check.ts

# 2. Verify Kelly voice
npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --quick

# 3. Check API credits
echo "ElevenLabs, Replicate, Sync Labs, HeyGen - verify credits"
```

### Phase 2: Generate Kelly Images (15 min)

```bash
# Generate authentic Kelly phase images with LoRA
npx tsx scripts/kelly-phase-visuals/phase-visual-generator.ts \
  --day 351 \
  --topic "visualization" \
  --poses "curious,explaining,thoughtful,excited,warm"

# Run quality check
cd tools/kelly-cop
python kelly_face_audit.py --html --limit 5
```

### Phase 3: Generate Audio (20 min)

```bash
# Generate all age variants
npx tsx scripts/generate-day-audio-elevenlabs.ts \
  --day 351 \
  --all

# This creates 36 audio files across 6 age groups
```

### Phase 4: Generate Lipsync Videos (30 min)

```bash
# Generate lipsync videos for adult track
npx tsx scripts/lipsync-pipeline/run-pipeline.ts \
  --day 351 \
  --age adult

# Or use HeyGen alternative
npx tsx scripts/generate-day-videos-heygen.ts --day 351
```

### Phase 5: Generate Infographics (15 min)

```bash
# Generate lesson infographics
npx tsx scripts/generate-day-infographics.ts --day 351

# Generate social media visuals
npx tsx scripts/kelly-visual-identity/generate-production.ts \
  --day 351 \
  --style social
```

### Phase 6: Generate Alignment Data (10 min)

```bash
# Generate word-level timing
python scripts/forced-alignment/align_audio.py \
  --audio generated-audio/day-351/ \
  --output generated-alignments/day-351.json
```

### Phase 7: Upload to CDN (10 min)

```bash
# Upload to Supabase Storage
npx tsx scripts/fill-supabase-with-assets.ts --day 351

# Backup to Cloudflare R2
# (handled automatically by unified-factory)
```

### Phase 8: Verify Production (5 min)

```bash
# Test assets load on production
curl https://curiouskelly.com/kelly/phases/351/hook.png -I
curl https://curiouskelly.com/lessons/day-351.json -I

# Verify data pack
curl https://curiouskelly.com/data/day-351-complete.js | head -20
```

---

## 📦 UNIFIED FACTORY (ONE COMMAND)

For maximum efficiency, use the unified factory:

```bash
npx tsx scripts/lesson-factory/unified-factory.ts \
  --day 351 \
  --full

# This orchestrates:
# ✓ Visual Plans (Gemini)
# ✓ Infographics (Imagen/Flux Pro)
# ✓ Kelly Source Images (Flux + LoRA)
# ✓ Motion Videos (MiniMax)
# ✓ Audio (ElevenLabs)
# ✓ Lipsync (Sync Labs)
# ✓ Supabase Upload
# ✓ Cloudflare R2 Backup
```

---

## 🌍 MULTI-PLATFORM ASSET MATRIX

### Email Assets
| Asset | Size | Format | Location |
|-------|------|--------|----------|
| Hero Image | 600×400 | PNG | CDN |
| Kelly Avatar | 80×80 | PNG | CDN |
| Fact Icons | 48×48 | SVG | Inline |
| Wisdom Quote BG | 560×200 | PNG | CDN |

### Instagram Assets
| Asset | Size | Format | Count |
|-------|------|--------|-------|
| Carousel Slides | 1080×1350 | PNG | 10 |
| Story Slides | 1080×1920 | PNG | 5 |
| Quote Card | 1080×1080 | PNG | 1 |

### TikTok Assets
| Asset | Size | Format | Count |
|-------|------|--------|-------|
| Hook Video | 1080×1920 | MP4 | 1 |
| Explainer Video | 1080×1920 | MP4 | 1 |
| Thumbnail | 1080×1920 | PNG | 1 |

### Twitter Assets
| Asset | Size | Format | Count |
|-------|------|--------|-------|
| Thread Images | 1200×675 | PNG | 3 |
| Quote Card | 1200×675 | PNG | 1 |

### YouTube Assets
| Asset | Size | Format | Count |
|-------|------|--------|-------|
| Shorts Video | 1080×1920 | MP4 | 1 |
| Thumbnail | 1280×720 | PNG | 1 |

---

## ⏱️ ESTIMATED TIMELINE

| Phase | Duration | Parallelizable |
|-------|----------|----------------|
| Pre-flight | 5 min | No |
| Kelly Images | 15 min | No |
| Audio Generation | 20 min | Yes (with images) |
| Lipsync Videos | 30 min | Yes (with audio) |
| Infographics | 15 min | Yes |
| Alignment | 10 min | Yes (with video) |
| CDN Upload | 10 min | No |
| Verification | 5 min | No |

**Total Sequential:** ~110 minutes  
**Total Parallelized:** ~60 minutes  
**API Cost Estimate:** ~$15-25

---

## 🔧 ENVIRONMENT REQUIREMENTS

### Required API Keys
```env
# ElevenLabs (Voice)
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_KELLY_VOICE_ID=wAdymQH5YucAkXwmrdL0

# Replicate (Images/Video)
REPLICATE_API_TOKEN=r8_...

# Sync Labs (Lipsync)
SYNC_LABS_API_KEY=...

# HeyGen (Alternative Video)
HEYGEN_API_KEY=...

# Google AI (Gemini/Imagen)
GOOGLE_AI_API_KEY=...

# Supabase (Storage/DB)
PUBLIC_SUPABASE_URL=https://...
SUPABASE_SERVICE_ROLE_KEY=...

# Cloudflare R2 (CDN Backup)
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_R2_ACCESS_KEY_ID=...
CLOUDFLARE_R2_SECRET_ACCESS_KEY=...
```

### Required Local Tools
- Node.js 18+
- Python 3.10+
- Rhubarb Lip Sync (for alignment)
- ffmpeg (for video processing)

---

## ✅ SUCCESS CHECKLIST

### Minimum Viable Launch (P0)
- [ ] 5 Kelly phase images (authenticated via Kelly Cop)
- [ ] 1 age variant audio set (adult)
- [ ] Lesson JSON + Data Pack deployed
- [ ] Email template working

### Full Launch (P1)
- [ ] All 6 age variant audio sets
- [ ] Lipsync video for each phase
- [ ] Infographics (3-5)
- [ ] Social media visuals (10+)
- [ ] Alignment data for captions

### Premium Experience (P2)
- [ ] Full iClone lesson video
- [ ] All 36 audio files
- [ ] Animated Kelly motion clips
- [ ] Interactive lesson player working

---

## 🎯 COMMANDS QUICK REFERENCE

```bash
# === ONE COMMAND TO RULE THEM ALL ===
npx tsx scripts/lesson-factory/unified-factory.ts --day 351

# === INDIVIDUAL PIPELINES ===

# Images
npx tsx scripts/kelly-phase-visuals/phase-visual-generator.ts --day 351

# Audio
npx tsx scripts/generate-day-audio-elevenlabs.ts --day 351 --all

# Videos
npx tsx scripts/lipsync-pipeline/run-pipeline.ts --day 351
npx tsx scripts/generate-day-videos-heygen.ts --day 351

# Infographics
npx tsx scripts/generate-day-infographics.ts --day 351

# Quality Check
python tools/kelly-cop/kelly_face_audit.py --html

# Upload
npx tsx scripts/fill-supabase-with-assets.ts --day 351
```

---

## 🚀 LET'S GO

**The pipelines are ready.**  
**The content is written.**  
**The world is waiting.**

Run the factory. Generate the assets. Launch Day 351.

*"The mind that rehearses grows stronger than the mind that merely waits."*

---

*Generated: December 16, 2025*  
*For: Curious Kelly Launch Day*
