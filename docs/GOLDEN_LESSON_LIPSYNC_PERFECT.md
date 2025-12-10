# 🏆 GOLDEN LESSON LIPSYNC - PERFECTION GUIDE

**Created:** December 7, 2025  
**Purpose:** Achieve film-quality lipsync for Day 1 "Starting Fresh"  
**Target:** 99% lip-sync accuracy with natural expressions

---

## 📊 Executive Summary

The Golden Lesson (Day 1 "Starting Fresh") uses our **most advanced lipsync pipeline** with:

| Component | Technology | Quality | Status |
|-----------|------------|---------|--------|
| Audio | ElevenLabs Premium | 99% natural | ✅ Ready |
| Image | Flux Dev + Kelly LoRA | Character-consistent | ✅ Ready |
| Base Video | LivePortrait | Natural motion | ✅ Ready |
| Lipsync | Sync Labs lipsync-2-pro | 95%+ accuracy | ✅ Available |
| Upscale | Real-ESRGAN + CodeFormer | 4K clarity | ✅ Ready |
| Fallback | Unity Blendshapes | Phoneme-based | ✅ Ready |

---

## 🎬 Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        GOLDEN LESSON LIPSYNC PIPELINE                           │
│                        ─────────────────────────────────                         │
│                                                                                 │
│   ┌──────────────┐                                                              │
│   │   SUPABASE   │  Fetch Day 1 atoms (15 total: 5 phases × 3 archetypes)      │
│   │   Database   │───────────────────────────────────────────────────────┐      │
│   └──────────────┘                                                       │      │
│                                                                          ▼      │
│   ┌──────────────┐    ┌──────────────┐                                          │
│   │  ElevenLabs  │    │   Kelly      │                                          │
│   │   Premium    │    │   LoRA       │                                          │
│   │    TTS       │    │  (Flux Dev)  │                                          │
│   └──────┬───────┘    └──────┬───────┘                                          │
│          │                   │                                                  │
│          │  Kelly's Voice    │  Character-Consistent Image                      │
│          │  (archetype-      │  (template based on phase)                       │
│          │   specific)       │                                                  │
│          ▼                   ▼                                                  │
│   ┌──────────────────────────────────────┐                                      │
│   │           LIVEPORTRAIT               │                                      │
│   │    Natural Motion + Expressions      │                                      │
│   │    - Eye retargeting enabled         │                                      │
│   │    - Lip retargeting enabled         │                                      │
│   │    - Natural head movement           │                                      │
│   └──────────────────┬───────────────────┘                                      │
│                      │                                                          │
│                      │  Base Video (with lip approximation)                     │
│                      ▼                                                          │
│   ┌──────────────────────────────────────┐                                      │
│   │          SYNC LABS LIPSYNC-2         │  ← Premium tier (if available)       │
│   │      95%+ Lip-Sync Accuracy          │                                      │
│   │   - Frame-by-frame mouth sync        │                                      │
│   │   - Preserves original expressions   │                                      │
│   └──────────────────┬───────────────────┘                                      │
│                      │                                                          │
│                      │  High-Quality Lipsynced Video                            │
│                      ▼                                                          │
│   ┌──────────────────────────────────────┐                                      │
│   │      REAL-ESRGAN + CODEFORMER        │                                      │
│   │          4K Upscaling                │                                      │
│   │   - Face enhancement                 │                                      │
│   │   - Artifact removal                 │                                      │
│   └──────────────────┬───────────────────┘                                      │
│                      │                                                          │
│                      ▼                                                          │
│   ┌──────────────────────────────────────┐                                      │
│   │        SUPABASE STORAGE              │                                      │
│   │   kelly-templates/production/videos  │                                      │
│   └──────────────────────────────────────┘                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Video Output Structure

```
kelly-templates/production/videos/
├── day_001_Hook_The_Explorer.mp4
├── day_001_Hook_The_Rebel.mp4
├── day_001_Hook_The_Scientist.mp4
├── day_001_Fact1_The_Explorer.mp4
├── day_001_Fact1_The_Rebel.mp4
├── day_001_Fact1_The_Scientist.mp4
├── day_001_Fact2_The_Explorer.mp4
├── day_001_Fact2_The_Rebel.mp4
├── day_001_Fact2_The_Scientist.mp4
├── day_001_Fact3_The_Explorer.mp4
├── day_001_Fact3_The_Rebel.mp4
├── day_001_Fact3_The_Scientist.mp4
├── day_001_Wisdom_The_Explorer.mp4
├── day_001_Wisdom_The_Rebel.mp4
└── day_001_Wisdom_The_Scientist.mp4
```

---

## 🚀 Quick Start

### Generate All Golden Lesson Videos

```bash
# Full pipeline - highest quality
npx tsx scripts/golden-lesson-lipsync-generator.ts

# Preview mode - faster, lower quality
npx tsx scripts/golden-lesson-lipsync-generator.ts --preview

# Dry run - see what would be generated
npx tsx scripts/golden-lesson-lipsync-generator.ts --dry-run
```

### Generate Specific Content

```bash
# Single archetype
npx tsx scripts/golden-lesson-lipsync-generator.ts --archetype "The Explorer"

# Single phase
npx tsx scripts/golden-lesson-lipsync-generator.ts --phase Hook

# Specific combination
npx tsx scripts/golden-lesson-lipsync-generator.ts --archetype "The Rebel" --phase Wisdom
```

### Generate Unity Fallback Alignments

```bash
# Pre-compute phoneme alignments for Unity blendshapes
npx tsx scripts/golden-lesson-alignment-generator.ts
```

---

## 🔑 Required API Keys

Add these to your `.env` file:

```env
# Required
REPLICATE_API_TOKEN=r8_xxxxx
ELEVENLABS_API_KEY=sk_xxxxx
PUBLIC_SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJxxxxx

# Premium (highly recommended)
SYNC_LABS_API_KEY=xxxxx     # https://sync.so - 95%+ lipsync quality

# Optional (alternative tiers)
HEDRA_API_KEY=xxxxx         # https://hedra.com
FAL_KEY=xxxxx               # https://fal.ai (OmniHuman)

# Kelly's Voice
ELEVENLABS_KELLY_VOICE_ID=wAdymQH5YucAkXwmrdL0
```

---

## 📋 Quality Checklist

### Lip-Sync (99% accuracy target)
- [ ] Every phoneme matches audio timing
- [ ] No drift over duration
- [ ] Natural transitions between sounds
- [ ] Correct jaw movement weight
- [ ] Mouth corners move naturally

### Eyes (Natural expression)
- [ ] Micro-saccades present (2-4 per second when speaking)
- [ ] Natural blinks (10-15 per minute)
- [ ] Eye direction appropriate for content
- [ ] Emotion-matched expressions

### Face Overall
- [ ] No uncanny valley artifacts
- [ ] Subsurface scattering on skin (realistic translucency)
- [ ] Natural skin texture movement
- [ ] Appropriate lighting response

### Character Consistency
- [ ] Kelly's brown hair consistent
- [ ] Brown eyes with catchlights
- [ ] Powder blue sweater (NOT pink, red, or other colors)
- [ ] Face matches reference (85%+ similarity)

---

## 💰 Cost Estimates

### Per Video (Production Quality)

| Step | API | Cost |
|------|-----|------|
| Audio | ElevenLabs | ~$0.03 |
| Image | Replicate | ~$0.003 |
| LivePortrait | Replicate | ~$0.10 |
| Sync Labs | Sync Labs | ~$0.20 |
| Upscale | Replicate | ~$0.05 |
| **Total** | | **~$0.38** |

### Full Golden Lesson (15 videos)

| Quality | Time | Cost |
|---------|------|------|
| Preview | ~30 min | ~$3 |
| Standard | ~60 min | ~$4 |
| Production | ~75 min | ~$6 |

---

## 🎯 Phase-to-Template Mapping

The pipeline uses different visual templates for each lesson phase:

| Phase | Template | Kelly's State |
|-------|----------|---------------|
| Hook | `excited` | Eyes sparkling, joyful expression, expressive gestures |
| Fact1 | `curious` | Head tilted, raised eyebrow, warm smile |
| Fact2 | `explain` | Animated expression, conceptual gesture, leaning forward |
| Fact3 | `thoughtful` | Contemplative, chin on hand, soft knowing smile |
| Wisdom | `heartfelt` | Hand on heart, genuine warmth, direct eye contact |

---

## 🎤 Voice Settings by Archetype

| Archetype | Stability | Style | Notes |
|-----------|-----------|-------|-------|
| The Explorer | 0.45 | 0.3 | More expressive, wonder-filled |
| The Rebel | 0.40 | 0.4 | Most expressive, challenging tone |
| The Scientist | 0.55 | 0.15 | Measured, analytical |

---

## 🔄 Fallback System

When full video isn't available (loading, network issues), the system falls back to:

### 1. Unity WebGL (Blendshapes)

```javascript
// Kelly Alignment Player handles this automatically
const player = new KellyAlignmentPlayer();
await player.loadAlignment(1, '6-12', 'en', 'Hook');
player.playWithAudio(audioElement);
```

### 2. 2D Avatar (Mouth States)

The alignment player also sends simplified mouth states to the 2D Kelly:
- `speaking` - jawOpen > 30
- `talking` - jawOpen > 10
- `idle` - jawOpen ≤ 10

---

## 📊 Database Tables

### `kelly_video_assets`
Stores video URLs and metadata for each generated video.

```sql
SELECT * FROM kelly_video_assets 
WHERE day_number = 1 
AND status = 'validated';
```

### `lipsync_alignments`
Stores pre-computed phoneme alignments for Unity fallback.

```sql
SELECT * FROM lipsync_alignments 
WHERE day_number = 1 
AND language = 'en';
```

---

## 🐛 Troubleshooting

### "Character doesn't look like Kelly"

1. Check LoRA is loading: `CONFIG.KELLY_LORA_URL` should be valid
2. Verify LoRA scale: Should be 0.85
3. Check negative prompt includes sweater color constraints

### "Lipsync drifts over time"

1. Ensure audio quality is high (no artifacts)
2. Try increasing Sync Labs model to `lipsync-2-pro`
3. Verify audio duration matches video duration

### "Generation fails with timeout"

1. Check API rate limits aren't exceeded
2. Increase poll timeout in config
3. Try `--preview` mode first to validate setup

### "Sync Labs not available"

The pipeline will automatically fall back to LivePortrait output. Quality will be ~80% instead of 95%, but still good.

---

## 📚 Related Documentation

| Document | Purpose |
|----------|---------|
| `GOLDEN_LESSON_DEEP_DIVE.md` | Day 1 content structure & features |
| `KELLY_VIDEO_PERFECTION_PLAN.md` | Research & architecture decisions |
| `docs/kelly-video-system/ARCHITECTURE.md` | Full video system overview |
| `scripts/lipsync-pipeline/README.md` | Standard lipsync pipeline |

---

## ✅ Success Criteria

A Golden Lesson video is "perfect" when:

1. ✅ Lip movements match audio at phoneme level
2. ✅ Eyes have natural micro-movements
3. ✅ Kelly's identity is unmistakably consistent
4. ✅ Blue sweater is correct color (not pink/red)
5. ✅ No visual artifacts or glitches
6. ✅ 4K resolution, 30+ FPS
7. ✅ Emotional expression matches script content
8. ✅ Passes face audit (similarity > 0.85)

---

## 🎬 Example Output

After running the pipeline, you'll have:

```
generated-videos/golden-lesson/
├── day_001_Hook_The_Explorer/
│   ├── audio.mp3              # ElevenLabs audio
│   ├── final_4k.mp4           # Production video
│   └── metadata.json          # Generation metadata
├── day_001_Hook_The_Rebel/
│   └── ...
└── generation_results_xxx.json  # Full run report
```

---

## 🚀 Next Steps After Generation

1. **Verify Quality**: Watch each video, check checklist above
2. **Test in Player**: Load `learn.html?day=1` and test all archetypes
3. **Check Fallback**: Test Unity blendshape player works
4. **Monitor Analytics**: Track user engagement with video vs. fallback
5. **Iterate**: Generate response videos for each option

---

**The Golden Lesson sets the standard for Kelly's quality. Make it perfect.** ✨

---

*Last Updated: December 7, 2025*  
*Document: `docs/GOLDEN_LESSON_LIPSYNC_PERFECT.md`*



