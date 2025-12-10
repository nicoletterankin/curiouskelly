# 📚 VISUAL ORCHESTRATION INDEX
## Quick Reference for Curious Kelly Asset Generation

---

## 📂 Document Map

| Document | Purpose | Use When |
|----------|---------|----------|
| **🏆 [UNIFIED_LESSON_FACTORY_FINAL.md](./UNIFIED_LESSON_FACTORY_FINAL.md)** | **THE FINAL PROMPT** — Seeds + Expansion + Kelly spatial awareness | **START HERE** for overnight generation runs |
| **[UNIFIED_LESSON_FACTORY_PROMPT.md](./UNIFIED_LESSON_FACTORY_PROMPT.md)** | Simplified version (108 assets, no expansion) | Quick reference for single-language generation |
| **[GOLDEN_LESSON_PIPELINE_PROMPT.md](./GOLDEN_LESSON_PIPELINE_PROMPT.md)** | Video-only pipeline (original) | Reference for video generation specifics |
| **[../docs/VISUAL_ORCHESTRATION_MASTER.md](../docs/VISUAL_ORCHESTRATION_MASTER.md)** | Visual philosophy & templates | Designing new infographic prompts |
| **[../docs/COMPLETE_VISUAL_ASSET_MANIFEST.md](../docs/COMPLETE_VISUAL_ASSET_MANIFEST.md)** | Complete file/size specifications | Checking what assets are needed |
| **[../content/visual-prompts/INFOGRAPHIC_TEMPLATES.md](../content/visual-prompts/INFOGRAPHIC_TEMPLATES.md)** | Infographic prompt templates | Creating infographics like Day 5 |
| **[../schemas/visual-plan-v2-schema.json](../schemas/visual-plan-v2-schema.json)** | JSON schema for visual plans | Validating visual-plan.json files |
| **[../content/visual-plans/day-005-visual-plan-v2.json](../content/visual-plans/day-005-visual-plan-v2.json)** | Complete example (Day 5 Sound) | Reference for how visual plans should look |

---

## 🔢 Asset Counts at a Glance

### SEED TEMPLATES (Base Assets Per Day)

| Asset Type | Count | Resolution | Format |
|------------|-------|------------|--------|
| HD Videos (main) | 15 | 1920×1080 | MP4 |
| HD Videos (responses) | 36 | 1920×1080 | MP4 |
| Infographics | 15 | 1920×1080 | WebP |
| Option Card Images | 36 | 512×512 | WebP |
| Thumbnails | 3 | 640×360 | WebP |
| Social Share | 3 | 1200×630 | WebP |
| **SEED TOTAL** | **108** | | |

### EXPANSION MATRIX

```
Languages:   EN, ES, FR = 3×
Age Buckets: 5-7, 8-12, 13-17, 18-35, 36-60, 61+ = 6×
Tones:       Playful, Conversational, Reflective = 3×
─────────────────────────────────────────────────
EXPANSION FACTOR: 54× (for videos)
                  18× (for images - lang×age)
```

### Full Scale Per Day

| Asset Type | Seed | Expansion | Total |
|------------|------|-----------|-------|
| Videos | 51 | ×54 | 2,754 |
| Images (generated) | 54 | ×18 avg | 918 |
| Images (reused) | 105 | ×1 | 105 |
| Images (thumbnails/social) | 6 | ×3 | 18 |
| **DAILY TOTAL** | **216** | | **3,795** |

### Full Year (365 Days)

| Metric | Seed Only | Full Expansion |
|--------|-----------|----------------|
| Videos | 18,615 | 1,005,210 |
| Images | 60,225 | 379,785 |
| **TOTAL** | **78,840** | **1,384,995** |
| **Cost** | ~$6,800 | ~$224,000 |

### Recommended Rollout

| Phase | What | Cost | Time |
|-------|------|------|------|
| 1. Seeds Only | EN + default age/tone | ~$19/day | Tonight |
| 2. Languages | Add ES, FR | +$50/day | Week 2 |
| 3. Age Buckets | 6 age variants | +$200/day | Month 2 |
| 4. Tones | Full expansion | +$340/day | Month 3 |

---

## 🎯 The Day 5 Gold Standard

The screenshot you showed is our BEST infographic ever. It's stored at:

```
https://tvja2xxyyjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/phases/005/hook-full.webp
```

### The Prompt That Created It:

```
Educational infographic: Split-scene comparison showing sound transmission.

LEFT: Deep underwater - humpback whale with visible sound waves traveling 
far through crystal-clear blue water, "1500 m/s" label, "Communicating Whale".

RIGHT: Railroad tracks at sunset - train approaching, young woman listening 
near tracks, sound waves through rails, "Hear the train coming".

Header: "Sound Travels Faster in Water & Solids"

Style: Photorealistic cinematic, 8K, warm-cool color transition.
```

### Why It Works:
1. **Split scene** = instant visual comparison
2. **Relatable subjects** = whale + train everyone knows
3. **Human POV** = girl listening creates identification
4. **Data callout** = 1500 m/s adds scientific credibility
5. **Color story** = blue→orange guides eye + creates mood

---

## 🚀 Quick Start Commands

### Generate Day N Complete:

```bash
npx ts-node scripts/lesson-factory/generate-complete-lesson.ts 1
```

### Generate Visual Plan Only:

```bash
npx ts-node scripts/lesson-factory/stages/visual-plan.ts 1 "The Explorer"
```

### Generate Videos Only (if images exist):

```bash
npx ts-node scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts --day 1 --archetype explorer
```

### Verify Day N:

```bash
npx ts-node scripts/lesson-factory/stages/verify.ts 1
```

---

## 📋 Database Content Structure

After factory runs, each `lesson_atoms.content` looks like:

```json
{
  "script": "Kelly's teaching...",
  "script_video_url": "https://.../hook_main.mp4",
  "infographic_url": "https://.../hook-infographic.webp",
  "kellyPose": "explaining",
  "kellyEmotion": "curious",
  "options": [
    {
      "letter": "A",
      "text": "Option A text",
      "quality": "good",
      "response": "Kelly's response A...",
      "response_video_url": "https://.../hook_response_a.mp4",
      "option_image_url": "https://.../hook-option-a.webp"
    },
    {
      "letter": "B",
      "text": "Option B text", 
      "quality": "best",
      "response": "Kelly's response B...",
      "response_video_url": "https://.../hook_response_b.mp4",
      "option_image_url": "https://.../hook-option-b.webp"
    },
    {
      "letter": "C",
      "text": "Option C text",
      "quality": "redirect",
      "response": "Kelly's response C...",
      "response_video_url": "https://.../hook_response_c.mp4",
      "option_image_url": "https://.../hook-option-c.webp"
    }
  ]
}
```

---

## 🔗 Storage Paths

### Supabase Buckets:
- `kelly-videos` → All HD lipsync videos
- `lesson-visuals` → All images (infographics, options, thumbnails)

### Path Patterns:
```
kelly-videos/day-{XXX}/{archetype}/{phase}_{type}.mp4
lesson-visuals/phases/{XXX}/{archetype}/{phase}-infographic.webp
lesson-visuals/phases/{XXX}/{archetype}/options/{phase}-option-{a|b|c}.webp
lesson-visuals/thumbnails/{XXX}-{archetype}.webp
lesson-visuals/social/{XXX}-{archetype}.webp
```

### Cloudflare R2 Backup:
```
https://assets.curiouskelly.com/{same paths as above}
```

---

## ⚡ TL;DR

**To generate seed templates (tonight):**

1. Open fresh Claude session
2. Paste contents of `UNIFIED_LESSON_FACTORY_FINAL.md`
3. Say: "Generate Day 1 seeds"
4. Wait ~3.5 hours for 108 seed assets
5. Test at `curiouskelly.com/learn?day=1`

**To expand to all variants (later):**

1. Say: "Expand Day 1 to all languages and age buckets"
2. Wait for 3,780 total assets
3. Frontend auto-selects based on user profile

## 🎯 Key Specs Confirmed

| Spec | Value |
|------|-------|
| **Lipsync Model** | `lipsync-2-pro` ✅ (premium) |
| **Kelly Spatial Awareness** | ✅ Looks at diagrams, gestures to options |
| **Motion Videos** | Reusable (lipsync with different audio) |
| **Languages** | EN, ES, FR (precomputed per CLAUDE.md) |
| **Age Buckets** | 5-7, 8-12, 13-17, 18-35, 36-60, 61+ |
| **Tone Variants** | Playful, Conversational, Reflective |

## 🌙 Tonight's Run

```bash
# Generate Day 1 seeds (EN only)
npx ts-node scripts/lesson-factory/generate-seeds.ts --day 1

# Expected: 108 assets
# Expected time: ~3.5 hours  
# Expected cost: ~$19
```

---

*Index v2.0 — December 9, 2025*  
*"Kelly is WITH the learner, not above. And she knows where the content is."*

