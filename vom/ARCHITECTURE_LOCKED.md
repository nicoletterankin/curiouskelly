# 🔒 ARCHITECTURE LOCKED — December 9, 2025

> **This architecture is COMMITTED. Do not deviate without explicit approval.**

---

## The Vision

**Kelly is WITH the learner, not above.** She's spatially aware of the content around her. She looks at diagrams, gestures to options, pushes rails down to reveal things, and pulls content out to make points. The experience is interactive, choreographed, and works together visually.

---

## The Numbers

```
SEED TEMPLATES (Base):       216 assets/day
FULL EXPANSION:            3,795 assets/day
FULL YEAR EXPANDED:    1,385,000 assets total
```

---

## The Architecture

### Seed → Expansion Model

```
SEED (Generate Once)
├── Motion Videos (MiniMax) — 51 per day
├── Kelly Source Images — 51 per day (for video generation)
├── Kelly Response Images — 36 per day (different expressions)
├── Infographics (Imagen) — 15 per day  
├── Option Cards (Imagen) — 36 per day
├── Backgrounds — 15 per day
└── Thumbnails/Social — 6 per day
= 210 seed assets

EXPANSION (Multiply)
├── Videos: 51 × 54 (lang×age×tone) = 2,754
├── Infographics: 15 × 18 (lang×age) = 270
├── Option Cards: 36 × 18 = 648
├── Kelly Images: 87 × 1 (reused) = 87
├── Backgrounds: 15 × 1 (reused) = 15
├── Thumbnails/Social: 6 × 3 (lang) = 18
= 3,792 total assets per day
```

### Key Innovation

**Motion videos are REUSABLE.** Generate once, then lipsync with different audio for each language/age/tone variant. One motion video → 54 final videos.

**Kelly images are REUSED.** Same visual, different audio. No need to regenerate her face for each language.

---

## The Tech Stack

| Component | Technology | Model |
|-----------|------------|-------|
| Voice | ElevenLabs | `eleven_multilingual_v2` |
| Voice ID | Kelly | `wAdymQH5YucAkXwmrdL0` |
| Images | Flux + LoRA | `lucataco/flux-dev-lora` |
| Motion | MiniMax | `video-01` |
| **Lipsync** | **Sync Labs** | **`lipsync-2-pro`** ✅ |
| Infographics | Google Imagen | `imagen-3.0-generate-002` |
| Visual Plans | Google Gemini | `gemini-pro` |
| Storage | Supabase + Cloudflare R2 | — |

---

## Kelly's Spatial Awareness

### Gaze Targets
- **Camera** — Direct learner connection
- **Up-right** — Looking at diagram/infographic
- **Right** — Acknowledging options rail
- **Down** — Contemplative moment

### Gestures
- **Push rail** — Slide panel down to reveal
- **Pull content** — Bring element into focus
- **Point** — Direct attention to data
- **Open palm** — Welcome choices
- **Hands to heart** — Wisdom moments

---

## Options Are IMAGES

Not just text. 512×512 image cards with:
- Visual representation of the answer
- Icon (emoji) top-right
- 2-4 word label bottom
- Green glow border for "best" quality

---

## File Locations

```
vom/
├── UNIFIED_LESSON_FACTORY_FINAL.md  ← THE MASTER PROMPT
├── UNIFIED_LESSON_FACTORY_PROMPT.md ← Simplified version
├── GOLDEN_LESSON_PIPELINE_PROMPT.md ← Original video-only
├── VISUAL_ORCHESTRATION_INDEX.md    ← Quick reference
└── ARCHITECTURE_LOCKED.md           ← This file

docs/
├── VISUAL_ORCHESTRATION_MASTER.md
├── COMPLETE_VISUAL_ASSET_MANIFEST.md
└── ...

content/visual-prompts/
├── INFOGRAPHIC_TEMPLATES.md
└── ...

schemas/
└── visual-plan-v2-schema.json
```

---

## Tonight's Run

```bash
npx ts-node scripts/lesson-factory/generate-seeds.ts --day 1

# Output: 210 seed assets
#   - 51 motion videos
#   - 51 Kelly source images
#   - 36 Kelly response images
#   - 15 infographics
#   - 36 option cards
#   - 15 backgrounds
#   - 6 thumbnails/social
# Time: ~3.5 hours
# Cost: ~$25
```

---

## Cost Summary

| Phase | Per Day | Full Year |
|-------|---------|-----------|
| Seeds Only | $19 | $6,935 |
| + Languages | +$50 | +$18,250 |
| + Age Buckets | +$200 | +$73,000 |
| + Tones | +$340 | +$124,100 |
| **Full Expansion** | **$615** | **$224,355** |

---

## Sacred Rules

1. **Kelly is humble** — "I don't have all the answers"
2. **No option is wrong** — Every choice leads to learning
3. **Motion videos are reusable** — Don't regenerate for each variant
4. **Use `lipsync-2-pro`** — Premium quality, no compromises
5. **Kelly is spatially aware** — She knows where the content is
6. **Backup everything** — Supabase + Cloudflare R2
7. **Precompute languages** — EN/ES/FR per CLAUDE.md
8. **Age-appropriate delivery** — 6 buckets, not one-size-fits-all

---

## Memory ID

This architecture is saved in Claude's memory as: **`12066006`**

---

*Locked December 9, 2025*  
*"Every frame teaches. Kelly is WITH you."*

