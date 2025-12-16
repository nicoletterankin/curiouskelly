# 🏭 UNIFIED LESSON FACTORY — Complete Build Specification

> **To:** Fresh Claude Session (Expert Mode)  
> **From:** Chief Academic Officer  
> **Date:** December 9, 2025  
> **Mission:** Generate ONE complete, perfect, interactive lesson with ALL assets — videos, infographics, option images, thumbnails — for ALL archetypes, saved to Supabase, backed up to Cloudflare, fully wired to frontend.

---

## 🎯 THE FACTORY OUTPUT

When this prompt runs successfully, it produces:

```
DAY {N} COMPLETE LESSON PACKAGE
├── 51 HD Lipsync Videos (17 per archetype × 3 archetypes)
├── 15 Educational Infographics (5 phases × 3 archetypes)
├── 36 Option Card Images (3 options × 4 phases × 3 archetypes)
├── 3 Thumbnails (1 per archetype)
├── 3 Social Share Images (1 per archetype)
├── Database records fully populated
├── Cloudflare backup complete
└── Frontend wired and tested
```

**Total Assets Per Day: 108**  
**Total Assets for 365 Days: 39,420**

---

## PART 0: KELLY'S SACRED VOICE (READ FIRST)

Before generating ANY content, audio, or video, internalize this:

> *"I don't have all the answers. But I love finding them. And I think learning is better together."*

### Core Attributes

| Attribute | What It Means | What It's NOT |
|-----------|---------------|---------------|
| **Humble** | "I don't have all the answers" | Not superior, not a know-it-all |
| **Curious** | "But I love finding them" | Not bored, not performative |
| **Collaborative** | "Learning is better together" | Not hierarchical, not transactional |
| **Warm** | Like a friend, like Mr. Rogers | Not corporate, not distant |
| **Simple** | Clear, honest, direct | Not jargon, not clever, not tryhard |

### The Voice Test

Before any Kelly content ships:
1. **Would Mr. Rogers say this?** — Warm, kind, inclusive
2. **Is Kelly WITH the learner, not above?** — No hierarchy
3. **Does it sound like a friend, not a brand?** — Human first
4. **Is it simple without being dumb?** — Respect intelligence
5. **Does it invite, not demand?** — Choice, not obligation

### Response Quality Philosophy

| Quality | Kelly's Tone | Expression | Purpose |
|---------|-------------|------------|---------|
| **"best"** | Genuinely delighted | Eyes bright, warm excitement | Celebrate curiosity |
| **"good"** | Supportive, validating | Gentle, accepting | Warm encouragement |
| **"redirect"** | Understanding, never judgmental | Thoughtful, patient | Guide back gently |

**CRITICAL: No option is "wrong"!** Every choice leads to learning.

---

## PART 1: THE COMPLETE LESSON STRUCTURE

### 5-Phase Interactive Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: HOOK (5-8 seconds)                                                │
│  Purpose: Spark curiosity, introduce topic                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ASSETS GENERATED:                                                          │
│  ├── hook_main.mp4 (Kelly video)                                           │
│  ├── hook_response_a.mp4, hook_response_b.mp4, hook_response_c.mp4         │
│  ├── hook-infographic.webp (educational visual)                            │
│  ├── hook-option-a.webp, hook-option-b.webp, hook-option-c.webp (512×512)  │
│  └── Database: lesson_atoms.content with all URLs                          │
│                                                                             │
│  FLOW:                                                                      │
│  1. Kelly VIDEO plays Hook script                                           │
│  2. Options appear (3 IMAGE CARDS + text)                                  │
│  3. User clicks option → Kelly VIDEO responds                              │
│  4. Auto-advance to Fact1                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: FACT1 (8-12 seconds)                                              │
│  Purpose: First scientific/evidence-based fact                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ASSETS: Same structure as Hook (1 main + 3 responses + infographic + 3 options)
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: FACT2 (8-12 seconds)                                              │
│  Purpose: Deepen understanding                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ASSETS: Same structure as Hook
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: FACT3 (8-12 seconds)                                              │
│  Purpose: Peak revelation, "aha" moment                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ASSETS: Same structure as Hook
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: WISDOM (5-8 seconds)                                              │
│  Purpose: Universal truth, memorable takeaway                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ASSETS:                                                                    │
│  ├── wisdom_main.mp4 (Kelly video) — NO RESPONSES                          │
│  ├── wisdom-infographic.webp (emotional backdrop)                          │
│  └── Database: lesson_atoms.content with video URL                         │
│                                                                             │
│  FLOW:                                                                      │
│  1. Kelly VIDEO plays Wisdom script                                         │
│  2. NO options (wisdom is universal)                                        │
│  3. Show completion celebration                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PART 2: COMPLETE ASSET MANIFEST

### Per Archetype, Per Day

| Asset Type | Count | Resolution | Format | Storage Location |
|------------|-------|------------|--------|------------------|
| **VIDEOS** |
| Main Script Videos | 5 | 1920×1080 | MP4 H.264 | kelly-videos/day-{N}/{arch}/ |
| Response Videos | 12 | 1920×1080 | MP4 H.264 | kelly-videos/day-{N}/{arch}/ |
| **IMAGES** |
| Phase Infographics | 5 | 1920×1080 | WebP | lesson-visuals/phases/{N}/ |
| Option Card A | 4 | 512×512 | WebP | lesson-visuals/phases/{N}/options/ |
| Option Card B | 4 | 512×512 | WebP | lesson-visuals/phases/{N}/options/ |
| Option Card C | 4 | 512×512 | WebP | lesson-visuals/phases/{N}/options/ |
| **META** |
| Thumbnail | 1 | 640×360 | WebP | lesson-visuals/thumbnails/ |
| Social Share | 1 | 1200×630 | WebP | lesson-visuals/social/ |

**Per Archetype Total: 36 assets**  
**Per Day Total (3 archetypes): 108 assets**

### Complete File Structure

```
supabase-storage/
├── kelly-videos/
│   └── day-001/
│       ├── explorer/
│       │   ├── hook_main.mp4
│       │   ├── hook_response_a.mp4
│       │   ├── hook_response_b.mp4
│       │   ├── hook_response_c.mp4
│       │   ├── fact1_main.mp4
│       │   ├── fact1_response_a.mp4
│       │   ├── fact1_response_b.mp4
│       │   ├── fact1_response_c.mp4
│       │   ├── fact2_main.mp4
│       │   ├── fact2_response_a.mp4
│       │   ├── fact2_response_b.mp4
│       │   ├── fact2_response_c.mp4
│       │   ├── fact3_main.mp4
│       │   ├── fact3_response_a.mp4
│       │   ├── fact3_response_b.mp4
│       │   ├── fact3_response_c.mp4
│       │   └── wisdom_main.mp4
│       ├── rebel/
│       │   └── (same 17 videos)
│       └── scientist/
│           └── (same 17 videos)
│
├── lesson-visuals/
│   └── phases/
│       └── 001/
│           ├── explorer/
│           │   ├── hook-infographic.webp
│           │   ├── fact1-infographic.webp
│           │   ├── fact2-infographic.webp
│           │   ├── fact3-infographic.webp
│           │   ├── wisdom-infographic.webp
│           │   └── options/
│           │       ├── hook-option-a.webp
│           │       ├── hook-option-b.webp
│           │       ├── hook-option-c.webp
│           │       ├── fact1-option-a.webp
│           │       ├── fact1-option-b.webp
│           │       ├── fact1-option-c.webp
│           │       ├── fact2-option-a.webp
│           │       ├── fact2-option-b.webp
│           │       ├── fact2-option-c.webp
│           │       ├── fact3-option-a.webp
│           │       ├── fact3-option-b.webp
│           │       └── fact3-option-c.webp
│           ├── rebel/
│           │   └── (same structure)
│           └── scientist/
│               └── (same structure)
│
└── cloudflare-backup/
    └── (mirror of above)
```

---

## PART 3: DATABASE SCHEMA

### Tables Involved

```sql
-- Core lesson metadata (365 rows)
core_lessons
├── id: uuid
├── day_number: 1-365
├── topic: "Starting Fresh"
├── universal_truth: "Fresh starts provide psychological permission to change"
├── icon_emoji: "🌅"
├── thumbnail_url: "https://..."
└── social_share_url: "https://..."

-- Content atoms (20,341 rows: 365 days × 5 phases × 3 archetypes + wisdom)
lesson_atoms
├── id: uuid
├── core_lesson_id: fk → core_lessons.id
├── archetype: "The Explorer" | "The Rebel" | "The Scientist"
├── phase: "Hook" | "Fact1" | "Fact2" | "Fact3" | "Wisdom"
├── content: jsonb (COMPLETE structure below)
├── hd_video_url: "https://..." (main script video)
└── visual_url: "https://..." (infographic)
```

### Complete content JSONB Structure

```json
{
  "script": "Kelly's main teaching content for this phase",
  "script_video_url": "https://[supabase]/kelly-videos/day-001/explorer/hook_main.mp4",
  "infographic_url": "https://[supabase]/lesson-visuals/phases/001/explorer/hook-infographic.webp",
  "kellyPose": "explaining",
  "kellyEmotion": "curious",
  "options": [
    {
      "letter": "A",
      "text": "What learner sees as Option A text",
      "quality": "good",
      "response": "Kelly's spoken response when user picks A",
      "response_video_url": "https://[supabase]/kelly-videos/day-001/explorer/hook_response_a.mp4",
      "option_image_url": "https://[supabase]/lesson-visuals/phases/001/explorer/options/hook-option-a.webp"
    },
    {
      "letter": "B",
      "text": "What learner sees as Option B text",
      "quality": "best",
      "response": "Kelly's spoken response when user picks B",
      "response_video_url": "https://[supabase]/kelly-videos/day-001/explorer/hook_response_b.mp4",
      "option_image_url": "https://[supabase]/lesson-visuals/phases/001/explorer/options/hook-option-b.webp"
    },
    {
      "letter": "C",
      "text": "What learner sees as Option C text",
      "quality": "redirect",
      "response": "Kelly's spoken response when user picks C",
      "response_video_url": "https://[supabase]/kelly-videos/day-001/explorer/hook_response_c.mp4",
      "option_image_url": "https://[supabase]/lesson-visuals/phases/001/explorer/options/hook-option-c.webp"
    }
  ]
}
```

---

## PART 4: THE GENERATION PIPELINE

### Pipeline Architecture

```
INPUT: Day Number + Core Lesson Data
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 1: VISUAL PLAN GENERATION (Gemini)                                    ║
║  Generate complete visual-plan.json for all phases and archetypes            ║
╚══════════════════════════════════════════════════════════════════════════════╝
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 2: INFOGRAPHIC GENERATION (Imagen 3)                                  ║
║  Generate 5 infographics per archetype × 3 archetypes = 15 images            ║
╚══════════════════════════════════════════════════════════════════════════════╝
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 3: OPTION CARD GENERATION (Imagen 3)                                  ║
║  Generate 12 option cards per archetype × 3 archetypes = 36 images           ║
╚══════════════════════════════════════════════════════════════════════════════╝
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 4: VIDEO GENERATION (ElevenLabs + Flux + MiniMax + SyncLabs)         ║
║  Generate 17 videos per archetype × 3 archetypes = 51 videos                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 5: THUMBNAIL & SOCIAL GENERATION                                      ║
║  Generate 1 thumbnail + 1 social per archetype × 3 = 6 images                ║
╚══════════════════════════════════════════════════════════════════════════════╝
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 6: UPLOAD TO SUPABASE                                                 ║
║  Upload all 108 assets to Supabase Storage                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 7: BACKUP TO CLOUDFLARE R2                                            ║
║  Mirror all assets to Cloudflare for redundancy                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 8: DATABASE UPDATE                                                    ║
║  Populate lesson_atoms.content with all URLs                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
           │
           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 9: VERIFICATION                                                       ║
║  Test all URLs, validate video playback, confirm database integrity          ║
╚══════════════════════════════════════════════════════════════════════════════╝

OUTPUT: Complete lesson ready for frontend consumption
```

---

## PART 5: STAGE-BY-STAGE SPECIFICATIONS

### STAGE 1: Visual Plan Generation

**API:** Google Gemini Pro  
**Input:** Day number, topic, universal truth, lesson atoms content  
**Output:** `visual-plan-v2.json`

```typescript
interface VisualPlan {
  day: number;
  topic: string;
  theme: {
    colorPalette: string;
    mood: 'wonder' | 'curiosity' | 'excitement' | 'warmth' | 'discovery' | 'reflection';
    environment: string;
  };
  archetypes: {
    explorer: ArchetypeVisualPlan;
    rebel: ArchetypeVisualPlan;
    scientist: ArchetypeVisualPlan;
  };
}

interface ArchetypeVisualPlan {
  phases: PhaseVisual[];
}

interface PhaseVisual {
  phase: 'hook' | 'fact1' | 'fact2' | 'fact3' | 'wisdom';
  infographic: {
    type: 'split-scene' | 'before-after' | 'process' | 'scale' | 'anatomy' | 'kelly-hero';
    prompt: string;           // Full Imagen prompt
    textOverlay: string;      // Max 50 chars
    style: string;
  };
  optionCards?: {             // Not present for Wisdom
    optionA: OptionCardSpec;
    optionB: OptionCardSpec;
    optionC: OptionCardSpec;
  };
}

interface OptionCardSpec {
  imagePrompt: string;        // Full 512×512 generation prompt
  icon: string;               // Emoji
  label: string;              // 2-4 words
  quality: 'best' | 'good' | 'redirect';
}
```

**Gemini Prompt Template:**

```
You are an expert educational visual designer for Curious Kelly.

Given this lesson:
- Day: {day_number}
- Topic: {topic}
- Universal Truth: {universal_truth}
- Archetype: {archetype}
- Phase Content: {phase_content_from_database}

Generate a detailed visual plan following these requirements:

1. INFOGRAPHIC (1920×1080):
   - Type: Choose from split-scene, before-after, process, scale, anatomy
   - Must teach the core concept VISUALLY
   - Include data callouts where applicable
   - Use color palette: {theme_colors}

2. OPTION CARDS (512×512 each):
   - Option A ({quality}): Visual representation of "{option_text}"
   - Option B ({quality}): Visual representation of "{option_text}"  
   - Option C ({quality}): Visual representation of "{option_text}"
   - Each card needs: icon, 2-4 word label, visual scene
   - "best" options get subtle green glow border
   - Must be instantly readable on mobile

Output valid JSON matching the VisualPlan interface.
```

---

### STAGE 2: Infographic Generation

**API:** Google Imagen 3.0 via Vertex AI  
**Model:** `imagen-3.0-generate-002`  
**Resolution:** 1920×1080 (16:9)

**Infographic Template Types:**

#### Type 1: SPLIT-SCENE COMPARISON (Like Day 5 Sound)

```
Educational infographic: Split-scene comparison showing {CONCEPT}.

LEFT SIDE: {SCENE_A}
- Environment: {SETTING}
- Subject: {VISUAL_ELEMENT}
- Visual effect: {HOW_CONCEPT_MANIFESTS}
- Label: "{TEXT_LABEL}"
- Data: "{STATISTIC}"

RIGHT SIDE: {SCENE_B}
- Environment: {SETTING}
- Subject: {VISUAL_ELEMENT}
- Visual effect: {HOW_CONCEPT_MANIFESTS}
- Label: "{TEXT_LABEL}"
- Human element: {RELATABLE_PERSON}

Color transition: {COLOR_A} smoothly blending to {COLOR_B}.
Header text: "{EDUCATIONAL_TAKEAWAY - 8 WORDS MAX}"

Style: Photorealistic cinematic, 8K resolution, educational diagram 
aesthetic, clean typography, dramatic lighting.

Negative: blurry, low quality, distorted text, misspelled words, 
cluttered, confusing, cartoon, anime, watermark, logo.
```

#### Type 2: BEFORE/AFTER TRANSFORMATION

```
Educational infographic: Before/After transformation showing {CONCEPT}.

BEFORE STATE (left):
- Scene: {INITIAL_CONDITION}
- Key visual: {BEFORE_APPEARANCE}
- Label: "Before: {STATE}"

TRANSFORMATION ARROW:
- Visual: {ARROW_OR_TIMELINE}
- Label: "{CAUSE_OF_CHANGE}"

AFTER STATE (right):
- Scene: {FINAL_CONDITION}
- Key visual: {AFTER_APPEARANCE}
- Label: "After: {STATE}"

Header: "{TRANSFORMATION_DESCRIBED}"
Style: Documentary photography, clean educational overlays, 8K.
```

#### Type 3: PROCESS/CYCLE DIAGRAM

```
Educational infographic: {PROCESS_NAME} shown as visual cycle.

STEP 1: {VISUAL} → Label: "{STEP}"
STEP 2: {VISUAL} → Label: "{STEP}"
STEP 3: {VISUAL} → Label: "{STEP}"
[3-6 steps maximum]

Layout: Circular for cycles, linear for sequences.
Background: {CONTEXTUAL_ENVIRONMENT}
Header: "{PROCESS_NAME}"
Style: Modern infographic, clean icons, clear arrows, 8K.
```

---

### STAGE 3: Option Card Generation

**API:** Google Imagen 3.0  
**Resolution:** 512×512 (1:1 square)

**Option Card Prompt Template:**

```
512×512 educational choice card for "{OPTION_TEXT}".

Visual scene: {VISUAL_REPRESENTATION_OF_ANSWER}
Icon: {EMOJI} in top-right corner (48px)
Label: "{2-4_WORD_LABEL}" in bold at bottom center

Design specifications:
- White text on dark gradient band at bottom
- {Border: subtle green glow if quality="best", neutral otherwise}
- High contrast for mobile visibility
- Clean, tappable design
- Educational infographic style
- Rounded corners (16px radius)

Negative: blurry, text errors, low contrast, cluttered.
```

**Quality-Specific Styling:**

| Quality | Border | Indicator |
|---------|--------|-----------|
| best | 4px green glow (#4CAF50) | Subtle checkmark icon |
| good | 2px neutral gray (#666) | None |
| redirect | 2px neutral gray (#666) | None |

---

### STAGE 4: Video Generation Pipeline

**Per Video Steps:**

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4.1: AUDIO GENERATION (ElevenLabs)                          │
├──────────────────────────────────────────────────────────────────┤
│ API: https://api.elevenlabs.io/v1/text-to-speech/{voice_id}     │
│ Model: eleven_multilingual_v2                                    │
│ Voice ID: wAdymQH5YucAkXwmrdL0 (Kelly)                          │
│                                                                  │
│ Voice Settings by Archetype:                                     │
│   Explorer: stability=0.45, similarity=0.85, style=0.25, speed=1.05
│   Rebel:    stability=0.40, similarity=0.85, style=0.35, speed=1.10
│   Scientist: stability=0.55, similarity=0.85, style=0.15, speed=0.95
│                                                                  │
│ Voice Modifiers by Response Quality:                             │
│   best:     style += 0.10, stability -= 0.05 (more enthusiasm)  │
│   good:     no change (standard warmth)                         │
│   redirect: style -= 0.05, stability += 0.10 (more thoughtful)  │
│                                                                  │
│ Output: MP3 file, ~5-15 seconds                                  │
│ Cost: ~$0.02 per generation                                      │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4.2: SOURCE IMAGE (Replicate Flux + Kelly LoRA)             │
├──────────────────────────────────────────────────────────────────┤
│ Model: lucataco/flux-dev-lora                                    │
│ LoRA: https://huggingface.co/CuriousKellycom/curious-kelly-lora │
│ Scale: 0.90                                                      │
│                                                                  │
│ ═══════════════════════════════════════════════════════════════  │
│ KELLY MASTER IDENTITY (LOCKED - NEVER CHANGE)                    │
│ ═══════════════════════════════════════════════════════════════  │
│                                                                  │
│ "kelly, calm confident female teacher, warm brown wavy           │
│ shoulder-length hair with subtle caramel highlights              │
│ center-parted, hazel-brown eyes with steady direct gaze,         │
│ soft natural features, light natural makeup, wearing soft        │
│ powder blue cashmere crewneck sweater, {EXPRESSION_FOR_CONTEXT}, │
│ looking directly at camera, professional warm classroom setting, │
│ soft natural lighting, shallow depth of field,                   │
│ professional portrait photography, 85mm lens, 4K UHD"            │
│                                                                  │
│ Expression Variants:                                             │
│   Main Script: "warm welcoming expression, composed posture"     │
│   Response best: "genuinely delighted, eyes bright with joy"     │
│   Response good: "warm supportive, gentle approving smile"       │
│   Response redirect: "thoughtful understanding, compassionate"   │
│                                                                  │
│ NEGATIVE (LOCKED):                                               │
│ "pink sweater, red sweater, purple sweater, teal sweater,        │
│ green sweater, yellow sweater, auburn hair, chestnut hair,       │
│ deformed, blurry, bad anatomy, extra fingers, mutated hands,     │
│ poorly drawn face, mutation, disfigured, low quality,            │
│ wandering eyes, looking away, darting gaze"                      │
│                                                                  │
│ Output: 1344×768 PNG (16:9)                                      │
│ Cost: ~$0.02                                                     │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4.3: MOTION VIDEO (MiniMax Video-01)                        │
├──────────────────────────────────────────────────────────────────┤
│ Model: minimax/video-01                                          │
│ Input: Source image from Step 4.2                                │
│                                                                  │
│ Motion Prompt (Main Script):                                     │
│ "Professional female teacher speaking directly to camera.        │
│ She is TALKING and her mouth is moving naturally as she speaks.  │
│ Steady direct eye contact with camera throughout.                │
│ Natural breathing, soft occasional blinking.                     │
│ Smooth cinematic quality, warm professional lighting.            │
│ CRITICAL: Mouth must open and move naturally while speaking.     │
│ Eyes stay focused on camera. Maintain calm composed presence.    │
│ AVOID: closed mouth, frozen face, wandering eyes, looking away,  │
│ excessive head movement, jerky motions."                         │
│                                                                  │
│ Motion Prompt (Response best):                                   │
│ "...responding with genuine delight and enthusiasm.              │
│ Eyes crinkled slightly with authentic joy.                       │
│ Slight forward lean showing engagement..."                       │
│                                                                  │
│ Motion Prompt (Response good):                                   │
│ "...responding with warm encouragement.                          │
│ Gentle approving nod. Open accepting posture..."                 │
│                                                                  │
│ Motion Prompt (Response redirect):                               │
│ "...responding with understanding and patience.                  │
│ Compassionate gaze. Slight empathetic head tilt..."              │
│                                                                  │
│ Output: ~6 second MP4                                            │
│ Cost: ~$0.12                                                     │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4.4: LIP-SYNC (Sync Labs)                                   │
├──────────────────────────────────────────────────────────────────┤
│ API: https://api.sync.so/v2/generate                            │
│ Model: lipsync-2-pro                                             │
│ Input: Motion video (4.3) + Audio (4.1)                          │
│                                                                  │
│ Fallback: Replicate wav2lip if Sync Labs unavailable             │
│                                                                  │
│ Output: Final HD video with synced lip movements                 │
│ Cost: ~$0.20                                                     │
└──────────────────────────────────────────────────────────────────┘

TOTAL PER VIDEO: ~$0.36
TOTAL PER DAY (51 videos): ~$18.36
```

---

### STAGE 5: Thumbnail & Social Generation

**Thumbnail (640×360):**

```
Thumbnail for Day {N}: {TOPIC}

Composition:
- Left 40%: Kelly portrait (medium shot, welcoming expression)
- Right 60%: Key visual from hook infographic (simplified)
- Top-left: Day badge "Day {N}" with icon emoji
- Bottom: Topic text "{TOPIC}" in bold

Style: Eye-catching, high contrast, readable at small sizes.
Colors: Match lesson theme palette.
```

**Social Share (1200×630):**

```
Social media preview for Day {N}: {TOPIC}

Composition:
- Full-width: Blurred hook infographic as background
- Center: Kelly portrait with slight glow
- Top: "✨ Curious Kelly" branding
- Bottom: "Day {N}: {TOPIC}"
- Subtitle: "{UNIVERSAL_TRUTH}" (abbreviated)

Style: Professional, shareable, curiosity-inducing.
```

---

### STAGE 6: Upload to Supabase

**Bucket Structure:**

```typescript
const BUCKETS = {
  videos: 'kelly-videos',
  visuals: 'lesson-visuals'
};

const uploadToSupabase = async (file: Buffer, path: string, bucket: string) => {
  const { data, error } = await supabase.storage
    .from(bucket)
    .upload(path, file, {
      contentType: getContentType(path),
      cacheControl: '31536000', // 1 year cache
      upsert: true
    });
  
  if (error) throw error;
  
  // Get public URL
  const { data: urlData } = supabase.storage
    .from(bucket)
    .getPublicUrl(path);
  
  return urlData.publicUrl;
};

// Upload patterns
await uploadToSupabase(videoBuffer, `day-001/explorer/hook_main.mp4`, 'kelly-videos');
await uploadToSupabase(imageBuffer, `phases/001/explorer/hook-infographic.webp`, 'lesson-visuals');
await uploadToSupabase(optionBuffer, `phases/001/explorer/options/hook-option-a.webp`, 'lesson-visuals');
```

---

### STAGE 7: Backup to Cloudflare R2

**R2 Configuration:**

```typescript
const R2_CONFIG = {
  accountId: process.env.CLOUDFLARE_ACCOUNT_ID,
  accessKeyId: process.env.R2_ACCESS_KEY_ID,
  secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  bucket: 'kelly-assets-backup'
};

const backupToR2 = async (file: Buffer, path: string) => {
  const S3 = new AWS.S3({
    endpoint: `https://${R2_CONFIG.accountId}.r2.cloudflarestorage.com`,
    accessKeyId: R2_CONFIG.accessKeyId,
    secretAccessKey: R2_CONFIG.secretAccessKey,
    signatureVersion: 'v4',
  });
  
  await S3.putObject({
    Bucket: R2_CONFIG.bucket,
    Key: path,
    Body: file,
    ContentType: getContentType(path)
  }).promise();
  
  return `https://assets.curiouskelly.com/${path}`;
};
```

**Backup Strategy:**
- Primary: Supabase Storage (fast, integrated)
- Backup: Cloudflare R2 (redundancy, CDN)
- All URLs point to Supabase; R2 is disaster recovery

---

### STAGE 8: Database Update

**SQL Update Pattern:**

```sql
-- Update main script video URL
UPDATE lesson_atoms 
SET content = jsonb_set(
  content, 
  '{script_video_url}', 
  '"https://[supabase]/kelly-videos/day-001/explorer/hook_main.mp4"'
)
WHERE core_lesson_id = '{lesson_id}' 
  AND archetype = 'The Explorer' 
  AND phase = 'Hook';

-- Update infographic URL
UPDATE lesson_atoms 
SET content = jsonb_set(
  content, 
  '{infographic_url}', 
  '"https://[supabase]/lesson-visuals/phases/001/explorer/hook-infographic.webp"'
)
WHERE core_lesson_id = '{lesson_id}' 
  AND archetype = 'The Explorer' 
  AND phase = 'Hook';

-- Update response video URLs (for each option)
UPDATE lesson_atoms 
SET content = jsonb_set(
  jsonb_set(
    jsonb_set(
      content,
      '{options,0,response_video_url}',
      '"https://[supabase]/kelly-videos/day-001/explorer/hook_response_a.mp4"'
    ),
    '{options,1,response_video_url}',
    '"https://[supabase]/kelly-videos/day-001/explorer/hook_response_b.mp4"'
  ),
  '{options,2,response_video_url}',
  '"https://[supabase]/kelly-videos/day-001/explorer/hook_response_c.mp4"'
)
WHERE core_lesson_id = '{lesson_id}' 
  AND archetype = 'The Explorer' 
  AND phase = 'Hook';

-- Update option image URLs
UPDATE lesson_atoms 
SET content = jsonb_set(
  jsonb_set(
    jsonb_set(
      content,
      '{options,0,option_image_url}',
      '"https://[supabase]/lesson-visuals/phases/001/explorer/options/hook-option-a.webp"'
    ),
    '{options,1,option_image_url}',
    '"https://[supabase]/lesson-visuals/phases/001/explorer/options/hook-option-b.webp"'
  ),
  '{options,2,option_image_url}',
  '"https://[supabase]/lesson-visuals/phases/001/explorer/options/hook-option-c.webp"'
)
WHERE core_lesson_id = '{lesson_id}' 
  AND archetype = 'The Explorer' 
  AND phase = 'Hook';
```

---

### STAGE 9: Verification

**Verification Checklist:**

```typescript
interface VerificationResult {
  day: number;
  archetype: string;
  phase: string;
  checks: {
    // Video checks
    mainVideoExists: boolean;
    mainVideoPlayable: boolean;
    mainVideoDuration: number;
    responseAVideoExists: boolean;
    responseBVideoExists: boolean;
    responseCVideoExists: boolean;
    
    // Image checks
    infographicExists: boolean;
    infographicSize: { width: number; height: number };
    optionAImageExists: boolean;
    optionBImageExists: boolean;
    optionCImageExists: boolean;
    
    // Database checks
    scriptVideoUrlPopulated: boolean;
    infographicUrlPopulated: boolean;
    responseVideoUrlsPopulated: boolean;
    optionImageUrlsPopulated: boolean;
    
    // Quality checks
    kellyFaceConsistent: boolean;
    lipSyncAccurate: boolean;
    audioQualityClear: boolean;
  };
  passed: boolean;
}

const verify = async (day: number): Promise<VerificationResult[]> => {
  const results: VerificationResult[] = [];
  
  for (const archetype of ['The Explorer', 'The Rebel', 'The Scientist']) {
    for (const phase of ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom']) {
      // Run all checks...
      results.push(result);
    }
  }
  
  return results;
};
```

---

## PART 6: FRONTEND INTEGRATION

### Player HTML Structure (Updated with Images)

```html
<div class="lesson-player" data-state="loading">
  <!-- Video Container -->
  <div class="video-container">
    <video id="kellyVideo" autoplay playsinline>
      <source src="{script_video_url}" type="video/mp4">
    </video>
  </div>
  
  <!-- Infographic Overlay (shows during video) -->
  <div class="infographic-overlay">
    <img src="{infographic_url}" alt="{phase} diagram" class="phase-infographic">
  </div>
  
  <!-- Options Container (IMAGE CARDS) -->
  <div class="options-container hidden">
    <p class="options-intro">What interests you most?</p>
    <div class="option-cards">
      
      <!-- OPTION A - Now with IMAGE -->
      <button class="option-card" data-option="A" data-quality="{options[0].quality}">
        <img src="{options[0].option_image_url}" alt="Option A" class="option-image">
        <div class="option-content">
          <span class="option-letter">A</span>
          <span class="option-text">{options[0].text}</span>
        </div>
      </button>
      
      <!-- OPTION B - Now with IMAGE -->
      <button class="option-card" data-option="B" data-quality="{options[1].quality}">
        <img src="{options[1].option_image_url}" alt="Option B" class="option-image">
        <div class="option-content">
          <span class="option-letter">B</span>
          <span class="option-text">{options[1].text}</span>
        </div>
      </button>
      
      <!-- OPTION C - Now with IMAGE -->
      <button class="option-card" data-option="C" data-quality="{options[2].quality}">
        <img src="{options[2].option_image_url}" alt="Option C" class="option-image">
        <div class="option-content">
          <span class="option-letter">C</span>
          <span class="option-text">{options[2].text}</span>
        </div>
      </button>
      
    </div>
  </div>
</div>
```

### Option Card CSS (Image-Based)

```css
.option-card {
  display: flex;
  flex-direction: column;
  border-radius: 16px;
  overflow: hidden;
  background: var(--card-bg);
  border: 3px solid transparent;
  transition: all 0.3s ease;
  cursor: pointer;
}

.option-card[data-quality="best"] {
  border-color: rgba(76, 175, 80, 0.5);
  box-shadow: 0 0 20px rgba(76, 175, 80, 0.2);
}

.option-card:hover {
  transform: scale(1.02);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.option-image {
  width: 100%;
  height: 200px;
  object-fit: cover;
}

.option-content {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.option-letter {
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--accent);
}

.option-text {
  font-size: 1rem;
  color: var(--text);
  line-height: 1.4;
}

/* Mobile: Stack cards vertically */
@media (max-width: 768px) {
  .option-cards {
    flex-direction: column;
    gap: 12px;
  }
  
  .option-image {
    height: 150px;
  }
}
```

---

## PART 7: ENVIRONMENT VARIABLES

```bash
# Supabase
PUBLIC_SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...

# ElevenLabs
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_KELLY_VOICE_ID=wAdymQH5YucAkXwmrdL0

# Replicate
REPLICATE_API_TOKEN=r8_...

# Sync Labs
SYNC_LABS_API_KEY=...

# Google AI (Gemini + Imagen)
GOOGLE_AI_API_KEY=...
GOOGLE_CLOUD_PROJECT=...

# Cloudflare R2
CLOUDFLARE_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=kelly-assets-backup
```

---

## PART 8: COST ANALYSIS

### Per Day Cost Breakdown

| Stage | Items | Cost Each | Total |
|-------|-------|-----------|-------|
| Visual Plan (Gemini) | 3 | $0.01 | $0.03 |
| Infographics (Imagen) | 15 | $0.04 | $0.60 |
| Option Cards (Imagen) | 36 | $0.02 | $0.72 |
| Audio (ElevenLabs) | 51 | $0.02 | $1.02 |
| Source Images (Flux) | 51 | $0.02 | $1.02 |
| Motion Videos (MiniMax) | 51 | $0.12 | $6.12 |
| Lipsync (Sync Labs) | 51 | $0.20 | $10.20 |
| Thumbnails | 3 | $0.02 | $0.06 |
| Social Images | 3 | $0.02 | $0.06 |
| **TOTAL PER DAY** | **108** | | **~$19.83** |

### Full Year Cost

| Metric | Value |
|--------|-------|
| Days | 365 |
| Total Assets | 39,420 |
| Total Videos | 18,615 |
| Total Images | 20,805 |
| **TOTAL COST** | **~$7,238** |

---

## PART 9: EXECUTION SCRIPT

### Master Factory Runner

```typescript
// scripts/lesson-factory/generate-complete-lesson.ts

import { generateVisualPlan } from './stages/visual-plan';
import { generateInfographics } from './stages/infographics';
import { generateOptionCards } from './stages/option-cards';
import { generateVideos } from './stages/videos';
import { generateThumbnails } from './stages/thumbnails';
import { uploadToSupabase } from './stages/upload';
import { backupToCloudflare } from './stages/backup';
import { updateDatabase } from './stages/database';
import { verifyLesson } from './stages/verify';

interface LessonFactoryConfig {
  dayNumber: number;
  archetypes: string[];
  skipExisting: boolean;
  verifyOnly: boolean;
}

async function runLessonFactory(config: LessonFactoryConfig) {
  const { dayNumber, archetypes } = config;
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`🏭 LESSON FACTORY — Day ${dayNumber}`);
  console.log(`${'═'.repeat(60)}\n`);
  
  // Fetch lesson data from Supabase
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();
  
  const { data: atoms } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id);
  
  for (const archetype of archetypes) {
    console.log(`\n📦 Processing: ${archetype}`);
    
    // STAGE 1: Visual Plan
    console.log('  🎨 Stage 1: Generating visual plan...');
    const visualPlan = await generateVisualPlan(lesson, atoms, archetype);
    
    // STAGE 2: Infographics
    console.log('  🖼️  Stage 2: Generating infographics...');
    const infographics = await generateInfographics(visualPlan);
    
    // STAGE 3: Option Cards
    console.log('  🎴 Stage 3: Generating option cards...');
    const optionCards = await generateOptionCards(visualPlan);
    
    // STAGE 4: Videos
    console.log('  🎬 Stage 4: Generating videos...');
    const videos = await generateVideos(atoms, archetype);
    
    // STAGE 5: Thumbnails
    console.log('  🖼️  Stage 5: Generating thumbnails...');
    const thumbnails = await generateThumbnails(lesson, archetype);
    
    // STAGE 6: Upload
    console.log('  ☁️  Stage 6: Uploading to Supabase...');
    const urls = await uploadToSupabase({
      infographics,
      optionCards,
      videos,
      thumbnails
    });
    
    // STAGE 7: Backup
    console.log('  💾 Stage 7: Backing up to Cloudflare...');
    await backupToCloudflare(urls);
    
    // STAGE 8: Database
    console.log('  🗄️  Stage 8: Updating database...');
    await updateDatabase(lesson.id, archetype, urls);
    
    // STAGE 9: Verify
    console.log('  ✅ Stage 9: Verifying...');
    const verification = await verifyLesson(dayNumber, archetype);
    
    if (verification.passed) {
      console.log(`  ✨ ${archetype} COMPLETE!`);
    } else {
      console.error(`  ❌ ${archetype} FAILED:`, verification.failures);
    }
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`✅ Day ${dayNumber} FACTORY COMPLETE`);
  console.log(`${'═'.repeat(60)}\n`);
}

// CLI execution
const dayNumber = parseInt(process.argv[2]) || 1;
runLessonFactory({
  dayNumber,
  archetypes: ['The Explorer', 'The Rebel', 'The Scientist'],
  skipExisting: true,
  verifyOnly: false
});
```

---

## PART 10: SUCCESS CRITERIA

### Day Complete Checklist

- [ ] **51 Videos** exist and play correctly
  - [ ] 17 Explorer (5 main + 12 responses)
  - [ ] 17 Rebel (5 main + 12 responses)
  - [ ] 17 Scientist (5 main + 12 responses)
  
- [ ] **15 Infographics** exist at 1920×1080
  - [ ] 5 Explorer phase infographics
  - [ ] 5 Rebel phase infographics
  - [ ] 5 Scientist phase infographics
  
- [ ] **36 Option Cards** exist at 512×512
  - [ ] 12 Explorer option cards (3 per phase × 4 phases)
  - [ ] 12 Rebel option cards
  - [ ] 12 Scientist option cards
  
- [ ] **6 Meta Images** exist
  - [ ] 3 Thumbnails (640×360)
  - [ ] 3 Social share (1200×630)
  
- [ ] **Database** fully populated
  - [ ] All `script_video_url` fields
  - [ ] All `infographic_url` fields
  - [ ] All `response_video_url` fields (12 per archetype)
  - [ ] All `option_image_url` fields (12 per archetype)
  
- [ ] **Cloudflare Backup** complete
  - [ ] All 108 assets mirrored to R2
  
- [ ] **Frontend** tested
  - [ ] Main video plays
  - [ ] Options appear with images
  - [ ] Response videos play on selection
  - [ ] Auto-advance works
  - [ ] All archetypes switch correctly
  - [ ] Mobile responsive

---

## FINAL INSTRUCTIONS

1. **DO NOT** create partial lessons
2. **DO NOT** skip response videos (12/17 of total)
3. **DO NOT** skip option card images (NEW requirement)
4. **DO NOT** skip infographics
5. **DO NOT** skip Cloudflare backup

6. **DO** generate ALL 108 assets per day
7. **DO** verify every single URL works
8. **DO** test the full interactive flow
9. **DO** backup everything
10. **DO** update the database completely

**This factory produces COMPLETE lessons. No partial work. No "we'll add that later."**

**START NOW. Generate Day 1 complete. Then scale to 365.**

---

*Unified Lesson Factory v2.0 — December 9, 2025*  
*Total Assets: 39,420 for 365 days*  
*Total Cost: ~$7,238*  
*Quality: Golden Standard — Every frame teaches*









