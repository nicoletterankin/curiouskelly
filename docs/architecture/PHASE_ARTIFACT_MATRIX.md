# Phase Artifact Matrix

## Overview

Every phase in a Curious Kelly lesson can have multiple artifacts. This document maps ALL possible artifacts per phase and defines how they're stored, displayed, and generated.

---

## The Complete Phase Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                           PHASE                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CORE CONTENT                                                        │
│  ├── script (text Kelly speaks)                                      │
│  ├── title (for Q1/Q2/Q3)                                            │
│  ├── duration (seconds)                                              │
│  └── prompt (question for learner)                                   │
│                                                                      │
│  CHOICE SYSTEM (all phases can have this)                            │
│  ├── option_a                                                        │
│  │   ├── text (button label)                                         │
│  │   ├── quality (good/best/neutral)                                 │
│  │   ├── response (Kelly's feedback)                                 │
│  │   ├── visual_url (image for this choice)                          │
│  │   └── audio_url (audio for feedback)                              │
│  │                                                                   │
│  └── option_b                                                        │
│      ├── text                                                        │
│      ├── quality                                                     │
│      ├── response                                                    │
│      ├── visual_url                                                  │
│      └── audio_url                                                   │
│                                                                      │
│  MEDIA ASSETS                                                        │
│  ├── kelly_video_url (HeyGen avatar video)                          │
│  ├── kelly_audio_url (ElevenLabs VO)                                 │
│  ├── scene_visual_url (wallpaper/background)                         │
│  ├── infographic_url (detailed diagram)                              │
│  └── celebration_url (outro only)                                    │
│                                                                      │
│  COMMONS LINKS                                                       │
│  ├── visual_commons_id (link to visual_commons table)                │
│  ├── audio_commons_id (future: shared audio)                         │
│  └── video_commons_id (future: shared video)                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Phase-by-Phase Breakdown

### HOOK 🎬

**Purpose:** Spark curiosity, set the scene

| Artifact | Required? | Display | Commons Link |
|----------|-----------|---------|--------------|
| script | ✅ | Subtitles / Kelly speaks | - |
| kelly_video | ✅ | Main stage | video_commons |
| kelly_audio | ✅ | Audio track | audio_commons |
| scene_visual | ✅ | 100% wallpaper behind Kelly | visual_commons |
| prompt | ✅ | Question for learner | - |
| option_a | ✅ | Choice button A | visual_commons |
| option_b | ✅ | Choice button B | visual_commons |

**Visual Behavior:**
- Scene visual fades in as wallpaper (dimmed 40%)
- Kelly appears centered over wallpaper
- Sets emotional tone for lesson
- Choice cards appear with A/B visuals

---

### CLIFF 🔀

**Purpose:** Present choice, create engagement

| Artifact | Required? | Display | Commons Link |
|----------|-----------|---------|--------------|
| script | ✅ | Kelly introduces choice | - |
| kelly_video | ✅ | Upper portion | video_commons |
| kelly_audio | ✅ | Audio track | audio_commons |
| prompt | ✅ | Question text above choices | - |
| option_a.text | ✅ | Choice button A | - |
| option_a.visual | 🎯 | Image on button A | visual_commons |
| option_a.response | ✅ | Kelly says after choice | - |
| option_a.audio | 🎯 | Audio for response | audio_commons |
| option_b.text | ✅ | Choice button B | - |
| option_b.visual | 🎯 | Image on button B | visual_commons |
| option_b.response | ✅ | Kelly says after choice | - |
| option_b.audio | 🎯 | Audio for response | audio_commons |

**Visual Behavior:**
```
┌─────────────────────────────────────────┐
│         Kelly asks question             │
├───────────────────┬─────────────────────┤
│                   │                     │
│   OPTION A        │      OPTION B       │
│   [Visual]        │      [Visual]       │
│   "Path of..."    │      "Path of..."   │
│                   │                     │
└───────────────────┴─────────────────────┘
```

---

### Q1/Q2/Q3 💡 (Fact Phases)

**Purpose:** Teach concepts, build understanding

| Artifact | Required? | Display | Commons Link |
|----------|-----------|---------|--------------|
| title | ✅ | Phase header | - |
| script | ✅ | Kelly teaches | - |
| kelly_video | ✅ | 50% or full | video_commons |
| kelly_audio | ✅ | Audio track | audio_commons |
| scene_visual | ✅ | 50% side or wallpaper | visual_commons |
| infographic | 🎯 | Expandable diagram | visual_commons |
| prompt | ✅ | Quiz question | - |
| option_a | ✅ | Quiz answer A + visual | visual_commons |
| option_b | ✅ | Quiz answer B + visual | visual_commons |

**All Q phases have choices** - typically testing comprehension or addressing misconceptions

**Visual Behavior (Q phases):**
```
┌────────────────────┬────────────────────┐
│                    │                    │
│   Kelly Video      │   Scene Visual     │
│   50% width        │   50% width        │
│                    │   (educational)    │
│                    │                    │
├────────────────────┴────────────────────┤
│   Script text / Title                   │
│   [Optional: Quiz A] [Quiz B]           │
└─────────────────────────────────────────┘
```

**Q3 Special (Wow Moment):**
```
┌─────────────────────────────────────────┐
│                                         │
│   VISUAL (full, dimmed 30%)             │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │  "The surprising truth is..."   │   │
│   │        KEY INSIGHT TEXT         │   │
│   └─────────────────────────────────┘   │
│                                         │
│   Kelly (small, corner)                 │
└─────────────────────────────────────────┘
```

---

### WISDOM ✨

**Purpose:** Inspire, connect to life

| Artifact | Required? | Display | Commons Link |
|----------|-----------|---------|--------------|
| script | ✅ | Kelly reflects | - |
| kelly_video | ✅ | Centered | video_commons |
| kelly_audio | ✅ | Audio track | audio_commons |
| scene_visual | ✅ | 100% wallpaper (warm) | visual_commons |
| prompt | ✅ | Reflection question | - |
| option_a | ✅ | Wisdom path A + visual | visual_commons |
| option_b | ✅ | Wisdom path B + visual | visual_commons |

**Visual Behavior:**
- Warm, contemplative wallpaper
- Kelly centered, slightly smaller
- Choice cards for reflection themes

---

### OUTRO 🎉

**Purpose:** Celebrate, tease tomorrow

| Artifact | Required? | Display | Commons Link |
|----------|-----------|---------|--------------|
| script | ✅ | Kelly celebrates | - |
| kelly_video | ✅ | Full celebration | video_commons |
| kelly_audio | ✅ | Audio track | audio_commons |
| scene_visual | ✅ | Celebration visual | visual_commons |
| prompt | ✅ | Takeaway question | - |
| option_a | ✅ | Action A + visual | visual_commons |
| option_b | ✅ | Action B + visual | visual_commons |
| tomorrow_teaser | ✅ | Next lesson preview | - |

---

## Database Schema

### lesson_phase_artifacts

```sql
CREATE TABLE lesson_phase_artifacts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Identity
  day_number INTEGER NOT NULL,
  phase TEXT NOT NULL, -- hook, cliff, q1, q2, q3, wisdom, outro
  
  -- Core Content
  title TEXT,
  script TEXT NOT NULL,
  duration INTEGER DEFAULT 15,
  prompt TEXT, -- Question for learner
  
  -- Option A
  option_a_text TEXT,
  option_a_quality TEXT, -- good, best, neutral
  option_a_response TEXT,
  option_a_visual_id UUID REFERENCES visual_commons(id),
  option_a_audio_url TEXT,
  
  -- Option B
  option_b_text TEXT,
  option_b_quality TEXT,
  option_b_response TEXT,
  option_b_visual_id UUID REFERENCES visual_commons(id),
  option_b_audio_url TEXT,
  
  -- Media Assets
  kelly_video_url TEXT,
  kelly_audio_url TEXT,
  scene_visual_id UUID REFERENCES visual_commons(id),
  infographic_id UUID REFERENCES visual_commons(id),
  
  -- Display Configuration
  visual_placement TEXT DEFAULT 'thumbnail', -- thumbnail, wallpaper-50, wallpaper-100, choice-cards, overlay
  kelly_position TEXT DEFAULT 'center', -- center, left, corner
  
  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_phase_artifacts_day ON lesson_phase_artifacts(day_number);
CREATE UNIQUE INDEX idx_phase_artifacts_day_phase ON lesson_phase_artifacts(day_number, phase);
```

---

## Visual Commons Extended

```sql
-- Extend visual_commons for artifact linking
ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  artifact_type TEXT DEFAULT 'scene'; 
  -- scene, choice_a, choice_b, infographic, celebration

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  placement_hint TEXT DEFAULT 'thumbnail';
  -- thumbnail, wallpaper-50, wallpaper-100, choice-card, overlay, expandable
```

---

## Complete Artifact Count Per Lesson

| Artifact Type | Count | Notes |
|---------------|-------|-------|
| Scripts | 7 | One per phase |
| Kelly Videos | 7 | One per phase (+ option responses) |
| Kelly Audios | 7+ | One per phase + option feedbacks |
| Scene Visuals | 7 | One per phase |
| Choice A Visuals | **7** | **REQUIRED for every phase** |
| Choice B Visuals | **7** | **REQUIRED for every phase** |
| Infographics | 0-3 | Q phases optional |
| Celebration | 1 | Outro |

**EVERY phase has:**
- A prompt (question for learner)
- Option A (text, quality, response, visual)
- Option B (text, quality, response, visual)

**This means 21 choice visuals per lesson** (7 phases × 2 options + 7 scenes)

---

## Generation Priority

### Phase 1: Core (What we have now)
1. ✅ Scene visuals for all 7 phases
2. ⏳ Kelly audio for all phases
3. ⏳ Kelly video for all phases

### Phase 2: Choices
4. Choice A visuals for cliff
5. Choice B visuals for cliff
6. Feedback audio for cliff options

### Phase 3: Enhanced Q Phases
7. Quiz prompts for Q1/Q2/Q3
8. Quiz option visuals
9. Infographics

### Phase 4: Full Coverage
10. Choice options for ALL phases
11. Personal reflection prompts
12. Activity visuals

---

## BYOK Integration Points

Every artifact with a `visual_commons` link can be:
1. **Viewed** - Learner sees existing visual
2. **Personalized** - Learner generates their own
3. **Contributed** - Learner shares to commons

```
┌─────────────────────────────────────────────────────┐
│  Visual Display                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │                                             │   │
│  │           [Current Visual]                  │   │
│  │                                             │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  [🎨 Personalize]  [🔄 Try Another]  [❤️ Save]     │
│                                                     │
│  "3 learners contributed visuals for this phase"   │
│  [See all styles →]                                 │
└─────────────────────────────────────────────────────┘
```

---

## Fallback Chain

For any missing artifact:

```
1. Check lesson_phase_artifacts for specific asset
   ↓ (missing)
2. Check visual_commons for day+phase+type match
   ↓ (missing)
3. Check visual_commons for phase-generic (day=null)
   ↓ (missing)
4. Use placeholder with BYOK prompt
   ↓ (user generates)
5. Save to user_personal_visuals
   ↓ (user shares)
6. Add to visual_commons as contributed
```

---

## Summary

Each lesson phase is a rich container of artifacts, not just a single visual. The infrastructure must:

1. **Store** all artifact types with proper relationships
2. **Display** artifacts according to phase-specific layouts
3. **Fallback** gracefully when artifacts are missing
4. **Enable** learner contribution at every point
5. **Track** usage and quality across the commons

This is the foundation for a truly personalized, community-enhanced learning experience.
