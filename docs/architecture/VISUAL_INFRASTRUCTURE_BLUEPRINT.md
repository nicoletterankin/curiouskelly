# Visual Infrastructure Blueprint

## Overview

This document defines the foundational infrastructure for lesson visuals - where they appear, how they're managed, how they fail gracefully, and how learners can contribute.

---

## Part 1: Visual Placement Wireframes

### A) Current State (Small Thumbnail)

```
┌─────────────────────────────────────────────────────────────┐
│  Phase Bar: [Hook] [Cliff] [Q1] [Q2] [Q3] [Wisdom] [Outro]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                                                             │
│                    🧑‍🏫 KELLY VIDEO                          │
│                    (Full screen)                            │
│                                                             │
│                                                ┌──────────┐ │
│                                                │ VISUAL   │ │
│                                                │ 180x180  │ │
│                                                └──────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Script text / Subtitles                                    │
│  [Choice A]  [Choice B]                                     │
└─────────────────────────────────────────────────────────────┘
```

**Pros:** Non-intrusive, always visible
**Cons:** Easy to ignore, not immersive

---

### B) Wallpaper Mode (50% Screen)

```
┌─────────────────────────────────────────────────────────────┐
│  Phase Bar                                                  │
├──────────────────────────┬──────────────────────────────────┤
│                          │                                  │
│    🧑‍🏫 KELLY VIDEO        │      VISUAL WALLPAPER           │
│    (50% width)           │      (50% width)                │
│                          │                                  │
│                          │      Topic-relevant image        │
│                          │      with subtle animation       │
│                          │                                  │
├──────────────────────────┴──────────────────────────────────┤
│  Script text / Subtitles                                    │
└─────────────────────────────────────────────────────────────┘
```

**Best for:** Q1, Q2, Q3 phases (educational content)
**Pros:** Immersive, educational context visible
**Cons:** Reduces Kelly's presence

---

### C) Wallpaper Mode (100% Behind Kelly)

```
┌─────────────────────────────────────────────────────────────┐
│  Phase Bar                                                  │
├─────────────────────────────────────────────────────────────┤
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░  VISUAL WALLPAPER  ░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░  (100% - dimmed)   ░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░  🧑‍🏫 KELLY VIDEO  ░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░  (centered, 60% size) ░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────────────────────────────────────────────────┤
│  Script text                                                │
└─────────────────────────────────────────────────────────────┘
```

**Best for:** Hook (set the scene), Wisdom (contemplative)
**Pros:** Cinematic, immersive, sets emotional tone
**Cons:** Visual competes with Kelly if not dimmed properly

---

### D) Choice Visual Mode (Cliff Phase)

```
┌─────────────────────────────────────────────────────────────┐
│  Phase Bar                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    🧑‍🏫 KELLY VIDEO                          │
│                    "Which path calls to you?"               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │                     │    │                     │        │
│  │   VISUAL OPTION A   │    │   VISUAL OPTION B   │        │
│  │                     │    │                     │        │
│  │   "Path of..."      │    │   "Path of..."      │        │
│  └─────────────────────┘    └─────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Best for:** Cliff phase (choice point)
**Pros:** Visual choices feel tangible, engaging
**Cons:** Requires 2 visuals per cliff

---

### E) Text Overlay Mode

```
┌─────────────────────────────────────────────────────────────┐
│  Phase Bar                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │                 VISUAL (dimmed 30%)                 │   │
│  │                                                     │   │
│  │  ┌───────────────────────────────────────────────┐ │   │
│  │  │                                               │ │   │
│  │  │   "The surprising fact is that water         │ │   │
│  │  │    molecules you drink today passed          │ │   │
│  │  │    through dinosaurs 65 million years ago."  │ │   │
│  │  │                                               │ │   │
│  │  └───────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│                    🧑‍🏫 Kelly (small, corner)                │
└─────────────────────────────────────────────────────────────┘
```

**Best for:** Q3 (wow moment), key facts that need emphasis
**Pros:** Maximum visual impact for important content
**Cons:** Kelly minimized, requires high-quality visual

---

### F) Infographic Expansion Mode

```
┌─────────────────────────────────────────────────────────────┐
│  [X] Close                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │                                                     │   │
│  │              FULL SCREEN VISUAL                     │   │
│  │              (tap thumbnail to expand)              │   │
│  │                                                     │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Day 2: The Three Lives of Water                           │
│  Phase: Q1 - The Water Cycle                               │
│                                                             │
│  [🎨 More Styles]  [🔄 Regenerate]  [❤️ Save]              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Best for:** User-initiated deep dive, complex infographics
**Pros:** Full detail visible, user has control
**Cons:** Interrupts lesson flow

---

## Part 2: Phase-to-Placement Mapping

| Phase | Primary Placement | Secondary | Visual Purpose |
|-------|-------------------|-----------|----------------|
| **Hook** | 100% Wallpaper (dimmed) | Thumbnail | Set scene, spark curiosity |
| **Cliff** | Choice Cards (A/B) | 50% side | Make choice tangible |
| **Q1** | 50% Side-by-side | Thumbnail | Illustrate concept |
| **Q2** | 50% Side-by-side | Thumbnail | Deepen understanding |
| **Q3** | Text Overlay | 100% Wallpaper | Maximum wow impact |
| **Wisdom** | 100% Wallpaper (warm) | Thumbnail | Contemplative mood |
| **Outro** | Celebration Animation | Thumbnail | Achievement feeling |

---

## Part 3: Visual Types Required Per Lesson

| Visual Type | Count | Description |
|-------------|-------|-------------|
| **Scene** | 7 | One per phase (hook, cliff, q1, q2, q3, wisdom, outro) |
| **Choice A** | 1 | Cliff phase option A visual |
| **Choice B** | 1 | Cliff phase option B visual |
| **Infographic** | 1-3 | Optional detailed diagrams |
| **Celebration** | 1 | Outro achievement visual |

**Total per lesson:** 9-12 visuals for full coverage

---

## Part 4: Resilience Architecture

### Visual State Machine

```
┌─────────────┐     Generate      ┌─────────────┐
│   MISSING   │ ───────────────▶ │   PENDING   │
└─────────────┘                   └─────────────┘
      ▲                                  │
      │                                  │ Review
      │ Regenerate                       ▼
      │                           ┌─────────────┐
┌─────────────┐    Flag Issue     │   ACTIVE    │
│  REJECTED   │ ◀──────────────── └─────────────┘
└─────────────┘                          │
      │                                  │ Learner flags
      │ Regenerate                       ▼
      │                           ┌─────────────┐
      └─────────────────────────▶ │  FLAGGED    │
                                  └─────────────┘
```

### Fallback Hierarchy

```javascript
// When displaying a visual:
1. Check for approved visual → Use it
2. Check for pending visual → Use with "preview" badge
3. Check for topic-generic visual → Use with topic name
4. Check for phase-generic visual → Use phase placeholder
5. No visual → Hide slot entirely, Kelly adapts script
```

### Database Schema Updates

```sql
-- Extend visual_commons for resilience
ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  placement_mode TEXT DEFAULT 'thumbnail'; -- thumbnail, wallpaper, choice, overlay

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  fallback_for TEXT; -- Links to primary visual this is fallback for

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  learner_flags INTEGER DEFAULT 0; -- Count of learner issue flags

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  regeneration_priority INTEGER DEFAULT 0; -- Higher = regenerate sooner

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  last_shown_at TIMESTAMPTZ; -- Track usage

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  contributed_by_user_id UUID; -- BYOK contributor tracking
```

---

## Part 5: BYOK (Bring Your Own Key) System

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LEARNER SEES VISUAL                                      │
│    - Current visual shown during lesson                     │
│    - [🎨 Personalize] button visible                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. BYOK MODAL                                               │
│    - Enter your Google AI API key                           │
│    - Choose style preference                                │
│    - Optional: Add personal context                         │
│    - [Generate My Version]                                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. GENERATION                                               │
│    - Uses learner's API key                                 │
│    - Prompt includes lesson context + personal preferences  │
│    - Shows generation progress                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. CONTRIBUTION CHOICE                                      │
│    - [Keep Private] - Only you see this                     │
│    - [Share to Commons] - Help future learners              │
│                                                             │
│    If shared:                                               │
│    - Saved to visual_commons with contributor credit        │
│    - Available for other learners to discover               │
│    - Contributor gets "helped X learners" badge             │
└─────────────────────────────────────────────────────────────┘
```

### Personal Visual Storage

```sql
-- User's personal visuals (not shared)
CREATE TABLE IF NOT EXISTS user_personal_visuals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  day_number INTEGER NOT NULL,
  phase TEXT NOT NULL,
  public_url TEXT NOT NULL,
  prompt_used TEXT,
  style_preference TEXT,
  personal_context TEXT, -- "I'm a nurse, make it medical"
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookup
CREATE INDEX idx_personal_visuals_user_day 
ON user_personal_visuals(user_id, day_number);
```

### Visual Preference System

```javascript
// User can set preferences that affect ALL their visuals
const visualPreferences = {
  style: 'photorealistic', // or 'illustrated', 'artistic', etc.
  complexity: 'detailed', // or 'simple', 'moderate'
  colorPalette: 'warm', // or 'cool', 'vibrant', 'muted'
  personalContext: 'I work in healthcare', // Applied to prompts
  showByokOption: true, // Show generate button
  preferPersonal: true // Use personal visuals over commons
};
```

---

## Part 6: Regeneration Procedures

### Automatic Regeneration Triggers

| Trigger | Priority | Action |
|---------|----------|--------|
| 3+ learner flags | High | Queue for immediate regeneration |
| Low engagement (shown but never expanded) | Medium | Review and possibly regenerate |
| Style doesn't match new guidelines | Low | Batch regenerate in next cycle |
| Missing phase visual | Critical | Generate on next API key availability |

### Manual Regeneration Flow

```
1. Admin flags visual for regeneration
2. Visual marked as "regenerating" (still shows but with badge)
3. Next generation cycle picks it up
4. New visual generated with improved prompt
5. Old visual archived (not deleted)
6. New visual becomes active
7. A/B test engagement if both available
```

### Regeneration Queue

```sql
CREATE TABLE IF NOT EXISTS visual_regeneration_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  visual_id UUID REFERENCES visual_commons(id),
  reason TEXT NOT NULL, -- 'learner_flags', 'admin_flag', 'style_update', 'missing'
  priority INTEGER DEFAULT 0,
  prompt_override TEXT, -- Optional improved prompt
  created_at TIMESTAMPTZ DEFAULT NOW(),
  processing_started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  new_visual_id UUID -- Result
);
```

---

## Part 7: Implementation Phases

### Phase 1: Foundation (Current Sprint)
- [x] Basic visual display (thumbnail)
- [x] Visual Commons database
- [x] BYOK modal (basic)
- [ ] Phase-to-placement mapping
- [ ] Fallback hierarchy

### Phase 2: Enhanced Placement
- [ ] Wallpaper mode (50% and 100%)
- [ ] Choice card visuals for cliff
- [ ] Text overlay mode
- [ ] Mobile-optimized layouts

### Phase 3: Resilience
- [ ] Visual state machine
- [ ] Learner flagging system
- [ ] Regeneration queue
- [ ] Fallback visuals

### Phase 4: BYOK Enhancement
- [ ] Personal visual storage
- [ ] Contribution to commons
- [ ] Style preferences
- [ ] Contributor badges

### Phase 5: Analytics & Optimization
- [ ] Visual engagement tracking
- [ ] A/B testing infrastructure
- [ ] Auto-regeneration based on engagement
- [ ] Quality scoring

---

## Part 8: API Endpoints Required

```typescript
// Visual display
GET  /api/visual/phase?day=1&phase=hook
     → Returns best visual(s) for phase

// Visual management
POST /api/visual/flag
     → Learner flags issue with visual

POST /api/visual/regenerate
     → Admin queues regeneration

// BYOK
POST /api/visual/generate-personal
     → Generate using user's API key

POST /api/visual/contribute
     → Share personal visual to commons

// User preferences
GET  /api/visual/preferences
POST /api/visual/preferences
     → Get/set user's visual preferences
```

---

## Part 9: Kelly Script Integration

Kelly's speech adapts based on visual state:

```javascript
const kellyVisualScripts = {
  // Visual available
  hasVisual: {
    hook: "Take a look at what's behind me...",
    cliff: "See these two paths? Each one leads somewhere different.",
    q1: "Notice this visual - it shows exactly what I mean.",
    // ...
  },
  
  // No visual available
  noVisual: {
    hook: "Picture this in your mind...",
    cliff: "Imagine two different directions you could go.",
    q1: "Let me paint a picture with words...",
    // ...
  },
  
  // BYOK prompt
  byokPrompt: {
    hook: "Want to see this your way? You can create your own visual.",
    // Only shown when user has enabled BYOK
  }
};
```

---

## Summary

This infrastructure provides:

1. **Multiple placement options** for different phase needs
2. **Resilience** through state machine and fallbacks
3. **User contribution** via BYOK and commons sharing
4. **Maintainability** through regeneration procedures
5. **Personalization** through preferences and custom generation

All visuals link back to `visual_commons` table with proper tracking, enabling analytics, quality control, and continuous improvement.
