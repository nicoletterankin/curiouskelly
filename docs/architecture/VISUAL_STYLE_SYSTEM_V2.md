# Visual Style System V2 - Curious Kelly

## Problem Analysis from V1

### Issues Identified:
1. **Inconsistent prompts** - verbose vs terse, conflicting instructions
2. **Style confusion** - "Ultra photorealistic diagram" makes no sense
3. **Text handling chaos** - some include text, some don't, AI can't render text reliably
4. **Vague educational content** - "show the concept" without specifics
5. **No unified visual language** - random aesthetics per image
6. **No composition guardrails** - no safe zones for UI overlay
7. **No quality filtering** - bad images slip through

---

## V2 Solution: The Curious Kelly Visual System

### Core Principles

1. **ONE consistent style** - Illustrated educational (not photorealistic)
2. **NEVER ask AI to render text** - Text is a UI overlay, not part of the image
3. **SPECIFIC visual subjects** - Always describe exactly what to draw
4. **CONSISTENT composition** - Safe zones for UI, predictable framing
5. **QUALITY GATES** - Size, aspect ratio, and content validation

---

## The Curious Kelly Visual Style

### Style Definition: "Illustrated Educational"

```
STYLE: Curious Kelly Educational Illustration
- Modern flat illustration with subtle depth and soft shadows
- Warm, friendly color palette (teals, corals, warm yellows, soft purples)
- Clean lines, approachable characters when shown
- Slightly stylized (not cartoonish, not photorealistic)
- Think: Headspace app, Duolingo illustrations, Khan Academy visuals
- Consistent lighting: soft, even, welcoming
```

### Color Palette (Hex):
- Primary Teal: #4ECDC4
- Coral Accent: #FF6B6B
- Warm Yellow: #FFE66D
- Soft Purple: #A78BFA
- Deep Navy: #1A1A2E
- Cream Background: #FFF9F0
- Forest Green: #27AE60

### Composition Rules:

```
COMPOSITION RULES:
- 16:9 aspect ratio ONLY (1920x1080 or 1280x720)
- Main subject takes 50-70% of frame, centered or rule-of-thirds
- LEFT 70% contains main visual content
- RIGHT 30% is simpler (for potential UI overlay)
- Clean, uncluttered backgrounds
- Generous whitespace/breathing room
- No busy patterns or textures that compete with subject
```

---

## Phase-Specific Visual Templates

### HOOK Phase
**Purpose**: Spark curiosity, create "wait, what?" moment
**Visual Type**: Surprising juxtaposition or unexpected angle

```
PROMPT TEMPLATE - HOOK:
Create an illustrated educational scene showing [SPECIFIC SURPRISING VISUAL].

SUBJECT: [Exact description of what to show - the unexpected angle on the topic]

STYLE:
- Modern flat illustration with soft shadows
- Warm friendly colors (teals, corals, yellows)
- Slightly mysterious or intriguing mood
- Clean composition, uncluttered

COMPOSITION:
- 16:9 aspect ratio
- Main subject centered, 60% of frame
- Right 30% kept simpler
- Soft gradient or solid background

DO NOT include any text, labels, numbers, or writing.
```

### CLIFF Phase
**Purpose**: Show the tension/choice, create anticipation
**Visual Type**: Two paths, contrast, or decision moment

```
PROMPT TEMPLATE - CLIFF:
Create an illustrated educational scene showing [VISUAL CONTRAST OR CHOICE].

SUBJECT: [Two contrasting elements or a clear decision point related to the topic]

STYLE:
- Modern flat illustration with soft shadows
- Warm friendly colors (teals, corals, yellows)
- Visual tension between two elements
- Clean split composition or comparison

COMPOSITION:
- 16:9 aspect ratio
- Clear visual division or comparison
- Right 30% kept simpler
- Balanced but with clear focal hierarchy

DO NOT include any text, labels, numbers, or writing.
```

### FACT/QUESTION Phases (Q1, Q2, Q3)
**Purpose**: Illustrate the core concept clearly
**Visual Type**: Explanatory, showing the mechanism or relationship

```
PROMPT TEMPLATE - FACT:
Create an illustrated educational scene showing [SPECIFIC CONCEPT VISUALIZATION].

SUBJECT: [Concrete visual representation of the fact - what physically demonstrates this?]

STYLE:
- Modern flat illustration with soft shadows
- Warm friendly colors (teals, corals, yellows)
- Clear, educational, easy to understand
- Approachable and friendly

COMPOSITION:
- 16:9 aspect ratio
- Main concept takes center stage
- Simple supporting elements
- Right 30% kept simpler for UI

DO NOT include any text, labels, numbers, or writing.
```

### WISDOM Phase
**Purpose**: Inspire, create lasting impression
**Visual Type**: Aspirational, peaceful, universal

```
PROMPT TEMPLATE - WISDOM:
Create an illustrated educational scene showing [INSPIRATIONAL VISUAL].

SUBJECT: [Universal, timeless visual that embodies the lesson's wisdom]

STYLE:
- Modern flat illustration with soft shadows
- Warm, golden-hour color palette
- Peaceful, inspiring, uplifting mood
- Timeless and universal appeal

COMPOSITION:
- 16:9 aspect ratio
- Open, breathing composition
- Sense of possibility and growth
- Right 30% kept simpler

DO NOT include any text, labels, numbers, or writing.
```

### OUTRO Phase
**Purpose**: Celebrate completion
**Visual Type**: Achievement, forward momentum

```
PROMPT TEMPLATE - OUTRO:
Create an illustrated educational scene showing [CELEBRATORY VISUAL].

SUBJECT: [Visual sense of accomplishment and "what's next" energy]

STYLE:
- Modern flat illustration with soft shadows
- Bright, energetic colors
- Celebratory, forward-looking mood
- Uplifting and motivating

COMPOSITION:
- 16:9 aspect ratio
- Dynamic but not chaotic
- Sense of completion with forward momentum
- Right 30% kept simpler

DO NOT include any text, labels, numbers, or writing.
```

---

## Quality Validation Pipeline

### Pre-Generation Checks:
1. ✅ Prompt follows template exactly
2. ✅ Subject is CONCRETE (not abstract like "show the concept")
3. ✅ No text/label requests in prompt
4. ✅ 16:9 aspect ratio specified

### Post-Generation Checks:
1. ✅ Image dimensions are exactly 1920x1080 or 1536x864 (16:9)
2. ✅ File size > 100KB (not a broken generation)
3. ✅ File size < 5MB (not bloated)
4. ✅ No visible text artifacts (common AI failure)
5. ✅ Dominant colors match palette (no neon, no harsh)
6. ✅ Subject clearly visible (not muddy/confused)

### Rejection Criteria (Auto-filter):
- ❌ Wrong aspect ratio
- ❌ Visible text/numbers/letters in image
- ❌ File too small (< 100KB = failed generation)
- ❌ Predominantly dark/muddy colors
- ❌ Multiple competing focal points
- ❌ Photorealistic when should be illustrated

---

## UI Placement Strategy

### Desktop Layout:
```
┌─────────────────────────────────────────────────────────────┐
│  Header Bar                                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     ┌────────────────────────────────────────┐              │
│     │                                        │              │
│     │         PHASE VISUAL (main)            │              │
│     │           Max 60% height               │              │
│     │                                        │              │
│     └────────────────────────────────────────┘              │
│                                                             │
│     ┌────────────────────────────────────────┐              │
│     │  Question/Content Text Overlay         │              │
│     └────────────────────────────────────────┘              │
│                                                             │
│     [Choice A]          [Choice B]          [Choice C]      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Phase Progress Bar                                         │
└─────────────────────────────────────────────────────────────┘
```

### Mobile Layout:
```
┌──────────────────────┐
│  Header              │
├──────────────────────┤
│ ┌──────────────────┐ │
│ │   PHASE VISUAL   │ │
│ │   (40% height)   │ │
│ └──────────────────┘ │
│                      │
│  Question/Content    │
│                      │
│  ┌────────────────┐  │
│  │   Choice A     │  │
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │   Choice B     │  │
│  └────────────────┘  │
│                      │
├──────────────────────┤
│  Phase Progress      │
└──────────────────────┘
```

### Visual Sizing CSS:
```css
.phase-visual {
  width: 100%;
  max-height: 45vh;
  object-fit: cover;
  object-position: center left; /* Keep important content visible */
  border-radius: 12px;
  margin-bottom: 16px;
}

@media (max-width: 768px) {
  .phase-visual {
    max-height: 35vh;
  }
}
```

---

## Subject Extraction Rules

### How to Generate SPECIFIC Subjects from Lesson Content

For each lesson, we extract:
1. **The surprising element** → Hook visual
2. **The contrast/choice** → Cliff visual  
3. **The mechanism/how-it-works** → Fact visuals
4. **The universal truth** → Wisdom visual
5. **The achievement** → Outro visual

### Example: Day 1 "Starting Fresh"

| Phase | Content | Visual Subject |
|-------|---------|----------------|
| Hook | "Every January 1st, millions try to change" | A diverse group of people at a starting line, all facing the same direction with determined expressions, calendar pages floating in the wind |
| Cliff | "Is it the date that matters, or something else?" | Split scene: left shows a calendar with January 1 circled, right shows a lightbulb moment happening on an ordinary Tuesday |
| Q1 | "62% more likely after fresh start dates" | A path that's steep and rocky vs a path that starts fresh with open gates and smooth beginning |
| Q2 | "Wharton School discovered the fresh start effect in 2014" | A researcher having an "aha" moment while looking at data on a whiteboard (illustrated, no text on whiteboard) |
| Q3 | "Any meaningful date works" | Multiple doorways labeled with visual symbols (birthday cake, Monday calendar icon, etc) all leading to the same bright destination |
| Wisdom | "The power to start fresh is always available" | A person standing at dawn, looking at an open horizon, peaceful and full of possibility |
| Outro | "You now understand fresh starts" | A person taking their first confident step forward, energy lines suggesting momentum |

---

## Implementation Checklist

- [ ] Create `lib/visual-prompts-v2.ts` with new templates
- [ ] Create `scripts/validate-visual.ts` for quality checks
- [ ] Create subject extraction function per lesson
- [ ] Update generator to use V2 prompts
- [ ] Add post-generation validation
- [ ] Update UI components for strategic placement
- [ ] Regenerate all visuals with new system
- [ ] Manual review pass for educational accuracy

---

## Model Selection

### Recommended: Imagen 4 (Standard or Ultra)
- Best at illustrated styles
- Consistent quality
- Good at following composition rules

### Fallback: Gemini Flash Image
- Faster
- Lower quality but acceptable
- Good for drafts/testing

### DO NOT USE for final:
- Any model for text rendering (they all fail)
- Photorealistic prompts with illustrated expectations
