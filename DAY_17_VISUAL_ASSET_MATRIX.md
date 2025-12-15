# DAY 17 - LAUNCH DAY VISUAL ASSET MATRIX

**Date:** December 17, 2025  
**Topic:** Why Bodies Need to Move  
**Universal Truth:** Human bodies are designed for constant motion, not sitting in chairs.

---

## CURRENT STATUS ✅

### Content Atoms: 84/84 COMPLETE
| Archetype | Hook | Cliff | Fact1 | Fact2 | Fact3 | Wisdom | Outro |
|-----------|------|-------|-------|-------|-------|--------|-------|
| The Architect | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Diplomat | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Empath | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Explorer | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The MacGyver | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Mystic | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Provider | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Rebel | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Scientist | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Storyteller | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Strategist | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| The Survivor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Motion Clips: 335/420 (from generic library)
- All 7 phases covered
- 12 personas × 4 age buckets (teen, adult, elder, super_elder)
- Clips are DAY-AGNOSTIC (same clips serve all 365 lessons)

### Images: ✅
- Thumbnail: https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/thumbnails/017-why-bodies-need-to-move.png
- Hero Image: Same as thumbnail

---

## VISUAL ASSET REQUIREMENTS

### 1. INFOGRAPHICS (7 per lesson) ❌ NOT YET GENERATED

| Phase | Type | Description | Status |
|-------|------|-------------|--------|
| Hook | `question_visual` | Intrigue visual, question/mystery motif | ❌ Missing |
| Cliff | `choice_paths` | Two branching paths, fork visual | ❌ Missing |
| Fact1 | `concept_breakdown` | 600 muscles working together | ❌ Missing |
| Fact2 | `process_flow` | How movement strengthens body | ❌ Missing |
| Fact3 | `cross_section` | Sitting vs moving effects on body | ❌ Missing |
| Wisdom | `wisdom_card` | Movement wisdom quote | ❌ Missing |
| Outro | `summary_visual` | Recap of 3 facts + wisdom | ❌ Missing |

**Key Insight:** Infographics are LESSON-SPECIFIC, not PERSONA-SPECIFIC.
- All 12 archetypes see the SAME infographic
- What changes is HOW KELLY EXPLAINS IT (via different atom text)

### 2. THUMBNAIL (1 per lesson) ✅ COMPLETE
- Size: 16:9, ~800x450px
- Shows: Kelly walking with movement/vitality theme
- URL: Already uploaded to Supabase

### 3. HERO IMAGE (1 per lesson) ✅ COMPLETE
- Size: 21:9, ~1400x600px
- Currently using thumbnail (can upgrade later)

---

## WHAT VARIES BY DIMENSION

### By Archetype (12 types)
| Asset | Changes? | Notes |
|-------|----------|-------|
| Lesson atom TEXT | ✅ YES | Different explanation style |
| Kelly motion clip | ❌ NO | Same generic clip from library |
| Infographic | ❌ NO | Same visual for all archetypes |

### By Age Bucket (5 types)
| Asset | Changes? | Notes |
|-------|----------|-------|
| Lesson atom TEXT | ✅ YES | Age-appropriate language |
| Kelly motion clip | ✅ YES | Age-appropriate avatar |
| Infographic | ❌ NO | Same visual for all ages |

### By Phase (7 phases)
| Asset | Changes? | Notes |
|-------|----------|-------|
| Lesson atom TEXT | ✅ YES | Different content per phase |
| Kelly motion clip | ✅ YES | Different gestures/energy |
| Infographic | ✅ YES | Different visual per phase |

---

## INFOGRAPHIC GENERATION SPECS

### For Gemini App Input

**Lesson Data:**
```json
{
  "day": 17,
  "topic": "Why Bodies Need to Move",
  "universal_truth": "Human bodies are designed for constant motion, not sitting in chairs.",
  "fun_facts": [
    "The average person walks five times around the world in their lifetime!",
    "The human body has over 600 muscles responsible for movement.",
    "Dance has been used for communication for thousands of years.",
    "Newton's laws of motion explain the physics behind sports.",
    "Astronauts exercise in space to counteract weightlessness."
  ],
  "phases": {
    "hook": "Bodies are made for movement, not sitting",
    "cliff": "Choose: explore through data OR experimentation",
    "fact1": "600 muscles working in harmony",
    "fact2": "Movement prevents disease, boosts metabolism",
    "fact3": "Sitting weakens muscles and bones",
    "wisdom": "Bodies are engines needing movement",
    "outro": "Movement investment = health investment"
  }
}
```

### Expected Output per Phase

**Hook (question_visual):**
- Creates intrigue about movement
- Question mark or mystery motif
- Shows silhouette in motion vs static

**Cliff (choice_paths):**
- Two clear paths: Data vs Experimentation
- Fork/branch visual metaphor
- Invites learner decision

**Fact1 (concept_breakdown):**
- 600 muscles diagram
- Body silhouette with muscle groups
- "Dance together" metaphor visual

**Fact2 (process_flow):**
- Sequential: Move → Strong Muscles → Prevent Disease → Boost Metabolism
- Timeline or flowchart style
- Cause and effect arrows

**Fact3 (cross_section):**
- Side-by-side: Active body vs Sedentary body
- Shows muscle/bone degradation
- Before/after comparison

**Wisdom (wisdom_card):**
- Quote: "Bodies are engines needing movement"
- Engine/machine metaphor visual
- Gold accent color (#f59e0b)

**Outro (summary_visual):**
- Grid of 3 facts + wisdom
- Clean recap layout
- Clear takeaway message

---

## BRAND REQUIREMENTS

### Colors
- Primary: Kelly Blue (#2563eb)
- Background: Dark (#09090b - #18181b)
- Text: White (#fafafa) or Gray (#a1a1aa)
- Wisdom Accent: Gold (#f59e0b)

### Typography
- Headlines: Clear, bold, readable
- Body: Professional, not decorative
- NO all-caps abuse, NO Comic Sans

### Style
- Clean, modern, educational
- Icons > emojis
- Diagrams > illustrations
- Professional quality

### Forbidden
- ❌ Emoji as main elements
- ❌ Cartoon mascots
- ❌ Cluttered layouts
- ❌ Gradient abuse
- ❌ Random colors

---

## ASSET COUNTS SUMMARY

### For Day 17 Only
| Asset Type | Count | Status |
|------------|-------|--------|
| Lesson Atoms | 84 | ✅ Complete |
| Infographics | 7 | ❌ Not generated |
| Thumbnail | 1 | ✅ Complete |
| Hero Image | 1 | ✅ Complete |
| Motion Clips | 0 (use library) | ✅ Available |

### For All 365 Days (scaling)
| Asset Type | Total Needed | Notes |
|------------|--------------|-------|
| Lesson Atoms | 30,660 | 84 per day × 365 |
| Infographics | 2,555 | 7 per day × 365 |
| Thumbnails | 365 | 1 per day |
| Hero Images | 365 | 1 per day (optional) |
| Motion Clips | 420 | Generic, serves all days |

---

## NEXT STEPS

1. **Generate 7 Infographics for Day 17**
   - Use Gemini App with lesson data above
   - Output: 7 JSON briefs or SVGs

2. **Upload to Supabase Storage**
   - Path: `lesson-visuals/day_017/{phase}.svg`

3. **Wire visual_url to lesson_atoms**
   - Update all 84 atoms with same 7 URLs (per phase)

4. **Test in Browser**
   - URL: https://www.curiouskelly.com/learn.html?day=17
   - Verify infographics appear at each phase

---

## TEST URL

**https://www.curiouskelly.com/learn.html?day=17**

Test all 7 phases with different archetypes to verify:
- Same infographic appears for all archetypes
- Text changes based on archetype selection
- Motion clips play correctly
