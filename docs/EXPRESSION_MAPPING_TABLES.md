# Expression Mapping Quick Reference Tables

This document provides quick lookup tables for the Curious Kelly expression generation system.

---

## Archetype → Expression Style Matrix

| Archetype | Style | Intensity | Key Expression | Signature Gesture |
|-----------|-------|-----------|----------------|-------------------|
| **The Scientist** | Subtle | 0.50 | eyebrowRaise + slight smile | `chin_touch` |
| **The Explorer** | Animated | 0.85 | wide eyes + big smile | `arms_wide_open` |
| **The Storyteller** | Dramatic | 0.90 | expressive mouth | `hands_flowing` |
| **The Empath** | Gentle | 0.60 | warm smile + soft gaze | `heart_open` |
| **The Mystic** | Contemplative | 0.55 | serene + distant gaze | `hands_prayer_position` |
| **The Provider** | Nurturing | 0.65 | warm smile + encouraging | `open_arms_welcome` |
| **The Diplomat** | Balanced | 0.60 | composed + agreeable | `balance_hands` |
| **The Architect** | Focused | 0.65 | focused gaze | `building_blocks` |
| **The Rebel** | Bold | 0.80 | smirk + confident | `fist_pump` |
| **The Strategist** | Controlled | 0.55 | calculated gaze | `chess_move` |
| **The MacGyver** | Inventive | 0.75 | eureka expression | `light_bulb` |
| **The Survivor** | Grounded | 0.50 | steady + determined | `grounding_stance` |

---

## Tone → Blend Shape Adjustments

| Tone | Smile Δ | Eyebrow Δ | Eyes Δ | Energy | Gestures |
|------|---------|-----------|--------|--------|----------|
| **Enthusiastic** | +25 | +20 | +15 wide | 1.3x | 1.4x freq |
| **Serious** | -20 | +0 | focused | 0.8x | 0.6x freq |
| **Playful** | +30 | +15 | normal | 1.2x | 1.3x freq |
| **Thoughtful** | +5 | +15 | gaze shift | 0.9x | 0.8x freq |
| **Warm** | +20 | +10 | soft | 1.0x | 0.9x freq |
| **Confident** | +15 | +10 | steady | 1.1x | 1.1x freq |

---

## Age → Expression Intensity Matrix

| Age Bucket | Label | Intensity | Movement | Duration | Character |
|------------|-------|-----------|----------|----------|-----------|
| **2-5** | Toddler | 1.50x | 1.40x | 0.70x | Bouncy |
| **6-12** | Child | 1.25x | 1.20x | 0.85x | Curious |
| **13-17** | Teen | 0.90x | 0.85x | 1.00x | Cool |
| **18-35** | Adult | 1.00x | 1.00x | 1.00x | Balanced |
| **36-60** | Mature | 0.85x | 0.90x | 1.15x | Subtle |
| **61-102** | Elder | 0.75x | 0.70x | 1.30x | Wise |

### Age Baseline Shifts

| Age | smile Δ | eyebrow Δ | eyes Δ | Special |
|-----|---------|-----------|--------|---------|
| 2-5 | +30 | +20 | +25 wide | bouncy = true |
| 6-12 | +20 | +15 | +15 wide | curious = true |
| 13-17 | -5 | +5 | — | smirk +15 |
| 18-35 | +0 | +0 | — | — |
| 36-60 | +5 | -5 | — | confident = true |
| 61-102 | +10 | +5 | — | warmGaze +20 |

---

## Language → Cultural Modifiers

| Language | Gesture Intensity | Expression | Style | Special Gestures |
|----------|-------------------|------------|-------|------------------|
| **English** | 1.00x | 1.00x | Clear | thumbs_up, wave |
| **Spanish** | 1.30x | 1.15x | Expressive | embrace, heart_touch |
| **French** | 0.90x | 1.05x | Elegant | subtle_shrug, expressive_mouth |

---

## Phase → Default Expressions

| Phase | Energy | Smile | Eyebrow | Eyes | Primary Gestures |
|-------|--------|-------|---------|------|------------------|
| **Welcome** | 1.10 | 65 | 35 | 25 wide | open_arms, wave |
| **Q1 (Teaching)** | 1.05 | 45 | 50 | 35 wide | point_up, chin_touch |
| **Q2 (Practice)** | 1.15 | 55 | 40 | normal | encourage, balance |
| **Q3 (Synthesis)** | 1.10 | 50 | 45 | focused | connect, light_bulb |
| **Wisdom** | 0.95 | 60 | 30 | warm | heart_open, gentle_nod |

---

## Emotion → Blend Shape Presets

| Emotion | smile | eyebrowRaise | eyesWide | Other |
|---------|-------|--------------|----------|-------|
| **neutral** | 20 | 10 | 0 | — |
| **warm** | 50 | 20 | 10 | warmGaze |
| **excited** | 85 | 70 | 75 | mouthOpen: 25 |
| **curious** | 40 | 60 | 55 | headTilt: 20 |
| **emphatic** | 50 | 60 | 40 | — |
| **amused** | 80 | 45 | 30 | — |
| **awed** | 45 | 85 | 80 | mouthOpen: 40 |
| **encouraging** | 75 | 40 | 20 | nod: 35 |
| **contemplative** | 25 | 40 | 10 | eyesClosed: 30 |
| **dramatic** | 35 | 80 | 60 | — |
| **serene** | 45 | 20 | eyesClosed: 40 | — |
| **proud** | 70 | 35 | 15 | cheekRaise: 50 |
| **skeptical** | 25 | 55 | squint: 40 | — |

---

## Gesture Library by Context

### Thinking/Analytical

| Gesture | Duration | Best For |
|---------|----------|----------|
| `chin_touch` | 1.5-3.0s | Scientist, Strategist |
| `hands_steepled` | 2.0-4.0s | Scientist, Architect |
| `thoughtful_chin` | 1.5-2.5s | Diplomat |
| `gaze_up` | 1.0-2.0s | All |

### Excitement/Energy

| Gesture | Duration | Best For |
|---------|----------|----------|
| `point_up_dramatic` | 0.8-1.5s | Explorer, MacGyver |
| `arms_wide_open` | 1.5-2.5s | Explorer, Storyteller |
| `hands_clasp_excited` | 0.8-1.2s | Explorer |
| `bounce_hop` | 0.5-0.8s | Ages 2-12 only |
| `fist_pump` | 0.8-1.2s | Rebel (avoid for 61+) |

### Warmth/Connection

| Gesture | Duration | Best For |
|---------|----------|----------|
| `heart_open` | 2.0-3.5s | Empath, Provider |
| `gentle_reach` | 1.5-2.5s | Empath |
| `embrace_gesture` | 2.0-3.0s | Provider |
| `soft_nod` | 1.0-1.5s | All warm contexts |

### Presentation/Explanation

| Gesture | Duration | Best For |
|---------|----------|----------|
| `hands_open_presenting` | 2.0-3.0s | All |
| `balance_hands` | 1.5-2.5s | Diplomat |
| `building_blocks` | 2.0-3.5s | Architect |
| `connect_points` | 1.5-2.0s | Architect, Scientist |
| `count_fingers` | 2.0-3.5s | Strategist |

### Dramatic/Storytelling

| Gesture | Duration | Best For |
|---------|----------|----------|
| `hands_flowing` | 2.0-4.0s | Storyteller |
| `theatrical_pause` | 1.5-2.5s | Storyteller |
| `expansive_reveal` | 1.2-2.0s | Storyteller |
| `character_mime` | 2.0-3.5s | Storyteller |

### Mystical/Contemplative

| Gesture | Duration | Best For |
|---------|----------|----------|
| `hands_prayer_position` | 2.0-4.0s | Mystic |
| `slow_raise` | 2.5-4.0s | Mystic |
| `breath_gesture` | 2.0-3.5s | Mystic |
| `palms_up_receiving` | 2.5-4.0s | Mystic |

---

## Gesture Avoidance Rules

### By Age

| Age Bucket | Avoid These Gestures |
|------------|---------------------|
| 2-5 | chin_touch, hands_steepled, thoughtful_chin |
| 6-12 | chess_move, blueprint_trace |
| 13-17 | bounce_hop, clap_hands, embrace_gesture |
| 61-102 | fist_pump, bounce_hop, rock_on, provocative_point |

### By Archetype

| Archetype | Avoid These Gestures |
|-----------|---------------------|
| The Mystic | fist_pump, provocative_point |
| The Survivor | theatrical_pause, dramatic gestures |
| The Scientist | bounce_hop, rock_on |

### By Language

| Language | Avoid These Gestures |
|----------|---------------------|
| French | fist_pump (culturally inappropriate) |

---

## Expression Intensity Formula

```
Final Intensity = Base Intensity 
                  × Tone Multiplier 
                  × Age Intensity Multiplier 
                  × Language Expression Multiplier
                  × Phase Energy Level
```

### Example Calculation

For an "excited" expression with:
- Base: 0.8
- Tone (enthusiastic): 1.3
- Age (6-12): 1.25
- Language (Spanish): 1.15
- Phase (welcome): 1.1

```
Final = 0.8 × 1.3 × 1.25 × 1.15 × 1.1 = 1.64 → capped at 1.0
```

---

## Text Trigger Patterns

| Emotion | Trigger Patterns |
|---------|------------------|
| **Excitement** | `!`, `amazing`, `incredible`, `wow`, `fantastic`, `awesome`, `wonderful` |
| **Curiosity** | `?`, `wonder`, `curious`, `how`, `why`, `what if`, `imagine` |
| **Warmth** | `love`, `heart`, `dear`, `special`, `precious`, `care`, `beautiful` |
| **Emphasis** | `CAPS`, `incredibly`, `absolutely`, `definitely`, `so` |
| **Humor** | `haha`, `😄`, `funny`, `silly`, `joke`, `laugh` |
| **Awe** | `cosmic`, `universe`, `infinity`, `eternal`, `profound`, `vast` |
| **Encouragement** | `you can`, `great job`, `well done`, `excellent`, `brilliant` |
| **Challenge** | `try`, `challenge`, `think about`, `consider`, `test` |

---

## Quick Decision Tree

```
START
  │
  ├─► Get archetype → Load expression profile
  │
  ├─► Get tone → Apply modifier
  │
  ├─► Get age → Apply intensity/movement adjustments
  │
  ├─► Get language → Apply cultural modifiers
  │
  ├─► Get phase → Set baseline expressions
  │
  ├─► Analyze text → Detect emotions/pauses/emphasis
  │
  ├─► Generate expressions at detected timestamps
  │
  ├─► Generate gestures at pause/emphasis points
  │
  ├─► Apply conflict resolution (no overlapping gestures)
  │
  └─► Output: { expressions[], gestures[], blendShapeTimeline[] }
```












