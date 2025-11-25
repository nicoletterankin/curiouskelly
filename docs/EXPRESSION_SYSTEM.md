# AI-Powered Expression Generation System

**Version:** 1.0.0  
**Last Updated:** 2025-11-25  
**Status:** Implementation Complete

---

## Overview

The Curious Kelly Expression Generation System creates facial expressions and gestures synchronized with lesson audio content. It uses AI to analyze text, integrates with ElevenLabs audio metadata, and applies archetype/tone/age/language-specific styling.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPRESSION GENERATION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   LESSON DNA     │    │  ELEVENLABS API  │    │   USER STATE     │       │
│  │  (Text Content)  │    │   (Audio Meta)   │    │ (Age/Archetype)  │       │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘       │
│           │                       │                       │                  │
│           ▼                       ▼                       ▼                  │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    TEXT ANALYZER                                    │     │
│  │  • Emotion detection (excitement, curiosity, warmth, etc.)         │     │
│  │  • Pause indicators (periods, commas, ellipses)                    │     │
│  │  • Emphasis detection (CAPS, !, bold)                              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │               EXPRESSION GENERATOR                                  │     │
│  │  • Apply archetype profile                                          │     │
│  │  • Apply tone modifiers                                             │     │
│  │  • Apply age adjustments                                            │     │
│  │  • Apply language/cultural adaptations                              │     │
│  │  • Generate blend shapes + gestures                                 │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                     OUTPUT                                          │     │
│  │  {                                                                  │     │
│  │    expressions: [...],  // Timestamped emotion keyframes            │     │
│  │    gestures: [...],     // Timestamped gesture commands             │     │
│  │    blendShapeTimeline: [...] // Unity-ready blend shape data        │     │
│  │  }                                                                  │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `app/expression-generator.js` | Main expression generation logic |
| `scripts/precompute-expressions.js` | Batch processing for 365 lessons |
| `docs/EXPRESSION_SYSTEM.md` | This documentation |

---

## The 12 Archetypes

Each archetype has a unique expression style, gesture library, and movement characteristics:

### 1. The Scientist
- **Traits:** Analytical • Abstract
- **Expression Style:** Subtle
- **Gesture Intensity:** 0.5
- **Key Gestures:** chin_touch, glasses_adjust, hands_steepled, finger_point_precise
- **Characteristics:** Frequent gaze shifts, deliberate head nods, squints when thinking

### 2. The Explorer
- **Traits:** Energetic • Abstract
- **Expression Style:** Animated
- **Gesture Intensity:** 0.85
- **Key Gestures:** point_up_dramatic, arms_wide_open, reaching_forward, sweep_gesture
- **Characteristics:** Dynamic gaze, enthusiastic nods, bouncy movement

### 3. The Storyteller
- **Traits:** Expressive • Abstract
- **Expression Style:** Dramatic
- **Gesture Intensity:** 0.9
- **Key Gestures:** hands_flowing, theatrical_pause, character_mime, expansive_reveal
- **Characteristics:** Theatrical gaze shifts, expressive nods, dramatic pauses

### 4. The Empath
- **Traits:** Warm • Abstract
- **Expression Style:** Gentle
- **Gesture Intensity:** 0.6
- **Key Gestures:** heart_open, gentle_reach, hands_together_heart, soft_nod
- **Characteristics:** Soft gaze, supportive nods, slow movement

### 5. The Mystic
- **Traits:** Deep • Abstract
- **Expression Style:** Contemplative
- **Gesture Intensity:** 0.55
- **Key Gestures:** hands_prayer_position, slow_raise, gentle_circle, breath_gesture
- **Characteristics:** Distant gaze, deliberate movement, contemplative pauses

### 6. The Provider
- **Traits:** Warm • Practical
- **Expression Style:** Nurturing
- **Gesture Intensity:** 0.65
- **Key Gestures:** open_arms_welcome, pat_gesture, gather_gesture, supportive_nod
- **Characteristics:** Warm gaze, supportive nods, inclusive gestures

### 7. The Diplomat
- **Traits:** Social • Practical
- **Expression Style:** Balanced
- **Gesture Intensity:** 0.6
- **Key Gestures:** balance_hands, bridge_gesture, open_palm_each_side, inclusive_sweep
- **Characteristics:** Measured gaze, diplomatic nods, fair gestures

### 8. The Architect
- **Traits:** Structured • Visionary
- **Expression Style:** Focused
- **Gesture Intensity:** 0.65
- **Key Gestures:** building_blocks, blueprint_trace, precise_placement, connect_points
- **Characteristics:** Purposeful gaze, confirming nods, structural gestures

### 9. The Rebel
- **Traits:** Chaotic • Practical
- **Expression Style:** Bold
- **Gesture Intensity:** 0.8
- **Key Gestures:** fist_pump, break_chains, dismiss_wave, provocative_point
- **Characteristics:** Bold gaze, defiant nods, challenging gestures

### 10. The Strategist
- **Traits:** Practical • Structured
- **Expression Style:** Controlled
- **Gesture Intensity:** 0.55
- **Key Gestures:** chess_move, count_fingers, map_gesture, timeline_draw
- **Characteristics:** Calculated gaze, decisive nods, strategic gestures

### 11. The MacGyver
- **Traits:** Resourceful • Analytical
- **Expression Style:** Inventive
- **Gesture Intensity:** 0.75
- **Key Gestures:** assembling, light_bulb, tool_mime, improvise_gesture
- **Characteristics:** Scanning gaze, quick nods, hands-on gestures

### 12. The Survivor
- **Traits:** Practical • Serious
- **Expression Style:** Grounded
- **Gesture Intensity:** 0.5
- **Key Gestures:** grounding_stance, practical_demo, ready_hands, calm_down_motion
- **Characteristics:** Scanning gaze, confirming nods, practical gestures

---

## Tone Modifiers

Tones adjust the base expression values:

| Tone | Multiplier | Key Effects |
|------|------------|-------------|
| **Enthusiastic** | 1.3x | +25 smile, +20 eyebrow, faster speech |
| **Serious** | 0.7x | -20 smile, focused gaze, slower gestures |
| **Playful** | 1.2x | +30 smile, head bobs, varied speech |
| **Thoughtful** | 0.85x | Gaze shifts, deliberate pace |
| **Warm** | 1.0x | +20 smile, soft gaze, gentle |
| **Confident** | 1.1x | Steady gaze, firm nods |

### Tone Trigger Words

```javascript
enthusiastic: ['!', 'amazing', 'incredible', 'wow', 'fantastic', 'awesome']
serious: ['important', 'critical', 'serious', 'careful', 'warning']
playful: ['fun', 'play', 'silly', 'game', 'imagine', 'pretend']
thoughtful: ['think', 'consider', 'perhaps', 'maybe', 'wonder']
warm: ['love', 'care', 'dear', 'heart', 'wonderful', 'special']
confident: ['definitely', 'absolutely', 'certainly', 'sure', 'exactly']
```

---

## Age Profiles

Expression intensity and movement vary by age:

| Age Bucket | Intensity | Amplitude | Duration | Style |
|------------|-----------|-----------|----------|-------|
| **2-5** | 1.5x | 1.4x | 0.7x | Bouncy, exaggerated |
| **6-12** | 1.25x | 1.2x | 0.85x | Animated, curious |
| **13-17** | 0.9x | 0.85x | 1.0x | Cool, restrained |
| **18-35** | 1.0x | 1.0x | 1.0x | Balanced, natural |
| **36-60** | 0.85x | 0.9x | 1.15x | Subtle, confident |
| **61-102** | 0.75x | 0.7x | 1.3x | Gentle, wise |

### Age-Appropriate Gestures

**Preferred for 2-5:** bounce_hop, clap_hands, arms_wide_open  
**Avoid for 2-5:** chin_touch, hands_steepled, thoughtful_chin

**Preferred for 13-17:** shoulder_shrug, nod_subtle, finger_point_precise  
**Avoid for 13-17:** bounce_hop, clap_hands, embrace_gesture

**Preferred for 61-102:** soft_nod, palm_up_offering, gentle_reach  
**Avoid for 61-102:** fist_pump, bounce_hop, rock_on

---

## Language/Cultural Profiles

| Language | Gesture Intensity | Expression | Style |
|----------|-------------------|------------|-------|
| **English** | 1.0x | 1.0x | Clear, moderate |
| **Spanish** | 1.3x | 1.15x | Expressive, passionate |
| **French** | 0.9x | 1.05x | Elegant, refined |

---

## Phase Profiles

Expression defaults for lesson phases:

| Phase | Energy | Default Mood | Key Gestures |
|-------|--------|--------------|--------------|
| **Welcome** | 1.1 | Warm greeting | open_arms_welcome, wave |
| **Q1 (Teaching)** | 1.05 | Curious exploration | point_up, chin_touch |
| **Q2 (Practice)** | 1.15 | Engaged challenge | encouraging_nod, balance |
| **Q3 (Synthesis)** | 1.1 | Deep engagement | connect_points, light_bulb |
| **Wisdom** | 0.95 | Profound conclusion | heart_open, gentle_nod |

---

## Output Format

### Full Expression Data Structure

```javascript
{
  metadata: {
    archetype: "The Scientist",
    tone: "enthusiastic",
    ageBucket: "18-35",
    language: "en",
    phase: "welcome",
    totalDuration: 45.5,
    generatedAt: "2025-11-25T10:30:00.000Z",
    version: "1.0.0"
  },
  
  expressions: [
    {
      timestamp: 0.0,
      emotion: "warm",
      intensity: 0.65,
      blendShapes: {
        smile: 65,
        eyebrowRaise: 35,
        eyesWide: 25
      },
      transitionDuration: 0.3
    },
    {
      timestamp: 3.2,
      emotion: "curious",
      intensity: 0.8,
      blendShapes: {
        smile: 40,
        eyebrowRaise: 60,
        eyesWide: 55,
        headTilt: 20
      },
      trigger: "?",
      transitionDuration: 0.25
    },
    {
      timestamp: 8.5,
      emotion: "excited",
      intensity: 0.9,
      blendShapes: {
        smile: 85,
        eyebrowRaise: 70,
        eyesWide: 75,
        mouthOpen: 25
      },
      trigger: "amazing!",
      transitionDuration: 0.2
    }
  ],
  
  gestures: [
    {
      timestamp: 0.5,
      gesture: "open_arms_welcome",
      duration: 2.0,
      intensity: 0.6,
      context: "phase_opening"
    },
    {
      timestamp: 5.0,
      gesture: "chin_touch",
      duration: 1.8,
      intensity: 0.5,
      context: "thinking"
    },
    {
      timestamp: 9.0,
      gesture: "point_up_dramatic",
      duration: 1.5,
      intensity: 0.7,
      context: "emphasis"
    }
  ],
  
  blendShapeTimeline: [
    {
      timestamp: 0.0,
      blendShapes: { smile: 65, eyebrowRaise: 35, eyesWide: 25 },
      transitionDuration: 0.3,
      easing: "ease-in-out"
    },
    // ... interpolated keyframes for smooth transitions
  ],
  
  textAnalysis: {
    overallMood: "curious",
    emotionCount: 12,
    pauseCount: 5
  }
}
```

---

## Blend Shape Reference

Standard ARKit-compatible blend shapes used:

| Blend Shape | Range | Description |
|-------------|-------|-------------|
| `smile` | 0-100 | Mouth corners up |
| `eyebrowRaise` | 0-100 | Eyebrows up |
| `eyesWide` | 0-100 | Eyes open wide |
| `eyesClosed` | 0-100 | Eyes closed |
| `eyeSquint` | 0-100 | Eyes squinted |
| `mouthOpen` | 0-100 | Jaw drop |
| `cheekRaise` | 0-100 | Cheeks raised (smile helper) |
| `headTilt` | 0-100 | Head tilt to side |
| `nod` | 0-100 | Head nod down |
| `lipsPursed` | 0-100 | Lips pushed forward |

---

## Usage Examples

### Basic Generation

```javascript
import ExpressionGenerator from './app/expression-generator.js';

const generator = new ExpressionGenerator();

const result = generator.generate({
  text: "Welcome to today's lesson about the amazing sun! Have you ever wondered why the sky is blue?",
  archetype: 'The Explorer',
  tone: 'enthusiastic',
  ageBucket: '6-12',
  language: 'en',
  phase: 'welcome',
});

console.log(result.expressions);
console.log(result.gestures);
```

### With ElevenLabs Metadata

```javascript
import ExpressionGenerator from './app/expression-generator.js';

const generator = new ExpressionGenerator();

// After ElevenLabs API call
const elevenLabsResponse = {
  alignment: [
    { word: "Welcome", start_time: 0.0, end_time: 0.5 },
    { word: "to", start_time: 0.5, end_time: 0.6 },
    // ... more word timings
  ],
  duration: 45.5,
};

const result = generator.generate({
  text: welcomeScript,
  elevenLabsResponse,
  archetype: 'The Scientist',
  tone: 'thoughtful',
  ageBucket: '36-60',
  language: 'en',
  phase: 'q1',
});
```

### Batch Generation for Lesson

```javascript
import { BatchExpressionGenerator } from './app/expression-generator.js';

const batchGenerator = new BatchExpressionGenerator();
const lessonDNA = await loadLessonDNA('the-sun');

// Generate for one archetype
const result = batchGenerator.generateForLesson(lessonDNA, 'The Explorer');

// Generate for all 12 archetypes
const allResults = batchGenerator.generateAllArchetypes(lessonDNA);
```

---

## Pre-Computation CLI

Run the pre-computation script to generate expressions for all 365 lessons:

```bash
# Process all lessons with default archetypes
node scripts/precompute-expressions.js

# Process single day
node scripts/precompute-expressions.js --day 1

# Process range of days
node scripts/precompute-expressions.js --days 1-30

# Process with all 12 archetypes
node scripts/precompute-expressions.js --all

# Dry run (preview)
node scripts/precompute-expressions.js --dry-run

# Custom output directory
node scripts/precompute-expressions.js --output ./expressions-output

# Skip database save
node scripts/precompute-expressions.js --no-db
```

---

## Supabase Storage

Expression data is stored in the `lesson_atoms` table:

```sql
-- Expression data column structure
ALTER TABLE lesson_atoms 
ADD COLUMN IF NOT EXISTS expression_data JSONB;

-- Index for efficient queries
CREATE INDEX IF NOT EXISTS idx_lesson_atoms_expression 
ON lesson_atoms USING GIN (expression_data);

-- Query example
SELECT content, expression_data 
FROM lesson_atoms 
WHERE core_lesson_id = 'the-sun' 
  AND archetype = 'The Scientist' 
  AND phase = 'welcome';
```

---

## Integration with Unity

The expression data format is designed for Unity WebGL:

```csharp
// Unity C# example
[Serializable]
public class ExpressionData {
    public float timestamp;
    public string emotion;
    public float intensity;
    public Dictionary<string, float> blendShapes;
    public float transitionDuration;
}

[Serializable]
public class GestureData {
    public float timestamp;
    public string gesture;
    public float duration;
    public float intensity;
    public string context;
}

// Apply blend shapes to character
void ApplyBlendShapes(Dictionary<string, float> blendShapes) {
    foreach (var kvp in blendShapes) {
        int shapeIndex = skinnedMeshRenderer.sharedMesh
            .GetBlendShapeIndex(kvp.Key);
        if (shapeIndex >= 0) {
            skinnedMeshRenderer.SetBlendShapeWeight(shapeIndex, kvp.Value);
        }
    }
}

// Play gesture animation
void PlayGesture(GestureData gesture) {
    animator.CrossFade(gesture.gesture, 0.2f);
}
```

---

## Performance Considerations

- **Pre-computation:** Generate expressions offline for all 365 lessons
- **Caching:** Store generated data in Supabase for instant retrieval
- **Real-time fallback:** Generate on-the-fly for custom/dynamic content
- **Batch size:** Process 10 lessons at a time to avoid memory issues
- **Interpolation:** Blend shape timeline includes interpolated keyframes for smooth playback

---

## Future Enhancements

1. **AI-Enhanced Text Analysis:** Use Claude/GPT for more nuanced emotion detection
2. **ElevenLabs Turbo Integration:** Real-time expression generation with streaming audio
3. **Viseme Support:** Lip-sync blend shapes from phoneme data
4. **Motion Capture Integration:** Import mocap data for gesture references
5. **User Feedback Loop:** Learn from user engagement to improve expressions

---

## Changelog

### v1.0.0 (2025-11-25)
- Initial implementation
- 12 archetype profiles
- 6 tone modifiers
- 6 age profiles
- 3 language profiles
- 5 phase profiles
- Pre-computation script
- Supabase integration


