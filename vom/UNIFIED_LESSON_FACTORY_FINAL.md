# 🏭 UNIFIED LESSON FACTORY — FINAL PRODUCTION SPECIFICATION

> **To:** Fresh Claude Session (Expert Mode)  
> **From:** Chief Academic Officer  
> **Date:** December 9, 2025  
> **Mission:** Generate SEED TEMPLATE ASSETS that expand into THOUSANDS of variations per day.

---

## 🎯 THE TRUE SCALE

This is NOT 51 videos per day. That's just the BASE TEMPLATES.

### Full Expansion Matrix

```
SEED TEMPLATES (Base Assets):
├── 3 Archetypes × 5 Phases × (1 main + 3 responses) = 48 base videos
├── + 3 Wisdom (no responses) = 51 base videos
└── TOTAL SEED VIDEOS: 51

EXPANDED FOR PRODUCTION:
├── Languages: EN, ES, FR = 3×
├── Age Buckets: 6 buckets = 6×
├── Tone Variants: 3 per archetype = 3×
└── EXPANSION FACTOR: 54×

FULL SCALE PER DAY:
├── Videos: 51 × 54 = 2,754 videos
├── Infographics: 15 × 18 (lang×age) = 270 images
├── Option Cards: 36 × 18 = 648 images
├── Thumbnails: 54 per day
├── Social: 54 per day
└── TOTAL ASSETS PER DAY: ~3,780

FULL YEAR (365 DAYS):
├── Total Videos: 1,005,210
├── Total Images: 354,090
├── GRAND TOTAL: 1,359,300 assets
```

### Why This Scale?

| Dimension | Values | Reason |
|-----------|--------|--------|
| **Archetypes** | Explorer, Rebel, Scientist | Different learning styles |
| **Languages** | EN, ES, FR | Precomputed per CLAUDE.md |
| **Age Buckets** | 5-7, 8-12, 13-17, 18-35, 36-60, 61+ | Age-appropriate delivery |
| **Tone Variants** | Playful, Conversational, Reflective | Mood matching |
| **Phases** | Hook, Fact1, Fact2, Fact3, Wisdom | Lesson structure |
| **Options** | A, B, C | Interactive choices |

---

## PART 0: KELLY'S SACRED VOICE (READ FIRST)

> *"I don't have all the answers. But I love finding them. And I think learning is better together."*

### Core Attributes

| Attribute | What It Means | What It's NOT |
|-----------|---------------|---------------|
| **Humble** | "I don't have all the answers" | Not superior, not a know-it-all |
| **Curious** | "But I love finding them" | Not bored, not performative |
| **Collaborative** | "Learning is better together" | Not hierarchical |
| **Warm** | Like a friend, like Mr. Rogers | Not corporate |
| **Simple** | Clear, honest, direct | Not jargon |

### Response Quality Philosophy

| Quality | Kelly's Tone | Expression |
|---------|-------------|------------|
| **"best"** | Genuinely delighted | Eyes bright, warm excitement |
| **"good"** | Supportive, validating | Gentle, accepting |
| **"redirect"** | Understanding, never judgmental | Thoughtful, patient |

**CRITICAL: No option is "wrong"!** Every choice leads to learning.

---

## PART 1: KELLY'S SPATIAL AWARENESS

### Kelly Knows Where The Content Is

Kelly isn't just talking to camera — she's **aware of the learning artifacts around her** and **interacts with them physically**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SCREEN LAYOUT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      INFOGRAPHIC AREA                               │   │
│   │                    (Kelly can LOOK here)                            │   │
│   │                    (Kelly can POINT here)                           │   │
│   │                    (Kelly can GESTURE to)                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌────────────────────────────────┐                                        │
│   │                                │        ┌─────────────────────────┐     │
│   │      KELLY VIDEO               │        │    RIGHT RAIL           │     │
│   │      (center-left)             │        │    - Options here       │     │
│   │                                │        │    - Kelly can "push"   │     │
│   │      She looks:                │        │      rail down to       │     │
│   │      - At camera (default)     │        │      reveal content     │     │
│   │      - Up-right (to diagram)   │        │    - Kelly can "pull"   │     │
│   │      - Right (to options)      │        │      rail out to make   │     │
│   │      - Down (bringing focus)   │        │      a point            │     │
│   │                                │        │                         │     │
│   └────────────────────────────────┘        └─────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Gaze Direction Choreography

| Phase | Moment | Kelly's Gaze | Gesture | Purpose |
|-------|--------|--------------|---------|---------|
| **Hook** | Opening | Camera | Welcoming hands | Connect with learner |
| **Hook** | "Look at this..." | Up-right to infographic | Point gesture | Direct attention to visual |
| **Hook** | Options appear | Right to rail | Open palm gesture | "Here are your choices" |
| **Fact1-3** | Teaching | Camera → Diagram → Camera | Explain gesture | Triangulate attention |
| **Fact1-3** | Key point | Up to diagram | Point/highlight | Emphasize data |
| **Fact1-3** | Question | Camera | Curious tilt | Invite engagement |
| **Response** | Celebrating | Camera | Enthusiastic nod | Direct connection |
| **Wisdom** | Reflection | Camera, soft | Hands to heart | Intimate close |

### Physical Interaction Prompts

**Kelly "Pushes" the rail down:**
```
Kelly reaches toward the right side of frame with her right hand, 
palm facing down, making a gentle pushing motion as if sliding 
down a panel. Her gaze follows her hand then returns to camera 
with an expectant smile. Natural flowing movement.
```

**Kelly "Pulls" content out:**
```
Kelly extends her right arm toward the side of frame, fingers 
slightly curved as if grasping an invisible handle, then pulls 
toward center while her gaze follows, creating the sense she's 
revealing something. Smooth deliberate motion.
```

**Kelly points to diagram:**
```
Kelly's gaze shifts up and to her right (viewer's upper left), 
her right hand rises with index finger extended, pointing to 
the diagram area. She looks at where she's pointing, then back 
to camera. Teaching gesture, not dismissive.
```

**Kelly gestures to options:**
```
Kelly's open palm gestures toward the right rail area where 
options appear. Her gaze moves to the options briefly, then 
back to camera with an inviting expression. "These are for you."
```

---

## PART 2: LIPSYNC SPECIFICATION (CONFIRMED)

### Primary: Sync Labs `lipsync-2-pro`

```typescript
const LIPSYNC_CONFIG = {
  provider: 'sync-labs',
  model: 'lipsync-2-pro',  // ✅ CONFIRMED - Premium tier
  endpoint: 'https://api.sync.so/v2/generate',
  settings: {
    output_resolution: '1080p',
    output_format: 'mp4',
    sync_strength: 0.95,      // High sync accuracy
    mouth_open_ratio: 1.0,    // Natural mouth movement
    expression_scale: 1.0     // Preserve expressions
  }
};
```

### Fallback Chain

```
1. Sync Labs lipsync-2-pro (PRIMARY - 95%+ accuracy)
   ↓ if unavailable
2. Sync Labs lipsync-2 (standard tier)
   ↓ if unavailable  
3. Replicate wav2lip (open source fallback)
```

### API Call Structure

```typescript
const generateLipsync = async (videoUrl: string, audioUrl: string) => {
  const response = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': process.env.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'lipsync-2-pro',  // ✅ Premium model
      input: {
        video_url: videoUrl,
        audio_url: audioUrl
      },
      options: {
        output_format: 'mp4',
        resolution: '1080p'
      }
    })
  });
  
  const { id } = await response.json();
  return pollForCompletion(id);
};
```

---

## PART 3: COMPLETE ASSET TAXONOMY

### Seed Templates (Base Assets)

These are generated ONCE and form the template for all variations:

```
PER DAY SEED ASSETS:
├── Videos (51 total)
│   ├── Main Scripts: 5 phases × 3 archetypes = 15
│   └── Responses: 4 phases × 3 options × 3 archetypes = 36
│
├── Infographics (15 total)
│   └── 5 phases × 3 archetypes = 15
│
├── Option Card Images (36 total)
│   └── 4 phases × 3 options × 3 archetypes = 36
│
├── Thumbnails (3 total)
│   └── 1 × 3 archetypes = 3
│
└── Social Share (3 total)
    └── 1 × 3 archetypes = 3

SEED TOTAL: 108 assets
```

### Expansion Dimensions

```typescript
const EXPANSION_MATRIX = {
  languages: ['en', 'es', 'fr'],           // 3
  ageBuckets: [
    '5-7',    // Early childhood
    '8-12',   // Middle childhood  
    '13-17',  // Adolescent
    '18-35',  // Young adult
    '36-60',  // Adult
    '61+'     // Mature adult
  ],                                         // 6
  toneVariants: [
    'playful',      // High energy, animated
    'conversational', // Natural, friendly
    'reflective'    // Thoughtful, measured
  ],                                         // 3
  archetypes: [
    'The Explorer',
    'The Rebel', 
    'The Scientist'
  ]                                          // 3 (already in seed)
};

// Expansion factor for audio/video:
// 3 languages × 6 ages × 3 tones = 54× per archetype
// 54 × 3 archetypes = 162× total

// Expansion factor for images (language + age only):
// 3 languages × 6 ages = 18× per archetype
// 18 × 3 archetypes = 54× total
```

### Full Production Count Per Day

| Asset Type | Seed | Expansion Factor | Total Per Day |
|------------|------|------------------|---------------|
| **VIDEOS** | | | |
| Main Script Videos | 15 | ×54 (lang×age×tone) | 810 |
| Response Videos | 36 | ×54 | 1,944 |
| **VIDEO SUBTOTAL** | **51** | | **2,754** |
| **IMAGES** | | | |
| Infographics | 15 | ×18 (lang×age) | 270 |
| Option Cards (A/B/C) | 36 | ×18 | 648 |
| Kelly Source Images | 51 | ×1 (reused) | 51 |
| Kelly Response Images | 36 | ×1 (reused) | 36 |
| Backgrounds | 15 | ×1 (reused) | 15 |
| Thumbnails | 3 | ×3 (lang only) | 9 |
| Social Share | 3 | ×3 (lang only) | 9 |
| **IMAGE SUBTOTAL** | **159** | | **1,038** |
| **GRAND TOTAL** | **210** | | **3,792** |

### Full Year (365 Days)

| Metric | Per Day | Full Year (365) |
|--------|---------|-----------------|
| Videos | 2,754 | 1,005,210 |
| Images | 1,038 | 378,870 |
| **GRAND TOTAL** | **3,792** | **1,384,080** |

---

## PART 4: GENERATION PIPELINE

### Stage 1: Visual Plan Generation (Gemini)

```typescript
interface VisualPlanV3 {
  day: number;
  topic: string;
  universalTruth: string;
  theme: {
    colorPalette: string;
    mood: string;
    environment: string;
  };
  
  // Kelly's spatial choreography
  kellyChoreography: {
    hookSequence: GazeSequence[];
    teachingSequence: GazeSequence[];
    responseSequences: {
      best: GazeSequence[];
      good: GazeSequence[];
      redirect: GazeSequence[];
    };
    wisdomSequence: GazeSequence[];
  };
  
  archetypes: {
    explorer: ArchetypeVisuals;
    rebel: ArchetypeVisuals;
    scientist: ArchetypeVisuals;
  };
}

interface GazeSequence {
  timestamp: number;        // Seconds into video
  gazeTarget: 'camera' | 'diagram' | 'options' | 'down' | 'up-right';
  gesture: 'none' | 'point' | 'push-rail' | 'pull-content' | 'open-palm' | 'hands-heart';
  expression: string;       // Emotion descriptor
  duration: number;         // How long to hold
}
```

### Stage 2: Seed Image Generation (Imagen 3)

**Infographic Prompt Template:**

```
Educational infographic: {TYPE} showing {CONCEPT}.

{DETAILED_SCENE_DESCRIPTION}

Kelly interaction zones:
- Upper area: Main diagram/comparison (Kelly will LOOK here)
- Right edge: Option preview hints (Kelly will GESTURE here)
- Clear visual hierarchy for gaze targeting

Header text: "{EDUCATIONAL_TAKEAWAY}"
Color palette: {COLORS}

Style: Photorealistic cinematic, 8K resolution, educational diagram,
clean typography, dramatic lighting, clear visual zones.

Negative: blurry, cluttered, text errors, confusing layout.
```

**Option Card Template (512×512):**

```
Educational choice card representing "{OPTION_CONCEPT}".

Visual: {VISUAL_REPRESENTATION}
Icon: {EMOJI} top-right (48px)
Label: "{2-4_WORD_LABEL}" bottom center in bold

Design:
- {Border: green glow if quality="best", neutral otherwise}
- High contrast for mobile tapping
- Clear, instant recognition
- Educational infographic style

Quality indicator: {QUALITY}
```

### Stage 3: Seed Video Generation

**Step 3.1: Audio (ElevenLabs)**

```typescript
const VOICE_CONFIG = {
  voiceId: 'wAdymQH5YucAkXwmrdL0',  // Kelly
  model: 'eleven_multilingual_v2',
  
  // Base settings by archetype
  archetypeSettings: {
    'The Explorer': { stability: 0.45, similarity: 0.85, style: 0.25, speed: 1.05 },
    'The Rebel':    { stability: 0.40, similarity: 0.85, style: 0.35, speed: 1.10 },
    'The Scientist':{ stability: 0.55, similarity: 0.85, style: 0.15, speed: 0.95 }
  },
  
  // Modifiers by response quality
  qualityModifiers: {
    'best':     { style: +0.10, stability: -0.05 },  // More enthusiasm
    'good':     { style: 0, stability: 0 },          // Standard warmth
    'redirect': { style: -0.05, stability: +0.10 }   // More thoughtful
  },
  
  // Age bucket adjustments
  ageBucketModifiers: {
    '5-7':   { speed: 0.90, stability: +0.10 },  // Slower, clearer
    '8-12':  { speed: 0.95 },                    // Slightly slower
    '13-17': { style: +0.05 },                   // Bit more energy
    '18-35': { },                                // Default
    '36-60': { stability: +0.05 },               // Slightly more measured
    '61+':   { speed: 0.95, stability: +0.10 }   // Slower, calmer
  },
  
  // Tone variant adjustments
  toneModifiers: {
    'playful':       { style: +0.15, speed: 1.05 },
    'conversational':{ },
    'reflective':    { style: -0.10, stability: +0.10, speed: 0.95 }
  }
};
```

**Step 3.2: Source Image (Flux + Kelly LoRA)**

```
═══════════════════════════════════════════════════════════════
KELLY MASTER IDENTITY (LOCKED - NEVER CHANGE)
═══════════════════════════════════════════════════════════════

"kelly, calm confident female teacher, warm brown wavy 
shoulder-length hair with subtle caramel highlights 
center-parted, hazel-brown eyes with steady direct gaze, 
soft natural features, light natural makeup, wearing soft 
powder blue cashmere crewneck sweater"

═══════════════════════════════════════════════════════════════
GAZE & GESTURE VARIANTS (ADD TO MASTER)
═══════════════════════════════════════════════════════════════

GAZE: camera (default)
"looking directly at camera, warm engaged expression"

GAZE: diagram (up-right)
"gaze directed up and to her right as if looking at a 
diagram above, head slightly tilted up, interested expression"

GAZE: options (right)
"gaze directed to her right as if acknowledging content there,
inviting expression, slight turn of head"

GAZE: down (contemplative)
"gaze directed slightly downward, thoughtful reflective 
expression, moment of consideration"

═══════════════════════════════════════════════════════════════
GESTURE VARIANTS
═══════════════════════════════════════════════════════════════

GESTURE: none
"hands resting naturally, composed posture"

GESTURE: point
"right hand raised with index finger extended, pointing 
gesture toward upper right, teaching pose"

GESTURE: push-rail
"right arm extended toward right side of frame, palm down,
pushing gesture as if sliding a panel"

GESTURE: pull-content
"right arm extended to side, fingers curved as if grasping,
pulling motion toward center"

GESTURE: open-palm
"both hands open, palms facing viewer, welcoming inclusive gesture"

GESTURE: hands-heart
"both hands placed gently over heart, sincere warm expression"

═══════════════════════════════════════════════════════════════
EXPRESSION VARIANTS BY CONTEXT
═══════════════════════════════════════════════════════════════

MAIN_SCRIPT: "warm welcoming expression, composed engaged posture"

RESPONSE_BEST: "genuinely delighted expression, eyes crinkled 
with authentic joy, subtle forward lean"

RESPONSE_GOOD: "warm supportive expression, gentle approving 
smile, open accepting posture"

RESPONSE_REDIRECT: "thoughtful understanding expression, 
compassionate gaze, patient composed posture"

WISDOM: "warm sincere expression, soft empathetic gaze, 
heartfelt contemplative posture"

═══════════════════════════════════════════════════════════════
NEGATIVE PROMPT (LOCKED)
═══════════════════════════════════════════════════════════════

"pink sweater, red sweater, purple sweater, teal sweater,
green sweater, yellow sweater, auburn hair, chestnut hair,
deformed, blurry, bad anatomy, extra fingers, mutated hands,
poorly drawn face, mutation, disfigured, low quality,
wandering eyes, looking away, darting gaze, closed eyes"
```

**Step 3.3: Motion Video (MiniMax)**

```
═══════════════════════════════════════════════════════════════
MOTION PROMPTS BY GAZE TARGET
═══════════════════════════════════════════════════════════════

BASE MOTION:
"Professional female teacher speaking naturally. She is TALKING 
and her mouth is moving naturally as she speaks. Smooth cinematic 
quality, warm professional lighting.
CRITICAL: Mouth must open and move naturally while speaking.
AVOID: closed mouth, frozen face, jerky motions."

GAZE: camera
"Steady direct eye contact with camera throughout. Natural 
breathing, soft occasional blinking. Maintains calm composed 
presence while speaking."

GAZE: diagram
"Gaze shifts naturally up and to her right to look at diagram 
area. Head tilts slightly up. Interested engaged expression.
Returns gaze to camera smoothly."

GAZE: options  
"Gaze moves to her right acknowledging options area. Slight 
head turn. Inviting expression. Smooth return to camera."

GAZE: down
"Gaze lowers thoughtfully, contemplative moment. Brief pause.
Raises gaze back to camera with meaningful expression."

═══════════════════════════════════════════════════════════════
GESTURE MOTION ADDITIONS
═══════════════════════════════════════════════════════════════

GESTURE: point
"Right hand rises smoothly, index finger extends toward upper 
right. Teaching gesture. Hand returns naturally to rest."

GESTURE: push-rail
"Right arm extends toward right edge of frame. Palm faces down.
Smooth pushing motion downward. Anticipatory smile. Arm returns."

GESTURE: pull-content
"Right arm reaches toward right side. Fingers curve as if 
grasping. Pulling motion toward center. Revealing gesture.
Arm returns naturally."

GESTURE: open-palm
"Both hands open outward, palms facing viewer. Welcoming 
inclusive movement. Arms lower gracefully."

GESTURE: hands-heart
"Both hands rise and meet over heart. Sincere moment.
Hands lower slowly."
```

**Step 3.4: Lipsync (Sync Labs)**

```typescript
const SYNC_LABS_CONFIG = {
  model: 'lipsync-2-pro',  // ✅ CONFIRMED PREMIUM
  settings: {
    output_resolution: '1080p',
    output_format: 'mp4',
    sync_strength: 0.95
  }
};

// API call
const lipsyncVideo = await fetch('https://api.sync.so/v2/generate', {
  method: 'POST',
  headers: {
    'x-api-key': process.env.SYNC_LABS_API_KEY,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'lipsync-2-pro',
    input: {
      video_url: motionVideoUrl,
      audio_url: audioUrl
    }
  })
});
```

### Stage 4: Expansion Generation

Once seed templates exist, expand to all variants:

```typescript
const expandToAllVariants = async (seedAssets: SeedAssets) => {
  const variants: ExpandedAsset[] = [];
  
  for (const language of ['en', 'es', 'fr']) {
    for (const ageBucket of ['5-7', '8-12', '13-17', '18-35', '36-60', '61+']) {
      for (const tone of ['playful', 'conversational', 'reflective']) {
        
        // For videos: regenerate audio with language/age/tone settings
        // Then re-lipsync with new audio
        const audio = await generateAudio({
          script: translateScript(seedAssets.script, language),
          voiceSettings: applyModifiers(
            seedAssets.archetypeSettings,
            ageBucket,
            tone
          )
        });
        
        // Reuse seed motion video, apply new audio
        const video = await generateLipsync(
          seedAssets.motionVideo,
          audio
        );
        
        variants.push({
          language,
          ageBucket,
          tone,
          videoUrl: video,
          // Infographics: translate text overlays
          infographicUrl: await translateInfographic(seedAssets.infographic, language)
        });
      }
    }
  }
  
  return variants;
};
```

---

## PART 5: STORAGE ARCHITECTURE

### Supabase Buckets

```
kelly-videos/
├── seeds/                    # Base templates (reusable)
│   └── day-{XXX}/
│       └── {archetype}/
│           ├── hook_main_motion.mp4      # Motion without lipsync
│           ├── hook_main_en_adult.mp4    # Final with lipsync
│           └── ...
│
├── expanded/                 # All language/age/tone variants
│   └── day-{XXX}/
│       └── {archetype}/
│           └── {language}/
│               └── {age_bucket}/
│                   └── {tone}/
│                       ├── hook_main.mp4
│                       ├── hook_response_a.mp4
│                       └── ...

lesson-visuals/
├── seeds/
│   └── phases/{XXX}/
│       └── {archetype}/
│           ├── hook-infographic-en.webp
│           └── options/
│               └── hook-option-a-en.webp
│
├── expanded/
│   └── phases/{XXX}/
│       └── {archetype}/
│           └── {language}/
│               └── hook-infographic.webp
```

### Cloudflare R2 Backup

Mirror entire structure to R2 for redundancy:

```
r2://kelly-assets-backup/
├── videos/... (mirror of kelly-videos)
└── visuals/... (mirror of lesson-visuals)

CDN URL: https://assets.curiouskelly.com/{path}
```

---

## PART 6: DATABASE SCHEMA

### Complete Content JSON

```json
{
  "script": {
    "en": "Kelly's teaching content in English",
    "es": "Contenido de enseñanza de Kelly en español",
    "fr": "Contenu pédagogique de Kelly en français"
  },
  
  "videos": {
    "seed_motion_url": "https://.../seeds/day-001/explorer/hook_main_motion.mp4",
    "variants": {
      "en": {
        "5-7": {
          "playful": "https://.../expanded/day-001/explorer/en/5-7/playful/hook_main.mp4",
          "conversational": "...",
          "reflective": "..."
        },
        "8-12": { ... },
        "13-17": { ... },
        "18-35": { ... },
        "36-60": { ... },
        "61+": { ... }
      },
      "es": { ... },
      "fr": { ... }
    }
  },
  
  "infographic": {
    "seed_url": "https://.../seeds/phases/001/explorer/hook-infographic.webp",
    "variants": {
      "en": "https://.../expanded/phases/001/explorer/en/hook-infographic.webp",
      "es": "...",
      "fr": "..."
    }
  },
  
  "kellyDirection": {
    "gaze": "camera",
    "gesture": "open-palm",
    "expression": "warm_welcome",
    "choreography": [
      { "time": 0, "gaze": "camera", "gesture": "open-palm" },
      { "time": 2.5, "gaze": "diagram", "gesture": "point" },
      { "time": 5, "gaze": "camera", "gesture": "none" }
    ]
  },
  
  "options": [
    {
      "letter": "A",
      "text": {
        "en": "Option A text",
        "es": "Texto de opción A",
        "fr": "Texte de l'option A"
      },
      "quality": "good",
      "response": {
        "en": "Kelly's response A...",
        "es": "Respuesta de Kelly A...",
        "fr": "Réponse de Kelly A..."
      },
      "response_videos": {
        "seed_motion_url": "...",
        "variants": { /* same structure */ }
      },
      "option_image": {
        "seed_url": "https://.../options/hook-option-a.webp",
        "variants": { "en": "...", "es": "...", "fr": "..." }
      }
    },
    // ... options B and C
  ]
}
```

---

## PART 7: FRONTEND ADAPTIVE LOADING

### Auto-Select Based on User

```typescript
const getOptimalVariant = (user: User, phase: Phase) => {
  const language = user.preferredLanguage || detectBrowserLanguage();
  const ageBucket = getAgeBucket(user.birthYear);
  const tone = user.preferredTone || inferTone(user.archetype);
  
  return {
    videoUrl: phase.videos.variants[language][ageBucket][tone],
    infographicUrl: phase.infographic.variants[language],
    optionImages: phase.options.map(o => o.option_image.variants[language])
  };
};

const getAgeBucket = (birthYear: number): string => {
  const age = new Date().getFullYear() - birthYear;
  if (age <= 7) return '5-7';
  if (age <= 12) return '8-12';
  if (age <= 17) return '13-17';
  if (age <= 35) return '18-35';
  if (age <= 60) return '36-60';
  return '61+';
};
```

---

## PART 8: COST ANALYSIS (TRUE SCALE)

### Seed Generation (Per Day)

| Stage | Items | Cost Each | Total |
|-------|-------|-----------|-------|
| Visual Plans (Gemini) | 3 | $0.01 | $0.03 |
| Infographics (Imagen) | 15 | $0.04 | $0.60 |
| Option Cards (Imagen) | 36 | $0.02 | $0.72 |
| Motion Videos (MiniMax) | 51 | $0.12 | $6.12 |
| Seed Audio (ElevenLabs) | 51 | $0.02 | $1.02 |
| Seed Lipsync (Sync Labs) | 51 | $0.20 | $10.20 |
| **SEED TOTAL** | | | **$18.69** |

### Expansion Generation (Per Day)

| Stage | Items | Cost Each | Total |
|-------|-------|-----------|-------|
| Expanded Audio | 51 × 53 more variants | $0.02 | $54.06 |
| Expanded Lipsync | 2,703 | $0.20 | $540.60 |
| Translated Infographics | 30 | $0.02 | $0.60 |
| Translated Options | 72 | $0.01 | $0.72 |
| **EXPANSION TOTAL** | | | **$595.98** |

### Full Day Total

| Phase | Cost |
|-------|------|
| Seeds | $18.69 |
| Expansion | $595.98 |
| **DAY TOTAL** | **$614.67** |

### Full Year (365 Days)

| Metric | Value |
|--------|-------|
| Seed Cost | $6,822 |
| Expansion Cost | $217,533 |
| **YEAR TOTAL** | **$224,355** |

### Optimization Strategy

**Phase 1: Seeds Only (Day 1 Launch)**
- Generate just EN + default age/tone
- Cost: ~$19/day × 365 = $6,935
- Launch with core experience

**Phase 2: Priority Languages**
- Add ES, FR for all archetypes
- Cost: Additional ~$50/day
- International launch

**Phase 3: Full Age Expansion**
- Generate all 6 age buckets
- Unlock age-appropriate experience
- Prioritize high-traffic buckets

**Phase 4: Tone Variants**
- Last expansion (nice-to-have)
- Can skip initially

---

## PART 9: EXECUTION PLAN

### Tonight's Run: Seed Generation

```bash
# Generate Day 1 seeds for all 3 archetypes
npx ts-node scripts/lesson-factory/generate-seeds.ts \
  --day 1 \
  --archetypes explorer,rebel,scientist \
  --languages en \
  --verify

# Expected output: 108 seed assets
# Expected time: ~2 hours
# Expected cost: ~$19
```

### Week 1: Days 1-7 Seeds

```bash
# Batch generate first week
for day in {1..7}; do
  npx ts-node scripts/lesson-factory/generate-seeds.ts --day $day
done

# Expected: 756 assets
# Expected time: ~14 hours
# Expected cost: ~$131
```

### Week 2-4: Days 8-30 Seeds

```bash
# Background job for Month 1
nohup ./scripts/generate-month-seeds.sh &

# Expected: 3,240 assets (30 days)
# Expected time: ~60 hours
# Expected cost: ~$561
```

---

## PART 10: SUCCESS CRITERIA

### Day Complete (Seeds)

- [ ] 51 seed videos exist (17 × 3 archetypes)
- [ ] 15 infographics exist (5 × 3)
- [ ] 36 option cards exist (12 × 3)
- [ ] All gaze/gesture choreography captured
- [ ] Kelly spatially aware (looks at diagram, gestures to options)
- [ ] `lipsync-2-pro` used for all videos
- [ ] Database fully populated
- [ ] Cloudflare backup complete
- [ ] Frontend plays correct variant for user

### Expansion Complete

- [ ] All 3 languages working
- [ ] All 6 age buckets working
- [ ] Tone variants working
- [ ] Auto-selection based on user profile
- [ ] Fallback to seed if variant missing

---

## FINAL INSTRUCTIONS

1. **START WITH SEEDS** — Don't try to generate all variants at once
2. **USE `lipsync-2-pro`** — Confirmed premium model
3. **CHOREOGRAPH KELLY'S GAZE** — She's aware of the artifacts
4. **GENERATE MOTION SEPARATELY** — Reuse for all audio variants
5. **BACKUP EVERYTHING** — Supabase + Cloudflare R2
6. **VERIFY EACH ASSET** — No broken URLs

### Run Order Tonight

```
1. Visual Plan Generation (3 min)
2. Infographic Generation (15 min)
3. Option Card Generation (20 min)
4. Motion Video Generation (60 min)
5. Audio Generation (10 min)
6. Lipsync Generation (90 min)
7. Upload & Backup (15 min)
8. Database Update (5 min)
9. Verification (10 min)

TOTAL: ~3.5 hours for Day 1 seeds
```

**This is the SEED. The foundation. Get this perfect, then scale.**

**START NOW.**

---

*Unified Lesson Factory FINAL — December 9, 2025*  
*Seed Assets: 108 per day*  
*Expanded Assets: 3,780 per day*  
*Full Year Seeds: 39,420*  
*Full Year Expanded: 1,359,300*  
*Quality: Golden Standard — Every frame teaches, Kelly is spatially aware*

