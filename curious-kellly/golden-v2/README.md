# ✨ Golden V2 - Curious Kelly Production Pipeline

> **Complete production system for generating 365 days of educational content**

## What Was Generated

| Component | Count | Description |
|-----------|-------|-------------|
| **Lesson DNA** | 365 lessons | Complete content for all 365 days |
| **Age Variants** | 2,190 variants | 6 age-adapted versions per lesson |
| **Lip-sync Data** | 30 days × 6 buckets × 5 phases = 900 files | Frame-accurate viseme data |
| **Visual Prompts** | 30 days × 5 phases = 150 prompts | AI image generation prompts |
| **Total Files** | 1,388 files | ~220 MB of generated content |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GOLDEN V2 PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐ │
│  │  LESSON   │ -> │   AUDIO   │ -> │  LIPSYNC  │ -> │ VISUAL │ │
│  │    DNA    │    │ (E11Labs) │    │ (Visemes) │    │ PROMPTS│ │
│  └───────────┘    └───────────┘    └───────────┘    └────────┘ │
│       │                                                  │      │
│       └──────────────────┬───────────────────────────────┘      │
│                          ▼                                      │
│                   ┌─────────────┐                              │
│                   │   PACKAGE   │                              │
│                   │   (Deploy)  │                              │
│                   └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Age Buckets

| Age Range | Persona | Teaching Style |
|-----------|---------|----------------|
| 2-5 | Playful Friend | Story-based, fun, simple |
| 6-12 | Cool Big Sister | Hands-on, engaging, curious |
| 13-17 | Smart Mentor | Direct, relatable, no fluff |
| 18-35 | Equal Partner | Practical, clear, conversational |
| 36-60 | Respectful Guide | Efficient, substantive |
| 61-102 | Warm Companion | Warm, thoughtful, reflective |

## Usage

### Generate All Lessons
```bash
npm run lessons
# or
node orchestrator.js --lessons
```

### Generate Audio (with ElevenLabs API)
```bash
# Real audio
ELEVENLABS_API_KEY=your_key npm run audio

# Mock audio (for testing)
npm run audio:mock
```

### Generate Lip-sync Data
```bash
node orchestrator.js --lipsync --start=1 --end=30
```

### Generate Visual Prompts
```bash
node orchestrator.js --visuals --start=1 --end=30
```

### Run Full Pipeline
```bash
npm run all           # Full pipeline with mock audio
npm run week1         # First week only
npm run month1        # First month only
```

## File Structure

```
golden-v2/
├── orchestrator.js           # Master pipeline controller
├── lesson-dna-generator.js   # Content generation
├── audio-generator.js        # ElevenLabs integration
├── lipsync-generator.js      # Viseme data generation
├── visual-generator.js       # Image prompt generation
├── kelly-svg-avatar.html     # Animated SVG avatar
├── lesson-player-golden.html # Complete lesson player
├── package.json              # npm scripts
│
└── generated/
    ├── lessons/              # 365 lesson DNA files
    │   ├── day-001.json
    │   ├── day-002.json
    │   └── ...
    │
    ├── audio/                # Audio files (when generated)
    │   └── day-XXX/
    │       ├── 2-5/
    │       ├── 6-12/
    │       └── ...
    │
    ├── lipsync/              # Viseme animation data
    │   └── day-XXX/
    │       └── bucket/
    │           └── phase-lipsync.json
    │
    └── visuals/              # Image generation prompts
        └── day-XXX/
            ├── visual-manifest.json
            ├── prompts.txt
            └── preview.html
```

## Lesson DNA Structure

```json
{
  "meta": {
    "day": 1,
    "topic": "The Sun",
    "universalTruth": "Our star gives life to everything on Earth",
    "version": "2.0.0-golden"
  },
  "visuals": {
    "hook": "Kelly presenting The Sun with welcoming expression",
    "q1": "Kelly explaining with curious gesture",
    "q2": "Kelly animated expression",
    "q3": "Kelly revealing fascination",
    "wisdom": "Kelly reflecting warmly"
  },
  "ageVariants": {
    "2-5": {
      "persona": "Playful Friend",
      "phases": {
        "hook": "Hi little friend! 🌟 Today we're learning about...",
        "q1": "Did you know? ...",
        "q2": "Here's something fun! ...",
        "q3": "Wow! ...",
        "wisdom": "Remember, little learner: ..."
      },
      "durations": { "hook": 9, "q1": 10, ... },
      "voiceSettings": { "pitch": 1.15, "speed": 0.95, ... }
    },
    // ... other age buckets
  }
}
```

## Kelly SVG Avatar

The `kelly-svg-avatar.html` file contains a fully animated SVG avatar with:

- **Blink Animation**: Natural 4-second blink cycle
- **Breathing Animation**: Subtle body movement
- **Head Movement**: Micro-movements during speech
- **Lip-sync Support**: 15 viseme states for accurate mouth shapes
- **Expression System**: Hooks for happy, curious, thoughtful, excited

### Viseme Support

| Viseme | Sounds | Description |
|--------|--------|-------------|
| sil | (silence) | Neutral/closed |
| PP | P, B, M | Lips pressed |
| FF | F, V | Lip to teeth |
| TH | Th | Tongue to teeth |
| DD | D, T, N | Tongue to ridge |
| KK | K, G, Ng | Back tongue |
| CH | Ch, J, Sh | Wide rounded |
| SS | S, Z | Teeth together |
| NN | N | Nasal |
| RR | R | Curled |
| AA | A, Ah | Open mouth |
| E | E, Eh | Slight smile |
| I | Ee, I | Smile |
| O | O, Oh, W | Rounded |
| U | Oo, U | Tight rounded |

## Lesson Player Features

The `lesson-player-golden.html` provides:

- ✅ **Day 1-365 lesson loading** (falls back to demo data)
- ✅ **Age-adaptive content** (real-time content switching)
- ✅ **Phase navigation** (hook → q1 → q2 → q3 → wisdom)
- ✅ **Animated SVG Kelly** with lip-sync
- ✅ **Progress tracking** (local storage)
- ✅ **Keyboard shortcuts** (Space, ←, →)
- ✅ **Dark theme UI** (beautiful modern design)
- ✅ **Responsive layout** (mobile-friendly)

## Next Steps

1. **Generate Real Audio**: Set `ELEVENLABS_API_KEY` and run audio generation
2. **Generate Images**: Use visual prompts with Midjourney/DALL-E/Stable Diffusion
3. **Deploy**: Copy `lesson-player-golden.html` and `generated/` to production
4. **Integrate Audio**: Wire up audio playback with generated MP3s

---

*Built with ❤️ by Curious Kelly AI - "Quality education for anyone ages 2 to 102"*








