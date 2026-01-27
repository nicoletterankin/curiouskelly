# ElevenLabs Metadata → Kelly Avatar Driving System

**Comprehensive Technical Report**  
**Voice ID:** `wAdymQH5YucAkXwmrdL0` (Kelly25)  
**Last Updated:** December 26, 2025

---

## Executive Summary

This report documents how Curious Kelly extracts and leverages metadata from ElevenLabs API responses to drive Kelly's avatar with precise lip-sync, expressions, and gestures. The system uses a multi-tier architecture combining:

1. **Pre-computed Forced Alignment** (production pipeline)
2. **ElevenLabs Response Metadata Processing** (word/character timings)
3. **Real-time Audio Analysis** (conversational fallback)
4. **Expression System Integration** (emotions, gestures)

---

## Table of Contents

1. [Kelly's Voice Configuration](#1-kellys-voice-configuration)
2. [ElevenLabs API Metadata Types](#2-elevenlabs-api-metadata-types)
3. [Metadata Extraction Pipeline](#3-metadata-extraction-pipeline)
4. [Lip-Sync Data Flow](#4-lip-sync-data-flow)
5. [Expression Generation from Metadata](#5-expression-generation-from-metadata)
6. [Production Architecture](#6-production-architecture)
7. [API Reference](#7-api-reference)
8. [Implementation Checklist](#8-implementation-checklist)

---

## 1. Kelly's Voice Configuration

### Primary Voice ID

```javascript
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';  // Kelly25 - trained voice
`

### Age-Adaptive Voice Settings

Kelly's voice is dynamically adjusted based on learner age bucket:

```javascript
// From curious-kellly/golden-v2/audio-generator.js
const VOICE_SETTINGS = {
  '2-5': {
    stability: 0.4,          // More expressive for young children
    similarity_boost: 0.8,
    style: 0.7,              // Playful
    speed: 0.95,             // Slightly slower
    use_speaker_boost: true
  },
  '6-12': {
    stability: 0.5,
    similarity_boost: 0.75,
    style: 0.6,
    speed: 1.0,
    use_speaker_boost: true
  },
  '13-17': {
    stability: 0.6,
    similarity_boost: 0.7,
    style: 0.4,              // More direct
    speed: 1.05,             // Slightly faster
    use_speaker_boost: true
  },
  '18-35': {
    stability: 0.65,
    similarity_boost: 0.75,
    style: 0.35,
    speed: 1.0,
    use_speaker_boost: true
  },
  '36-60': {
    stability: 0.7,
    similarity_boost: 0.8,
    style: 0.3,
    speed: 0.95,
    use_speaker_boost: true
  },
  '61-102': {
    stability: 0.75,         // More stable/calm
    similarity_boost: 0.85,
    style: 0.25,             // Warm, gentle
    speed: 0.9,              // Slower, thoughtful
    use_speaker_boost: true
  }
};
```

### Model Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `model_id` | `eleven_multilingual_v2` | High-quality multilingual |
| `output_format` | `mp3_44100_128` | CD-quality audio |
| Voice ID | `wAdymQH5YucAkXwmrdL0` | Kelly's trained voice |

---

## 2. ElevenLabs API Metadata Types

### 2.1 Standard TTS Response

The basic TTS endpoint returns audio without timing metadata:

```http
POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
Accept: audio/mpeg
```

**Response:** Raw audio bytes (MP3/WAV)

**Headers Extracted:**
- `x-character-count` - Characters processed

### 2.2 Streaming with Timestamps (Recommended for Lipsync)

For word-level timing, use the streaming endpoint with timestamps:

```http
POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream/with-timestamps
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Hello, curious learners!",
  "model_id": "eleven_multilingual_v2",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.35,
    "use_speaker_boost": true
  },
  "output_format": "mp3_44100_128"
}
```

**Response (Server-Sent Events / Chunks):**

Each chunk contains JSON with audio AND alignment data:

```json
{
  "audio_base64": "//uQxAAAAAAA...",
  "alignment": {
    "characters": ["H", "e", "l", "l", "o"],
    "character_start_times_seconds": [0.0, 0.04, 0.08, 0.12, 0.16],
    "character_end_times_seconds": [0.04, 0.08, 0.12, 0.16, 0.20]
  },
  "normalized_alignment": {
    "characters": ["H", "e", "l", "l", "o"],
    "character_start_times_seconds": [0.0, 0.04, 0.08, 0.12, 0.16],
    "character_end_times_seconds": [0.04, 0.08, 0.12, 0.16, 0.20]
  }
}
```

### 2.3 Metadata Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `alignment.characters` | `string[]` | Individual characters from input |
| `alignment.character_start_times_seconds` | `number[]` | Start time for each character |
| `alignment.character_end_times_seconds` | `number[]` | End time for each character |
| `normalized_alignment` | Same structure | Normalized to audio duration |

---

## 3. Metadata Extraction Pipeline

### 3.1 ElevenLabsMetadataProcessor Class

**Location:** `app/expression-generator.js` (lines 940-1075)

This class processes ElevenLabs API responses into usable timing data:

```javascript
export class ElevenLabsMetadataProcessor {
  /**
   * Parse ElevenLabs audio generation response
   * @param {Object} response - ElevenLabs API response
   * @returns {Object} Processed timing and emphasis data
   */
  process(response) {
    const result = {
      duration: 0,
      wordTimings: [],
      emphasisMarkers: [],
      pauseMarkers: [],
      pitchVariations: [],
    };

    // Handle alignment field (word-level timing)
    if (response.alignment) {
      result.wordTimings = this.parseAlignment(response.alignment);
      result.duration = this.calculateDuration(result.wordTimings);
    }

    // Handle normalized_alignment (emphasis detection)
    if (response.normalized_alignment) {
      result.emphasisMarkers = this.extractEmphasis(response.normalized_alignment);
    }

    // Extract from character timings
    if (response.characters) {
      const charData = this.parseCharacterTimings(response.characters);
      result.pauseMarkers = charData.pauses;
      result.pitchVariations = charData.pitchVariations;
    }

    // Fallback: estimate if no timing data
    if (result.wordTimings.length === 0 && response.text) {
      result.wordTimings = this.estimateWordTimings(response.text, response.duration || 60);
      result.duration = response.duration || 60;
    }

    return result;
  }

  parseAlignment(alignment) {
    const timings = [];
    if (Array.isArray(alignment)) {
      for (const item of alignment) {
        timings.push({
          word: item.word || item.text,
          start: item.start_time || item.start,
          end: item.end_time || item.end,
          confidence: item.confidence || 1.0,
        });
      }
    }
    return timings;
  }

  parseCharacterTimings(characters) {
    const result = { pauses: [], pitchVariations: [] };
    let lastEnd = 0;
    
    for (const char of characters) {
      const gap = (char.start_time || char.start) - lastEnd;
      
      // Detect pauses (gaps > 200ms)
      if (gap > 0.2) {
        result.pauses.push({
          timestamp: lastEnd,
          duration: gap,
          type: gap > 0.5 ? 'long' : 'short',
        });
      }

      // Track pitch variations
      if (char.pitch) {
        result.pitchVariations.push({
          timestamp: char.start_time || char.start,
          pitch: char.pitch,
        });
      }

      lastEnd = char.end_time || char.end || lastEnd + 0.05;
    }
    return result;
  }
}
```

### 3.2 Extracted Data Structure

After processing, the metadata is structured as:

```typescript
interface ProcessedMetadata {
  duration: number;           // Total audio duration in seconds
  wordTimings: WordTiming[];  // Word-level timestamps
  emphasisMarkers: Emphasis[]; // Detected emphasis points
  pauseMarkers: Pause[];      // Detected pauses
  pitchVariations: Pitch[];   // Pitch changes
}

interface WordTiming {
  word: string;
  start: number;     // Start time in seconds
  end: number;       // End time in seconds
  confidence: number; // 0.0-1.0 alignment confidence
}

interface Emphasis {
  timestamp: number;
  intensity: number; // 0.0-1.0
  word: string;
}

interface Pause {
  timestamp: number;
  duration: number;
  type: 'short' | 'long';
}
```

---

## 4. Lip-Sync Data Flow

### 4.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KELLY AVATAR DRIVING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │  Lesson      │    │   ElevenLabs     │    │   Forced Alignment      │   │
│  │  Script      │───▶│   TTS API        │───▶│   (Montreal Forced      │   │
│  │  (text)      │    │   (Kelly voice)  │    │   Aligner / Gentle)     │   │
│  └──────────────┘    └──────────────────┘    └─────────────────────────┘   │
│                                │                         │                   │
│                                ▼                         ▼                   │
│                    ┌──────────────────┐    ┌─────────────────────────┐     │
│                    │  Audio File      │    │  Phoneme Alignment      │     │
│                    │  (MP3/WAV)       │    │  [{phone, start, end}]  │     │
│                    └──────────────────┘    └─────────────────────────┘     │
│                                                          │                   │
│                                                          ▼                   │
│                                             ┌─────────────────────────┐     │
│                                             │  Phoneme → Viseme Map   │     │
│                                             │  (ARPAbet → ARKit)      │     │
│                                             └─────────────────────────┘     │
│                                                          │                   │
│                                                          ▼                   │
│                                             ┌─────────────────────────┐     │
│                                             │  Blendshape Timeline    │     │
│                                             │  [{timestamp, shapes}]  │     │
│                                             └─────────────────────────┘     │
│                                                          │                   │
│                                                          ▼                   │
│                                             ┌─────────────────────────┐     │
│                                             │  Kelly Avatar Renderer  │     │
│                                             │  (Unity/HTML/Video)     │     │
│                                             └─────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Phoneme to Viseme Mapping

**Location:** `app/lipsync/phoneme-viseme-map.js`

The system maps ARPAbet phonemes to viseme categories:

```javascript
const PHONEME_TO_VISEME = {
  // Vowels
  'AA': 'A', 'AE': 'A', 'AH': 'A',   // Wide open
  'AO': 'O', 'AW': 'O', 'OW': 'O',   // Round O
  'EH': 'E', 'EY': 'E',              // Teeth smile
  'IH': 'I', 'IY': 'I',              // Tight I
  'UH': 'U', 'UW': 'U',              // Pursed U
  'ER': 'R',                          // Retroflex
  
  // Consonants
  'P': 'M', 'B': 'M', 'M': 'M',      // Bilabial
  'F': 'F', 'V': 'F',                // Labiodental
  'TH': 'C', 'DH': 'C',              // Dental
  'T': 'C', 'D': 'C', 'N': 'C',      // Alveolar
  'S': 'C', 'Z': 'C',                // Sibilant
  'SH': 'SH', 'ZH': 'SH',            // Postalveolar
  'CH': 'SH', 'JH': 'SH',            // Affricates
  'K': 'C', 'G': 'C', 'NG': 'C',     // Velar
  'HH': 'A',                          // Glottal
  'L': 'L', 'R': 'R',                // Liquids
  'W': 'U', 'Y': 'I',                // Glides
  
  // Silence
  'SIL': 'REST', 'SP': 'REST'
};
```

### 4.3 Blendshape Generation

Each viseme maps to ARKit-compatible blendshape values:

```javascript
const VISEME_BLENDSHAPES = {
  'A': {
    jawOpen: 0.6,
    mouthOpen: 0.5,
    mouthWide: 0.3,
    tongueOut: 0.0
  },
  'O': {
    jawOpen: 0.4,
    mouthOpen: 0.6,
    mouthPucker: 0.4,
    mouthShrugUpper: 0.2
  },
  'E': {
    jawOpen: 0.2,
    mouthSmile_L: 0.4,
    mouthSmile_R: 0.4,
    mouthStretch_L: 0.3,
    mouthStretch_R: 0.3
  },
  'M': {
    jawOpen: 0.0,
    mouthClose: 0.8,
    mouthPress_L: 0.5,
    mouthPress_R: 0.5
  },
  // ... full mapping in phoneme-viseme-map.js
};
```

---

## 5. Expression Generation from Metadata

### 5.1 ExpressionGenerator Class

**Location:** `app/expression-generator.js` (lines 1084+)

The expression generator combines:
- **Text analysis** (emotion detection from script)
- **ElevenLabs metadata** (timing, emphasis)
- **Archetype profiles** (Kelly's teaching style)
- **Age adaptations** (audience-appropriate expressions)

```javascript
export class ExpressionGenerator {
  constructor(options = {}) {
    this.textAnalyzer = new TextAnalyzer();
    this.metadataProcessor = new ElevenLabsMetadataProcessor();
    this.options = {
      transitionDuration: 0.3,
      minExpressionDuration: 0.5,
      maxExpressionDuration: 5.0,
      ...options,
    };
  }

  generate(params) {
    const {
      text,
      elevenLabsResponse = null,
      archetype = 'The Scientist',
      tone = 'enthusiastic',
      ageBucket = '18-35',
      language = 'en',
      phase = 'welcome',
      totalDuration = null,
    } = params;

    // Process ElevenLabs metadata
    let audioMetadata = {
      duration: totalDuration || 60,
      wordTimings: [],
      emphasisMarkers: [],
      pauseMarkers: [],
    };

    if (elevenLabsResponse) {
      audioMetadata = this.metadataProcessor.process(elevenLabsResponse);
    }

    // Generate expressions based on metadata
    return this.buildExpressionTimeline(text, audioMetadata, {
      archetype, tone, ageBucket, language, phase
    });
  }
}
```

### 5.2 Expression Events from Metadata

The system generates expression events at key moments:

| Metadata Signal | Expression Event |
|-----------------|------------------|
| Word emphasis (pitch > 1.1) | Raised eyebrows, wider eyes |
| Long pause (> 0.5s) | Contemplative look, slight smile |
| Question mark detected | Curious expression, head tilt |
| Exclamation detected | Excited expression, bright smile |
| Phase transition | Full expression reset |

### 5.3 Conversational AI Expression Bridge

**Location:** `public/js/kelly-conversation.js`

For real-time conversations, expressions are driven by conversation state:

```javascript
const EXPRESSION_MAP = {
  'listening':  { expression: 'listening', avatar: 'attentive' },
  'thinking':   { expression: 'thinking', avatar: 'curious' },
  'speaking':   { expression: 'explaining', avatar: 'lipsync_active' },
  'idle':       { expression: 'hello', avatar: 'neutral' }
};

// Called when ElevenLabs conversation state changes
function updateKellyExpression(state) {
  const mapping = EXPRESSION_MAP[state];
  kellyAvatar.setExpression(mapping.expression);
  kellyAvatar.setAvatarState(mapping.avatar);
}
```

---

## 6. Production Architecture

### 6.1 Pre-computed Lipsync Pipeline

For lesson videos, lipsync is pre-computed during content generation:

**Pipeline Location:** `scripts/lipsync-pipeline/`

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRE-COMPUTATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Audio Generation (ElevenLabs)                               │
│     └── scripts/lipsync-pipeline/generate-lesson-audio.ts      │
│                                                                  │
│  2. Forced Alignment (Montreal Forced Aligner / Gentle)        │
│     └── scripts/forced-alignment/align_audio.py                 │
│                                                                  │
│  3. Blendshape Timeline Generation                              │
│     └── scripts/lipsync-pipeline/generate-alignments.ts        │
│                                                                  │
│  4. Database Storage                                            │
│     └── Supabase: lipsync_alignments table                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Database Schema

**Table:** `lipsync_alignments`

```sql
CREATE TABLE lipsync_alignments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  day_number INTEGER NOT NULL,
  age_bucket TEXT NOT NULL,           -- '2-5', '6-12', etc.
  language TEXT NOT NULL,              -- 'en', 'es', 'fr'
  phase TEXT NOT NULL,                 -- 'script', 'response_A', etc.
  transcript TEXT NOT NULL,
  words JSONB,                         -- Word-level timing
  phones JSONB,                        -- Phoneme-level timing
  blendshape_timeline JSONB,          -- Pre-computed blendshapes
  duration_seconds NUMERIC,
  method TEXT,                         -- 'mfa', 'gentle', 'estimation'
  confidence NUMERIC,
  fps INTEGER DEFAULT 30,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  UNIQUE(day_number, age_bucket, language, phase)
);
```

### 6.3 Runtime Playback

**Client Component:** `public/js/kelly-alignment-player.js`

```javascript
class KellyAlignmentPlayer {
  async loadAlignment(day, ageBucket, language, phase) {
    const response = await fetch(
      `/api/lipsync-alignment?day=${day}&age=${ageBucket}&lang=${language}&phase=${phase}`
    );
    const alignment = await response.json();
    
    this.words = alignment.words;
    this.phones = alignment.phones;
    this.blendshapeTimeline = alignment.blendshapeTimeline;
    this.duration = alignment.duration;
    
    return alignment;
  }

  playWithAudio(audioElement) {
    audioElement.addEventListener('timeupdate', () => {
      const currentTime = audioElement.currentTime;
      const frame = this.getFrameAtTime(currentTime);
      kellyAvatar.applyBlendshapes(frame.blendshapes);
    });
  }

  getFrameAtTime(time) {
    // Find and interpolate blendshapes at current time
    const timeline = this.blendshapeTimeline;
    const fps = 30;
    const frameIndex = Math.floor(time * fps);
    return timeline[Math.min(frameIndex, timeline.length - 1)];
  }
}
```

---

## 7. API Reference

### 7.1 TTS Generation

**Endpoint:** `POST /api/tts`

```javascript
// Request
{
  "text": "Hello, curious learner!",
  "voiceId": "wAdymQH5YucAkXwmrdL0"  // Optional, defaults to Kelly
}

// Response: audio/mpeg binary
```

### 7.2 Alignment Retrieval

**Endpoint:** `GET /api/lipsync-alignment`

```javascript
// Request
GET /api/lipsync-alignment?day=1&age=6-12&lang=en&phase=script

// Response
{
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.35, "confidence": 0.95},
    {"word": "curious", "start": 0.40, "end": 0.82, "confidence": 0.92}
  ],
  "phones": [
    {"phone": "HH", "start": 0.0, "end": 0.05, "viseme": "A"},
    {"phone": "EH", "start": 0.05, "end": 0.12, "viseme": "E"},
    {"phone": "L", "start": 0.12, "end": 0.18, "viseme": "L"},
    {"phone": "OW", "start": 0.18, "end": 0.35, "viseme": "O"}
  ],
  "blendshapeTimeline": [
    {"timestamp": 0.000, "blendshapes": {"jawOpen": 0.2, "mouthOpen": 0.1}},
    {"timestamp": 0.033, "blendshapes": {"jawOpen": 0.4, "mouthOpen": 0.3}}
  ],
  "duration": 12.5,
  "method": "mfa",
  "confidence": 0.95,
  "fps": 30,
  "meta": {
    "day": 1,
    "ageBucket": "6-12",
    "language": "en",
    "phase": "script"
  }
}
```

### 7.3 Real-time Alignment

**Endpoint:** `POST /api/align`

```javascript
// Request
{
  "transcript": "Hello everyone, welcome to today's lesson!",
  "audio_url": "https://storage.supabase.co/.../audio.mp3"
}

// Response
{
  "words": [...],
  "phones": [...],
  "duration": 5.2,
  "method": "gentle",  // or "estimation" if no aligner available
  "confidence": 0.85,
  "transcript": "Hello everyone..."
}
```

---

## 8. Implementation Checklist

### 8.1 Required Environment Variables

```bash
# ElevenLabs
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=wAdymQH5YucAkXwmrdL0  # Kelly25

# Supabase
PUBLIC_SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Optional: External Aligner
GENTLE_API_URL=http://localhost:8765  # If running Gentle locally
```

### 8.2 Generation Workflow

1. **Generate Audio**
   ```bash
   npx ts-node scripts/lipsync-pipeline/generate-lesson-audio.ts --days 1-30
   ```

2. **Generate Alignments**
   ```bash
   npx ts-node scripts/lipsync-pipeline/generate-alignments.ts --input ./generated-audio
   ```

3. **Full Pipeline**
   ```bash
   npx ts-node scripts/lipsync-pipeline/run-pipeline.ts --days 1-365
   ```

### 8.3 Quality Verification

- [ ] Verify Kelly Voice ID is correct: `wAdymQH5YucAkXwmrdL0`
- [ ] Check audio generation uses correct age-bucket settings
- [ ] Validate alignment confidence > 0.8 for production
- [ ] Test blendshape playback at 30 FPS
- [ ] Verify expression transitions are smooth (< 300ms)
- [ ] Check pause detection triggers appropriate expressions
- [ ] Validate multilingual support (EN/ES/FR)

---

## Appendix A: Voice Settings Quick Reference

| Age Bucket | Stability | Similarity | Style | Speed |
|------------|-----------|------------|-------|-------|
| 2-5        | 0.4       | 0.8        | 0.7   | 0.95  |
| 6-12       | 0.5       | 0.75       | 0.6   | 1.0   |
| 13-17      | 0.6       | 0.7        | 0.4   | 1.05  |
| 18-35      | 0.65      | 0.75       | 0.35  | 1.0   |
| 36-60      | 0.7       | 0.8        | 0.3   | 0.95  |
| 61-102     | 0.75      | 0.85       | 0.25  | 0.9   |

---

## Appendix B: Viseme Category Reference

| Viseme | Phonemes | Mouth Shape |
|--------|----------|-------------|
| A | AA, AE, AH | Wide open |
| O | AO, AW, OW, OY | Rounded O |
| E | EH, EY, IH, IY | Teeth showing |
| U | UH, UW, W | Pursed lips |
| M | P, B, M | Closed/bilabial |
| F | F, V | Lower lip in |
| L | L | Tongue up |
| R | R, ER | Retroflex |
| C | TH, T, D, N, S, Z, K, G | Alveolar/dental |
| SH | SH, ZH, CH, JH | Rounded fricative |
| REST | SIL, SP | Neutral/closed |

---

## Appendix C: File Locations

| Purpose | Path |
|---------|------|
| Voice Engine | `app/elevenlabs-voice-engine.js` |
| Metadata Processor | `app/expression-generator.js` |
| Phoneme-Viseme Map | `app/lipsync/phoneme-viseme-map.js` |
| Lipsync Orchestrator | `app/lipsync/kelly-lipsync-orchestrator.js` |
| TTS API | `api/tts.ts` |
| Alignment API | `api/align.ts` |
| Lipsync Alignment API | `api/lipsync-alignment.ts` |
| Audio Generator | `curious-kellly/golden-v2/audio-generator.js` |
| Pipeline Runner | `scripts/lipsync-pipeline/run-pipeline.ts` |
| Forced Alignment | `scripts/forced-alignment/align_audio.py` |
| Client Player | `public/js/kelly-alignment-player.js` |

---

**Document Status:** Complete  
**Maintained By:** Curious Kelly Engineering Team  
**Contact:** hello@curiouskelly.com

© 2025 Lesson of the Day PBC

