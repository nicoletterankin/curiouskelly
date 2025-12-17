# HeyGen Multi-Motion Video Generation System

## Mission
Build a production-ready video generation pipeline that creates natural, non-robotic Kelly videos by intelligently stitching multiple motion variants together, eliminating the "uncanny valley" effect caused by HeyGen's 10-second base motion loop.

---

## The Problem We're Solving

HeyGen's Talking Photo feature applies a 10-second "Kling engine" base motion treatment to static images. When generating videos longer than 10 seconds, this motion **loops visibly** - Kelly's head subtly shakes or resets every 10 seconds, creating an unnatural, robotic feel that breaks immersion.

**The Solution:** Use HeyGen's multi-scene `video_inputs` array to stitch together multiple shorter clips (each under 10 seconds), each using a DIFFERENT avatar ID with a unique base motion. This creates 30+ seconds of varied, natural movement.

---

## Assets & Resources

### Motion Library (COMPLETE)
**File:** `generated-images/kelly-motion-library.json`

Contains 36 unique HeyGen avatar IDs:
- **12 archetypes:** scientist, explorer, rebel, architect, diplomat, empath, macgyver, mystic, provider, storyteller, strategist, survivor
- **3 motions each:**
  - **A** = "Warm Welcoming" (hooks, intros) - friendly, inviting energy
  - **B** = "Talk Talk Talk" (main teaching) - engaged, expressive teaching
  - **C** = "Filler" (transitions, outros) - settled, grounded presence

### Lesson Content
**File pattern:** `public/lessons/day-{N}.json`

Each lesson contains phases with scripts and durations:
```json
{
  "phases": {
    "hook": { "script": "...", "duration": 8 },
    "cliff": { "script": "...", "duration": 11 },
    "fact1": { "script": "...", "duration": 16 },
    "fact2": { "script": "...", "duration": 14 },
    "fact3": { "script": "...", "duration": 12 },
    "wisdom": { "script": "...", "duration": 15 },
    "outro": { "script": "...", "duration": 9 }
  }
}
```

### Kelly Voice
**HeyGen Voice ID:** `0015ce4f932b405b9fc3a5e2f5e92c46`
(Kelly's trained ElevenLabs voice cloned into HeyGen)

### Motion Prompts Reference
**File:** `docs/KELLY_36_MOTION_PROMPTS.md`
Contains the specific motion prompts used to create each avatar variant. Useful for understanding the personality/phase context of each motion.

---

## Technical Specifications

### HeyGen API Endpoint
```
POST https://api.heygen.com/v2/video/generate
Authorization: Bearer {HEYGEN_API_KEY}
```

### Multi-Scene Request Structure
```json
{
  "video_inputs": [
    {
      "character": {
        "type": "talking_photo",
        "talking_photo_id": "{AVATAR_ID_FROM_LIBRARY}"
      },
      "voice": {
        "type": "text",
        "input_text": "{SCRIPT_SEGMENT}",
        "voice_id": "0015ce4f932b405b9fc3a5e2f5e92c46"
      },
      "background": {
        "type": "color",
        "value": "#1a1a2e"
      }
    },
    // ... more scenes
  ],
  "dimension": { "width": 1920, "height": 1080 }
}
```

### Critical Timing Rules
- **MAX_SCENE_SECONDS = 8** — Cut BEFORE the 10-second loop seam (buffer for safety)
- **Speech rate:** ~150 words per minute (~2.5 words/second)
- **Natural breaks:** Split at sentence ends (. ! ?) or clause breaks (, ; :)

---

## Motion Rotation Strategy

### By Phase Type:
| Phase Type | Motion Sequence | Rationale |
|------------|-----------------|-----------|
| hook | A → B → A | Open warm, get engaging, back to warm |
| cliff | B → A → B | Provocative question energy |
| fact1-3 | B → C → B | Teaching with grounded pauses |
| wisdom | A → C → A | Warm insight with centered reflection |
| outro | A → C → A | Warm close with settled ending |

### Implementation Logic:
```typescript
function getMotionForSegment(phaseType: string, segmentIndex: number, archetype: string): string {
  const library = loadMotionLibrary();
  const patterns = {
    hook: ['A', 'B', 'A'],
    cliff: ['B', 'A', 'B'],
    fact: ['B', 'C', 'B'],
    wisdom: ['A', 'C', 'A'],
    outro: ['A', 'C', 'A']
  };
  const pattern = patterns[phaseType] || patterns.fact;
  const motionKey = pattern[segmentIndex % pattern.length];
  return library[archetype][motionKey];
}
```

---

## Script Splitting Algorithm

```typescript
function splitScript(script: string, maxSeconds: number = 8): string[] {
  const words = script.split(' ');
  const wordsPerSecond = 2.5;
  const maxWords = Math.floor(maxSeconds * wordsPerSecond); // ~20 words per segment
  
  const segments: string[] = [];
  let currentSegment: string[] = [];
  
  for (const word of words) {
    currentSegment.push(word);
    
    // Check if we've hit capacity AND we're at a natural break
    if (currentSegment.length >= maxWords) {
      const text = currentSegment.join(' ');
      if (endsWithBreak(text)) {
        segments.push(text);
        currentSegment = [];
      }
    }
  }
  
  // Don't forget remaining words
  if (currentSegment.length > 0) {
    segments.push(currentSegment.join(' '));
  }
  
  return segments;
}

function endsWithBreak(text: string): boolean {
  return /[.!?;,:]$/.test(text.trim());
}
```

---

## Deliverables

### 1. Core Generator Script
**File:** `scripts/heygen-video-generator.ts`

Features:
- Reads motion library from JSON
- Reads lesson content from day JSON
- Splits scripts intelligently at natural breaks
- Rotates motions based on phase type
- Handles API errors with retry logic
- Logs video IDs for status polling
- Saves output manifest for tracking

### 2. Status Poller Script
**File:** `scripts/heygen-check-status.ts`

Features:
- Takes video ID(s) as input
- Polls HeyGen API for completion
- Downloads completed videos to local folder
- Updates manifest with final URLs

### 3. Batch Generator
**File:** `scripts/heygen-batch-generate.ts`

Features:
- Generates videos for all 12 archetypes for a given day
- Rate limiting to respect API quotas
- Progress tracking with resume capability
- Cost estimation before running

### 4. Output Manifest
**File:** `generated-videos/day-{N}-manifest.json`

Structure:
```json
{
  "day": 351,
  "generated": "2025-12-17T...",
  "videos": {
    "scientist": {
      "video_id": "...",
      "status": "completed",
      "url": "https://...",
      "phases": ["hook", "cliff", "fact1", "fact2", "fact3", "wisdom", "outro"],
      "total_scenes": 14,
      "credits_used": 2.5
    }
  }
}
```

---

## Quality Checklist

Before submitting any video generation:

- [ ] All avatar IDs are valid (32-char hex strings from motion library)
- [ ] Voice ID is correct: `0015ce4f932b405b9fc3a5e2f5e92c46`
- [ ] No single scene exceeds 8 seconds of speech
- [ ] Script splits occur at natural language breaks
- [ ] Motion rotation follows the phase-appropriate pattern
- [ ] API key is loaded from environment (never hardcoded)
- [ ] Error handling includes retry with exponential backoff
- [ ] Output includes video IDs for status tracking

---

## Constraints (From CLAUDE.md)

- **Never use browser TTS** — All audio via HeyGen's voice synthesis
- **Respect rate limits** — Batch requests, add delays between calls
- **Cache and reuse** — Don't regenerate existing videos
- **No secrets in code** — API keys from .env only
- **Cost awareness** — Log credit usage, estimate before batch runs

---

## Example Usage

```bash
# Generate single archetype video for Day 351
npx ts-node scripts/heygen-video-generator.ts --day 351 --archetype scientist

# Check video status
npx ts-node scripts/heygen-check-status.ts --video-id abc123

# Batch generate all archetypes for Day 351
npx ts-node scripts/heygen-batch-generate.ts --day 351

# Estimate costs before running
npx ts-node scripts/heygen-batch-generate.ts --day 351 --dry-run
```

---

## Success Criteria

A successful implementation will:

1. **Eliminate the uncanny valley** — No visible motion loop at 10-second marks
2. **Produce natural transitions** — Scene cuts feel like natural pauses, not jumps
3. **Be deterministic** — Same inputs produce same motion sequence
4. **Be efficient** — Minimize API calls and credit usage
5. **Be observable** — Clear logging, progress tracking, cost reporting
6. **Be resumable** — Can pick up where it left off if interrupted

---

## Start Here

1. Read `generated-images/kelly-motion-library.json` to understand the avatar ID structure
2. Read `public/lessons/day-351.json` to see the script content and phase durations
3. Review `docs/KELLY_36_MOTION_PROMPTS.md` to understand the motion personalities
4. Build `scripts/heygen-video-generator.ts` following this specification
5. Test with a single phase (e.g., `hook`) before running full generation
6. Verify the output video has smooth, varied motion
7. Scale to batch generation once single-video quality is confirmed
