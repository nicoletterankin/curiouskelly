# Kelly Lipsync Production Pipeline

Complete pipeline for generating pre-computed lipsync alignments for Kelly's lessons.

## Overview

This pipeline creates high-quality lipsync data by:
1. **Generating audio** using ElevenLabs (Kelly's trained voice)
2. **Computing phoneme alignments** using Montreal Forced Aligner (or estimation fallback)
3. **Creating blendshape timelines** for frame-by-frame playback
4. **Storing everything** in Supabase for efficient retrieval

## Quick Start

### Prerequisites

```bash
# Install dependencies
npm install

# Required environment variables
ELEVENLABS_API_KEY=your_key
ELEVENLABS_VOICE_ID=kelly_voice_id  # Optional, uses default
PUBLIC_SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
```

### Run the Pipeline

```bash
# Full pipeline for Days 1-30 (all 6 age buckets, English)
npx ts-node scripts/lipsync-pipeline/run-pipeline.ts --days 1-30

# Dry run to see what would happen
npx ts-node scripts/lipsync-pipeline/run-pipeline.ts --days 1-5 --dry-run

# Single day, single age
npx ts-node scripts/lipsync-pipeline/run-pipeline.ts --day 1 --ages 6-12

# Skip audio generation (rerun alignment only)
npx ts-node scripts/lipsync-pipeline/run-pipeline.ts --skip-audio
```

## Pipeline Components

### 1. Audio Generation (`generate-lesson-audio.ts`)

Generates MP3 audio files using ElevenLabs API.

```bash
npx ts-node scripts/lipsync-pipeline/generate-lesson-audio.ts \
  --days 1-30 \
  --ages all \
  --lang en
```

**Output:**
- `./generated-audio/day-{N}/` - Audio files per day
- `./generated-audio/manifest.json` - Processing manifest

**Naming Convention:**
```
{day}_{age-bucket}_{lang}_{phase}.mp3
Example: 1_6-12_en_script.mp3
```

### 2. Alignment Generation (`generate-alignments.ts`)

Processes audio files to create phoneme alignments.

```bash
npx ts-node scripts/lipsync-pipeline/generate-alignments.ts \
  --input ./generated-audio
```

**Output:**
- `./generated-alignments/day-{N}/` - Alignment JSON files
- Stored in `lipsync_alignments` table in Supabase

### 3. Combined Runner (`run-pipeline.ts`)

Orchestrates both steps with error handling.

```bash
npx ts-node scripts/lipsync-pipeline/run-pipeline.ts --days 1-30
```

## Database Schema

### `lipsync_alignments` Table

```sql
CREATE TABLE lipsync_alignments (
  id UUID PRIMARY KEY,
  day_number INTEGER NOT NULL,
  age_bucket TEXT NOT NULL,      -- '2-5', '6-12', etc.
  language TEXT NOT NULL,         -- 'en', 'es', 'fr'
  phase TEXT NOT NULL,            -- 'script', 'response_A', etc.
  transcript TEXT NOT NULL,
  words JSONB,                    -- Word-level timing
  phones JSONB,                   -- Phoneme-level timing
  blendshape_timeline JSONB,      -- Pre-computed blendshapes
  duration_seconds NUMERIC,
  method TEXT,                    -- 'mfa', 'estimation'
  confidence NUMERIC,
  fps INTEGER DEFAULT 30
);
```

## API Endpoint

### `GET /api/lipsync-alignment`

Fetch pre-computed alignment for a lesson segment.

```
GET /api/lipsync-alignment?day=1&age=6-12&lang=en&phase=script
```

**Response:**
```json
{
  "words": [{"word": "Hello", "start": 0.0, "end": 0.35}],
  "phones": [{"phone": "HH", "start": 0.0, "end": 0.05, "viseme": "A"}],
  "blendshapeTimeline": [{"timestamp": 0, "blendshapes": {"jawOpen": 30}}],
  "duration": 12.5,
  "method": "mfa",
  "confidence": 0.95
}
```

## Client Integration

### Using KellyAlignmentPlayer

```html
<script src="/js/kelly-alignment-player.js"></script>
<script>
  const player = new KellyAlignmentPlayer();
  
  // Load alignment for Day 1, ages 6-12
  await player.loadAlignment(1, '6-12', 'en', 'script');
  
  // Play synchronized with audio
  const audio = document.getElementById('lesson-audio');
  player.playWithAudio(audio);
  audio.play();
</script>
```

### Preloading Multiple Segments

```javascript
// Preload all segments for a lesson
await player.preloadAlignments([
  { day: 1, ageBucket: '6-12', language: 'en', phase: 'script' },
  { day: 1, ageBucket: '6-12', language: 'en', phase: 'response_A' },
  { day: 1, ageBucket: '6-12', language: 'en', phase: 'response_B' },
  { day: 1, ageBucket: '6-12', language: 'en', phase: 'response_C' },
]);
```

## Cost Estimation

### Audio Generation (ElevenLabs)

- **Per file:** ~$0.015-$0.03 (depending on length)
- **Per day (6 ages × 4 phases):** ~$0.36-$0.72
- **30 days:** ~$10-$22
- **365 days:** ~$130-$260

### Alignment Processing

- **Montreal Forced Aligner:** Free (runs locally)
- **Estimation fallback:** Free (pure computation)

## File Structure

```
scripts/lipsync-pipeline/
├── README.md                   # This file
├── run-pipeline.ts             # Main orchestrator
├── generate-lesson-audio.ts    # Audio generation
└── generate-alignments.ts      # Alignment processing

generated-audio/                 # Created by pipeline
├── day-1/
│   ├── 1_2-5_en_script.mp3
│   ├── 1_2-5_en_response_A.mp3
│   ├── ...
├── day-2/
├── ...
└── manifest.json

generated-alignments/            # Created by pipeline
├── day-1/
│   ├── 1_2-5_en_script_alignment.json
│   ├── ...
└── ...

api/
└── lipsync-alignment.ts        # API endpoint

public/js/
└── kelly-alignment-player.js   # Client-side player
```

## Troubleshooting

### "ELEVENLABS_API_KEY not set"
Set the environment variable or add to `.env`:
```
ELEVENLABS_API_KEY=your_key_here
```

### "No shard found for day X"
The lesson content isn't in the database. Check:
```sql
SELECT * FROM lesson_shards ls
JOIN core_lessons cl ON cl.id = ls.core_lesson_id
WHERE cl.day_number = X AND ls.region = 'en';
```

### MFA Not Available
The pipeline will fall back to estimation. Estimation quality is acceptable but not perfect. To install MFA:
```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

### Rate Limiting
ElevenLabs has rate limits. The pipeline includes delays between requests. If you hit limits, reduce the batch size:
```bash
npx ts-node run-pipeline.ts --days 1-5  # Then 6-10, etc.
```

## Next Steps After Running

1. **Upload Audio to CDN**
   ```bash
   # Upload to Supabase Storage or Cloudflare R2
   # Update lesson_assets table with URLs
   ```

2. **Test Playback**
   - Open lesson player
   - Verify alignment loads
   - Check lipsync quality

3. **Monitor Performance**
   - Check API response times
   - Verify cache hits
   - Monitor blendshape frame rate

---

Built for Curious Kelly © 2025 Lesson of the Day PBC

