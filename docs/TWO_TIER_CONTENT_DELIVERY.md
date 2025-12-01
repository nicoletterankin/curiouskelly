# Two-Tier Content Delivery System

## Overview

The Curious Kelly platform uses a two-tier content delivery system to provide **instant playback** for daily lessons while allowing **flexible custom lesson generation** for exploration.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TWO-TIER ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TIER 1: Pre-Computed Daily Lessons (365)                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  📚 Audio files → CDN/Supabase Storage                      │   │
│  │  😊 Expression data → lesson_atoms.expression_data          │   │
│  │  ⚡ Loading: INSTANT (like Spotify)                         │   │
│  │  💰 Cost: One-time ~$2,000 for all 32,850 variants         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  TIER 2: Custom Lessons (On-Demand)                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  🎙️ Audio: Generated via ElevenLabs API                    │   │
│  │  😊 Expression: Generated via expression-generator.js       │   │
│  │  ⏳ Loading: Shows loading state (acceptable for exploration)│   │
│  │  💾 Caching: Saves to Supabase for future users            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
app/
├── phase-loader.js        # Main loading logic for both tiers
├── cache-manager.js       # Multi-tier caching (memory → IndexedDB → Supabase)
├── elevenlabs-voice-engine.js  # Audio generation with age/archetype settings
└── expression-generator.js     # AI-powered expression generation

scripts/
└── precompute-365-lessons.js   # Batch pre-computation script

prisma/migrations/
└── 20251125_two_tier_content_delivery/
    └── migration.sql      # Database schema for audio/expression storage
```

## Storage Architecture

### Recommended: Supabase Storage + CDN

```
Supabase Storage Bucket: lesson-audio
├── precomputed/                          # TIER 1: Pre-computed daily lessons
│   ├── the-sun/
│   │   ├── 2-5-en-welcome.mp3
│   │   ├── 2-5-en-q1.mp3
│   │   ├── 2-5-en-q2.mp3
│   │   ├── 2-5-en-q3.mp3
│   │   ├── 2-5-en-wisdom.mp3
│   │   ├── 2-5-es-welcome.mp3
│   │   ├── ... (90 files per lesson: 6 ages × 3 languages × 5 phases)
│   │   └── 61-102-fr-wisdom.mp3
│   ├── gravity/
│   │   └── ... (90 files)
│   └── ... (365 lessons total)
│
└── generated/                            # TIER 2: Cached custom lessons
    ├── custom-dinosaurs/
    │   └── 18-35-en-welcome.mp3
    └── ...
```

### Cost Analysis

| Item | Calculation | Cost |
|------|-------------|------|
| **Audio Generation** | 32,850 files × 200 chars × $0.30/1000 | **~$1,971** |
| **Storage** | 32,850 files × 100KB = 3.3GB | **~$0.08/month** |
| **CDN Bandwidth** | 1,000 users × 30 lessons × 5 phases × 100KB | ~$0.50/month |
| **Total One-Time** | | **~$2,000** |
| **Total Monthly** | | **~$1/month** |

### Alternative: Cloudflare R2

For higher traffic, consider Cloudflare R2:

```javascript
// Update STORAGE_CONFIG in phase-loader.js
const STORAGE_CONFIG = {
  precomputedAudioBase: 'https://lesson-audio.curiouskelly.com/precomputed',
  generatedAudioBase: 'https://lesson-audio.curiouskelly.com/generated',
  // ...
};
```

Benefits:
- Zero egress fees (vs. $0.09/GB for S3/Supabase)
- Global CDN with edge caching
- S3-compatible API

## Database Schema

### lesson_atoms (Extended)

```sql
-- Added columns for expression storage
ALTER TABLE public.lesson_atoms
ADD COLUMN expression_data JSONB DEFAULT '{}';
ADD COLUMN audio_url TEXT;

-- expression_data structure:
{
  "18-35-en": {
    "expressions": [
      { "timestamp": 0, "emotion": "warm", "intensity": 0.7 },
      { "timestamp": 2.5, "emotion": "excited", "intensity": 0.8 }
    ],
    "gestures": [
      { "timestamp": 0.5, "gesture": "open_arms_welcome", "duration": 2.0 }
    ],
    "metadata": {
      "generatedAt": "2025-11-25T10:00:00Z",
      "version": "1.0.0"
    }
  },
  "2-5-es": { ... },
  ...
}
```

### audio_cache (New Table)

```sql
CREATE TABLE public.audio_cache (
  id UUID PRIMARY KEY,
  lesson_slug TEXT NOT NULL,
  age_bucket TEXT NOT NULL,
  language TEXT NOT NULL,
  phase TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  public_url TEXT,
  file_size_bytes INTEGER,
  duration_seconds NUMERIC,
  created_at TIMESTAMPTZ,
  last_accessed_at TIMESTAMPTZ,
  UNIQUE (lesson_slug, age_bucket, language, phase)
);
```

## Phase Loading Strategy

### 1. Instant Loading for Daily Lessons

```javascript
async function loadPhase(phase) {
  const state = getState();
  
  // Daily lesson = pre-computed content
  if (state.selectedLesson.isDailyLesson) {
    // Load from CDN (instant, like Spotify)
    const audioUrl = getPrecomputedAudioUrl(state, phase);
    const expressions = getPrecomputedExpressions(state, phase);
    
    sendToUnity(audioUrl, expressions);
    return;
  }
  
  // Custom lesson = generate on-demand
  showLoadingState();
  const { audioUrl, expressions } = await generatePhaseContent(state, phase);
  hideLoadingState();
  
  // Cache for future users
  await cacheGeneratedContent(state, phase, audioUrl, expressions);
  sendToUnity(audioUrl, expressions);
}
```

### 2. Prefetching Strategy

```javascript
// Prefetch next phase when user is 75% through current
function onPhaseProgress(progress) {
  if (progress >= 0.75) {
    prefetchNextPhase();
  }
}

// When loading phase 1, prefetch phase 2 in background
async function loadWithPrefetch(currentPhase) {
  const [result] = await Promise.all([
    loadPhase(currentPhase),            // Load current (awaited)
    prefetchPhase(getNextPhase()),      // Prefetch next (background)
  ]);
  return result;
}
```

### 3. State Change Handling

```javascript
// When user changes age or language mid-lesson
async function handleStateChange(changeType, newValue) {
  cancelPrefetch(); // Cancel any pending prefetch
  
  const currentPhase = getState().currentPhase;
  
  if (isDailyLesson) {
    // Instant switch to different pre-computed variant
    const audioUrl = getPrecomputedAudioUrl(newState, currentPhase);
    sendToUnity(audioUrl, expressions);
  } else {
    // Regenerate with new settings
    showLoadingState();
    const content = await generatePhaseContent(newState, currentPhase);
    hideLoadingState();
    sendToUnity(content.audioUrl, content.expressions);
  }
}
```

## Pre-Computation Script

### Usage

```bash
# Dry run to estimate costs
node scripts/precompute-365-lessons.js --dry-run

# Process all lessons
node scripts/precompute-365-lessons.js

# Process specific range
node scripts/precompute-365-lessons.js --start-day=1 --end-day=30

# Process single language
node scripts/precompute-365-lessons.js --language=en

# Resume after interruption
node scripts/precompute-365-lessons.js --resume

# Skip audio (expressions only)
node scripts/precompute-365-lessons.js --skip-audio
```

### Environment Variables

```env
ELEVENLABS_API_KEY=sk_xxx...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJxxx...
KELLY_VOICE_ID=voice_xxx
```

### Output

```
═══════════════════════════════════════════════════════════════
  Curious Kelly - 365 Lesson Pre-Computation Script
═══════════════════════════════════════════════════════════════

📚 Found 365 lessons in database
🎯 Processing days 1 to 365 (365 lessons)
📊 Total variants to process: 32,850

📖 Processing: Day 1 - The Sun
   Slug: the-sun
   🔄 Processing: 2-5-en-welcome
   ✅ Complete: 2-5-en-welcome
   ...

═══════════════════════════════════════════════════════════════
  Pre-Computation Complete
═══════════════════════════════════════════════════════════════

Summary:
  ✅ Processed: 32,850
  🎵 Audio files: 32,850
  😊 Expression sets: 32,850
  📝 Total characters: 6,570,000
  💰 Estimated cost: $1,971.00
  ⏱️  Duration: 54,750 seconds (~15 hours)
```

## Caching Strategy

### Multi-Tier Cache (client-side)

```
┌────────────────────────────────────────────────────────────┐
│                    CLIENT CACHING                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Layer 1: Memory Cache (Map)                              │
│  ├── Fastest access                                       │
│  ├── Volatile (cleared on page refresh)                   │
│  └── Max 50 entries, LRU eviction                         │
│                      ↓                                     │
│  Layer 2: IndexedDB Cache                                 │
│  ├── Persistent browser storage                           │
│  ├── 7-day TTL for audio, 30-day for expressions         │
│  └── Max 100 audio, 500 expression entries               │
│                      ↓                                     │
│  Layer 3: Supabase Storage (Global)                       │
│  ├── Shared across all users                              │
│  ├── Pre-computed = permanent                             │
│  └── Generated = cached for future users                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Cache Key Format

```javascript
// Pre-computed daily lesson
`precomputed-${lessonSlug}-${ageBucket}-${language}-${phase}`
// Example: precomputed-the-sun-18-35-en-welcome

// Generated custom lesson
`generated-${lessonSlug}-${ageBucket}-${language}-${phase}`
// Example: generated-custom-dinosaurs-6-12-es-q1
```

## API Reference

### PhaseLoader

```javascript
import PhaseLoader from './app/phase-loader.js';

const loader = new PhaseLoader({
  stateManager,
  unityBridge,
  elevenLabsApiKey: 'sk_xxx',
  voiceId: 'Kelly_v1',
  onLoadingStart: (phase) => showSpinner(phase),
  onLoadingEnd: (phase, result, error) => hideSpinner(),
});

// Load a phase
await loader.loadPhase('welcome');

// Handle state changes
await loader.handleStateChange('age', 25);
await loader.handleStateChange('language', 'es');

// Get statistics
const stats = loader.getStats();
console.log(`Cache hit rate: ${stats.cacheHitRate}`);
```

### CacheManager

```javascript
import CacheManager from './app/cache-manager.js';

const cache = new CacheManager();

// Store and retrieve
await cache.set('my-key', data, 'lessons');
const cached = await cache.get('my-key', 'lessons');

// Audio-specific methods
const url = await cache.cacheAudio('audio-key', audioBlob);
const audio = await cache.getAudio('audio-key');

// Statistics
const stats = cache.getStats();
console.log(`Memory hit rate: ${stats.memoryHitRate}`);
```

## Performance Targets

| Metric | TIER 1 (Pre-computed) | TIER 2 (Generated) |
|--------|----------------------|-------------------|
| Phase load time | < 200ms | < 5s |
| Age/language switch | < 100ms | < 3s |
| Audio start | Instant | After generation |
| First byte | < 50ms | N/A |

## Monitoring

### Key Metrics to Track

```javascript
// In phase-loader.js
const stats = {
  tier1Loads: 0,        // Pre-computed loads
  tier2Loads: 0,        // Generated loads
  cacheHits: 0,         // Local cache hits
  cacheMisses: 0,       // Required generation
  avgLoadTimeMs: 0,     // Average load time
  prefetchHits: 0,      // Successful prefetches
};
```

### Analytics Events

```javascript
// Track in analytics
analytics.track('phase_loaded', {
  tier: result.metadata.tier,
  type: result.metadata.type,
  loadTimeMs: loadTime,
  fromCache: result.metadata.fromCache,
  phase: phase,
  ageBucket: state.ageBucket,
  language: state.language,
});
```

## Migration Steps

1. **Run database migration**
   ```bash
   psql $DATABASE_URL -f prisma/migrations/20251125_two_tier_content_delivery/migration.sql
   ```

2. **Create storage bucket**
   - Go to Supabase Dashboard → Storage
   - Create bucket: `lesson-audio`
   - Set to public
   - Set CORS policy for your domain

3. **Pre-compute content**
   ```bash
   # First, dry run to verify
   node scripts/precompute-365-lessons.js --dry-run
   
   # Then run for real
   node scripts/precompute-365-lessons.js
   ```

4. **Update frontend**
   ```javascript
   // In your main app initialization
   import PhaseLoader from './app/phase-loader.js';
   
   const phaseLoader = new PhaseLoader({
     stateManager,
     unityBridge,
   });
   
   // Replace existing phase loading logic
   window.loadPhase = (phase) => phaseLoader.loadPhase(phase);
   ```

## Troubleshooting

### Audio Not Loading

1. Check storage bucket permissions
2. Verify audio URL format matches storage path
3. Check browser console for CORS errors

### Expressions Not Animating

1. Verify expression_data exists in lesson_atoms
2. Check age-bucket-language key matches state
3. Ensure Unity bridge is receiving events

### High Costs

1. Enable request deduplication in voiceEngine
2. Check cache is working (should see cache hits)
3. Review prefetch settings (avoid over-fetching)

## See Also

- [CLAUDE.md](../CLAUDE.md) - Operating rules
- [UNITY_INTEGRATION_PLAN.md](../UNITY_INTEGRATION_PLAN.md) - Unity bridge documentation
- [TECHNICAL_ALIGNMENT_MATRIX.md](../TECHNICAL_ALIGNMENT_MATRIX.md) - Asset-to-requirement mapping








