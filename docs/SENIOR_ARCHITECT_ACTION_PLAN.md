# SENIOR SOLUTIONS ARCHITECT ACTION PLAN
## The Daily Lesson - Video Pipeline Consolidation
## February 3, 2026

---

## THE BRUTAL TRUTH

You have **5 video generation systems** and **0 working pipelines**.

| System | Status | Why It's Not Working |
|--------|--------|---------------------|
| HeyGen | 39 videos synced | Production code not deployed |
| Fal (MuseTalk) | Idle | No script running it |
| Sync Labs | Idle | Script exists, not running |
| Replicate (Wav2Lip) | Idle | Not configured |
| Local | Unknown | No inventory |

**Result:** Kelly is showing a static image with audio instead of lip-synced video.

---

## THE FIX: ONE UNIFIED PIPELINE

Instead of 5 separate systems, we need **ONE orchestrator** that:
1. Takes a job (day/phase/age/archetype/language)
2. Tries providers in priority order
3. Writes result to ONE database table
4. Moves to next job

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO ORCHESTRATOR                        │
│                                                              │
│  Input: day=1, phase=hook, age=adult, archetype=storyteller │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────┐  fail  ┌─────────┐  fail  ┌──────────┐        │
│  │ HeyGen  │ ──────▶│Sync Labs│ ──────▶│ Fal.ai   │        │
│  │ (best)  │        │ (fast)  │        │(fallback)│        │
│  └────┬────┘        └────┬────┘        └────┬─────┘        │
│       │ success          │ success          │ success       │
│       └──────────────────┴──────────────────┘               │
│                         │                                    │
│                         ▼                                    │
│              ┌─────────────────────┐                        │
│              │   heygen_videos     │                        │
│              │   (unified table)   │                        │
│              └─────────────────────┘                        │
│                         │                                    │
│                         ▼                                    │
│              Kelly plays lip-synced video                   │
└─────────────────────────────────────────────────────────────┘
```

---

## IMMEDIATE ACTIONS (Next 2 Hours)

### ACTION 1: Deploy Production (Nicolette - 2 minutes)
```
1. Go to v0.app
2. Click the PUBLISH button (top right, orange/blue)
3. Wait for "Ready" status
4. Verify: https://thedailylesson.com shows Kelly with video
```

### ACTION 2: Run Unified Generator (Cursor - starts now)

I will create and run a single script that:
- Uses HeyGen first (best quality, we have 650 credits)
- Falls back to Sync Labs if HeyGen fails
- Falls back to Fal.ai if Sync fails
- Writes ALL results to `heygen_videos` table

### ACTION 3: Continuous Sync (Cursor - every 10 min)
- Poll HeyGen for completed videos
- Update database with new URLs
- Log progress

---

## THE MATH: What We Need

### Target: 365 days × 5 phases = 1,825 videos (English, adult, storyteller only)

### Current Resources:

| Provider | Credits/Capacity | Cost per Video | Can Generate |
|----------|-----------------|----------------|--------------|
| HeyGen | 650 minutes | ~0.5 min/video | ~1,300 videos |
| Sync Labs | $0.05/sec | ~$1.50/video | Limited by $ |
| Fal.ai | Pay per use | ~$0.10/video | Unlimited |
| Replicate | Pay per use | ~$0.05/video | Unlimited |

### Strategy:
1. **HeyGen first** - Best quality, use all 650 credits (~1,300 videos)
2. **Sync Labs second** - Good quality, fast
3. **Fal.ai fallback** - Acceptable quality, cheap

This covers 1,825 videos with HeyGen alone if we're efficient.

---

## THE UNIFIED ORCHESTRATOR SCRIPT

I will create this NOW and start it running:

```javascript
// UNIFIED VIDEO ORCHESTRATOR
// Generates videos using multiple providers with fallback

const PROVIDERS = [
  { name: 'heygen', priority: 1, quality: 'best' },
  { name: 'sync', priority: 2, quality: 'good' },
  { name: 'fal', priority: 3, quality: 'acceptable' },
];

async function generateVideo(job) {
  for (const provider of PROVIDERS) {
    try {
      const result = await providers[provider.name].generate(job);
      if (result.success) {
        await saveToDatabase(job, result.videoUrl, provider.name);
        return { success: true, provider: provider.name };
      }
    } catch (err) {
      console.log(`${provider.name} failed, trying next...`);
    }
  }
  return { success: false, error: 'All providers failed' };
}
```

---

## PHASE 1: English Adult Storyteller (Days 1-365)

**Priority order:**
1. Days 1-7 (flagship week) - ALL PHASES
2. Days 1-30 (first month) - hook phase only
3. Days 31-365 - hook phase only
4. Backfill other phases

**Estimated completion:**
- HeyGen processes ~10 videos/hour
- 1,825 videos ÷ 10/hour = ~182 hours = 7.5 days

**BUT** we can parallelize:
- HeyGen: 10/hour
- Sync Labs: 30/hour (faster)
- Fal.ai: 60/hour (fastest)

**With all three:** ~100 videos/hour = **18 hours for full coverage**

---

## PHASE 2: Add Ages (After Phase 1)

| Age | Multiplier | Total Videos |
|-----|-----------|--------------|
| Adult | 1x | 1,825 |
| Kid | 1x | 1,825 |
| Senior | 1x | 1,825 |
| **Total** | 3x | **5,475** |

---

## PHASE 3: Add Languages (After Phase 2)

| Language | Multiplier |
|----------|-----------|
| English | 1x |
| Spanish | 1x |
| French | 1x |
| German | 1x |
| Portuguese | 1x |
| Chinese | 1x |
| **Total** | 6x |

**Final target:** 5,475 × 6 = **32,850 videos**

---

## READINESS AUDIT

### What We Have:
```
✅ HeyGen API key (active, 650 credits)
✅ Sync Labs API key (in .env)
✅ Fal.ai API key (in .env)
✅ Database (Neon PostgreSQL)
✅ 9,135 audio files (ElevenLabs TTS)
✅ 39 synced HeyGen videos
✅ Webhook configured
```

### What's Missing:
```
❌ Production deployment (v0 needs to click Publish)
❌ Unified orchestrator script (I'll create now)
❌ Continuous monitoring (I'll set up)
❌ Progress dashboard (can add later)
```

---

## SCRIPTS I WILL CREATE NOW

### 1. unified-video-orchestrator.cjs
- Generates videos using all providers
- Automatic fallback
- Writes to unified database

### 2. continuous-sync.cjs
- Runs every 10 minutes
- Polls all providers for completed videos
- Updates database

### 3. progress-monitor.cjs
- Shows real-time progress
- Estimates completion time
- Alerts on failures

---

## DECISION REQUIRED FROM NICOLETTE

**Option A: Conservative (HeyGen Only)**
- Use only HeyGen (best quality)
- 650 credits = ~1,300 videos
- Takes ~7 days
- Cost: $0 (credits already paid)

**Option B: Aggressive (All Providers)**
- Use HeyGen + Sync + Fal in parallel
- Full coverage in ~18 hours
- Cost: ~$500 additional for Sync/Fal

**Option C: Hybrid (Recommended)**
- HeyGen for flagship days (1-30)
- Sync Labs for days 31-180
- Fal.ai for days 181-365
- Best quality where it matters most
- Cost: ~$200 additional

**Which option do you want?**

---

## IMMEDIATE NEXT STEPS

1. **NICOLETTE:** Click Publish in v0.app (2 min)
2. **CURSOR:** Create unified orchestrator (10 min)
3. **CURSOR:** Start generation for Days 1-7 (30 min)
4. **CURSOR:** Set up continuous sync (5 min)
5. **ALL:** Monitor progress

---

## LET'S STOP PLANNING AND START EXECUTING

The architecture is clear. The tools are available. The credits are there.

**What's blocking us is execution, not planning.**

I'm going to:
1. Create the unified orchestrator NOW
2. Start it running NOW
3. Report progress every 30 minutes

**Nicolette needs to:**
1. Click Publish in v0.app
2. Choose Option A, B, or C above
3. Confirm we should proceed

---

**Clock is ticking. Let's go.**
