# REALISTIC VIDEO ACTION PLAN

**Date:** February 3, 2026  
**Status:** Ground Truth Established  
**Source:** V0-Cursor Sync Document

---

## THE BRUTAL TRUTH

| What | Count | Status |
|------|-------|--------|
| Lip-synced videos we HAVE | 40 | Day 18 + Day 20 only |
| Audio files we HAVE | 9,135 | Ready to lip-sync |
| Videos we NEED (full) | 394,200 | 99.99% missing |
| HeyGen credits remaining | 668 | NOT ENOUGH |
| Budget remaining | ~$0 | CRITICAL |

**Gap: 394,160 videos**

---

## TIERED APPROACH (Reality-Based)

### TIER 1: Minimum Viable Kelly (TODAY)
**Goal:** Kelly teaches with audio + fallback video

| Metric | Value |
|--------|-------|
| Audio coverage | 9,135 files |
| Video | Base fallback (single Kelly video) |
| Quality | B- (audio works, lips don't sync) |
| Cost | $0 |

**This is what we have NOW.** Kelly speaks with correct audio, but her lips don't match.

---

### TIER 2: English Adult Hook (THIS WEEK)
**Goal:** Perfect lip-sync for the most important videos

| Dimension | Value | Count |
|-----------|-------|-------|
| Days | 365 | 365 |
| Phases | hook only | 1 |
| Ages | adult only | 1 |
| Languages | EN only | 1 |
| **TOTAL** | | **365 videos** |

**Why hook only?**
- Hook is the first thing users see
- If hook is good, they're hooked
- Other phases can use fallback

**Cost estimate:**
- Sync Labs: 365 × $0.20 = **$73**
- OR MuseTalk (local GPU): **$0**

---

### TIER 3: English All Phases (NEXT WEEK)
**Goal:** Full lesson experience in English

| Dimension | Value | Count |
|-----------|-------|-------|
| Days | 365 | 365 |
| Phases | all 5 | 5 |
| Ages | adult only | 1 |
| Languages | EN only | 1 |
| **TOTAL** | | **1,825 videos** |

**Cost estimate:**
- Sync Labs: 1,825 × $0.20 = **$365**
- OR MuseTalk (local GPU): **$0** + 60 hours compute

---

### TIER 4: English All Ages (WEEK 2)
**Goal:** Kid, Adult, Elder in English

| Dimension | Value | Count |
|-----------|-------|-------|
| Days | 365 | 365 |
| Phases | all 5 | 5 |
| Ages | 3 | 3 |
| Languages | EN only | 1 |
| **TOTAL** | | **5,475 videos** |

**Cost estimate:**
- Sync Labs: 5,475 × $0.20 = **$1,095**

---

### TIER 5: Full Multilingual (MONTH 2+)
**Goal:** 6 languages × 3 ages × all phases

| Dimension | Value | Count |
|-----------|-------|-------|
| Days | 365 | 365 |
| Phases | all 5 | 5 |
| Ages | 3 | 3 |
| Languages | 6 | 6 |
| **TOTAL** | | **32,850 videos** |

**Cost estimate:**
- Sync Labs: 32,850 × $0.20 = **$6,570**

---

## IMMEDIATE ACTIONS (NEXT 24 HOURS)

### 1. Download Existing HeyGen Videos (URGENT)
The 40 HeyGen videos point to `files.heygen.ai` - these URLs may expire.

```bash
# Run this script
npx tsx scripts/download-heygen-videos.ts
```

**Script needs to:**
- Query `kelly_lesson_assets` for `video_url LIKE '%files.heygen.ai%'`
- Download each video
- Upload to Vercel Blob
- Update database with new URL

### 2. Verify Audio Coverage
```sql
-- Run via Neon console or script
SELECT 
  day_number,
  COUNT(*) as records,
  COUNT(audio_url) as with_audio,
  COUNT(video_url) as with_video
FROM kelly_lesson_assets
WHERE day_number <= 30
GROUP BY day_number
ORDER BY day_number;
```

### 3. Test Sync Labs Pipeline
Single video test before batch:
```bash
npx tsx scripts/test-sync-labs-single.ts --day 1 --phase hook --age adult
```

---

## SCRIPT: Download HeyGen Videos

```typescript
// scripts/download-heygen-videos.ts
import { createClient } from '@supabase/supabase-js';
import { put } from '@vercel/blob';

const NEON_URL = process.env.DATABASE_URL!;

async function downloadHeyGenVideos() {
  // 1. Query videos with HeyGen URLs
  const result = await fetch(NEON_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: `
        SELECT id, day_number, phase, age_group, video_url 
        FROM kelly_lesson_assets 
        WHERE video_url LIKE '%files.heygen.ai%'
      `
    })
  });
  
  const videos = await result.json();
  console.log(`Found ${videos.length} HeyGen videos to download`);
  
  for (const video of videos) {
    try {
      // 2. Download from HeyGen
      const response = await fetch(video.video_url);
      const blob = await response.blob();
      
      // 3. Upload to Vercel Blob
      const filename = `video/kelly/day-${String(video.day_number).padStart(3, '0')}/${video.phase}-${video.age_group}.mp4`;
      const uploaded = await put(filename, blob, { access: 'public' });
      
      // 4. Update database
      await fetch(NEON_URL, {
        method: 'POST',
        body: JSON.stringify({
          query: `UPDATE kelly_lesson_assets SET video_url = $1 WHERE id = $2`,
          params: [uploaded.url, video.id]
        })
      });
      
      console.log(`✓ ${video.day_number}/${video.phase}/${video.age_group}`);
    } catch (err) {
      console.error(`✗ ${video.day_number}: ${err.message}`);
    }
  }
}

downloadHeyGenVideos();
```

---

## COST COMPARISON

| Provider | Quality | Cost/Video | 365 Videos | 1,825 Videos | 32,850 Videos |
|----------|---------|------------|------------|--------------|---------------|
| HeyGen | A+ | ~1 credit | 365 credits | 1,825 credits | N/A (out of credits) |
| Sync Labs | A | $0.20 | $73 | $365 | $6,570 |
| fal.ai | B+ | $0.05 | $18 | $91 | $1,642 |
| MuseTalk (local) | B | $0 (GPU time) | 12 hrs | 60 hrs | 1,095 hrs |
| Wav2Lip | B- | $0.10 | $36 | $182 | $3,285 |

**Recommendation:** 
- Use remaining 668 HeyGen credits for **Day 1-7 hook videos** (flagship)
- Use Sync Labs for bulk generation ($365 for EN adult all phases)
- Fall back to MuseTalk for budget overflow

---

## PRIORITY ORDER

1. **NOW:** Download 40 existing HeyGen videos
2. **TODAY:** Generate Day 1 all phases (test pipeline)
3. **THIS WEEK:** Generate Days 1-30 hook phase
4. **NEXT WEEK:** Generate all 365 days hook phase
5. **WEEK 3:** Expand to all phases
6. **MONTH 2:** Add ages and languages

---

## SUCCESS METRICS

| Milestone | Target | Measure |
|-----------|--------|---------|
| Day 1 | 40 videos preserved | Download complete |
| Week 1 | 365 hook videos | Pipeline proven |
| Week 2 | 1,825 videos | EN adult complete |
| Month 1 | 5,475 videos | EN all ages complete |
| Month 2 | 32,850 videos | Full multilingual |

---

## DATABASE CONNECTION

**Neon PostgreSQL (NOT Supabase!)**

Connection string format:
```
postgresql://[user]:[password]@[host]/[database]?sslmode=require
```

The environment variable is likely `DATABASE_URL` in Vercel, not the Supabase variables.

---

**This is the realistic path forward. Execute in order.**
