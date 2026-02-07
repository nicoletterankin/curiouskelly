# V0.APP DEPLOYMENT SPECIFICATION

**Date:** February 4, 2026 05:15 UTC  
**Priority:** CRITICAL - ONE FIX NEEDED  
**Goal:** Make thedailylesson.com serve HeyGen lip-synced videos

---

## 🚨 ROOT CAUSE IDENTIFIED

**Production is connected to the WRONG DATABASE.**

v0 confirmed this - production queries return errors about missing `video_url` and `subtitle` columns. The correct database HAS these columns and HAS the videos.

**THE FIX IS ONE STEP:**

1. Go to **Vercel Dashboard** → **v0-the-dl** → **Settings** → **Environment Variables**
2. Update `DATABASE_URL` to:
```
postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require
```
3. Click **Save** → **Redeploy**

That's it. Videos will work immediately after redeploy.

---

## TODAY'S LESSON IS READY

**Day 35 (February 4, 2026) - "How Dolphins Sleep"**

| Age | hook | story | wonder | action | wisdom |
|-----|------|-------|--------|--------|--------|
| adult | ✅ | ✅ | ✅ | ✅ | ✅ |
| kid | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| senior | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

All 5 adult phases have HeyGen videos ready in the correct database.

---

## CURRENT PIPELINE STATUS

| Metric | Value |
|--------|-------|
| HeyGen Completed | 512 |
| HeyGen Processing | 5,332 |
| Synced to Assets | 392 |
| Days with Video | 80/365 |

**The videos exist. Production just needs to connect to the right database.**

---

## PART 1: VERIFY PRODUCTION DATABASE CONNECTION

### Step 1.1: Check Current Production State

Visit this URL in a browser:
```
https://thedailylesson.com/api/debug/heygen-check?day=34
```

**If it returns:**
```json
{
  "rowsForDay": 45,
  "dbHost": "ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech"
}
```
→ **PASS** - Production is correctly connected. Skip to Part 3.

**If it returns:**
```json
{
  "rowsForDay": 0,
  "dbHost": "ep-quiet-field..."
}
```
→ **FAIL** - Production is using wrong database. Continue to Step 1.2.

### Step 1.2: Fix DATABASE_URL in Vercel

1. Go to **Vercel Dashboard** → **v0-the-dl** project
2. Click **Settings** → **Environment Variables**
3. Find `DATABASE_URL`

**Required Value:**
```
postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require
```

4. If the current value contains `ep-quiet-field` or any other host, **delete it**
5. Add new `DATABASE_URL` with the value above
6. Ensure these boxes are checked:
   - ☑ Production
   - ☑ Preview  
   - ☑ Development
7. Click **Save**

### Step 1.3: Delete Conflicting Variables

If there are multiple `DATABASE_URL` or `DATABASE_URL_UNPOOLED` entries:
- Delete ALL of them
- Add back ONLY the single `DATABASE_URL` from Step 1.2

### Step 1.4: Trigger Redeploy

1. Go to **Deployments** tab
2. Find the latest deployment
3. Click **⋯** → **Redeploy**
4. Do NOT check "Use existing build cache"
5. Click **Redeploy**

### Step 1.5: Verify Fix

Wait 60 seconds after deploy completes, then test:
```
https://thedailylesson.com/api/debug/heygen-check?day=34
```

**Expected:**
- `rowsForDay: 45`
- `dbHost: "ep-fragrant-scene..."`

---

## PART 2: VERIFY VIDEO URL API

### Step 2.1: Test Video Endpoint

Visit:
```
https://thedailylesson.com/api/video/url?day=34&phase=hook&age=30&archetype=storyteller&language=en
```

**Expected Response:**
```json
{
  "url": "https://files.heygen.ai/video/v1/[uuid]/[uuid].mp4",
  "source": "heygen_videos",
  "status": "ready"
}
```

**Failure Indicators:**
- `url` contains `vercel-storage.com` → Fallback video being used
- `source: "verified_base_video"` → No HeyGen video found
- `url: null` → Query failed

### Step 2.2: Verify Query Logic

The `/api/video/url/route.ts` file must query `heygen_videos` table with:
- Column: `day_of_year` (NOT `day_number`)
- Column: `age_category` (NOT `age_group`)
- Status filter: `IN ('completed', 'placeholder', 'ready')`

**Correct Query Pattern:**
```typescript
const heygenResult = await sql`
  SELECT video_url, audio_url, script, status
  FROM heygen_videos
  WHERE day_of_year = ${day}
  AND phase = ${phase}
  AND status IN ('completed', 'placeholder', 'ready')
  AND video_url IS NOT NULL
  ORDER BY updated_at DESC
  LIMIT 1
`;
```

---

## PART 3: VERIFY FRONTEND PLAYBACK

### Step 3.1: Load Site

Visit: `https://thedailylesson.com`

### Step 3.2: Check Video Source

1. Open browser DevTools (F12)
2. Go to **Network** tab
3. Filter by "video" or "mp4"
4. Look at the video URL being loaded

**Expected:** URL contains `files.heygen.ai` or `files2.heygen.ai`
**Failure:** URL contains `vercel-storage.com` (fallback)

### Step 3.3: Visual Verification

- Kelly should appear with lip-synced speech
- Her mouth movements should match the audio
- The lesson should be Day 34 (How Magnets Work)

---

## PART 4: PULL LATEST CODE FROM GITHUB

The Cursor agent has pushed bug fixes to GitHub. Pull them:

### Step 4.1: Pull from GitHub

In your v0 project:
1. Go to **Git** tab
2. Click **Pull from origin/main**

**Recent commits to pull:**
- `9a09b0d` - Fix DATABASE_URL debug output when undefined
- `fd6e8a9` - Update heygen-check debug endpoint

### Step 4.2: Redeploy After Pull

After pulling, trigger a new deployment to include the fixes.

---

## PART 5: DATABASE GROUND TRUTH

### The Correct Database Contains (LIVE - 2026-02-04 05:00 UTC):

| Table | Count | Description |
|-------|-------|-------------|
| heygen_videos | 5,844 | Video generation queue |
| → processing | 5,364 | Being rendered by HeyGen |
| → completed | 480 | Ready to serve |
| → Day 34 | 1 complete | kid/story done, 14 processing |
| kelly_lesson_assets | 109,847 | TTS audio + video URLs |
| → has video | 372 | Synced from HeyGen |
| → has audio | 9,110 | Ready for Fal lipsync |
| lesson_perspectives | 39,420 | Scripts |
| lessons | 365 | Core lessons |

**Note:** 
- HeyGen is processing 5,364 videos (submitted 2026-02-04 03:40 UTC)
- Completions trickle in as HeyGen renders - expect ~50-100/hour
- Use `node scripts/full-status.cjs` for live status
- Use `node scripts/poll-day34.cjs` to check today's lesson specifically

### Correct Database Identifiers:

| Property | Value |
|----------|-------|
| Neon Project ID | soft-block-64917198 |
| Project Name | neon-rose-island |
| Database Host | ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech |
| Region | aws-us-east-1 |
| PostgreSQL | 17 |

### Wrong Database Identifiers (DO NOT USE):

| Property | Value |
|----------|-------|
| Database Host | ep-quiet-field-* |
| Database Host | ep-holy-wind-* |
| Database Host | ep-long-term-* |

---

## PART 6: CHECKLIST

Run through this before declaring success:

| # | Check | How to Verify | Status |
|---|-------|---------------|--------|
| 1 | Debug endpoint returns videos | `/api/debug/heygen-check?day=34` → `rowsForDay: 45` | ☐ |
| 2 | Correct database host | Response contains `ep-fragrant-scene` | ☐ |
| 3 | Video URL returns HeyGen | `/api/video/url?day=34&phase=hook` → `files.heygen.ai` | ☐ |
| 4 | Site loads Kelly video | Visit site, see lip-synced Kelly | ☐ |
| 5 | Network shows HeyGen URL | DevTools → Network → video URL | ☐ |

---

## PART 7: TROUBLESHOOTING

### Problem: `rowsForDay: 0`

**Cause:** Wrong DATABASE_URL
**Fix:** Update DATABASE_URL in Vercel Settings → Environment Variables

### Problem: `dbHost: "NOT_SET"`

**Cause:** DATABASE_URL not configured
**Fix:** Add DATABASE_URL in Vercel Settings → Environment Variables

### Problem: Video URL returns fallback

**Cause:** Query not finding videos OR wrong column names
**Fix:** Check that code uses `day_of_year` not `day_number`

### Problem: Site shows old code

**Cause:** Stale deployment
**Fix:** Pull from GitHub, then Redeploy without cache

### Problem: Vercel build fails

**Cause:** Missing environment variable during build
**Fix:** Ensure DATABASE_URL is set for ALL environments (Production + Preview + Development)

---

## PART 8: SUCCESS STATE

When everything is working:

1. **Debug Endpoint:**
```json
{
  "success": true,
  "totalHeygenVideos": 459,
  "rowsForDay": 45,
  "dbHost": "ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech"
}
```

2. **Video URL API:**
```json
{
  "url": "https://files.heygen.ai/video/v1/abc123/abc123.mp4",
  "source": "heygen_videos",
  "status": "ready"
}
```

3. **thedailylesson.com:**
- Kelly appears with lip-synced video
- Day 34 lesson plays correctly
- No fallback videos being used

---

## PART 9: FAL LIPSYNC PIPELINE (v0's Task)

While the HeyGen pipeline runs, v0 can process videos using the Fal lipsync pipeline.

### v0's Data Source:

Query `kelly_lesson_assets` for videos ready for Fal lipsync:

```sql
SELECT day_number, phase, age_group, audio_url, archetype
FROM kelly_lesson_assets
WHERE audio_url IS NOT NULL 
AND video_url IS NULL
ORDER BY day_number, phase;
```

**Returns:** 9,070 records with audio ready for lipsync

### Fal Pipeline Process:

1. Get audio URL from `kelly_lesson_assets.audio_url`
2. Get base Kelly video (from Vercel Blob or `kelly_base_videos`)
3. Call Fal lipsync API with audio + base video
4. Store result in `kelly_lesson_assets.video_url`
5. Set `video_source = 'fal'`

### Base Videos Location:

Vercel Blob: `https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/`

### Coordination:

- HeyGen videos use `video_source = 'heygen'`
- Fal videos use `video_source = 'fal'`
- Both pipelines can run simultaneously without conflict
- Frontend should prefer any `video_url` that exists

---

## CONTACT

If this specification doesn't resolve the issue:
- The Cursor agent has database access and can run diagnostic queries
- Do NOT create new databases - use the existing `ep-fragrant-scene` database
- All changes should be pushed to `nicoletterankin/v0-the-dl` GitHub repo

---

**END OF SPECIFICATION**

Execute Parts 1-6 in order. Report back with checklist status.

**Live Pipeline Status:** Run `node scripts/quick-status.cjs` for HeyGen progress.
