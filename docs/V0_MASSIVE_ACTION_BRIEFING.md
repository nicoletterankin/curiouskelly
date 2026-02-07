# V0.APP MASSIVE ACTION BRIEFING
## February 3, 2026 - CRITICAL DEPLOYMENT REQUIRED

**From:** Cursor Agent  
**To:** v0.app Agent  
**Priority:** URGENT - Deploy immediately

---

## EXECUTIVE SUMMARY

We have **39 lip-synced Kelly videos** ready in the database that are NOT appearing on production because the deployed code is stale. You need to:

1. **DEPLOY NOW** - Click Publish to push latest code
2. **VERIFY** - Check that `files2.heygen.ai` URLs are being served
3. **MONITOR** - Ensure webhook is receiving new video completions

---

## CURRENT DATABASE STATE

### heygen_videos Table - THE SOURCE OF TRUTH

**Total rows:** 49,550  
**With video_url:** 45  
**Newly synced today:** 39 videos with HeyGen URLs

```sql
-- These videos are READY and waiting to be served:
SELECT day_of_year, phase, age_category, archetype, 
       LEFT(video_url, 50) as url_preview,
       status, updated_at
FROM heygen_videos 
WHERE video_url IS NOT NULL 
  AND status = 'completed'
ORDER BY updated_at DESC
LIMIT 10;
```

**Sample result:**
| day | phase | age | archetype | url_preview | status |
|-----|-------|-----|-----------|-------------|--------|
| 30 | wisdom | adult | storyteller | https://files2.heygen.ai/aws_pacific/avatar... | completed |
| 30 | action | adult | storyteller | https://files2.heygen.ai/aws_pacific/avatar... | completed |
| 30 | wonder | adult | storyteller | https://files2.heygen.ai/aws_pacific/avatar... | completed |
| 29 | wisdom | adult | storyteller | https://files2.heygen.ai/aws_pacific/avatar... | completed |
| 1 | hook | adult | storyteller | https://files2.heygen.ai/aws_pacific/avatar... | completed |
| 1 | story | adult | storyteller | https://files2.heygen.ai/aws_pacific/avatar... | completed |

### Videos Available by Day:
- **Day 1:** All 5 phases ✅
- **Day 2:** hook only
- **Day 11:** hook, wisdom
- **Day 12:** action
- **Day 14:** hook, story, wisdom
- **Day 21:** wonder
- **Day 23:** wisdom
- **Day 24:** wonder
- **Day 25:** story
- **Day 26:** wonder, action, wisdom
- **Day 27-30:** All 5 phases ✅

---

## THE PROBLEM

When I query production:
```
GET https://thedailylesson.com/api/video/url?day=1&phase=hook&age=30&archetype=storyteller&nocache=true
```

**Expected response:**
```json
{
  "url": "https://files2.heygen.ai/aws_pacific/avatar_tmp/...",
  "source": "heygen_videos"
}
```

**Actual response:**
```json
{
  "url": "https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/...",
  "source": "heygen_videos"
}
```

The URL is wrong - it's returning an OLD Vercel Blob URL instead of our NEW HeyGen URL.

### Root Cause:
The production deployment is stale. The database has the correct data, but either:
1. Old code with different query logic is deployed
2. There's server-side caching at Vercel
3. The deployment failed silently

---

## REQUIRED ACTIONS

### Action 1: FORCE DEPLOY

1. In v0.app interface, click **"Publish"** or **"Update"** button
2. If there's a "Force redeploy" option, use it
3. Wait for deployment status to show "Ready"

### Action 2: VERIFY API RESPONSE

After deploy, test this URL:
```
https://thedailylesson.com/api/video/url?day=1&phase=hook&age=30&archetype=storyteller&language=en&nocache=true
```

Success criteria:
- `source` should be `"heygen_videos"`
- `url` should start with `https://files2.heygen.ai/`

### Action 3: CHECK VIDEO PLAYBACK

Go to https://thedailylesson.com and verify:
- Kelly video PLAYS (not static image)
- Lip movement matches audio
- No console errors

### Action 4: CONFIRM WEBHOOK IS ACTIVE

The webhook at `https://thedailylesson.com/api/webhooks/heygen` is verified in HeyGen dashboard. When new videos complete, HeyGen will POST to this endpoint.

Ensure the webhook handler updates the `heygen_videos` table:

```typescript
// In /api/webhooks/heygen/route.ts
// On video completion, this should INSERT/UPDATE heygen_videos:

if (payload.event_type === 'avatar_video.success' && video_url) {
  await sql`
    UPDATE heygen_videos 
    SET video_url = ${video_url}, 
        status = 'completed',
        updated_at = NOW()
    WHERE heygen_video_id = ${video_id}
  `;
}
```

---

## VIDEO API QUERY LOGIC

The `/api/video/url` route should query `heygen_videos` with this priority:

```sql
SELECT video_url, audio_url, script, thumbnail_url
FROM heygen_videos
WHERE day_of_year = $1
  AND phase = $2
  AND status IN ('completed', 'placeholder', 'ready')
  AND video_url IS NOT NULL
ORDER BY 
  CASE WHEN age_category = $3 THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END,
  CASE WHEN archetype = $4 THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END,
  created_at DESC
LIMIT 1
```

**Parameters:**
- $1 = day number (1-365)
- $2 = phase (hook, story, wonder, action, wisdom)
- $3 = age category (adult, kid, senior)
- $4 = archetype (storyteller, scientist, explorer, etc.)

---

## HEYGEN INTEGRATION STATUS

### API Key: ACTIVE ✅
- Credits remaining: ~39,000 seconds (650 minutes)
- Videos in queue: ~100 still processing

### Webhook: VERIFIED ✅
- URL: `https://thedailylesson.com/api/webhooks/heygen`
- Events: `avatar_video.success`, `avatar_video.fail`

### Avatar IDs (for reference):
```javascript
const ADULT_AVATARS = {
  architect: "afc54d3abfc04947bec026b9ec917ce8",
  diplomat: "433ad96bf5d647d9964cecf784d008f6",
  empath: "aa8b5eb1d711468a9a6e2085a4f8469c",
  explorer: "45e5ef8b651846e0b62b7477e552e87b",
  macgyver: "b9032c922c6e4e35b58a98abd499d060",
  mystic: "a2b31ed0b5f84b0fa02d15d411735d3a",
  provider: "06b78109ad22489ea2165ebbf180f77b",
  rebel: "e614671b193c40f99772f7de5d1c51f7",
  scientist: "7bb18cddacd44333813cc90ffa44f766",
  storyteller: "9ffd06bd986a4e3086612921f3ac87ea",
  strategist: "2411df8bdb0d40b088aa453d4c2a2d20",
  survivor: "3f44bd33bfd1494d916d2746808a1a39"
};
```

---

## DATABASE SCHEMA REFERENCE

### heygen_videos (Primary Video Table)
```sql
CREATE TABLE heygen_videos (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  day_of_year integer NOT NULL,
  phase varchar NOT NULL,           -- hook, story, wonder, action, wisdom
  age_category varchar NOT NULL,    -- child, teen, adult, middleAge, senior
  archetype varchar NOT NULL,       -- 12 archetypes
  language varchar DEFAULT 'en',
  heygen_video_id varchar,          -- HeyGen API video ID
  video_url text,                   -- Final video URL
  audio_url text,
  script text,
  status varchar DEFAULT 'queued',  -- queued, processing, completed, failed
  thumbnail_url text,
  error_message text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  UNIQUE(day_of_year, phase, age_category, archetype, language)
);
```

### kelly_lesson_assets (Audio/Fallback)
```sql
-- Contains ElevenLabs TTS audio when heygen_videos has no audio_url
-- Query for audio if heygen_videos.audio_url is NULL
SELECT audio_url FROM kelly_lesson_assets
WHERE day_number = $1 AND phase = $2 AND language = $3
  AND audio_url IS NOT NULL;
```

---

## MONITORING QUERIES

### Check video coverage:
```sql
SELECT day_of_year, COUNT(*) as phases_with_video
FROM heygen_videos 
WHERE video_url IS NOT NULL AND status = 'completed'
GROUP BY day_of_year
ORDER BY day_of_year;
```

### Check processing queue:
```sql
SELECT status, COUNT(*) 
FROM heygen_videos 
GROUP BY status;
```

### Find gaps:
```sql
SELECT day_of_year, phase
FROM heygen_videos
WHERE status = 'completed' AND video_url IS NULL
ORDER BY day_of_year, phase;
```

---

## SUCCESS CRITERIA

After deployment:

1. ✅ `GET /api/video/url?day=1&phase=hook&...&nocache=true` returns `files2.heygen.ai` URL
2. ✅ Kelly plays video on thedailylesson.com (not static image)
3. ✅ Debug HUD shows "VIDEO: heygen_videos" not "NOTHING"
4. ✅ Lip sync matches audio
5. ✅ No console errors

---

## WHAT HAPPENS NEXT

### Cursor (me) will:
- Run sync script every 30 minutes to capture newly completed videos
- Continue HeyGen batch generation for remaining days
- Download HeyGen temp URLs to Vercel Blob for permanence

### You (v0) should:
- Deploy immediately
- Verify the API returns correct URLs
- Monitor for errors in Vercel logs
- Report back status

### Nicolette will:
- Click Publish if you can't
- Verify Kelly is talking on the site
- Approve video quality

---

## IMMEDIATE NEXT STEPS

1. **RIGHT NOW:** Deploy/Publish the latest code
2. **AFTER DEPLOY:** Test the API endpoint with nocache=true
3. **VERIFY:** Check thedailylesson.com shows Kelly with lip-sync
4. **REPORT:** Confirm success or report any errors

---

**END OF BRIEFING**

Please acknowledge receipt and proceed with deployment. The videos are ready - we just need the code deployed to serve them.

Time is critical. Every minute delayed is a minute Kelly isn't teaching with lip-sync.

🚀 LET'S GO!
