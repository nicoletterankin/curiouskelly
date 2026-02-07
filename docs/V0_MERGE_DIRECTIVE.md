# V0.APP MERGE DIRECTIVE
## Consolidate All Chats Into One Deployable Project
## February 3, 2026

---

## THE PROBLEM

You have multiple v0.app chats with code scattered across them:
- **Polish player interface** - UI/player code
- **Full audit and evals** - Database queries, video API
- **Other chats** - Various features

The deployed production may be missing critical code from one of these chats.

---

## THE GOAL

**ONE unified codebase** that:
1. Serves lip-synced videos from `heygen_videos` table
2. Falls back gracefully when videos don't exist
3. Plays audio from `kelly_lesson_assets`
4. Has proper caching (minimize cache misses)

---

## CRITICAL FILE: /app/api/video/url/route.ts

This is the **MOST IMPORTANT FILE**. It must query the database correctly.

### REQUIRED QUERY LOGIC:

```typescript
// /app/api/video/url/route.ts
import { neon } from '@neondatabase/serverless'

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get('day') || '1')
  const phase = searchParams.get('phase') || 'hook'
  const age = parseInt(searchParams.get('age') || '30')
  const archetype = searchParams.get('archetype') || 'storyteller'
  const language = searchParams.get('language') || 'en'
  
  // Convert numeric age to category
  const ageCategory = age <= 12 ? 'child' : age <= 19 ? 'teen' : age <= 55 ? 'adult' : 'senior'
  
  // PRIORITY 1: heygen_videos - lip-synced videos
  const heygenVideo = await sql`
    SELECT video_url, audio_url, script
    FROM heygen_videos
    WHERE day_of_year = ${day}
      AND phase = ${phase}
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY 
      CASE WHEN age_category = ${ageCategory} THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END,
      CASE WHEN archetype = ${archetype} THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END,
      updated_at DESC
    LIMIT 1
  `.catch(() => [])
  
  if (heygenVideo.length > 0 && heygenVideo[0].video_url) {
    // Get audio from kelly_lesson_assets if not in heygen_videos
    let audioUrl = heygenVideo[0].audio_url
    if (!audioUrl) {
      const audio = await sql`
        SELECT audio_url FROM kelly_lesson_assets
        WHERE day_number = ${day} AND phase = ${phase} AND language = ${language}
          AND audio_url IS NOT NULL
        LIMIT 1
      `.catch(() => [])
      audioUrl = audio[0]?.audio_url || null
    }
    
    return Response.json({
      url: heygenVideo[0].video_url,
      audioUrl,
      script: heygenVideo[0].script || getDefaultScript(phase),
      source: 'heygen_videos',
      status: 'ready'
    })
  }
  
  // PRIORITY 2: kelly_lesson_assets - audio with base video fallback
  const kellyAsset = await sql`
    SELECT audio_url, video_url, script_text
    FROM kelly_lesson_assets
    WHERE day_number = ${day} AND phase = ${phase} AND language = ${language}
      AND (audio_url IS NOT NULL OR video_url IS NOT NULL)
    ORDER BY updated_at DESC
    LIMIT 1
  `.catch(() => [])
  
  if (kellyAsset.length > 0) {
    return Response.json({
      url: kellyAsset[0].video_url || getBaseVideoUrl(phase),
      audioUrl: kellyAsset[0].audio_url,
      script: kellyAsset[0].script_text || getDefaultScript(phase),
      source: 'kelly_lesson_assets',
      status: 'ready'
    })
  }
  
  // FALLBACK: Base video with default script
  return Response.json({
    url: getBaseVideoUrl(phase),
    audioUrl: null,
    script: getDefaultScript(phase),
    source: 'fallback',
    status: 'ready'
  })
}

function getBaseVideoUrl(phase: string): string {
  // Verified working Kelly base videos
  const videos = {
    excited: 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/uncategorized/0a437abfa17e46d2a3f2c9a8f27de9ee.mp4',
    default: 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/uncategorized/064dbfab193c461fbb2869f27d663c7b.mp4',
  }
  return ['hook', 'wonder'].includes(phase) ? videos.excited : videos.default
}

function getDefaultScript(phase: string): string {
  const scripts = {
    hook: "Welcome to today's lesson! Let's spark your curiosity.",
    story: "Let me tell you an incredible story.",
    wonder: "What do you wonder about?",
    action: "Let's put this into practice!",
    wisdom: "Here's an insight to carry with you."
  }
  return scripts[phase] || scripts.hook
}
```

---

## MERGE INSTRUCTIONS FOR V0

### Step 1: Identify Which Chat Has Video API

In v0.app, check each chat:
- Open **Polish player interface**
- Search for `/app/api/video/url/route.ts`
- Check if it has the `heygen_videos` query

Do the same for **Full audit and evals**.

### Step 2: Copy The Correct Code

The correct video API must:
1. ✅ Query `heygen_videos` table FIRST
2. ✅ Accept `status IN ('completed', 'placeholder', 'ready')`
3. ✅ Order by age_category and archetype match
4. ✅ Fall back to `kelly_lesson_assets` for audio
5. ✅ Use verified base video URLs as final fallback

### Step 3: Create New Unified Chat (if needed)

If neither chat has correct code:
1. Create a **New Chat** in v0.app
2. Copy all necessary files from both chats
3. Paste the video API code from above
4. Deploy from this new unified chat

### Step 4: Verify Database Connection

Make sure `DATABASE_URL` env var in Vercel points to:
```
postgresql://neondb_owner:...@ep-...neon.tech/neondb?sslmode=require
```

### Step 5: Clear Cache and Deploy

1. In v0.app, click **Update** (not just Publish)
2. In Vercel dashboard, go to Settings > Functions > Clear Cache
3. Test with `?nocache=true` parameter

---

## CACHING STRATEGY (Minimize Cache Misses)

### Current Problem:
- API caches responses for 1 minute
- But database changes faster than cache expires
- Results in stale data being served

### Solution: Smart Cache Keys

```typescript
// Cache key should include:
// 1. Day/phase/age/archetype/language (content key)
// 2. Database last_updated timestamp (freshness key)

const cacheKey = `video:${day}:${phase}:${ageCategory}:${archetype}:${language}`

// Check cache
const cached = await kv.get(cacheKey)
if (cached && !searchParams.get('nocache')) {
  return Response.json(cached)
}

// Query database
const result = await queryDatabase(...)

// Cache for 5 minutes (videos don't change that often)
await kv.set(cacheKey, result, { ex: 300 })

return Response.json(result)
```

### Cache Invalidation:

When webhook receives completed video:
```typescript
// In /api/webhooks/heygen/route.ts
await kv.del(`video:${day}:${phase}:*`) // Invalidate all variants for this day/phase
```

---

## WHAT TO TELL V0

Copy this into v0.app:

```
I need to merge code from multiple chats into one deployable project.

CRITICAL: The /app/api/video/url/route.ts must query heygen_videos table:

1. Query heygen_videos WHERE status IN ('completed', 'placeholder', 'ready') AND video_url IS NOT NULL
2. Order by age_category and archetype match
3. Fall back to kelly_lesson_assets for audio
4. Use verified base video URLs as final fallback

The database has 76+ videos with URLs in heygen_videos table.
These URLs start with https://files2.heygen.ai/

Current production is returning old Vercel Blob URLs instead of HeyGen URLs.

Please:
1. Check if /app/api/video/url/route.ts has the correct query
2. Update it if needed
3. Deploy with cache cleared

Test: https://thedailylesson.com/api/video/url?day=1&phase=hook&age=30&archetype=storyteller&nocache=true

Should return: {"url": "https://files2.heygen.ai/...", "source": "heygen_videos"}
```

---

## VERIFICATION CHECKLIST

After merge and deploy:

- [ ] API returns `source: "heygen_videos"` for Day 1
- [ ] URL starts with `https://files2.heygen.ai/`
- [ ] Kelly plays lip-synced video on site
- [ ] Debug HUD shows "VIDEO: heygen_videos"
- [ ] No console errors

---

## DATABASE CURRENT STATE

```sql
-- heygen_videos: 76 videos with URLs
SELECT COUNT(*) FROM heygen_videos WHERE video_url IS NOT NULL;
-- Result: 76

-- Days covered: 1, 2, 11, 12, 14, 21, 23-30
SELECT DISTINCT day_of_year FROM heygen_videos 
WHERE video_url IS NOT NULL ORDER BY day_of_year;
```

---

## IF ALL ELSE FAILS

1. Export the correct route.ts code from Cursor
2. Manually paste into v0.app code editor
3. Force deploy

The code is in: `c:\Users\user\UI-TARS-desktop\lib\v0-sync\api-video-url\route.ts`

---

**The videos are ready. The database is ready. We just need the right code deployed.**
