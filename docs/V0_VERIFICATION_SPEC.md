# V0.APP ZERO-TRUST VERIFICATION SPECIFICATION

**Date:** February 3, 2026  
**Purpose:** Verify database connection and video serving pipeline  
**Trust Level:** ZERO - Verify everything, assume nothing

---

## CRITICAL CONTEXT

You are managing `thedailylesson.com` deployed on Vercel. The site should serve HeyGen lip-synced videos of Kelly (an AI teacher) for 365 daily lessons.

**THE PROBLEM WE'RE SOLVING:**  
Production was connected to the WRONG database. This spec verifies the fix.

---

## STEP 1: VERIFY DATABASE CONNECTION

### 1.1 Check Environment Variable

Go to **Settings → Environment Variables** in your Vercel project.

Find `DATABASE_URL` and verify it contains:
```
ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech
```

**FAIL CONDITION:** If it contains `ep-quiet-field` or any other host, the database is WRONG.

### 1.2 Verify No Conflicting Variables

Ensure there are NOT multiple `DATABASE_URL` or `DATABASE_URL_UNPOOLED` entries pointing to different databases.

**CORRECT STATE:**
- ONE `DATABASE_URL` for "All Environments" pointing to `ep-fragrant-scene`
- OR individual entries all pointing to `ep-fragrant-scene`

**FAIL CONDITION:** Multiple entries pointing to different Neon instances.

---

## STEP 2: VERIFY DEBUG ENDPOINT

### 2.1 Create/Verify Debug Endpoint Exists

The file `/app/api/debug/heygen-check/route.ts` should exist with this code:

```typescript
import { neon } from "@neondatabase/serverless"
import { NextResponse } from "next/server"

export const dynamic = "force-dynamic"
export const revalidate = 0

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const day = parseInt(searchParams.get("day") || "34")
  
  try {
    const sql = neon(process.env.DATABASE_URL!)
    
    // Extract database host for verification
    const dbUrl = process.env.DATABASE_URL || ""
    const hostMatch = dbUrl.match(/@([^\/]+)\//)
    const dbHost = hostMatch ? hostMatch[1] : "unknown"
    
    // Count total videos
    const totalResult = await sql`
      SELECT COUNT(*) as count FROM heygen_videos 
      WHERE status = 'completed' AND video_url IS NOT NULL
    `
    
    // Get videos for specific day
    const dayResult = await sql`
      SELECT id, day_of_year, phase, age_category, archetype, video_url, status
      FROM heygen_videos 
      WHERE day_of_year = ${day} 
      AND status = 'completed' 
      AND video_url IS NOT NULL
      ORDER BY phase
    `
    
    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      dbHost: dbHost,
      dbUrlPrefix: dbUrl.substring(0, 80) + "...",
      totalCompletedVideos: parseInt(totalResult[0].count),
      requestedDay: day,
      rowsForDay: dayResult.length,
      sampleRows: dayResult.slice(0, 3).map(r => ({
        day: r.day_of_year,
        phase: r.phase,
        age: r.age_category,
        archetype: r.archetype,
        hasUrl: !!r.video_url,
        urlPrefix: r.video_url?.substring(0, 50)
      }))
    })
  } catch (error: any) {
    return NextResponse.json({
      success: false,
      error: error.message,
      dbUrlPrefix: (process.env.DATABASE_URL || "").substring(0, 50)
    }, { status: 500 })
  }
}
```

### 2.2 Test Debug Endpoint

After deployment, visit:
```
https://thedailylesson.com/api/debug/heygen-check?day=34
```

**EXPECTED RESPONSE (PASS):**
```json
{
  "success": true,
  "dbHost": "ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech",
  "totalCompletedVideos": 459,
  "requestedDay": 34,
  "rowsForDay": 45,
  "sampleRows": [
    { "day": 34, "phase": "action", "hasUrl": true, "urlPrefix": "https://files.heygen.ai/video/v1/" }
  ]
}
```

**FAIL CONDITIONS:**
- `dbHost` contains `ep-quiet-field` → Wrong database
- `rowsForDay: 0` → Database has no videos (wrong DB or data missing)
- `totalCompletedVideos: 0` → Definitely wrong database
- `success: false` → Connection error

---

## STEP 3: VERIFY VIDEO URL API

### 3.1 Test Main Video Endpoint

Visit:
```
https://thedailylesson.com/api/video/url?day=34&phase=hook&age=30&archetype=storyteller&language=en
```

**EXPECTED RESPONSE (PASS):**
```json
{
  "url": "https://files.heygen.ai/video/v1/[uuid]/[uuid].mp4",
  "source": "heygen_videos",
  "status": "ready"
}
```

**FAIL CONDITIONS:**
- `url` contains `vercel-storage.com` → Fallback video (DB query failed)
- `source: "verified_base_video"` → Fallback (no HeyGen video found)
- `url: null` → Complete failure

### 3.2 Verify Video URL Query Logic

The `/api/video/url/route.ts` should query `heygen_videos` table FIRST:

```typescript
// Priority 1: Check heygen_videos table
const heygenResult = await sql`
  SELECT video_url, audio_url, script, status
  FROM heygen_videos
  WHERE day_of_year = ${day}
  AND phase = ${phase}
  AND status IN ('completed', 'placeholder', 'ready')
  AND video_url IS NOT NULL
  ORDER BY updated_at DESC
  LIMIT 1
`

if (heygenResult.length > 0 && heygenResult[0].video_url) {
  return NextResponse.json({
    url: heygenResult[0].video_url,
    source: "heygen_videos",
    status: "ready"
  })
}
```

---

## STEP 4: VERIFY FRONTEND PLAYBACK

### 4.1 Load Production Site

Visit: `https://thedailylesson.com`

### 4.2 Navigate to Day 34

The app should default to today's lesson (Day 34 = February 3, 2026).

### 4.3 Verify Video Plays

**PASS:** Kelly appears with lip-synced speech matching the lesson script.

**FAIL:** 
- Kelly appears but mouth doesn't move (fallback video)
- Error message or blank screen
- Video URL in network tab shows `vercel-storage.com` instead of `files.heygen.ai`

---

## STEP 5: DATABASE GROUND TRUTH

### 5.1 Expected Data in `heygen_videos` Table

Run this query (via Neon console or Drizzle Studio):

```sql
SELECT 
  status, 
  COUNT(*) as count 
FROM heygen_videos 
GROUP BY status 
ORDER BY count DESC;
```

**EXPECTED:**
| status | count |
|--------|-------|
| queued | ~49,000 |
| completed | 459 |
| processing | 0-5 |
| placeholder | 1 |

### 5.2 Expected Day Coverage

```sql
SELECT 
  day_of_year, 
  COUNT(*) as videos 
FROM heygen_videos 
WHERE status = 'completed' AND video_url IS NOT NULL
GROUP BY day_of_year 
ORDER BY day_of_year;
```

**EXPECTED:** Days 1-60 should have videos (5-45 per day).

### 5.3 Verify Correct Database

```sql
SELECT current_database(), inet_server_addr();
```

The server should be in AWS us-east-1 region (Neon `ep-fragrant-scene` project).

---

## STEP 6: CORRECTIVE ACTIONS

### If Database is Wrong

1. Go to Vercel → Settings → Environment Variables
2. Delete ALL `DATABASE_URL` and `DATABASE_URL_UNPOOLED` entries
3. Add new `DATABASE_URL`:
   ```
   postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require
   ```
4. Select: ☑ Production ☑ Preview ☑ Development
5. Save
6. Redeploy (Deployments → Latest → ⋯ → Redeploy)

### If Debug Endpoint Missing

Create `/app/api/debug/heygen-check/route.ts` with the code from Step 2.1.

### If Video URL API Returns Fallbacks

Check that the query in `/api/video/url/route.ts` queries `heygen_videos` table with correct column names:
- `day_of_year` (not `day_number`)
- `age_category` (not `age_group`)
- `status IN ('completed', 'placeholder', 'ready')`

---

## VERIFICATION CHECKLIST

Run through this checklist after any deployment:

| Check | Command/URL | Expected | Actual |
|-------|-------------|----------|--------|
| Debug endpoint works | `/api/debug/heygen-check?day=34` | `rowsForDay: 45` | ___ |
| Correct database | Debug response `dbHost` | `ep-fragrant-scene` | ___ |
| Total videos | Debug response | `459+` | ___ |
| Video URL returns HeyGen | `/api/video/url?day=34&phase=hook` | `files.heygen.ai` URL | ___ |
| Frontend plays video | Visit site, Day 34 | Kelly speaks with lip-sync | ___ |

---

## CONTACT FOR ISSUES

If verification fails after following corrective actions:
- **Cursor agent** has direct database access and can run queries
- **Database connection string** is in this document (for `ep-fragrant-scene`)
- **Do NOT create new Neon databases** - use the existing one

---

**END OF SPECIFICATION**
