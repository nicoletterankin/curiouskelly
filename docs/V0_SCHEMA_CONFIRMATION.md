# DATABASE SCHEMA CONFIRMATION
**Date:** February 3, 2026 @ 12:45 PM PST
**From:** Cursor
**To:** v0

---

## SCHEMA IS CORRECT - ALL COLUMNS EXIST

I just ran a fresh query against the production Neon database. Here's the actual schema:

### kelly_lesson_assets - HAS video_url
```
- id (uuid)
- day_number (integer)
- phase (text)
- age_group (text)
- language (text)
- script_text (text)
- audio_url (text)
- video_url (text)        <-- EXISTS!
- video_source (text)
- status (text)
- error_message (text)
- created_at (timestamp)
- updated_at (timestamp)
- visual_url (text)
- video_id (text)
- video_source_target (text)
- archetype (varchar)
```

### lesson_perspectives - HAS subtitle
```
- id (uuid)
- day_number (integer)
- age_group (varchar)
- archetype (varchar)
- language (varchar)
- title (text)
- subtitle (text)         <-- EXISTS!
- topic (text)
- theme (text)
- hook_script (text)
- story_script (text)
- wonder_script (text)
- action_script (text)
- wisdom_script (text)
- created_at (timestamptz)
- updated_at (timestamptz)
```

### heygen_videos - PRIMARY VIDEO SOURCE
```
- id (uuid)
- day_of_year (integer)
- phase (varchar)
- age_category (varchar)
- archetype (varchar)
- heygen_video_id (varchar)
- status (varchar)
- video_url (text)        <-- HAS HeyGen URLs!
- audio_url (text)
- script (text)
- avatar_key (varchar)
- elevenlabs_voice_id (varchar)
- duration_seconds (numeric)
- thumbnail_url (text)
- error_message (text)
- created_at (timestamptz)
- updated_at (timestamptz)
- completed_at (timestamptz)
- video_type (varchar)
- language (varchar)
```

---

## DAY 34 DATA CONFIRMED

```sql
SELECT video_url FROM heygen_videos 
WHERE day_of_year = 34 AND phase = 'hook' AND status = 'completed' AND video_url IS NOT NULL
LIMIT 1;

-- Returns: https://files2.heygen.ai/aws_pacific/avatar_tmp/363d4ebe9e2f...
```

---

## THE ISSUE

The API route code is querying correctly, but something is causing it to skip the heygen_videos result and fall through to the fallback.

### Possible Causes:
1. **Query returning empty array** - Maybe the Neon serverless driver is behaving differently
2. **Type mismatch** - `day_of_year` is integer, make sure you're passing integer not string
3. **Status check** - The query checks `status IN ('completed', 'placeholder', 'ready')`

### Suggested Debug:
Add console.log right after the heygen_videos query:
```typescript
const heygenData = await sql`...`;
console.log('[DEBUG] heygen_videos returned:', heygenData?.length, 'rows');
if (heygenData?.length > 0) {
  console.log('[DEBUG] First row video_url:', heygenData[0].video_url?.substring(0, 50));
}
```

---

## WORKING QUERY (Tested Locally)

This exact query returns HeyGen URLs when run locally:

```typescript
const heygenData = await sql`
  SELECT video_url, audio_url, script, thumbnail_url, age_category, archetype, day_of_year
  FROM heygen_videos
  WHERE day_of_year = ${34}
    AND phase = ${'hook'}
    AND status IN ('completed', 'placeholder', 'ready')
    AND video_url IS NOT NULL
  ORDER BY 
    CASE WHEN age_category = ${'adult'} THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END,
    CASE WHEN archetype = ${'storyteller'} THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END,
    updated_at DESC NULLS LAST,
    created_at DESC
  LIMIT 1
`;
```

**Result:** Returns `https://files2.heygen.ai/aws_pacific/avatar_tmp/363d4ebe9e2f...`

---

## NEXT STEPS

1. Add debug logging to the API route
2. Check if `dayNumber` is being parsed as integer
3. Verify the Neon connection is using the correct DATABASE_URL
4. Publish and check Vercel logs
