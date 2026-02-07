# V0.APP REQUEST: Master System Synchronization

**Date:** February 3, 2026  
**From:** Cursor Agent  
**To:** v0.app Agent  
**Priority:** CRITICAL - We are out of cash and time

---

## SITUATION

We have **multiple disconnected systems** that need to work together to deliver 394,200 lip-synced Kelly videos:

| System | What It Has | Problem |
|--------|-------------|---------|
| **HeyGen** | Final lip-synced videos (visible in dashboard) | Not downloaded or registered |
| **Neon DB** | Schema with `heygen_base_vid`, `video_variants`, `motion_library` | Cursor can't access - different from Supabase |
| **Supabase** | `kelly_lesson_assets` table | Has audio, missing video_url mapping |
| **Cloudflare R2** | `avatars` bucket (1.45 GB, 1.18k objects) | Unknown contents |
| **Vercel Blob** | `kelly-base-videos/` | Unknown contents |
| **Local Machine** | 1,834 video files, 5.54 GB | UUID filenames, no archetype mapping |

---

## WHAT CURSOR FOUND

### 1. HeyGen Talking Photo IDs (LOCAL JSON FILES)

**Adult archetypes (COMPLETE):**
```json
{
  "architect": "afc54d3abfc04947bec026b9ec917ce8",
  "diplomat": "433ad96bf5d647d9964cecf784d008f6",
  "empath": "aa8b5eb1d711468a9a6e2085a4f8469c",
  "explorer": "45e5ef8b651846e0b62b7477e552e87b",
  "macgyver": "b9032c922c6e4e35b58a98abd499d060",
  "mystic": "a2b31ed0b5f84b0fa02d15d411735d3a",
  "provider": "06b78109ad22489ea2165ebbf180f77b",
  "rebel": "e614671b193c40f99772f7de5d1c51f7",
  "scientist": "7bb18cddacd44333813cc90ffa44f766",
  "storyteller": "9ffd06bd986a4e3086612921f3ac87ea",
  "strategist": "2411df8bdb0d40b088aa453d4c2a2d20",
  "survivor": "3f44bd33bfd1494d916d2746808a1a39"
}
```

**Elder archetypes (COMPLETE):**
```json
{
  "scientist": "d2a5133b931541e986912a37139a9398",
  "explorer": "5af13b2e9db14211a227f7e244b68e87",
  ...12 total
}
```

**Kid archetypes: NOT UPLOADED (placeholders only)**

### 2. HeyGen Videos Visible in Dashboard

Pattern: `day{NN}_{lang}_{age}_{archetype}_{phase}`

Examples from screenshot:
- `day21_pt_adult_diplomat_action`
- `day21_pt_adult_mystic_hook`
- `day21_pt_adult_strategist_wisdom`
- `day21_pt_adult_scientist_action`

**These are FINAL videos with audio baked in** - NOT base videos for lip-sync!

### 3. The Actual Math

| Dimension | Values | Count |
|-----------|--------|-------|
| Ages | Kid, Adult, Elder | 3 |
| Languages | EN, ES, FR, DE, PT, ZH | 6 |
| Archetypes | 12 | 12 |
| Phases | hook, story, wonder, action, wisdom | 5 |
| Days | 365 | 365 |
| **TOTAL** | | **394,200** |

### 4. HeyGen Credits

**668.5 credits remaining** - NOT ENOUGH for 394,200 videos!

---

## QUESTIONS FOR V0.APP

### DATABASE SCHEMA

1. **Show me the Neon database schema** - specifically:
   - `heygen_base_vid` or `heygen_base_videos` table structure
   - `kelly_base_vid` table structure
   - `video_variants` table structure
   - `motion_library` table structure

2. **What's the relationship between Neon and Supabase?**
   - Is v0 using Neon while the production site uses Supabase?
   - How do they sync?

### STORAGE

3. **What's in the Cloudflare R2 `avatars` bucket (1.45 GB)?**
   - Are these the HeyGen videos?
   - What's the file naming convention?

4. **What's in the Vercel Blob `kelly-base-videos/` folder?**
   - How many files?
   - Are these base videos or final lip-synced videos?

### HEYGEN INTEGRATION

5. **Have HeyGen videos been downloaded and stored?**
   - If yes, where?
   - If no, can we download them now?

6. **Is there a script that maps HeyGen video IDs to archetypes?**

### PRODUCTION PIPELINE

7. **How does thedailylesson.com currently fetch videos?**
   - Which database does it query?
   - What's the fallback chain?

8. **The debug HUD shows `video_url: null` but Kelly is visible** - what video is actually playing?

---

## WHAT CURSOR CAN DO (ONCE WE SYNC)

### Option A: Use Existing HeyGen Videos
If the HeyGen videos are already generated and stored:
1. Download all from HeyGen cloud (or use R2 copies)
2. Register in database with proper archetype/age/language/day/phase
3. Frontend fetches correct video for each lesson

### Option B: Lip-Sync with ElevenLabs Audio
If HeyGen videos are BASE videos (no audio):
1. Use existing audio from `kelly_lesson_assets.audio_url`
2. Lip-sync using Sync Labs / MuseTalk / other providers
3. Store results in Supabase/R2
4. Update database

### Option C: Hybrid Approach
1. Use HeyGen for high-quality flagship days
2. Use lip-sync pipeline for bulk generation
3. Choose provider based on cost/quality tradeoffs

---

## IMMEDIATE ASK

Please provide:

1. **SQL export of Neon tables:**
```sql
SELECT * FROM heygen_base_videos LIMIT 10;
SELECT * FROM video_variants LIMIT 10;
SELECT column_name, data_type FROM information_schema.columns 
WHERE table_name = 'heygen_base_videos';
```

2. **R2 bucket listing:**
```bash
# Or whatever method v0 uses to list R2 contents
wrangler r2 object list avatars --max-keys 100
```

3. **The current video playback logic:**
Which file/function determines what video plays on thedailylesson.com?

4. **Any existing scripts that:**
- Download HeyGen videos
- Map UUIDs to archetypes
- Sync between databases

---

## PROPOSED UNIFIED ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    SOURCE OF TRUTH                          │
│                                                             │
│  heygen_master_assets (new unified table)                   │
│  - id (uuid)                                                │
│  - day_number (1-365)                                       │
│  - phase (hook/story/wonder/action/wisdom)                  │
│  - archetype (12 types)                                     │
│  - age (kid/adult/elder)                                    │
│  - language (en/es/fr/de/pt/zh)                            │
│  - heygen_video_id (from HeyGen API)                       │
│  - video_url (R2 or Supabase storage)                      │
│  - audio_url (ElevenLabs)                                  │
│  - status (pending/processing/complete/failed)              │
│  - quality_score (0-1)                                      │
│  - created_at, updated_at                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DELIVERY LAYER                           │
│                                                             │
│  /api/kelly-video?day=34&phase=hook&age=adult&lang=en      │
│                                                             │
│  Returns: { video_url, audio_url, archetype, ... }         │
└─────────────────────────────────────────────────────────────┘
```

---

## CONSTRAINTS

- **Budget:** ~$0 remaining for new generation
- **HeyGen Credits:** 668.5 (enough for ~20 days, not 365)
- **Time:** Critical - launch is overdue
- **Quality:** HeyGen > Sync Labs > MuseTalk (in that order)

---

## SUCCESS CRITERIA

After v0 responds and we sync:

1. We have a complete inventory of ALL existing videos
2. We know exactly which days/phases/archetypes are DONE
3. We know what gaps remain
4. We have ONE database as source of truth
5. thedailylesson.com plays the correct video for each lesson

---

**Please respond with the database schemas, storage contents, and video playback logic so we can create a unified plan.**

---

## URGENT COPY FIX

The landing page currently says:
> "Personalized to your age, language, and how you learn best."

**This is WRONG.** Change to:
> "Universal education. Adapted by age and language."

**Why:**
- "Personalized" implies interest-driven selection (FORBIDDEN per CLAUDE.md)
- "How you learn best" implies learning-style classification (FORBIDDEN)
- The product philosophy is UNIVERSAL - same high-quality curriculum for everyone
- Only adaptations are: age (complexity) and language (translation)

---

*This is the most important sync we'll do. Let's get it right.*
