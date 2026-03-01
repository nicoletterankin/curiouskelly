# DATABASE DISCONNECT ANALYSIS
## Two Databases, Two Schemas, One Broken Product

> **Date:** 2026-02-08
> **Status:** CRITICAL — The v0 Next.js app (`2_6_2026/`) talks to the WRONG database
> **Impact:** v0.app and Claude Code Desktop cannot find tables/columns because the code queries
> tables that exist in Neon but NOT in Supabase, and vice versa.

---

## THE CORE PROBLEM

There are **TWO completely separate PostgreSQL databases** with **different schemas**:

| | Supabase (Primary) | Neon "wispy-resonance" (v0 App) |
|---|---|---|
| **Project** | `tvjalxxsyryjphkforjv` | `ep-fragrant-scene-a4lk0xwx` |
| **URL** | `https://tvjalxxsyryjphkforjv.supabase.co` | `ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech` |
| **Used by** | Root `api/` (Vercel serverless), Supabase Edge Functions, marketing site, all Cursor-created scripts | `2_6_2026/` Next.js app (v0-created), all `.cjs` scripts in `scripts/` |
| **Connected via** | `@supabase/supabase-js` client | `@neondatabase/serverless` via `DATABASE_URL` |
| **Env var** | `PUBLIC_SUPABASE_URL` + `PUBLIC_SUPABASE_ANON_KEY` | `DATABASE_URL` |

**The v0 Next.js app (`2_6_2026/`) ONLY connects to Neon.** It has zero Supabase connection awareness for its API routes. The `2_6_2026/lib/db.ts` file uses `@neondatabase/serverless` exclusively.

**The root `api/` serverless functions ONLY connect to Supabase.** They use `@supabase/supabase-js` via `api/lib/supabase.ts`.

**They share NO connection logic.** Code in one world cannot see data in the other.

---

## DATABASE-BY-DATABASE: WHAT'S ACTUALLY IN EACH

### NEON (wispy-resonance) — What Claude Code/v0 Found

| Table | Rows | Key Columns |
|---|---|---|
| `lessons` | 366 | `day_of_year` (NOT `day` or `day_number`), `title`, `topic`, `hook_script`, `story_script`, `wonder_script`, `action_script`, `wisdom_script`, `hook_fact`, `hook_correct_answer`, `kelly_age` |
| `lesson_atoms` | 3,385 | (archetype-specific data) |
| `lesson_audio` | 81 | `day_number`, `language`, `age_variant`, `audio_url`, `duration_seconds` |
| `heygen_videos` | 338 | `day_of_year`, `phase`, `age_category`, `archetype`, `video_url`, `audio_url`, `status` |
| `kelly_assets` | 0 | (empty) |
| `kelly_base_videos` | 0 | (empty) |
| `kelly_frames` | ? | |
| `lesson_perspectives` | ~39,000 | `day_number`, `age_group`, `archetype`, `language`, scripts |
| `kelly_lesson_assets` | ~109,000 | `day_number`, `phase`, `age_group`, `language`, `audio_url`, `video_url`, `script_text` |
| `generated_assets` | ? | `lesson_id`, `phase`, `asset_type`, `url`, `status` |
| `kelly_scripts` | ? | `day_number`, `variant`, `phase`, `script_text` |
| `active_translations` | ? | `day_number`, `phase`, `language`, `translated_text`, `audio_url` |
| `learners` | ? | `email`, `password_hash`, `name`, `age`, `language`, `archetype`, `day_of_year`, `subscription_status` |

**Neon has the SCRIPTS inline** (hook_script, story_script, etc.) and the **HeyGen video URLs**.
**Neon has the user auth table** called `learners` (not `users`).

### SUPABASE — What Cursor/MCP Has Access To

| Table | Rows | Key Columns |
|---|---|---|
| `core_lessons` | 730 | `day_number`, `topic`, `universal_truth`, `marketing_headline`, `icon_emoji` — but **NO `title`**, **NO `hook_script`**, **NO scripts at all** |
| `lesson_atoms` | 20,533 | `core_lesson_id` (UUID reference), `archetype`, `phase`, `content` (JSONB) — but **NO `day_number`**, **NO `kelly_script`** |
| `lessons` | 365 | `day_number`, `title`, `subtitle`, `content` (JSONB), `emoji` — but **NO `hook_script`**, **NO `day_of_year`**, NO inline scripts |
| `kelly_video_assets` | 2,265 | video asset records |
| `kelly_lesson_assets` | 1,888 | `day_number`, `phase`, `age_group` (INTEGER not text), `language`, `script`, `audio_url`, `video_url` |
| `kelly_motion_library` | 336 | motion clips |
| `lesson_scripts` | 60 | separate script records |
| `lesson_translations` | 20 | translation records |
| `users` | ? | Supabase Auth users (NOT `learners`) |
| `video_jobs` | ? | job queue for video generation |

**Supabase has NO inline scripts** on lessons — scripts live in `lesson_atoms.content` (JSONB) or `kelly_lesson_assets.script`.
**Supabase has NO `heygen_videos` table.**
**Supabase has NO `learners` table** — users are in `users` table (extends `auth.users`).

---

## COLUMN NAME MISMATCHES (The Detailed Kill List)

### `lessons` table — DIFFERENT SCHEMA in each DB

| Column | Neon | Supabase |
|---|---|---|
| Day identifier | `day_of_year` (int) | `day_number` (int) |
| Title | `title` (text) | `title` (text) |
| Topic/Subject | `topic` (text) | NO `topic` — lives in `core_lessons.topic` |
| Hook script | `hook_script` (text) | NO script columns at all |
| Story script | `story_script` (text) | NO script columns |
| Wonder script | `wonder_script` (text) | NO script columns |
| Action script | `action_script` (text) | NO script columns |
| Wisdom script | `wisdom_script` (text) | NO script columns |
| Hook fact | `hook_fact` (text) | NO `hook_fact` |
| Content | NO `content` | `content` (JSONB) |
| Kelly age | `kelly_age` (int) | NO `kelly_age` |

### `core_lessons` table — ONLY EXISTS IN SUPABASE

| What | Supabase | Neon |
|---|---|---|
| Exists? | YES (730 rows) | **NO** |
| Day column | `day_number` | N/A |
| Has `title`? | NO (has `marketing_headline`) | N/A |
| Has `subject`? | NO (has `topic`) | N/A |
| Has scripts? | NO | N/A |

### `lesson_atoms` table — DIFFERENT SCHEMA

| Column | Neon | Supabase |
|---|---|---|
| Day reference | `day_number` (int) | `core_lesson_id` (UUID) — NO direct day lookup |
| Script | `kelly_script` (text) | NO `kelly_script` |
| Emotion | `kelly_emotion` (text) | NO `kelly_emotion` |
| Active flag | `is_active` (bool) | NO `is_active` |
| Content | ? | `content` (JSONB) |
| Phase | ? | `phase` (text) |

### `kelly_lesson_assets` table — SIMILAR but different

| Column | Neon | Supabase |
|---|---|---|
| Age group | `age_group` (text: 'kid', 'adult', 'senior') | `age_group` (INTEGER) |
| Archetype | `archetype` (text) | NO `archetype` |
| Script | `script_text` (text) | `script` (text) |

### Tables that ONLY EXIST in Neon (not Supabase)

| Table | Purpose | Impact |
|---|---|---|
| `heygen_videos` | 338 HeyGen lip-synced videos | **CRITICAL** — No way to serve videos from Supabase |
| `lesson_perspectives` | 39K personalized scripts by age/archetype/language | **CRITICAL** — The personalization layer |
| `kelly_scripts` | Variant scripts (kid/adult/elder) | Missing word-by-word scripts |
| `generated_assets` | Generated video/audio assets | Missing asset pipeline data |
| `active_translations` | Active translations with audio | Missing translation + audio |
| `learners` | User auth table | Different auth system entirely |
| `kellyos_lessons` | Phased multilingual content | May not exist in Neon either (code has fallbacks) |
| `kellyos_audio` | Audio URLs by phase/language | May not exist in Neon either |
| `kellyos_facts` | True/false facts for hook game | May not exist in Neon either |
| `kelly_base_videos` | Base video references | Empty in Neon, doesn't exist in Supabase |

### Tables that ONLY EXIST in Supabase (not Neon)

| Table | Purpose |
|---|---|
| `core_lessons` | 730 lessons with marketing data, topics, icons |
| `kelly_video_assets` | 2,265 video assets |
| `kelly_motion_library` | 336 motion clips |
| `lesson_shards` | Demographic variants |
| `video_jobs` | Video generation queue |
| `affiliates` / `referrals` | Business tables |
| `analytics_events` | Analytics |
| `users` | User profiles (Supabase Auth) |
| All `commons_*` tables | Community features |
| All financial tables | Revenue, payouts, commissions |

---

## WHAT THE v0 APP CODE EXPECTS (and what breaks)

### `2_6_2026/app/api/lesson/today/route.ts`
Queries (via Neon `sql`):
1. `core_lessons` WHERE `day_number = X` → **FAILS** (doesn't exist in Neon)
2. Fallback: `lessons` WHERE `day_of_year = X` → **WORKS** in Neon
3. `kellyos_lessons` → **PROBABLY FAILS** (may not exist)
4. `kellyos_audio` → **PROBABLY FAILS**
5. `lesson_atoms` WHERE `day_number = X` AND `archetype = Y` AND `is_active = true` → Schema mismatch (Neon may have this, Supabase doesn't have `day_number` or `is_active`)
6. `kellyos_facts` → **PROBABLY FAILS**
7. Fallback: `lesson_perspectives` → **WORKS** in Neon (39K rows)
8. Fallback: `lessons.hook_script` etc → **WORKS** in Neon

### `2_6_2026/app/api/lessons/by-day/route.ts`
Queries (via Neon `sql`):
1. `active_translations` → May or may not exist in Neon
2. `lesson_translations` → May have different schema
3. `lessons WHERE day_of_year = X` → **WORKS** in Neon
4. `lesson_perspectives` → **WORKS** in Neon
5. `kelly_scripts` → **WORKS** in Neon
6. `kelly_lesson_assets` → **WORKS** in Neon (but `age_group` is text vs integer)
7. `heygen_videos` → **WORKS** in Neon (338 videos)

### `2_6_2026/app/api/video/url/route.ts`
Queries (via Neon `sql`):
1. `heygen_videos` → **WORKS** in Neon
2. `generated_assets` → **WORKS** in Neon
3. `kelly_lesson_assets` → **WORKS** in Neon
4. `lesson_perspectives` → **WORKS** in Neon
5. `lessons` → **WORKS** in Neon

### `2_6_2026/lib/auth.ts`
Queries (via Neon `sql`):
1. `learners` table → **EXISTS ONLY IN NEON** (Supabase has `users`)

---

## THE ROOT CAUSE CHAIN

1. **v0.app created the Next.js app** (`2_6_2026/`) with its own Neon database connection
2. **v0 seeded data into Neon** with its own schema (inline scripts, `day_of_year`, `learners` table)
3. **Cursor/I created everything in Supabase** with a different schema (`day_number`, JSONB content, UUID references, `users` table)
4. **The two were never unified** — they evolved independently
5. **30+ scripts in `scripts/*.cjs`** hardcode the Neon connection string directly
6. **The v0 app's DATABASE_URL** in Vercel points to Neon, not Supabase

---

## DATA OVERLAP ANALYSIS

**Day 39 comparison:**

| Field | Neon `lessons` | Supabase `core_lessons` | Supabase `lessons` |
|---|---|---|---|
| Day identifier | `day_of_year: 39` | `day_number: 39` | `day_number: 39` |
| Title | "Where Data Comes From" | (no title field) | "Where Rain Comes From" |
| Topic | "philosophy" | "How Questions" | (no topic field) |
| Hook | "Every drop of rain was once part of an ocean, a cloud, or inside a living thing" | (no hook) | (no hook) |
| Scripts | All 5 scripts inline | None | None |

**They have DIFFERENT CONTENT for the same day.** The Neon DB was seeded with v0's curriculum, while Supabase was seeded with Cursor's curriculum. These are not the same lessons.

---

## HARDCODED SECRETS FOUND IN SCRIPTS

**WARNING:** The following files have the Neon connection string AND HeyGen API key hardcoded (not from env):

```
scripts/poll-heygen-complete.cjs
scripts/sync-all-processing.cjs
scripts/reset-and-regenerate.cjs
scripts/debug-heygen-status.cjs
scripts/full-status-check.cjs
scripts/quick-status.cjs
scripts/continuous-poll.cjs
scripts/generate-day-full.cjs
scripts/batch-generate-all-videos.cjs
scripts/generate-day-v2.cjs
scripts/check-day34.cjs
scripts/full-status.cjs
scripts/bulk-sync-heygen.cjs
scripts/check-heygen-status.cjs
scripts/check-heygen-queue.cjs
scripts/check-heygen-credits.cjs
scripts/check-fal-ready.cjs
scripts/check-coverage.cjs
scripts/find-missed-completions.cjs
scripts/sync-all-completed.cjs
scripts/sync-completed-to-assets.cjs
scripts/live-dashboard.cjs
scripts/verify-lesson-day34.cjs
scripts/check-video-status.cjs
scripts/poll-day34.cjs
scripts/check-recent-heygen.cjs
scripts/add-constraint.cjs
scripts/audit-and-recover.cjs
scripts/check-day35.cjs
```

Connection string: `postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require`
HeyGen API key: `sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E`

---

## WHAT NEEDS TO HAPPEN (Decision Matrix)

### Option A: Consolidate to Supabase (Recommended)
- Migrate Neon data INTO Supabase
- Add missing columns/tables to Supabase (`hook_script` etc on `lessons`, `heygen_videos` table, `lesson_perspectives`, etc.)
- Rewrite `2_6_2026/lib/db.ts` to use `@supabase/supabase-js` instead of `@neondatabase/serverless`
- Update all `.cjs` scripts to use Supabase
- **PRO:** Single source of truth, Supabase has Auth/Storage/Edge Functions/RLS
- **CON:** Massive migration, need to reconcile conflicting data

### Option B: Consolidate to Neon
- Migrate Supabase-only tables INTO Neon
- Keep `@neondatabase/serverless` in the v0 app
- Add Supabase connection only for Auth/Storage
- **PRO:** v0 app already works with Neon
- **CON:** Lose Supabase Auth, Storage, Edge Functions, RLS; would need to recreate all of that

### Option C: Bridge Layer (Fastest, Most Risk)
- Keep both databases
- Create a connection abstraction that queries the right one
- Sync critical tables between them
- **PRO:** Minimum changes to existing code
- **CON:** Ongoing maintenance nightmare, data drift guaranteed

### Option D: Neon for v0 App, Supabase for Everything Else (Current Reality)
- Accept the split
- Make sure each app's env vars point to the right DB
- Don't pretend they share data
- **PRO:** Zero work
- **CON:** Two separate products that can't share users, lessons, or state

---

## IMMEDIATE ACTIONS (if merging v0 code)

Before any v0 PR is merged:

1. **DECIDE**: Which database is the source of truth for lessons?
   - Neon has scripts + videos + perspectives (39K rows, 338 HeyGen videos)
   - Supabase has marketing data + atoms + assets (730 core_lessons, 20K atoms, 2.2K video assets)

2. **DECIDE**: Which auth system?
   - Neon has `learners` table with password hashing (custom JWT)
   - Supabase has `users` table extending `auth.users` (Supabase Auth with OAuth)

3. **DECIDE**: Which video asset system?
   - Neon has `heygen_videos` (338 lip-synced videos from HeyGen)
   - Supabase has `kelly_video_assets` (2,265 video assets from pipeline)

4. **DO NOT MERGE** the v0 app without resolving which DB its API routes hit
   - Every route in `2_6_2026/app/api/` uses `import { sql } from '@/lib/db'` (Neon)
   - If Vercel's `DATABASE_URL` doesn't point to Neon, ALL routes break
   - If it does point to Neon, Supabase data is invisible to the app

---

## QUICK REFERENCE: Which Code Hits Which DB

| Code Path | Database | Connection Method |
|---|---|---|
| `2_6_2026/app/api/**/*.ts` | **NEON** | `2_6_2026/lib/db.ts` → `@neondatabase/serverless` → `DATABASE_URL` |
| `2_6_2026/lib/auth.ts` | **NEON** | `@neondatabase/serverless` → `DATABASE_URL` → `learners` table |
| `api/**/*.ts` (root) | **SUPABASE** | `api/lib/supabase.ts` → `@supabase/supabase-js` |
| `scripts/*.cjs` | **NEON** | Hardcoded connection string |
| `scripts/*.ts` (newer) | **SUPABASE** | `@supabase/supabase-js` via env vars |
| `supabase/functions/*` | **SUPABASE** | Supabase Edge Runtime |
| `daily-lesson-marketing/` | **SUPABASE** | `@supabase/supabase-js` |
| `public/js/lib/supabase.js` | **SUPABASE** | `@supabase/supabase-js` |
| `TEMPLATES/v0/lib/supabase.ts` | **SUPABASE** | Template reference (not executed) |

---

*This document must be resolved before merging any v0 code into production.*
