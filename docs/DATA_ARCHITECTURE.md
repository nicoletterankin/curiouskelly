# Data Architecture - Single Source of Truth

**Last Updated:** December 1, 2025  
**Status:** ✅ CANONICAL

---

## Overview

This document defines the authoritative data sources for the Curious Kelly / Daily Lesson platform. **Supabase is the single source of truth for all lesson content.**

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SUPABASE (Production)                            │
│                     Single Source of Truth                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │  core_lessons   │──▶│  lesson_atoms   │   │  lesson_shards  │       │
│  │    (365)        │   │   (21,915)      │   │   (38,700)      │       │
│  │                 │   │                 │   │                 │       │
│  │  - day_number   │   │  - archetype    │   │  - age          │       │
│  │  - topic        │   │  - phase        │   │  - region       │       │
│  │  - universal_   │   │  - content      │   │  - tone         │       │
│  │    truth        │   │    (script,     │   │  - script_      │       │
│  │  - marketing_*  │   │     options,    │   │    content      │       │
│  │                 │   │     responses)  │   │                 │       │
│  └────────┬────────┘   └─────────────────┘   └─────────────────┘       │
│           │                                                             │
└───────────┼─────────────────────────────────────────────────────────────┘
            │
            │ sync (scripts/sync_supabase_to_calendar.py)
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LOCAL MIRROR (Read-Only)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  lessons/365_day_calendar.json                                          │
│  ├── Synced from Supabase core_lessons                                  │
│  ├── Used for: offline reference, tooling, CI/CD                        │
│  └── DO NOT EDIT DIRECTLY - use sync script                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Primary: Supabase

| Table | Records | Purpose |
|-------|---------|---------|
| `core_lessons` | 365 | Daily lesson topics, universal truths, marketing copy |
| `lesson_atoms` | 21,915 | Interactive content pieces (script, options, responses) |
| `lesson_shards` | 38,700 | Demographic-specific variants (age, region, tone) |
| `users` | Variable | User accounts and progress |
| `user_progress` | Variable | Lesson completion tracking |

### Secondary: Local JSON (Mirror)

| File | Purpose |
|------|---------|
| `lessons/365_day_calendar.json` | Local cache of core_lessons for tooling |

---

## Data Flow

### Production (Lesson Player)

```
User Opens Lesson
       │
       ▼
┌──────────────────┐
│  Supabase Query  │  SELECT * FROM core_lessons 
│                  │  JOIN lesson_atoms
│                  │  WHERE day_number = ?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Render Lesson   │  Display script, choices, responses
└──────────────────┘
```

### Sync Workflow

```
Supabase Updated
       │
       ▼
python scripts/sync_supabase_to_calendar.py
       │
       ▼
lessons/365_day_calendar.json (updated)
       │
       ▼
python scripts/verify_calendar_alignment.py
       │
       ▼
✅ 100% Match Confirmed
```

---

## What Was Deprecated

### DNA Files (Archived Dec 1, 2025)

**Location:** `_archive/dna-legacy/`

DNA files were a rich content format with:
- Age-specific variants (2-5, 6-12, 13-17, etc.)
- Multilingual translations (EN, ES, FR)
- Voice profiles for ElevenLabs
- Interactive scripts

**Why Deprecated:**
1. Content was OUT OF SYNC with Supabase
2. Dual systems caused confusion
3. Supabase `lesson_atoms` now contains equivalent content
4. Maintenance burden of keeping two systems in sync

**If Needed:** The DNA schema is preserved in `_archive/dna-legacy/` and content can be regenerated from Supabase data.

---

## Scripts

### sync_supabase_to_calendar.py
```bash
python scripts/sync_supabase_to_calendar.py
```
Pulls `core_lessons` from Supabase → updates `lessons/365_day_calendar.json`

### verify_calendar_alignment.py
```bash
python scripts/verify_calendar_alignment.py
```
Compares local JSON against Supabase to ensure 100% alignment

---

## Rules

1. **NEVER edit `lessons/365_day_calendar.json` directly** - always sync from Supabase
2. **All curriculum changes happen in Supabase** via admin tools or SQL
3. **Run verification after any sync** to confirm alignment
4. **Backup before major changes** using `daily-lesson-marketing/backup_full.js`

---

## Supabase Connection

```javascript
// Environment Variables (never commit actual values)
PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
PUBLIC_SUPABASE_ANON_KEY=eyJhbG...
SUPABASE_SERVICE_KEY=eyJhbG...  // For admin operations only
```

---

## Content Structure

### core_lessons Record
```json
{
  "id": "uuid",
  "day_number": 1,
  "topic": "Starting Fresh",
  "universal_truth": "Nature renews itself...",
  "marketing_headline": "Leaf It to Nature!",
  "marketing_tagline": "Growth, Decay, & Renewal!",
  "marketing_pitch": "...",
  "learning_objectives": ["..."],
  "icon_emoji": "🍁",
  "difficulty_level": "Beginner",
  "estimated_duration": 8
}
```

### lesson_atoms Record
```json
{
  "id": "uuid",
  "core_lesson_id": "uuid",
  "archetype": "The Survivor",
  "phase": "Fact1",
  "content": {
    "script": "Kelly says this...",
    "options": ["Option A", "Option B", "Option C"],
    "responses": {
      "Option A": "Response to A...",
      "Option B": "Response to B...",
      "Option C": "Response to C..."
    }
  }
}
```

---

## Troubleshooting

### Calendar Out of Sync
```bash
python scripts/sync_supabase_to_calendar.py
python scripts/verify_calendar_alignment.py
```

### Missing Lesson Content
Check if `lesson_atoms` exist for the `core_lesson_id`:
```sql
SELECT COUNT(*) FROM lesson_atoms WHERE core_lesson_id = 'uuid';
```

### Supabase Connection Issues
Verify environment variables are set:
```bash
echo $PUBLIC_SUPABASE_URL
echo $PUBLIC_SUPABASE_ANON_KEY
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.0.0 | Dec 1, 2025 | DNA files archived, Supabase as single source |
| 2.0.0 | Nov 30, 2025 | Calendar synced from Supabase |
| 1.0.0 | Nov 13, 2025 | Initial calendar with DNA references |



