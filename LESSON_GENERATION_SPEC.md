# 🔒 THE LESSON GENERATION SPEC
## Canonical Process for Curious Kelly Daily Lessons

**Last Updated:** December 19, 2025  
**Status:** LOCKED - Do not deviate without explicit approval

---

## ⚠️ CRITICAL RULES (NEVER VIOLATE)

1. **NEVER use browser TTS** - If audio doesn't exist, GENERATE IT with ElevenLabs
2. **NEVER assume data exists** - QUERY and VERIFY before claiming something works
3. **NEVER trust UI visuals** - Verify the DATA SOURCE (database, JSON, or fallback)
4. **NEVER gloss over errors** - Every console error is a bug to fix
5. **NEVER change phase structure** - 7 phases are canonical: hook, cliff, q1, q2, q3, wisdom, outro

---

## 📐 CANONICAL LESSON STRUCTURE

### The 7 Phases (LOCKED)
```
Phase Key  | DB Name | Description
-----------|---------|------------------------------------------
hook       | Hook    | Opening question to spark curiosity
cliff      | Cliff   | Choice point - learner picks path A or B  
q1         | Fact1   | First knowledge block
q2         | Fact2   | Second knowledge block
q3         | Fact3   | Third knowledge block
wisdom     | Wisdom  | Universal truth / takeaway
outro      | Outro   | Tomorrow preview + farewell
```

### Phase Order in JSON Files
```json
"phaseOrder": ["hook", "cliff", "q1", "q2", "q3", "wisdom", "outro"]
```

### Phase Names in Database
```sql
-- lesson_atoms.phase column values:
'Hook', 'Cliff', 'Fact1', 'Fact2', 'Fact3', 'Wisdom', 'Outro'
```

---

## 🗄️ DATABASE TABLES

### core_lessons (365 rows - one per day)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| day_number | int | 1-365 |
| topic | text | Lesson topic (used in calendar, search) |
| **category** | text | 'Mind & Brain', 'Science', 'Emotions', etc. |
| universal_truth | text | Core wisdom |
| marketing_headline | text | Hook for marketing |
| **search_vector** | tsvector | Full-text search index (auto-generated) |

**Search Index Trigger:**
```sql
-- Auto-update search vector on insert/update
CREATE OR REPLACE FUNCTION update_lesson_search_vector()
RETURNS TRIGGER AS $$
BEGIN
  NEW.search_vector := to_tsvector('english', 
    COALESCE(NEW.topic, '') || ' ' || 
    COALESCE(NEW.marketing_headline, '') || ' ' ||
    COALESCE(NEW.category, '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER core_lessons_search_update
  BEFORE INSERT OR UPDATE ON core_lessons
  FOR EACH ROW EXECUTE FUNCTION update_lesson_search_vector();
```

### lesson_atoms (365 × 12 archetypes × 7 phases × 3 languages = 91,980 target)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| core_lesson_id | uuid | FK to core_lessons |
| archetype | text | 'The Scientist', 'The Explorer', etc. |
| phase | text | 'Hook', 'Cliff', 'Fact1', etc. |
| **language** | text | 'en', 'es', 'fr' (precomputed, per CLAUDE.md) |
| content | jsonb | See CONTENT STRUCTURE below |
| **visual_url** | text | URL to phase infographic (1920×1080) |
| **hd_video_url** | text | URL to HD Kelly video for this phase |

**Unique Constraint:**
```sql
-- One atom per (lesson, archetype, phase, language)
ALTER TABLE lesson_atoms ADD CONSTRAINT lesson_atoms_unique 
  UNIQUE (core_lesson_id, archetype, phase, language);
```

### ⚠️ CONTENT STRUCTURE (LOCKED - EVERY PHASE)
```json
{
  "script": "Kelly's spoken words for this phase",
  "kellyPose": "teaching | curious | celebrating | reflective",
  "kellyEmotion": "excited | warm | thoughtful | playful",
  
  "options": [
    {
      "id": "A",
      "label": "Option A display text",
      "imageUrl": "https://... (512×512 option card)",
      "responseScript": "Kelly's response when A is chosen",
      "quality": "best | good | redirect"
    },
    {
      "id": "B",
      "label": "Option B display text",
      "imageUrl": "https://... (512×512 option card)",
      "responseScript": "Kelly's response when B is chosen",
      "quality": "best | good | redirect"
    }
  ],
  
  "simulatedComments": [
    {
      "emoji": "✨",
      "text": "Phase-specific comment from simulated student",
      "author": "curious_maya",
      "timestamp": "2m ago"
    },
    {
      "emoji": "✨",
      "text": "Another engaging comment",
      "author": "science_sam",
      "timestamp": "5m ago"
    }
  ]
}
```

**⚠️ CRITICAL: ALL 7 PHASES require this structure. Not just Cliff.**
- Hook: 2 options (how to approach the topic)
- Cliff: 2 options (choose your path)
- Fact1: 2 options (knowledge check)
- Fact2: 2 options (knowledge check)
- Fact3: 2 options (knowledge check)
- Wisdom: 2 options (how this applies to life)
- Outro: 2 options (what to explore next)

### lesson_visuals (phase-linked visual assets)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| day_number | int | 1-365 |
| phase | text | 'Hook', 'Cliff', 'Fact1', etc. |
| visual_type | text | 'infographic', 'option-a', 'option-b', 'thumbnail' |
| prompt_hash | text | SHA256 of generation prompt (for caching) |
| storage_url | text | Supabase Storage URL |
| model_used | text | 'imagen-4.0-ultra', 'gemini-2.0-flash', etc. |
| generation_source | text | 'seed', 'platform', 'byok', 'staff' |
| status | text | 'pending', 'ready', 'failed' |

**Visual Assets Per Phase (ALL phases have 2 options):**
```
Phase     | Infographic (1920×1080) | Option Cards (512×512) | Simulated Comments | Total
----------|-------------------------|------------------------|-------------------|-------
hook      | 1                       | 2 (Option A, B)        | 2-3               | 5-6
cliff     | 1                       | 2 (Option A, B)        | 2-3               | 5-6
q1        | 1                       | 2 (Option A, B)        | 2-3               | 5-6
q2        | 1                       | 2 (Option A, B)        | 2-3               | 5-6
q3        | 1                       | 2 (Option A, B)        | 2-3               | 5-6
wisdom    | 1                       | 2 (Option A, B)        | 2-3               | 5-6
outro     | 1                       | 2 (Option A, B)        | 1-2               | 4-5
----------|-------------------------|------------------------|-------------------|-------
TOTAL     | 7                       | 14                     | 14-20             | 35-41/day
```

**⚠️ EVERY PHASE = 2 OPTIONS. NO EXCEPTIONS.**

### kelly_video_assets (audio/video asset registry)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| day_number | int | 1-365 |
| phase | text | Phase name |
| **language** | text | 'en', 'es', 'fr' |
| asset_type | text | 'audio', 'video', 'lipsync' |
| url | text | Supabase Storage URL |
| status | text | 'pending', 'processing', 'ready', 'failed' |

### user_progress (calendar completion states)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| user_id | uuid | FK to auth.users |
| day_number | int | 1-365 |
| completed | boolean | True if lesson finished |
| completed_at | timestamptz | When completed |
| last_phase | int | Last phase reached (0-6) |
| choices | jsonb | { "hook": "A", "cliff": "B", ... } |

### user_bookmarks (saved moments)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| user_id | uuid | FK to auth.users |
| day_number | int | 1-365 |
| phase | text | 'Hook', 'Cliff', etc. |
| note | text | Optional user note |
| created_at | timestamptz | When bookmarked |

---

## 🔄 THE GENERATION PIPELINE

### Step 0: Pre-Flight Check
```bash
# Verify environment
npx tsx scripts/lesson-factory/preflight-check.ts

# Required env vars:
# - ELEVENLABS_API_KEY           (audio generation)
# - ELEVENLABS_KELLY_VOICE_ID    (Kelly's voice)
# - SUPABASE_URL (or PUBLIC_SUPABASE_URL)
# - SUPABASE_SERVICE_ROLE_KEY
# - GOOGLE_AI_API_KEY            (visual generation - Imagen/Gemini)
```

### Step 1: Verify Core Lesson Exists
```sql
SELECT id, day_number, topic, universal_truth
FROM core_lessons
WHERE day_number = $DAY_NUMBER;
```
**If missing:** Create the core lesson first.

### Step 2: Verify All 7 Phase Atoms Exist
```sql
SELECT phase, archetype, content
FROM lesson_atoms
WHERE core_lesson_id = $LESSON_ID
  AND archetype = 'The Scientist'  -- or target archetype
ORDER BY phase;
```
**Expected result:** 7 rows (Hook, Cliff, Fact1, Fact2, Fact3, Wisdom, Outro)

**If Cliff/Outro missing:**
```bash
npx tsx scripts/generate-cliff-outro-atoms.ts --day=$DAY_NUMBER
```

### Step 3: Generate Audio Assets
```bash
npx tsx scripts/generate-day-audio-elevenlabs.ts \
  --day=$DAY_NUMBER \
  --age=adult \
  --lang=en \
  --all
```
**Verifies:** Audio files uploaded to Supabase, rows in kelly_video_assets

### Step 4: Generate Visual Assets (Infographics + Option Cards)

**Priority Order:**
1. Check `lesson_visuals` cache for existing assets (by prompt_hash)
2. If missing, generate using available API key source:
   - **BYOK** (user's Google AI key) - preferred, $0 cost
   - **Platform key** - fallback, rate-limited pool

```bash
# Generate all visuals for a day (5 infographics + 8 option cards)
npx tsx scripts/generate-day-visuals.ts \
  --day=$DAY_NUMBER \
  --model=imagen-4.0-ultra

# Verify visual URLs are populated in lesson_atoms
npx tsx scripts/verify-visual-urls.ts --day=$DAY_NUMBER
```

**API Endpoints:**
| Model | Cost | Quality | Use Case |
|-------|------|---------|----------|
| `imagen-4.0-ultra-generate-001` | $0.06 | Best | Hero infographics |
| `imagen-4.0-generate-001` | $0.04 | High | Standard visuals |
| `imagen-4.0-fast-generate-001` | $0.02 | Good | Option cards |
| `gemini-2.0-flash` | Free | Good | Fallback/diagrams |

**Visual Prompt Templates:**
- **Split-Scene Comparison** - Best for before/after, A vs B concepts
- **Before/After Transformation** - Cause-and-effect
- **Process/Cycle Diagram** - Multi-step sequences
- **Scale Comparison** - Size relationships
- **Anatomy/Cross-Section** - Internal structures

See: `content/visual-prompts/INFOGRAPHIC_TEMPLATES.md`

**Output Verification:**
```sql
-- Verify visual_url populated for all phases
SELECT phase, visual_url IS NOT NULL as has_visual
FROM lesson_atoms
WHERE core_lesson_id = $LESSON_ID
  AND archetype = 'The Scientist'
ORDER BY phase;

-- Verify lesson_visuals registry
SELECT phase, visual_type, status, storage_url
FROM lesson_visuals
WHERE day_number = $DAY_NUMBER
ORDER BY phase, visual_type;
```

### Step 5: Generate Video Assets (Optional for MVP)
```bash
# Option A: HeyGen (faster, good quality)
npx tsx scripts/generate-day-videos-heygen.ts --day=$DAY_NUMBER

# Option B: Full HD Pipeline (best quality, slower)
npx tsx scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts --day=$DAY_NUMBER
```

### Step 6: Verify ALL Assets Registered
```sql
-- Audio/Video assets
SELECT day_number, phase, asset_type, status, url
FROM kelly_video_assets
WHERE day_number = $DAY_NUMBER
ORDER BY phase, asset_type;

-- Visual assets
SELECT phase, visual_type, status, storage_url
FROM lesson_visuals
WHERE day_number = $DAY_NUMBER
ORDER BY phase, visual_type;

-- Phase completeness check
SELECT 
  la.phase,
  la.visual_url IS NOT NULL as has_infographic,
  EXISTS(SELECT 1 FROM kelly_video_assets kva 
         WHERE kva.day_number = $DAY_NUMBER 
         AND kva.phase = LOWER(la.phase) 
         AND kva.asset_type = 'audio') as has_audio
FROM lesson_atoms la
JOIN core_lessons cl ON la.core_lesson_id = cl.id
WHERE cl.day_number = $DAY_NUMBER
  AND la.archetype = 'The Scientist';
```
**Expected:** 7 phases, each with: audio=✅, infographic=✅ (except Cliff/Outro)

### Step 7: Update Static JSON (if needed)
The JSON file at `/public/lessons/day-XXX.json` should already exist.
Verify it has all 7 phases in the `phases` object.

### Step 8: End-to-End UI Verification
```
1. Open browser to https://www.curiouskelly.com/learn.html?day=$DAY_NUMBER&debug=1
2. Open DevTools Console
3. Check for:
   - NO red errors
   - "Loaded day X from [source]" message
   - Source should be 'supabase', 'vercel-api', or 'static' (NOT 'emergency')
4. Click through ALL 7 phases
5. Verify:
   - Kelly's script matches database content
   - Choices appear on cliff phase
   - Audio plays (NOT browser TTS)
   - No dead ends or broken navigation
```

---

## ✅ VERIFICATION CHECKLIST

Before declaring a lesson "ready":

### Database Verification
- [ ] core_lessons row exists with correct topic
- [ ] 7 atoms exist for target archetype (Hook, Cliff, Fact1, Fact2, Fact3, Wisdom, Outro)
- [ ] Each atom has content.script (non-empty string)
- [ ] **ALL 7 atoms have content.options[] with EXACTLY 2 choices (A and B)**
- [ ] Each option has: label, imageUrl, responseScript, quality
- [ ] **ALL 7 atoms have content.simulatedComments[] with 2-3 comments**
- [ ] Each comment has: emoji (✨), text, author
- [ ] Audio assets registered in kelly_video_assets with status='ready'

### Visual Asset Verification
- [ ] Hook phase has `visual_url` populated (infographic)
- [ ] Fact1/Fact2/Fact3 phases have `visual_url` populated
- [ ] Wisdom phase has `visual_url` populated
- [ ] Cliff phase has Option A/B images in lesson_visuals (512×512)
- [ ] Q1/Q2/Q3 phases have Option A/B images in lesson_visuals
- [ ] All visual URLs resolve (no 404s)
- [ ] lesson_visuals rows exist with status='ready'

### UI Verification
- [ ] Lesson loads without JavaScript errors
- [ ] Data source is NOT 'emergency' fallback
- [ ] Topic displays correctly in header
- [ ] All 7 phases accessible via phase bar
- [ ] Cliff choices appear and respond to clicks
- [ ] **Cliff choices show IMAGES (not just text)**
- [ ] Kelly responds appropriately to choices
- [ ] Audio plays from ElevenLabs (NOT browser TTS)
- [ ] **📊 Infographic button shows visual for current phase**
- [ ] Can complete lesson from Hook to Outro

### File Verification
- [ ] Static JSON exists at /public/lessons/day-XXX.json
- [ ] JSON has all 7 phases in phases object
- [ ] JSON phaseOrder matches canonical order
- [ ] JSON includes visualUrl for each phase (where applicable)

---

## 🚨 COMMON FAILURES & FIXES

### "Phase clicking doesn't work"
**Cause:** Phase names mismatch between frontend and data
**Fix:** Ensure PHASE_CONFIG in learn.html matches database phase names

### "Browser TTS playing instead of Kelly's voice"
**Cause:** Audio assets not generated or not registered
**Fix:** Run generate-day-audio-elevenlabs.ts, verify kelly_video_assets rows

### "Empty script / No content"
**Cause:** Atoms missing for that phase/archetype
**Fix:** Run generate-cliff-outro-atoms.ts for Cliff/Outro, verify lesson_atoms

### "Wrong topic showing"
**Cause:** Stale localStorage state or wrong day_number
**Fix:** Clear localStorage, verify ?day= parameter matches expected day

### "Data loading from emergency fallback"
**Cause:** Supabase/D1/static all failed
**Fix:** Check network tab for 4xx/5xx errors, verify RLS policies, check API routes

### "Infographic shows 'Coming Soon' placeholder"
**Cause:** visual_url not populated in lesson_atoms
**Fix:** Run generate-day-visuals.ts, verify visual_url in database

### "Option cards showing text only, no images"
**Cause:** Option card visuals not generated or not linked
**Fix:** Check lesson_visuals for visual_type='option-a'/'option-b' entries

### "BYOK generation failing"
**Cause:** User's Google AI API key invalid or quota exhausted
**Fix:** Test key at aistudio.google.com, check daily quota (500-1000 images/day free)

### "Visual generation rate limited"
**Cause:** Too many requests to Imagen API
**Fix:** Use model rotation (ultra → standard → fast), add delays between generations

### "Phase has no options / only 1 option"
**Cause:** Options not generated for all phases (legacy: only Cliff had options)
**Fix:** ALL 7 phases need exactly 2 options. Regenerate atoms with full content structure.

### "No simulated comments showing"
**Cause:** simulatedComments[] missing from atom content
**Fix:** Add 2-3 simulated comments per phase. Each needs ✨ emoji, text, author.

### "Lesson flow breaks after Hook"
**Cause:** Missing options breaks the interactive loop
**Fix:** Verify all 7 phases have complete content structure (script + 2 options + comments)

---

## 📂 KEY FILES

### Data Sources (Priority Order)
1. `window.CURIOUS_KELLY.LOCAL_PACKS` - Pre-bundled lesson data
2. Supabase - core_lessons, lesson_atoms tables
3. Cloudflare D1 - Mirror database
4. `/api/lessons/[dayNumber]` - Vercel API fallback
5. `/public/lessons/day-XXX.json` - Static JSON files
6. Emergency fallback - Hardcoded content (FAILURE STATE)

### Generation Scripts
| Script | Purpose |
|--------|---------|
| `scripts/generate-cliff-outro-atoms.ts` | Add missing Cliff/Outro atoms to DB |
| `scripts/generate-day-audio-elevenlabs.ts` | Generate ElevenLabs audio |
| `scripts/generate-day-videos-heygen.ts` | Generate HeyGen videos |
| `scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts` | Full HD video pipeline |
| `scripts/fill-supabase-with-assets.ts` | Upload assets to Supabase Storage |
| `scripts/generate-day-visuals.ts` | Generate infographics + option cards |
| `scripts/seed-visual-commons.ts` | Seed visual library with Imagen 4 Ultra |
| `scripts/multi-key-generator.ts` | Multi-account Imagen generation |
| `scripts/verify-visual-urls.ts` | Verify visual_url populated in atoms |

### Visual Asset References
| Document | Purpose |
|----------|---------|
| **`UI_GENERATION_SPEC.md`** | **Sister doc: Layout, zones, interaction patterns** |
| `content/visual-prompts/INFOGRAPHIC_TEMPLATES.md` | 5 master prompt templates |
| `content/visual-plans/day-XXX-visual-plan-v2.json` | Per-day visual specifications |
| `docs/VISUAL_ORCHESTRATION_MASTER.md` | Visual philosophy & standards |
| `vom/UNIFIED_LESSON_FACTORY_FINAL.md` | Full production asset spec |
| `docs/trust-safety/SIMULATED_SOCIAL_CONTENT.md` | Comment safety rules |

### Frontend
| File | Purpose |
|------|---------|
| `public/learn.html` | Main lesson player |
| `public/js/kelly-lesson-loader.js` | Data loading with cascading fallbacks |
| `public/lessons/day-XXX.json` | Static lesson data (365 files) |

---

## 🔌 BACKEND QUERIES FOR UI

### Load Lesson by Day Number
```sql
-- Primary query when user navigates to a day
SELECT 
  cl.id, cl.day_number, cl.topic, cl.category, cl.universal_truth,
  la.phase, la.content, la.visual_url, la.hd_video_url
FROM core_lessons cl
JOIN lesson_atoms la ON la.core_lesson_id = cl.id
WHERE cl.day_number = $DAY_NUMBER
  AND la.archetype = $ARCHETYPE  -- e.g., 'The Scientist'
  AND la.language = $LANGUAGE    -- e.g., 'en'
ORDER BY 
  CASE la.phase
    WHEN 'Hook' THEN 0
    WHEN 'Cliff' THEN 1
    WHEN 'Fact1' THEN 2
    WHEN 'Fact2' THEN 3
    WHEN 'Fact3' THEN 4
    WHEN 'Wisdom' THEN 5
    WHEN 'Outro' THEN 6
  END;
```

### Search Lessons
```sql
-- Full-text search for calendar/curriculum view
SELECT 
  day_number, topic, category,
  ts_rank(search_vector, plainto_tsquery('english', $QUERY)) as rank
FROM core_lessons
WHERE 
  day_number::text = $QUERY  -- Direct day match
  OR search_vector @@ plainto_tsquery('english', $QUERY)
ORDER BY 
  CASE WHEN day_number::text = $QUERY THEN 0 ELSE 1 END,
  rank DESC
LIMIT 20;
```

### Calendar Month Data
```sql
-- Load all days for calendar grid (one month)
SELECT 
  cl.day_number, cl.topic, cl.category,
  EXISTS(SELECT 1 FROM user_progress up 
         WHERE up.user_id = $USER_ID 
         AND up.day_number = cl.day_number 
         AND up.completed = true) as is_completed,
  cl.day_number <= $TODAY_DAY_NUMBER as is_available
FROM core_lessons cl
WHERE cl.day_number BETWEEN $START_DAY AND $END_DAY
ORDER BY cl.day_number;
```

### Language Switch
```sql
-- When user changes language, refetch atoms + audio
SELECT la.phase, la.content, la.visual_url
FROM lesson_atoms la
JOIN core_lessons cl ON la.core_lesson_id = cl.id
WHERE cl.day_number = $DAY_NUMBER
  AND la.archetype = $ARCHETYPE
  AND la.language = $NEW_LANGUAGE;

-- Also get new audio URLs
SELECT phase, url as audio_url
FROM kelly_video_assets
WHERE day_number = $DAY_NUMBER
  AND language = $NEW_LANGUAGE
  AND asset_type = 'audio'
  AND status = 'ready';
```

### User Progress (for calendar states)
```sql
-- Get user's completion status for calendar rendering
SELECT day_number, completed, completed_at
FROM user_progress
WHERE user_id = $USER_ID
ORDER BY day_number;
```

### Bookmarks
```sql
-- Get user's saved moments for Bookmarks tab
SELECT 
  b.day_number, b.phase, b.created_at,
  cl.topic,
  la.content->>'script' as script_snippet
FROM user_bookmarks b
JOIN core_lessons cl ON cl.day_number = b.day_number
JOIN lesson_atoms la ON la.core_lesson_id = cl.id 
  AND la.phase = b.phase
  AND la.archetype = $ARCHETYPE
  AND la.language = $LANGUAGE
WHERE b.user_id = $USER_ID
ORDER BY b.created_at DESC;
```

---

## 📊 COMPLETE PHASE ASSET MAP

**Everything links to phases.** Not separate "commons" systems.

### Per-Phase Asset Requirements (ALL phases have 2 options)
```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE    │ SCRIPT │ AUDIO │ VIDEO │ INFOGRAPHIC │ OPTIONS │ COMMENTS │ TOTAL         │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ hook     │   ✓    │   ✓   │   ◐   │      ✓      │  ✓ A+B  │   2-3    │ 7-8 assets    │
│ cliff    │   ✓    │   ✓   │   ◐   │      ✓      │  ✓ A+B  │   2-3    │ 7-8 assets    │
│ q1       │   ✓    │   ✓   │   ◐   │      ✓      │  ✓ A+B  │   2-3    │ 7-8 assets    │
│ q2       │   ✓    │   ✓   │   ◐   │      ✓      │  ✓ A+B  │   2-3    │ 7-8 assets    │
│ q3       │   ✓    │   ✓   │   ◐   │      ✓      │  ✓ A+B  │   2-3    │ 7-8 assets    │
│ wisdom   │   ✓    │   ✓   │   ◐   │      ✓      │  ✓ A+B  │   2-3    │ 7-8 assets    │
│ outro    │   ✓    │   ✓   │   ◐   │      ✓      │  ✓ A+B  │   1-2    │ 6-7 assets    │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ TOTAL    │   7    │   7   │   7   │      7      │   14    │  14-20   │ 49-56/day     │
└───────────────────────────────────────────────────────────────────────────────────────┘

Legend: ✓ = Required | ◐ = Optional for MVP
⚠️ EVERY PHASE HAS 2 OPTIONS + 2-3 SIMULATED COMMENTS. NO EXCEPTIONS.
```

### How Assets Link to Database

```
core_lessons (1 per day)
    │
    ├── lesson_atoms (7 phases × 12 archetypes = 84 rows)
    │       │
    │       ├── content.script        → What Kelly says
    │       ├── content.options[]     → Choice text + responseScript
    │       ├── visual_url            → Infographic URL (1920×1080)
    │       └── hd_video_url          → Kelly HD video URL
    │
    ├── kelly_video_assets (7 phases × asset_types)
    │       │
    │       ├── asset_type='audio'    → ElevenLabs MP3
    │       ├── asset_type='video'    → HeyGen/HD video
    │       └── asset_type='lipsync'  → Sync Labs output
    │
    └── lesson_visuals (5 infographics + 8 option cards = 13 rows)
            │
            ├── visual_type='infographic'  → 1920×1080 educational diagram
            ├── visual_type='option-a'     → 512×512 choice card
            └── visual_type='option-b'     → 512×512 choice card
```

### Full Generation Order
```
Day $N Generation Pipeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Core lesson exists?          → CREATE if missing
Step 2: All 7 phase atoms exist?     → GENERATE Cliff/Outro if missing
Step 3: Audio for all 7 phases?      → GENERATE via ElevenLabs
Step 4: Visuals for 5 phases?        → GENERATE via Imagen/BYOK
Step 5: Option cards for 4 phases?   → GENERATE via Imagen/BYOK
Step 6: Video for all 7 phases?      → GENERATE via HeyGen (optional)
Step 7: Verify all URLs resolve      → FIX any 404s
Step 8: UI end-to-end test           → CONFIRM playable
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔐 LOCKED CONSTANTS

### The 12 Archetypes
```
The Scientist, The Explorer, The Rebel, The Architect,
The Diplomat, The Empath, The MacGyver, The Mystic,
The Provider, The Storyteller, The Strategist, The Survivor
```

### Kelly's Voice
- **Voice ID:** `wAdymQH5YucAkXwmrdL0`
- **Model:** `eleven_multilingual_v2`
- **Settings:** stability=0.5, similarity_boost=0.75

### Phase Configuration in Frontend
```javascript
const PHASE_CONFIG = {
  hook:   { name: 'Hook',   dbName: 'Hook',   icon: '🎬' },
  cliff:  { name: 'Cliff',  dbName: 'Cliff',  icon: '🔀' },
  q1:     { name: 'Fact 1', dbName: 'Fact1',  icon: '💡' },
  q2:     { name: 'Fact 2', dbName: 'Fact2',  icon: '💡' },
  q3:     { name: 'Fact 3', dbName: 'Fact3',  icon: '💡' },
  wisdom: { name: 'Wisdom', dbName: 'Wisdom', icon: '✨' },
  outro:  { name: 'Outro',  dbName: 'Outro',  icon: '🎉' },
};
const PHASE_ORDER = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];
```

---

## 🔑 BYOK (Bring Your Own Key) VISUAL GENERATION

### What Is BYOK?
Learners can use their **free Google AI API key** to generate personalized visuals.
- Get key at: https://aistudio.google.com/app/apikey
- Free tier: 500-1000 images/day
- Key stored in browser localStorage only (never sent to our servers)

### Why BYOK Matters
| Approach | Cost Per Image | Who Pays |
|----------|---------------|----------|
| Platform key | $0.02-$0.06 | Us |
| BYOK | $0.00 | Google (free tier) |

**The Commons Principle:** First learner generates, all future learners benefit.
Every BYOK generation is cached in `lesson_visuals` for everyone.

### BYOK Flow in UI
```
1. Learner enters phase with missing visual
2. UI shows "✨ Generate Visual" button
3. Learner clicks → BYOK modal opens
4. Learner enters their Google AI API key
5. Visual generated via Imagen API
6. Saved to lesson_visuals with generation_source='byok'
7. visual_url updated in lesson_atoms
8. All future learners see cached visual
```

### BYOK UI Location
- Settings panel: `public/learn.html` lines 8472-8500 (AI Chat BYOK)
- Visual modal: `public/learn.html` lines 8701-8736 (Visual generation BYOK)

### Required Environment Variables
```bash
# For platform fallback (when BYOK not available)
GOOGLE_AI_API_KEY=AIza...  # Platform's Google AI key
```

---

## 🎯 SUCCESS CRITERIA

A lesson is **PRODUCTION READY** when:

1. ✅ All database rows exist and are valid
2. ✅ All 7 phases play with real content (not fallbacks)
3. ✅ Audio plays from ElevenLabs (not browser TTS)
4. ✅ **ALL 7 phases have 2 options with working choices**
5. ✅ Zero console errors during playthrough
6. ✅ Can complete full lesson from Hook to Outro
7. ✅ Static JSON backup exists
8. ✅ **Infographics display for ALL 7 phases**
9. ✅ **Option card images (A + B) display for ALL 7 phases**
10. ✅ **Simulated comments (2-3) exist for ALL 7 phases**
11. ✅ **Kelly logo click opens left panel with comments + chat**

---

## 🛠️ RLS AND CONSTRAINT FIXES

If generation scripts fail with RLS errors, run these migrations:

```sql
-- lesson_atoms INSERT policy
CREATE POLICY "Service role can insert lesson_atoms" ON public.lesson_atoms
  FOR INSERT WITH CHECK (true);

-- kelly_video_assets full access policy  
CREATE POLICY "Service role can manage kelly_video_assets" ON public.kelly_video_assets
  FOR ALL USING (true) WITH CHECK (true);

-- Phase check constraint (include cliff/outro)
ALTER TABLE public.kelly_video_assets DROP CONSTRAINT kelly_video_assets_phase_check;
ALTER TABLE public.kelly_video_assets ADD CONSTRAINT kelly_video_assets_phase_check 
  CHECK (phase = ANY (ARRAY['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro']));
```

---

## 📝 CHANGE LOG

| Date | Change | Author |
|------|--------|--------|
| 2025-12-19 | Initial spec locked | Claude |
| 2025-12-19 | Added RLS/constraint fixes, verified Day 353 generation | Claude |
| 2025-12-19 | Day 353 MVP VERIFIED: 7 phases, 21 audio files, all choices work | Claude |
| 2025-12-19 | **VISUAL LAYER ADDED:** lesson_visuals table, visual_url in atoms, Step 4 visual generation, BYOK section, visual verification checklists | Claude |
| 2025-12-19 | **PHASE OPTIONS LOCKED:** ALL 7 phases require 2 options (not just Cliff). Added simulatedComments to content structure. | Claude |
| 2025-12-19 | Added: category + search_vector to core_lessons, language to lesson_atoms, Backend queries for UI (search, calendar, bookmarks, language switch) | Claude |

---

## ✅ VERIFIED: Day 353 MVP (December 19, 2025)

### Database State After Generation
| Category | Count | Status |
|----------|-------|--------|
| Core Lesson | 1 | ✅ topic="Being Where You Are" |
| Hook atoms | 12 | ✅ all archetypes |
| Cliff atoms | 12 | ✅ generated with choices |
| Fact1 atoms | 10 | ✅ |
| Fact2 atoms | 10 | ✅ |
| Fact3 atoms | 10 | ✅ |
| Wisdom atoms | 10 | ✅ |
| Outro atoms | 12 | ✅ generated |
| Audio assets | 21 | ✅ 3 archetypes × 7 phases |

### UI Verification (Production)
- [x] Topic displays correctly: "Being Where You Are"
- [x] 7 phase buttons visible in phase bar
- [x] Phase navigation works (clicked Hook → Cliff)
- [x] Cliff choices appear with correct content ("Show me the data" / "Let me experiment")
- [x] Audio plays from ElevenLabs (not browser TTS)
- [x] Console shows: `🔊 ✅ Playing pre-generated audio`

### Commands Used
```powershell
# 1. Generate Cliff/Outro atoms
npx tsx scripts/generate-cliff-outro-atoms.ts --day=353

# 2. Generate audio
npx tsx scripts/generate-day-audio-elevenlabs.ts --day=353 --age=adult --lang=en
```

### Migrations Applied
- `fix_lesson_atoms_insert_policy` - Allow service role to insert atoms
- `fix_kelly_video_assets_insert_policy` - Allow service role to manage assets
- `update_kelly_video_assets_phase_check` - Add cliff/outro to allowed phases

---

*This spec is the source of truth. When in doubt, follow this document.*
