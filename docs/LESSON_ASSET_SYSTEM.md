# Lesson Asset System - Complete Guide

**Created:** December 3, 2025  
**Status:** Ready for deployment  
**Purpose:** Generate assets ONCE, cache forever, share across all learners

---

## 🎯 The Vision

When a learner opens a lesson, every asset (thumbnail, Kelly images, audio, video) should:
1. **Already exist** - pre-generated and cached
2. **Load instantly** - from CDN, not generated on-demand
3. **Never regenerate** - if another student already triggered generation

This is the **Lesson Creation Factory** - the core engine that powers Curious Kelly.

---

## 📋 What Was Built

### 1. Database Schema (Run in Supabase)

Two SQL migrations need to be run:

#### Migration 003: Lesson Assets System
```
sql/migrations/003_lesson_assets_system.sql
```

Creates:
- `thumbnail_slug` column on `core_lessons` table
- `lesson_assets` table - stores ALL cached content (images, audio, video)
- `lesson_variant_cache` table - tracks what's been generated per variant
- `get_or_create_variant_cache()` function - checks/creates cache entries
- `get_lesson_thumbnail()` function - gets thumbnail URL with fallbacks

#### Migration 003b: Populate Thumbnail Slugs
```
sql/migrations/003b_populate_thumbnail_slugs.sql
```

Populates `thumbnail_slug` for 81 lessons that have existing thumbnails.

### 2. Frontend Code

#### Updated: `public/curriculum.html`
- Now fetches `thumbnail_slug` from database
- `getThumbnailPath()` uses the slug instead of generating from topic

#### New: `public/js/lesson-assets.js`
- `LessonAssetManager` class with:
  - Thumbnail URL generation
  - Phase image lookups with fallback chains
  - Variant cache checking
  - Asset preloading
- `KELLY_PROMPT_LIBRARY` - prompts for each phase

### 3. Thumbnail Files

**81 thumbnails** now exist in `public/kelly/thumbnails/raw/`:
- Days 1-31 (January)
- Days 32-81 (February-March partial)

---

## 🚀 Deployment Steps

### Step 1: Run SQL Migrations in Supabase

1. Go to Supabase Dashboard → SQL Editor
2. Run `sql/migrations/003_lesson_assets_system.sql`
3. Run `sql/migrations/003b_populate_thumbnail_slugs.sql`
4. Verify with:
```sql
SELECT day_number, topic, thumbnail_slug 
FROM core_lessons 
WHERE thumbnail_slug IS NOT NULL 
ORDER BY day_number 
LIMIT 20;
```

### Step 2: Clear Vercel Build Cache

1. Go to Vercel Dashboard → curiouskelly project
2. Settings → General → Clear Build Cache
3. Redeploy (or wait for auto-deploy from git push)

### Step 3: Verify on Live Site

1. Go to https://curiouskelly.com/curriculum.html
2. Check browser console - should see "📸 Loaded X thumbnail mappings"
3. Thumbnails for days 1-81 should display (no more blue placeholders)

---

## 📁 File Structure

```
public/kelly/
├── thumbnails/
│   ├── raw/                    # 81 lesson thumbnails
│   │   ├── lesson-001-starting-fresh.png
│   │   ├── lesson-002-the-three-lives-of-water.png
│   │   └── ... (81 files)
│   ├── approved/               # QC-approved versions
│   ├── final/                  # Production-ready
│   └── rejected/               # Failed QC
├── lessons/
│   ├── 001/                    # Per-lesson phase images
│   │   ├── lesson-1-hero.png
│   │   ├── lesson-1-guide-point.png
│   │   └── ...
│   └── ... (1-31 complete)
├── poses/                      # Base Kelly poses (fallbacks)
│   ├── kelly_welcome.png
│   ├── kelly_thinking.png
│   └── ...
└── choices/                    # Choice buttons
    ├── choice_left.png
    └── choice_right.png
```

---

## 🔄 Asset Generation Pipeline

### For Thumbnails (Hero Images)

1. **Generation**: Use Flux/DALL-E with prompts from `KELLY_PROMPT_LIBRARY`
2. **Naming**: `lesson-{DDD}-{slug}.png` where slug matches `thumbnail_slug`
3. **Storage**: `public/kelly/thumbnails/raw/`
4. **Database**: Set `thumbnail_slug` in `core_lessons`

### For Phase Images

1. **Per-lesson images**: `public/kelly/lessons/{DDD}/lesson-{day}-{phase}.png`
2. **Phases**: hero, q1, q2, q3, hook, wisdom, reaction
3. **Fallback**: Uses base poses from `public/kelly/poses/`

### For Variants (Future)

The `lesson_variant_cache` table tracks:
- Language variants (en, es, fr)
- Age bucket variants (toddler → senior)
- Archetype variants (The Survivor, The Explorer, etc.)
- Tone variants (playful, curious, serious)

When a learner requests a variant:
1. Check `lesson_variant_cache` - does it exist?
2. If yes → serve from cache
3. If no → generate, save to `lesson_assets`, update cache

---

## 📊 Current Status

| Asset Type | Days Covered | Files |
|------------|--------------|-------|
| Thumbnails (raw) | 1-81 | 81 |
| Per-lesson images | 1-31 | ~150 |
| Base poses | Universal | 11 |
| **Total** | | **~242** |

### Remaining Work

| Task | Days | Files Needed |
|------|------|--------------|
| Thumbnails | 82-365 | 284 |
| Per-lesson images | 32-365 | ~1,670 |
| Audio (per phase) | 1-365 | ~2,555 |
| Video (animated Kelly) | 1-365 | ~2,555 |

---

## 🎨 Prompt Engineering

Each phase has a specific prompt in `KELLY_PROMPT_LIBRARY`:

```javascript
phases: {
    hero: (topic) => `Kelly introducing "${topic}". Curious, inviting expression.`,
    intro: (topic) => `Kelly warmly welcomes learner. Excited eyes, warm smile.`,
    q1: (topic) => `Kelly presents first question. Encouraging, head tilted.`,
    q2: (topic) => `Kelly asks second question. Leaning forward, interested.`,
    q3: (topic) => `Kelly presents final question. Thoughtful yet supportive.`,
    hook: (topic) => `Kelly reveals surprise. Wide eyes, delighted discovery.`,
    wisdom: (topic) => `Kelly shares final wisdom. Warm, contemplative look.`,
    correct: () => `Kelly shows pride and joy. Authentic teacher happiness.`,
    incorrect: () => `Kelly responds warmly. Understanding, encouraging, no judgment.`,
}
```

---

## 🔑 Key Functions

### getThumbnailUrl(lesson)
```javascript
// Uses canonical thumbnail_slug from database
// Fallback to per-lesson folder
// Returns: /kelly/thumbnails/raw/lesson-001-starting-fresh.png
```

### getPhaseImageUrl(dayNumber, phase)
```javascript
// Returns: { primary: lesson-specific, fallback: base-pose }
// Phases: hero, intro, q1, q2, q3, hook, wisdom, correct, incorrect
```

### checkVariantCache(dayNumber, phase, options)
```javascript
// Checks if variant (language/age/archetype/tone) exists
// Creates cache entry if not
// Returns: { cacheId, isComplete, assetsReady, needsGeneration }
```

---

## 🚨 Important Notes

1. **Don't regenerate existing thumbnails** - they're already styled
2. **Thumbnail slugs are canonical** - must match actual filenames
3. **Per-lesson images have different naming** - `lesson-{day}-{type}.png`
4. **Fallback chain is important** - always have base poses

---

## Next Steps for Tomorrow

1. ✅ Run SQL migrations in Supabase
2. ✅ Clear Vercel cache and redeploy
3. ⏳ Generate thumbnails for days 82-365
4. ⏳ Generate per-lesson phase images for days 32-365
5. ⏳ Integrate into learn.html with phase-specific Kelly images
6. ⏳ Set up audio generation pipeline (ElevenLabs)

---

*This system ensures Kelly's visual identity is consistent, cached, and shared across all learners forever.*


