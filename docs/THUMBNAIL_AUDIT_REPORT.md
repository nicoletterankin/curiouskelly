# Thumbnail & Visual Assets Audit Report

**Date:** December 19, 2025  
**Auditor:** Claude (AI Assistant)

---

## Executive Summary

| Metric | Count | Status |
|--------|-------|--------|
| Total days in journey | 365 | ✅ |
| Days with file thumbnails | 31 | ⚠️ January only |
| Days with Supabase `thumbnail_url` | 1 | ❌ Critical gap |
| Days with phase visuals | 5 | ⚠️ Days 1-5 only |
| **Days missing thumbnails** | **334** | ❌ Needs fix |

---

## Current Thumbnail Sources

### 1. File System Thumbnails (31/365)
**Location:** `public/assets/kelly/production/thumbnails/january/`

```
✅ January (Days 1-31): 31 files exist
   Format: lesson-{N}.webp (640x360)
   
❌ February (Days 32-59): 0 files
❌ March (Days 60-90): 0 files
❌ April-December (Days 91-365): 0 files
```

**Files found:**
- `lesson-1.webp` through `lesson-31.webp`

### 2. Supabase `core_lessons.thumbnail_url` (1/365)
Only **Day 17** has a populated thumbnail_url:
```
Day 17: https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/thumbnails/017-why-bodies-need-to-move.png
```

### 3. Supabase `kelly_video_assets` (25 images)
These are **phase-specific** images, NOT calendar thumbnails:
```
Days 1-5: hook, q1, q2, q3, wisdom images (5 images per day)
Location: kelly-templates/production/images/
Example: day_001_hook.png, day_001_q1.png, etc.
```

### 4. Manifest File
**Location:** `public/assets/kelly/production/thumbnails/manifest.json`

```json
{
  "coverage": {
    "total": 365,
    "available": 31,
    "missing": 334,
    "completedMonths": ["january"],
    "pendingMonths": ["february"]
  }
}
```

---

## How Thumbnails Are Currently Used

### Journey Calendar (`learn.html`)
```javascript
function getLessonThumbnailUrl(day) {
  // 1. Check THUMBNAIL_INDEX from manifest
  // 2. Fallback to January webp path
  // 3. Return null if not found
}
```

**Problem:** Days 32-365 return `null` → empty circles in calendar.

### Index Page (`index.html`)
```javascript
// Tries to fetch visual_url from lesson_atoms
// Falls back to hero_image_url
// Displays gradient if neither exists
```

---

## Root Causes

1. **Generation incomplete:** Only January thumbnails were generated via Replicate/FLUX
2. **Sync missing:** File thumbnails not synced to Supabase `core_lessons.thumbnail_url`
3. **No fallback chain:** UI shows empty state instead of generated placeholder

---

## Fix Plan

### Phase 1: Immediate (Sync Existing)
1. Sync 31 January file thumbnails to Supabase `core_lessons.thumbnail_url`
2. Update manifest to reflect actual file coverage

### Phase 2: Generate Missing (334 days)
Run `scripts/kelly-visual-identity/generate-all-365-thumbnails.ts`:
- Uses FLUX Dev + Kelly LoRA
- Outputs to `public/kelly/thumbnails/raw/`
- Requires Replicate API credits (~$30-50 for full run)

### Phase 3: Production Optimization
1. Convert raw PNGs to optimized WebP
2. Upload to Supabase Storage `lesson-visuals` bucket
3. Update `core_lessons.thumbnail_url` for each day
4. Update manifest with full coverage

### Phase 4: UI Fallback
Add smart fallback in `getLessonThumbnailUrl()`:
1. Supabase `thumbnail_url` (if populated)
2. Production webp file (`/assets/kelly/production/thumbnails/{month}/lesson-{N}.webp`)
3. Raw PNG file (`/kelly/thumbnails/raw/lesson-{N}-*.png`)
4. Generated CSS thumbnail (using `KellyThumbnailGenerator`)

---

## Recommended Immediate Actions

### Script: Sync January to Supabase
```sql
-- Run for each day 1-31
UPDATE core_lessons 
SET thumbnail_url = 'https://curiouskelly.com/assets/kelly/production/thumbnails/january/lesson-{N}.webp'
WHERE day_number = {N};
```

### Script: Generate Remaining Days
```bash
npx tsx scripts/kelly-visual-identity/generate-all-365-thumbnails.ts
```

### Manifest Update
Add entries for days 32-365 as they're generated.

---

## Cost Estimate

| Item | Count | Cost |
|------|-------|------|
| Replicate FLUX (334 images) | 334 | ~$30-50 |
| Storage (WebP @ 50KB avg) | 365 | ~18MB total |
| CDN bandwidth | - | Negligible |

---

## Files to Create/Modify

1. ✅ `docs/THUMBNAIL_AUDIT_REPORT.md` (this file)
2. 📝 `scripts/sync-thumbnails-to-supabase.ts` (new)
3. 📝 `scripts/generate-missing-thumbnails.ts` (wrapper)
4. 📝 Update `public/assets/kelly/production/thumbnails/manifest.json`
5. 📝 Add fallback in `public/js/` thumbnail utilities

---

## Appendix: Data Sources

### Supabase Tables Used
- `core_lessons` - Main curriculum (365 rows)
- `kelly_video_assets` - Generated assets (2,242 rows total)
- `lesson_atoms` - Phase content with `visual_url`

### Storage Buckets
- `kelly-templates` - Production assets
- `lesson-visuals` - Thumbnail storage

### Generation Scripts
- `scripts/kelly-visual-identity/generate-all-365-thumbnails.ts`
- `scripts/kelly-visual-identity/generate-thumbnails-january.ts`
- `scripts/kelly-visual-identity/generate-thumbnails-february.ts`
- `scripts/generate-day17-thumbnail.ts`

