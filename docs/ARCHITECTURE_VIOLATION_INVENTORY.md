# Architecture Violation Inventory
## Frontend Direct Supabase Access Points

**Date:** January 2025  
**Status:** AUDIT COMPLETE - FIX IN PROGRESS  
**Total Violations:** 71+ instances across 26 files

---

## 📊 Summary

| Category | Count | Files |
|----------|-------|-------|
| **Critical (Lesson Loading)** | 3 | kelly-lesson-loader.js, learn.html, player.html |
| **High Priority** | 8 | hub.html, app.html, lesson-resilience.js, etc. |
| **Medium Priority** | 15 | Audit/admin tools, test files, archive files |

---

## 🔴 CRITICAL FILES (Must Fix First)

### 1. `public/js/kelly-lesson-loader.js`
**Violations:** 3 direct Supabase calls
- Line 378: `fetchFromSupabaseWithTimeout()` method
- Line 484: `.from('core_lessons')`
- Line 497: `.from('lesson_atoms')`
- Line 503: `.from('lesson_shards')`

**Impact:** Core lesson loading - affects ALL lessons
**Priority:** CRITICAL
**Fix:** Replace with API calls to `/api/lessons/[dayNumber]-edge` → `/api/lessons/[dayNumber]`

---

### 2. `public/learn.html`
**Violations:** 3+ direct Supabase calls
- Line 10735: `window.supabase.createClient()`
- Line 15228: `.from('core_lessons')`
- Line 15241: `.from('lesson_atoms')`
- Additional calls for user_progress, kelly_video_assets

**Impact:** Main lesson player page
**Priority:** CRITICAL
**Fix:** Replace with API calls

---

### 3. `public/player.html`
**Violations:** 2 direct Supabase calls
- Line 544: `.from('core_lessons')`
- Line 568: `.from('lesson_atoms')`

**Impact:** Standalone lesson player
**Priority:** CRITICAL
**Fix:** Replace with API calls

---

## 🟡 HIGH PRIORITY FILES

### 4. `public/hub.html`
**Violations:** 2 direct Supabase calls
- Line 516: `.from('core_lessons')`
- Line 609: `.from('core_lessons')`

**Impact:** Lesson hub/explorer
**Priority:** HIGH

---

### 5. `public/app.html`
**Violations:** 5 direct Supabase calls
- Multiple `.from('core_lessons')`, `.from('lesson_atoms')`, `.from('users')`, `.from('user_progress')`

**Impact:** App interface
**Priority:** HIGH

---

### 6. `public/js/lesson-resilience.js`
**Violations:** 3 direct Supabase calls
- Line 78: `.from('core_lessons')`
- Line 91: `.from('lesson_atoms')`
- Line 97: `.from('lesson_shards')`

**Impact:** Lesson resilience/fallback system
**Priority:** HIGH

---

### 7. `public/js/kelly-lesson-audit.js`
**Violations:** 1+ direct Supabase calls
- Audit tool for lesson quality

**Impact:** Admin/audit tool
**Priority:** HIGH

---

### 8. `public/js/golden-v5-data-loader.js`
**Violations:** 2 direct Supabase calls
- Legacy data loader

**Impact:** Legacy lesson loading
**Priority:** HIGH

---

## 🟢 MEDIUM PRIORITY FILES

### Admin/Audit Tools:
- `public/commons.html` - 1 violation
- `public/lesson-detail.html` - 3 violations
- `public/supabase-visuals-audit.html` - 2 violations
- `public/picky-nicky.html` - 2 violations

### Archive/Test Files:
- `public/archive/curriculum-marketing.html` - 1 violation
- `public/archive/index-marketing.html` - 7 violations
- `public/index-backup-20251212-174223.html` - 7 violations
- `public/index-unified.html` - 2 violations
- `public/golden-v5.html` - 2 violations
- `public/debug-supabase.html` - 3 violations
- `public/supabase-test.html` - 8 violations
- `public/kelly.html` - 2 violations
- `public/live.html` - 1 violation
- `public/day/index.html` - 1 violation
- `public/golden-lesson-review.html` - 2 violations
- `public/lesson-commons.html` - 1 violation
- `public/js/lesson-assets.js` - 1 violation
- `public/js/api.js` - 4 violations

---

## 🔧 FIX STRATEGY

### Phase 1: Core Lesson Loading (CRITICAL)
1. ✅ Fix `kelly-lesson-loader.js` - Replace `fetchFromSupabaseWithTimeout()` with API calls
2. ✅ Fix `learn.html` - Replace direct queries with API calls
3. ✅ Fix `player.html` - Replace direct queries with API calls

### Phase 2: Supporting Systems (HIGH)
4. Fix `hub.html`
5. Fix `app.html`
6. Fix `lesson-resilience.js`
7. Fix other high-priority files

### Phase 3: Admin/Test Files (MEDIUM)
8. Fix admin/audit tools
9. Fix test/debug files
10. Archive or remove deprecated files

---

## ✅ API ENDPOINTS AVAILABLE

### Primary Endpoint:
- `/api/lessons/[dayNumber]-edge` - Edge-optimized (<5ms)
- `/api/lessons/[dayNumber]` - Standard API (fallback)

### Endpoint Format:
```javascript
// Edge API (try first)
fetch(`/api/lessons/${dayNumber}-edge?archetype=${archetype}&track=${track}`)

// Standard API (fallback)
fetch(`/api/lessons/${dayNumber}?archetype=${archetype}&track=${track}`)
```

### Response Format:
```json
{
  "source": "edge-config" | "supabase-admin" | "emergency-fallback",
  "lesson": { ... },
  "atoms": [ ... ],
  "shards": [ ... ],
  "dayNumber": 1,
  "archetype": "The Scientist",
  "ageBucket": "adult"
}
```

---

## 📝 PROGRESS TRACKING

- [ ] Step 1.1: Audit complete ✅
- [ ] Step 1.2: Fix kelly-lesson-loader.js ⏳ IN PROGRESS
- [ ] Step 1.3: Fix learn.html
- [ ] Step 1.4: Fix config.js
- [ ] Step 1.5: Fix other critical files
- [ ] Step 1.6: Remove Supabase SDK

---

**Last Updated:** January 2025  
**Status:** Fixing critical files first

