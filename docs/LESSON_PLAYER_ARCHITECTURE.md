# Lesson Player Architecture - AI Agent Guide

**Last Updated:** December 5, 2025  
**Purpose:** Prevent confusion between different lesson player implementations

---

## ⚠️ CRITICAL: There Are Multiple Lesson Players!

This codebase has **multiple lesson player implementations**. Before making changes, understand which one you're working with.

---

## The Players (In Order of Priority)

### 1. 🟢 `public/learn.html` - PRODUCTION PLAYER
**Status:** ✅ ACTIVE - This is what users see on curiouskelly.com

```
URL: https://www.curiouskelly.com/learn.html
Path: public/learn.html (also: daily-lesson-marketing/public/learn.html)
```

**What it does:**
- Full-featured lesson player
- Connects to Supabase for content
- Has age/tone/language personalization
- Has visual system (Kelly poses, backgrounds)
- Has phase navigation (Hook → Q1-3 → Wisdom)
- Has social features (comments, sharing)

**Key Code Sections:**
- Lines ~3741-3752: `TONE_TO_ARCHETYPE` mapping
- Lines ~3754-3920: Age/Language personalization functions (NEW!)
- Lines ~3886-4290: `loadLesson()` and `buildLessonFromAtoms()`
- Lines ~4540-4800: `renderPhase()` with personalization
- Lines ~5070-5140: `selectVariant()` with age/language reload

**When to edit:** This is the main player. Edit this for production features.

---

### 2. 🟡 `app/index.html` + `app/script.js` - UNIFIED SHELL (Development)
**Status:** ⚠️ DEVELOPMENT - Newer architecture, not yet production

```
Path: app/index.html (with app/script.js module)
```

**What it does:**
- Modular JavaScript architecture (ES modules)
- State management with StateManager class
- Unity bridge for 3D Kelly
- Vibe Tuner for archetype selection
- Session tracking

**Key files:**
- `app/script.js` - Main UnifiedLessonApp class
- `app/supabase-service.js` - Database queries
- `app/state-manager.js` - Centralized state
- `app/unity-bridge.js` - 3D Kelly communication

**When to edit:** For architectural improvements or Unity integration.

---

### 3. 🔴 `public/app.html` - LEGACY KELLY OS
**Status:** ❌ DEPRECATED - Do not use for new features

```
Path: public/app.html (also: daily-lesson-marketing/public/app.html)
```

**What it does:**
- Original "Kelly OS" concept
- Menu drawer navigation
- Checkout integration
- Some lesson loading (from local JSON)

**When to edit:** Only for bug fixes. Do not add new features.

---

### 4. ⚫ `public/learn-v1.html`, `public/learn-v2.html` - VERSION HISTORY
**Status:** 📦 ARCHIVED - Historical versions

**When to edit:** Never. These are backups.

---

## Quick Decision Guide

| Task | Edit This File |
|------|----------------|
| Fix user-facing bug in lesson player | `public/learn.html` |
| Add new personalization feature | `public/learn.html` |
| Change how age/tone/language works | `public/learn.html` |
| Improve Unity/3D Kelly integration | `app/script.js` |
| Change database schema usage | Both `learn.html` and `app/supabase-service.js` |
| Fix checkout/payments | `public/app.html` (legacy) or backend |

---

## Database Tables Used by Players

| Table | `learn.html` | `app/script.js` | `app.html` |
|-------|--------------|-----------------|------------|
| `core_lessons` | ✅ Yes | ✅ Yes | ❌ No |
| `lesson_atoms` | ✅ Yes | ✅ Yes | ❌ No |
| `lesson_shards` | ✅ Yes (NEW!) | ✅ Yes | ⚠️ Partial |
| `lesson_age_hooks` | ✅ Yes (NEW!) | ✅ Yes | ❌ No |

---

## Personalization Flow (learn.html)

```
User loads lesson
       ↓
loadLesson(dayNumber)
       ↓
┌──────────────────────────┐
│ 1. Query core_lessons    │
│ 2. Query lesson_atoms    │
│ 3. Build lesson object   │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│ PERSONALIZATION (NEW!)   │
│ 4. loadAgeHook()         │
│ 5. loadLessonShards()    │
└──────────────────────────┘
       ↓
renderPhase()
       ↓
getVariantText() checks:
  1. personalization.ageHook (for welcome/hook)
  2. personalization.currentShard.script_content
  3. phase.text object variants
  4. phase.text string (fallback)
```

---

## When User Changes Age/Language/Tone

### learn.html (NOW WORKING):
```javascript
selectVariant('age', '2-5')
  → closePopover()
  → reloadPersonalizedContent()
  → loadAgeHook() + findMatchingShard()
  → renderPhase() with new content
  → showToast("✨ Personalized for ages 2-5")
```

### app/script.js:
```javascript
ageSlider.addEventListener('input', ...)
  → stateManager.setState({ age, ageBucket })
  → setTimeout 400ms (debounce)
  → reloadPersonalizedContent()
  → loadAgeHook() + loadLessonShards()
  → renderPhase() with new content
```

---

## File Reference Map

```
UI-TARS-desktop/
├── app/                          # Unified Shell (development)
│   ├── index.html                # Shell HTML
│   ├── script.js                 # Main app class
│   ├── supabase-service.js       # DB queries
│   ├── state-manager.js          # State
│   └── unity-bridge.js           # 3D Kelly
│
├── public/
│   ├── learn.html                # 🟢 PRODUCTION PLAYER
│   ├── learn-v1.html             # ⚫ Archive
│   ├── learn-v2.html             # ⚫ Archive
│   ├── app.html                  # 🔴 DEPRECATED Kelly OS
│   ├── kelly.html                # Static Kelly demo
│   └── player.html               # Old player variant
│
└── daily-lesson-marketing/
    └── public/
        ├── learn.html            # Mirror of public/learn.html
        └── app.html              # Mirror of public/app.html
```

---

## Common Pitfalls

### ❌ DON'T: Edit `app.html` thinking it's the main player
It's not. `learn.html` is production.

### ❌ DON'T: Mix up `app/` folder with `public/app.html`
- `app/` folder = NEW modular architecture
- `public/app.html` = OLD Kelly OS concept

### ❌ DON'T: Forget to update both `public/learn.html` and `daily-lesson-marketing/public/learn.html`
These should stay in sync (or use symlinks/build process).

### ✅ DO: Check which player the user is asking about
Ask: "Are you on learn.html or app.html?" if unclear.

### ✅ DO: Test on production URL
`https://www.curiouskelly.com/learn.html` is the real user experience.

---

## Summary

| File | Status | Purpose |
|------|--------|---------|
| `public/learn.html` | 🟢 PRODUCTION | Main lesson player |
| `app/index.html` | 🟡 DEVELOPMENT | Future unified shell |
| `public/app.html` | 🔴 DEPRECATED | Legacy Kelly OS |
| `public/learn-v*.html` | ⚫ ARCHIVE | Version history |

**When in doubt, edit `public/learn.html`.**


