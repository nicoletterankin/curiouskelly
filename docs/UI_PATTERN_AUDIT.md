# UI Pattern Audit — learn.html

**Date:** December 17, 2025  
**Issue:** Duplicate UI patterns causing confusion and broken navigation

---

## Executive Summary

The learn.html file has **328 overlay/modal/panel references** and contains multiple duplicate UI systems that were created over time instead of extending existing ones. This causes:
- "Take Quiz" button not visible in Settings
- Privacy Policy showing over Kelly when opening Settings
- Conflicting navigation patterns
- User confusion

---

## Duplicate UI Systems Identified

### 1. ⚠️ Settings (FIXED)

| System | Element | Shows When | Content |
|--------|---------|------------|---------|
| **Original (correct)** | `#scene-settings` → right panel | `showScene('settings')` | Kelly selector, Age pills, Take Quiz, Learning settings |
| **Duplicate (incomplete)** | `#settings-overlay` | `data-ui-mode="settings"` | Master-detail: Profile, Privacy, Terms (missing Kelly selector) |

**Resolution:** Changed `showSettingsMode()` to use `showScene('settings')` instead of `setUiMode('settings')`.

---

### 2. ⚠️ Journey (NEEDS FIX)

| System | Element | Shows When | Content |
|--------|---------|------------|---------|
| **Original** | `#scene-journey` → right panel | `showScene('journey')` | Stats, Tabs (Calendar/Week/Curriculum/Bookmarks) |
| **Duplicate** | `#journey-overlay` | `data-ui-mode="journey"` | "Today-First" design with greeting, hero card |

**Current state:** Bottom nav Journey button calls `showJourneyMode()` → shows `#journey-overlay` (the newer design).  
**Issue:** Two different journey experiences with different features.

---

### 3. Home Mode

| System | Element | Content |
|--------|---------|---------|
| `#home-overlay` | Full-screen | Marketing scenes (About, Adults, Children, etc.) |

This appears intentional (different from lesson flow).

---

## All Scenes (6 total)

```
#scene-character  → Character select carousel
#scene-lesson     → Main lesson player
#scene-journey    → Journey (tabs + calendar) [MOVED TO PANEL]
#scene-settings   → Settings (Kelly selector) [MOVED TO PANEL]
#scene-complete   → Lesson completion
#scene-achievements → Achievements
```

---

## All Overlays (10 total)

```
loading-overlay     → Loading spinner
home-overlay        → In-app marketing hub (JS-rendered)
journey-overlay     → DUPLICATE of scene-journey (JS-rendered)
settings-overlay    → DUPLICATE of scene-settings (JS-rendered)
overlay-infographic → Infographic viewer
overlay-picker      → Settings pickers (language, voice, time)
overlay-about       → About page
overlay-pricing     → Subscription/pricing
parental-gate       → COPPA gate
paywall             → Paywall/upgrade
```

---

## Guardrails Implemented

### Code-Level Guardrails
1. **HTML Comments:** Each canonical scene has `(CANONICAL)` marker and deprecation warnings
2. **Deprecated Overlays:** `#journey-overlay` and `#settings-overlay` have `data-deprecated="true"` and HTML comments
3. **Runtime Warnings:** `setUiMode('journey'|'settings')` logs console.warn with deprecation notice
4. **Function Comments:** `showJourneyMode()` and `showSettingsMode()` have clear documentation

### The Rules
- ✅ **Journey:** Use `showScene('journey')` → `#scene-journey` panel
- ✅ **Settings:** Use `showScene('settings')` → `#scene-settings` panel  
- ✅ **Home:** Use `setUiMode('home')` → `#home-overlay` (this is correct)
- ❌ **NEVER** use `setUiMode('journey')` or `setUiMode('settings')`

---

## Recommended Actions

### Completed ✅
1. ✅ **Settings:** Fixed - now uses `#scene-settings` panel with Kelly selector
2. ✅ **Journey:** Fixed - now uses `#scene-journey` panel with tabs
3. ✅ **Deprecated overlays:** Marked with HTML comments and `data-deprecated` attribute
4. ✅ **Guardrails:** Added deprecation warnings and documentation

### Short-term (This Week)
5. 🔲 **Add Privacy/Terms links** to the `#scene-settings` panel
6. 🔲 **Delete deprecated overlay CSS/JS** once confirmed stable

### Medium-term (Next Sprint)
7. 🔲 **Refactor into components:** Break 14k-line file into modules
8. 🔲 **Add ESLint rule:** Warn on `setUiMode` calls with journey/settings

---

## Bottom Nav Behavior (Current)

| Button | Calls | Opens | Correct? |
|--------|-------|-------|----------|
| Home | `handleHomeButton()` → `showHomeState()` | `#home-overlay` | ✅ |
| Journey | `showJourneyMode()` | `#scene-journey` (panel) | ✅ Fixed |
| Play | `handlePlayButton()` | Lesson scene | ✅ |
| Settings | `showSettingsMode()` | `#scene-settings` (panel) | ✅ Fixed |

---

## Files Affected
- `public/learn.html` (14,129 lines)
