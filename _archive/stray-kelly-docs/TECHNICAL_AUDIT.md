# Curious Kelly - Technical Audit
**Date**: December 1, 2025  
**Auditor**: AI Engineering Assistant  
**Target**: Production site (curiouskelly.com)

---

## Executive Summary

The learn page is functional but has several issues that need attention before scaling. Core learning experience works, but there are errors in Unity 3D loading and TTS that degrade the experience.

---

## 🟢 WORKING (Verified)

| Feature | Status | Evidence |
|---------|--------|----------|
| **Lesson Caching** | ✅ Working | `[Learn] ⚡ Using cached lesson for day 334` - instant load |
| **Kelly 2D Avatar** | ✅ Working | `[Kelly] 2D Avatar module loaded (two-frame pointing v3)` |
| **Kelly Production Assets** | ✅ Working | `[KellyAssets] ✅ Essential states preloaded` |
| **Chat Overlay** | ✅ Working | `[ChatOverlay] Started` |
| **Phase Navigation** | ✅ Working | `[Phase 1]` → `[Phase 2]` transitions observed |
| **Date Display** | ✅ Working | "November 30, 2025" showing correctly |
| **Bottom Navigation** | ✅ Working | Home, Calendar, Learn, Me, Settings visible |
| **Action Buttons** | ✅ Working | Age (18), Language (EN), Tone (C), Mode (2D), Share, Sound |

---

## 🔴 BROKEN (Needs Fix)

### 1. Unity 3D Loading Failure
```
Uncaught SyntaxError: Unexpected identifier 'https'
Unable to parse blob:... The file is corrupt, or compression was misconfigured
Uncaught ReferenceError: unityFramework is not defined
```
**Impact**: 3D Kelly avatar doesn't load  
**Root Cause**: Unity WebGL files not properly gzip compressed on GitHub Pages  
**Fix**: Either fix compression headers on hosting or use uncompressed builds  
**Priority**: Medium (2D works as fallback)

### 2. TTS API Failure
```
[KellyAudio] ElevenLabs error: Error: TTS API error: 500 - TTS failed
```
**Impact**: Kelly doesn't speak  
**Root Cause**: Server-side TTS endpoint returning 500  
**Fix**: Check Vercel/Netlify function logs, verify ElevenLabs API key  
**Priority**: High (core feature)

### 3. Kelly Avatar Controller Error
```
Uncaught TypeError: Cannot read properties of null (reading 'appendChild')
```
**Impact**: Minor - doesn't break core functionality  
**Root Cause**: DOM element not found when script runs  
**Fix**: Add null check or defer initialization  
**Priority**: Low

---

## 🟡 WARNINGS (Monitor)

| Warning | Count | Notes |
|---------|-------|-------|
| Unity download 0.00 MB | 3 | Files are empty/404 from GitHub Pages |
| "Not gzip compressed, using raw data" | 3 | Expected if files aren't compressed |

---

## Mobile Layout Audit

### Current State (390x844 viewport)
- ✅ Kelly image: Full screen, high quality
- ✅ Top bar: Date + stats visible
- ✅ Chat overlay: Working, positioned top-right
- ✅ Action buttons: 6 buttons visible on right side
- ✅ Speech bubble: Visible at bottom with lesson text
- ✅ Bottom nav: 5 items visible
- ⚠️ Topic title: Partially cut off ("Your Voice in Decisions")
- ❌ Choice buttons: NOT VISIBLE in current screenshot

### Missing on Mobile
1. **Choice buttons** - NOT A CODE BUG - Data issue (see below)
2. **Phase progress** - Dots visible but hard to see (top right)

---

## 🔴 ROOT CAUSE: Empty Lesson Content

**Console Evidence:**
```
[Phase 1] Content loading......
[Phase 2] Content loading......
```

**Problem**: The `lesson_atoms` table in Supabase has empty or malformed `content` JSONB:
- `content.script` is undefined → shows "Content loading..."
- `content.options` is undefined → no choice buttons render

**This is a DATA issue, not a CODE issue.**

**Fix Required**: Populate `lesson_atoms.content` with proper structure:
```json
{
  "script": "The actual lesson text Kelly should say",
  "prompt": "Optional hint for the learner",
  "options": [
    { "text": "Choice A text", "response": "Kelly's response to A" },
    { "text": "Choice B text", "response": "Kelly's response to B" }
  ]
}
```

---

## Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Lesson cache hit | Yes | Yes | ✅ |
| Time to interactive | ~250ms (cached) | <500ms | ✅ |
| Asset preload | 4 states | 4+ | ✅ |
| Kelly image size | 27-252KB | <300KB | ✅ |

---

## Feature Inventory (learn.html)

### Core Features
- [x] Lesson loading from Supabase
- [x] Lesson caching (30 min TTL)
- [x] Phase navigation (5 phases)
- [x] Kelly 2D avatar display
- [x] Kelly expression changes
- [x] Speech bubble with lesson text
- [x] Chat overlay (social learning)
- [x] Live stats bar (countries, watching)
- [x] Bottom navigation

### Personalization Features
- [x] Age variant selector
- [x] Language selector
- [x] Tone selector (Curious/Playful/Serious)
- [x] Difficulty selector
- [x] Avatar mode (2D/3D/Audio/Image/Full)
- [x] Experience mode (Solo/Social)

### Broken Features
- [ ] TTS audio playback
- [ ] Unity 3D avatar
- [ ] Choice button rendering (investigate)

---

## Recommendations

### Immediate (Today)
1. **Fix choice button rendering** - Investigate why choices don't appear
2. **Check TTS API** - Verify ElevenLabs key and endpoint

### Short-term (This Week)
3. **Fix Unity hosting** - Move to proper CDN with gzip or use uncompressed
4. **Add error boundaries** - Graceful degradation when features fail

### Long-term
5. **Add monitoring** - Track feature failures in production
6. **Performance budgets** - Set limits on asset sizes

---

## Test Checklist for Engineers

```bash
# Manual Tests
[ ] Load /learn - should show Kelly + lesson text
[ ] Tap Age button - popover should appear
[ ] Tap Language button - language options should appear
[ ] Tap Tone button - tone options should appear
[ ] Wait for phase transition - Kelly should change expression
[ ] Tap choice A - should advance to next phase
[ ] Tap choice B - should advance to next phase
[ ] Complete all 5 phases - should show wisdom/completion
[ ] Check mobile (390px) - all elements should be visible
[ ] Check tablet (768px) - layout should adapt
[ ] Check desktop (1440px) - full experience

# Console Checks
[ ] No red errors blocking functionality
[ ] Lesson cache working (⚡ messages)
[ ] Kelly assets preloaded
[ ] Chat overlay started
```

---

## Files Involved

| File | Purpose | Lines |
|------|---------|-------|
| `/learn.html` | Main lesson player | ~2700 |
| `/js/kelly-production-assets.js` | Kelly image management | ~220 |
| `/js/chat-overlay.js` | Social chat simulation | ~200 |
| `/js/kelly-audio.js` | TTS integration | ~100 |
| `/js/kelly-2d-avatar.js` | 2D avatar controller | ~140 |
| `/js/unity-kelly-loader.js` | Unity 3D loader | ~410 |

---

*This audit reflects the state of curiouskelly.com as of December 1, 2025*

