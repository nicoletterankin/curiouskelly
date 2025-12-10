# Lesson Player UI/UX Recommendations

**Date:** December 5, 2025  
**Based on:** Testing of `learn.html` (production) and `app/index.html` (unified shell)

---

## Testing Summary

### What Works ✅
| Feature | Status | Notes |
|---------|--------|-------|
| Day Navigation (< >) | ✅ Working | URL updates, content loads correctly |
| Phase Navigation (Hook → Q1-3 → Complete) | ✅ Working | Phase indicator updates, content changes |
| Tone Selection | ⚠️ Partial | Maps to archetype, reloads lesson |
| Age Selection | ❌ Broken | UI updates but content doesn't change |
| Language Selection | ❌ Broken | UI updates but content doesn't change |
| Search | ✅ UI Present | Needs content testing |
| Talk to Kelly | ✅ UI Present | Voice interaction panel |
| Social Features | ✅ UI Present | Share, comments visible |

### What's Broken ❌
| Feature | Issue | Impact |
|---------|-------|--------|
| Age Personalization | Selecting "Toddler" shows adult content about "negotiation" and "conflict" | **Critical** - Core promise broken |
| Language Switch | No content change on selection | **Major** - Non-English speakers can't use |

---

## UI/UX Issues Identified

### 1. 🔴 Age Switcher Creates False Promise
**Problem:** The age popover is beautifully designed with:
- Smooth slider (2-102)
- Quick-select buttons (Toddler → Elder)  
- Emoji avatars for each stage
- Real-time age number display

**But:** Selecting "Toddler" (age 2) still shows:
> "Consider this: Every negotiation is, at its core, an information exchange. Unveiling data points can reshape the entire landscape of the discussion."

**Impact:** This is a **trust destroyer**. Parents selecting content for a 2-year-old will see adult content and feel deceived.

**Recommendations:**
1. **Disable age switcher** until content personalization works
2. Or show clear messaging: "Toddler content coming soon - showing default"
3. Add loading indicator when age changes: "Adjusting Kelly for ages 2-5..."
4. Show visual confirmation: Kelly's appearance/pose could change for different ages

### 2. 🟡 Phase Navigation Lacks Visual Feedback
**Current State:** 
- Vertical timeline on right rail
- Q1-Q3 dots are small and hard to see
- No clear indication of "where I am" vs "where I've been"

**Recommendations:**
1. **Larger phase indicators** - current dot is tiny
2. **Progress fill animation** - show a line filling between phases as user progresses
3. **Phase labels** - show "Hook", "Q1", "Q2", "Q3", "✨" inline or on hover
4. **Completion checkmarks** - show ✓ for completed phases

### 3. 🟡 Day Navigation is Hidden
**Current State:**
- Previous/Next buttons are edge arrows that fade into the image
- No indication of what day you're on or how many exist

**Recommendations:**
1. **Add day counter** - "Day 339 of 365" or "Dec 5 • Day 339"
2. **Progress ring** - small circular progress showing yearly completion
3. **Calendar preview** - clicking the date could show a mini calendar
4. **Swipe gesture hint** - "Swipe for next lesson" is nice, add "< Day 338 | Day 340 >"

### 4. 🟡 Left Sidebar Icon Overload
**Current State:** 12+ icons stacked vertically:
- 🔍 Search
- 📅 Calendar  
- 👤 Age (with badge "2")
- 🌐 Language (with badge "EN")
- 😊 Tone (with badge "C")
- ... and more

**Issues:**
- Too many options creates analysis paralysis
- Small badges are hard to read
- Icons aren't labeled
- Unclear hierarchy

**Recommendations:**
1. **Group related controls** - Settings cluster, Navigation cluster, Social cluster
2. **Use drawer/panel** instead of individual icons for settings
3. **Add tooltips** on hover for each icon
4. **Primary vs Secondary** - Make essential controls larger/more prominent
5. **Collapse non-essential** - Hide Learner Commons, Share behind a "..." menu

### 5. 🟡 Choice Cards Need Better Hierarchy
**Current State:**
- A/B choice cards are small and pushed to the corner
- Same visual weight makes it unclear which to focus on

**Recommendations:**
1. **Animate choices in** - stagger them appearing after Kelly speaks
2. **Larger touch targets** - especially on mobile
3. **Hover states** - more dramatic hover feedback
4. **Selection confirmation** - pulse/glow when selected before transitioning

### 6. 🟢 Caption Bar is Well Done
**Current State:**
- Clean dark background
- Large, readable text
- Good contrast

**Enhancement Opportunities:**
1. **Word-by-word highlight** during audio playback (karaoke style)
2. **Collapse when not speaking** - expand when Kelly talks
3. **Copy button** - let users copy the text

---

## Recommended Quick Wins

### Priority 1: Fix Age Personalization (Backend)
Without this, the age UI is actively harmful. Either:
- Hide the feature
- Show "coming soon" state
- Wire up the `lesson_shards` and `lesson_age_hooks` queries (code provided earlier)

### Priority 2: Add Loading States
When age/tone/language changes:
```
[Before] Click "Toddler" → Nothing happens (content stays same)
[After]  Click "Toddler" → "Adjusting Kelly for ages 2-5..." → Content updates
```

### Priority 3: Simplify Left Rail
Reduce from 12 icons to essential 6:
1. 🔍 Search
2. 📅 Calendar  
3. ⚙️ Settings (opens panel with Age/Language/Tone)
4. 👤 Profile/Me
5. 📤 Share
6. 🔊 Sound

### Priority 4: Add Day Context
Show user where they are:
```
[Current]  "How Money Works"        December 4, 2025
[Better]   "How Money Works"        Day 338 / 365  •  Dec 4
```

### Priority 5: Phase Progress Visualization
```
[Current]  ● ○ ○ ○ ✓   (tiny, hard to see)
[Better]   ●━━○━━○━━○━━✓   (larger, connected, shows progress)
```

---

## Mobile Considerations

The current UI is desktop-first. For mobile:

1. **Bottom navigation** - move essential controls to bottom thumb zone
2. **Swipe gestures** - swipe left/right for day, up/down for phase
3. **Collapse left rail** - use hamburger menu
4. **Full-width choices** - A/B cards should be stacked, not side-by-side
5. **Larger touch targets** - minimum 44x44px

---

## Summary Table

| Area | Current Score | After Fixes |
|------|---------------|-------------|
| Day Navigation | 7/10 | 9/10 |
| Phase Navigation | 6/10 | 9/10 |
| Age/Language/Tone | 2/10 | 9/10 (if backend works) |
| Visual Hierarchy | 5/10 | 8/10 |
| Mobile Experience | 4/10 | 8/10 |
| **Overall** | **4.8/10** | **8.6/10** |

The lesson player looks beautiful but the core personalization promise (age-appropriate content) is completely broken. This should be the #1 priority to fix before any UI polish.



