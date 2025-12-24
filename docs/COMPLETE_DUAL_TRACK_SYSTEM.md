# Complete Dual-Track Lesson System - Expert Directive
**Date:** December 22, 2025  
**Status:** ✅ IMPLEMENTED & DOCUMENTED

---

## 🎯 System Overview

**One lesson = Two tracks = 700+ hardcoded base lessons**

Every day (1-365) has:
- **Learn Track**: Traditional education (`lesson.topic`)
- **Grow Track**: AI fluency (`grow.topic`)

Both tracks are stored in `day-XXX-complete.js` files → `LOCAL_PACKS`

---

## 📦 Storage Architecture

### Static Files: `/public/data/day-XXX-complete.js`

**Structure:**
```javascript
window.CURIOUS_KELLY.DAY_001 = {
  meta: { day_number: 1, version: "v3.0-skeleton" },
  
  // LEARN TRACK
  lesson: {
    topic: "Starting Fresh",           // Learn topic
    headline: "New beginnings...",
    universal_truth: "...",
    emoji: "🍁",
    category: "general"
  },
  atoms: [
    { phase: "Hook", content: { script: "..." } },
    { phase: "Cliff", content: { script: "..." } },
    // ... 7 phases total
  ],
  
  // GROW TRACK
  grow: {
    topic: "I'm an AI - Understanding Your Digital Learning Partner",  // Grow topic
    objective: "Develop foundational AI literacy...",
    activity: "Practice asking AI questions..."
  },
  
  // Age variants (both tracks)
  ageVariants: {
    "2-5": { persona: "Playful Friend", phases: {...} },
    // ... 6 age buckets
  }
}
```

**Key Points:**
- ✅ 365 files (one per day)
- ✅ Both tracks in same file
- ✅ Stored in `window.CURIOUS_KELLY.LOCAL_PACKS`
- ✅ Offline-first (works without internet)
- ✅ Base layer (Supabase enriches, doesn't replace)

---

## 🔄 Loading Priority Chain

### For Learn Track:
1. `LOCAL_PACKS[dayNum].lesson` → Learn topic + atoms
2. Supabase `core_lessons` + `lesson_atoms` (track='learn')
3. JSON fallback (`/lessons/day-XXX.json`)
4. Emergency fallback

### For Grow Track:
1. `LOCAL_PACKS[dayNum].grow` → Grow topic + objective
2. Supabase `core_lessons` + `lesson_atoms` (track='grow')
3. JSON fallback
4. Emergency fallback

**Function:** `KellyLessonLoader.loadLesson(dayNumber, { track: 'learn' | 'grow' })`

---

## 📊 Completeness Calculation

### Scoring System:

**Learn Track (60% total):**
- Base Content (40%): topic + 7 phases with scripts
- Enhanced Content (20%): videos, visuals, multiple archetypes

**Grow Track (40% total):**
- Base Content (30%): topic + objective
- Enhanced Content (10%): activity or full content

### Status Levels:
- **Production** (80-100%): All content + videos + visuals
- **Complete** (60-80%): Base + some enhanced content
- **Basic** (40-60%): Learn + Grow base content
- **Skeleton** (0-40%): Base Learn content only
- **Missing** (0%): No data found

### Implementation:
```javascript
const completeness = LessonPreviewPopup.calculateCompleteness(dayNumber);
// Returns: {
//   completeness: 85,
//   status: 'production',
//   checks: { learnBase: true, learnEnhanced: true, growBase: true, growEnhanced: true },
//   stats: { phases: 7, videos: 5, visuals: 7, archetypes: 3 }
// }
```

---

## 🎨 Preview Popup Component

### File: `public/js/lesson-preview-popup.js`

### Features:
- ✅ Shows Learn + Grow topics side-by-side
- ✅ Displays completeness percentage (0-100%)
- ✅ Status badge (Production/Complete/Basic/Skeleton)
- ✅ Quick stats (phases, videos, visuals, archetypes)
- ✅ Track status indicators (Base ✓/✗, Enhanced ✓/✗)
- ✅ Actions: "Start Learn Track", "Start Grow Track", "View Full Details"

### Usage:
```javascript
// Show preview popup
window.LessonPreviewPopup.show(dayNumber);

// Calculate completeness
const completeness = window.LessonPreviewPopup.calculateCompleteness(dayNumber);

// Show full audit
window.LessonPreviewPopup.showFullAudit(dayNumber);
```

### Data Sources:
1. **LOCAL_PACKS** (primary): `window.CURIOUS_KELLY.LOCAL_PACKS[dayNumber]`
2. **JSON fallback**: `/lessons/day-${dayNumber}.json`
3. **Graceful degradation**: Shows "Loading..." if neither available

---

## 🏠 Homepage Calendar Integration

### File: `public/index.html`

### Behavior:
- **Single Click:** Shows full audit (LessonInspector - full screen)
- **Double Click:** Shows compact preview popup (LessonPreviewPopup - 600px card)
- **Mobile:** Tap = audit, Double-tap = preview

### Updated Handlers:
- Desktop double-click → `LessonPreviewPopup.show(day)`
- Mobile double-tap → `LessonPreviewPopup.show(day)`
- Tooltip text: "Double-click for preview"

---

## 📱 Journey Panel Enhancement

### File: `public/learn.html` (Line 14660)

### Updated `populateJourneyPanel()`:
- ✅ Checks `LOCAL_PACKS` first for static lessons
- ✅ Displays both Learn and Grow topics
- ✅ Falls back to curriculum JSON if needed
- ✅ Shows asset availability from Supabase

### Display Format:
```
Day 17 • Tue Dec 17
Learn: Why We Dream 🌙
Grow: Understanding AI Dreams 🤖
✓ 7 assets
```

---

## ✅ Implementation Checklist

### Core Architecture:
- [x] Understand dual-track structure (Learn + Grow)
- [x] Know where topics are stored (`lesson.topic` + `grow.topic`)
- [x] Understand completeness calculation
- [x] Know loading priority chain

### Preview Popup:
- [x] Create compact preview component
- [x] Calculate completeness accurately
- [x] Display Learn + Grow topics
- [x] Show status badges and stats
- [x] Provide track selection actions
- [x] Link to full audit

### Integration:
- [x] Update homepage double-click handler
- [x] Update mobile touch handler
- [x] Update tooltip text
- [x] Load preview popup script
- [x] Update journey panel to show both tracks

### Testing:
- [x] Test with Day 1 (skeleton)
- [x] Test with Day 17 (complete)
- [x] Test with missing days (fallback)
- [x] Test completeness calculation
- [x] Test popup display
- [x] Test track selection

---

## 🚀 Next Layer: Obvious Wiring

### 1. Track Selection in Lesson Player
**Current:** Lesson player loads Learn track by default  
**Needed:** Respect `?track=learn` / `?track=grow` URL param

**Implementation:**
```javascript
// In learn.html init()
const urlParams = new URLSearchParams(window.location.search);
const track = urlParams.get('track') || 'learn';
state.track = track;
state.currentTrack = track;
```

### 2. Track Completion Tracking
**Current:** Single completion per day  
**Needed:** Separate completion for Learn and Grow tracks

**Implementation:**
```javascript
state.completedLessons = {
  learn: [1, 2, 3, ...],
  grow: [1, 2, ...]
};
```

### 3. Track Toggle in Header
**Current:** Track toggle in Kelly Panel  
**Needed:** Quick track toggle in header CTA

**Implementation:**
- Add track indicator to `nav-lesson-cta`
- Click to toggle between Learn/Grow
- Update lesson content on toggle

### 4. Journey Panel Track Filter
**Current:** Shows both tracks  
**Needed:** Filter by track (Learn only / Grow only / Both)

**Implementation:**
- Add track filter buttons
- Filter lesson list by selected track
- Show track-specific completion stats

### 5. Completeness Badge in Calendar
**Current:** Calendar dots show completion  
**Needed:** Show completeness status (Production/Complete/Basic/Skeleton)

**Implementation:**
- Calculate completeness on hover
- Show status badge in tooltip
- Color-code dots by completeness

---

## 📝 Best Practices

### When Loading Lessons:
1. **Always check both tracks:**
   ```javascript
   const learnLesson = await loadLesson(dayNum, { track: 'learn' });
   const growLesson = await loadLesson(dayNum, { track: 'grow' });
   ```

2. **Display both topics:**
   ```javascript
   const learnTopic = pack.lesson?.topic || 'Loading...';
   const growTopic = pack.grow?.topic || 'Loading...';
   ```

3. **Track completion separately:**
   ```javascript
   state.completedLessons.learn = [...];
   state.completedLessons.grow = [...];
   ```

### When Calculating Completeness:
1. **Check base content first:**
   - Learn: topic + 7 phases
   - Grow: topic + objective

2. **Then enhanced content:**
   - Learn: videos, visuals, archetypes
   - Grow: activity, full content

3. **Show clear status:**
   - Production: Ready for users
   - Complete: All base content
   - Basic: Partial content
   - Skeleton: Minimal content

### When Displaying Lessons:
1. **Show dual-track preview:**
   - Learn topic (primary)
   - Grow topic (secondary)
   - Completeness for both

2. **Provide track selection:**
   - "Start Learn Track"
   - "Start Grow Track"
   - Track toggle in player

3. **Calculate completeness accurately:**
   - Check LOCAL_PACKS first
   - Fallback to JSON
   - Show clear status

---

## 🎯 Success Criteria

- [x] Understand dual-track architecture
- [x] Know where Learn/Grow topics are stored
- [x] Can calculate completeness accurately
- [x] Can display both tracks
- [x] Can show compact preview
- [x] Can provide track selection
- [x] Preview popup works on double-click
- [x] Journey panel shows both tracks
- [x] Completeness calculation is accurate
- [x] Fallback to JSON works

---

## 📚 Reference Documents

1. **`docs/DUAL_TRACK_LESSON_ARCHITECTURE_DIRECTIVE.md`**
   - Complete architecture documentation
   - Best practices
   - Function reference

2. **`docs/LESSON_ARCHITECTURE_EXPERT.md`**
   - Lesson storage architecture
   - Loading priority chain
   - Why we hardcode base parts

3. **`docs/DUAL_TRACK_IMPLEMENTATION_SUMMARY.md`**
   - Implementation summary
   - What was built
   - How to use

4. **`public/js/lesson-preview-popup.js`**
   - Preview popup component
   - Completeness calculation
   - Integration code

---

**Status:** ✅ Complete and ready for next layer of wiring


