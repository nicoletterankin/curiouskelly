# Calendar → Panel → Player Data Coherency Plan

**Date:** December 23, 2025  
**Status:** Planning Phase

## Problem Analysis

### Current Data Flow Issues

1. **Calendar (`index.html`)**
   - Loads from: `/lessons/day-${day}.json` (static JSON files)
   - Data: Basic metadata (topic, emoji, category)
   - Issue: Doesn't use unified loader

2. **Audit Panel (`lesson-audit-panel.js`)**
   - Loads from: `LOCAL_PACKS` → `LessonInspector.getFullAudit()` → JSON fallback
   - Data: Completeness, tracks, assets
   - Issue: **Learn track shows "Loading..."** because it's not using `KellyLessonLoader`
   - Issue: No visual previews or full copy display

3. **Learn Player (`learn.html`)**
   - Loads from: `KellyLessonLoader.loadLesson()` (unified loader)
   - Data: Full lesson with atoms, visuals, videos
   - Issue: Different data source than calendar/panel

### Root Cause

**Three different data sources = data coherency issues**

- Calendar: Static JSON files
- Panel: LOCAL_PACKS + Inspector
- Player: KellyLessonLoader (LOCAL_PACKS → Supabase → API → Emergency)

When a user clicks calendar → panel → player, they see different data because each component loads independently.

## Solution: Unified Data Layer

### Architecture

```
┌─────────────────────────────────────────────────┐
│     Unified Lesson Data Service (New)            │
│  Uses KellyLessonLoader as single source        │
└─────────────────────────────────────────────────┘
           │
           ├─── Calendar (metadata only)
           ├─── Panel (full preview)
           └─── Player (full lesson)
```

### Implementation Plan

#### Phase 1: Create Unified Lesson Service

**File:** `public/js/kelly-unified-lesson-service.js`

**Purpose:** Single source of truth for all lesson data

**API:**
```javascript
KellyUnifiedLessonService = {
  // Get lesson metadata (for calendar)
  async getMetadata(dayNumber) {
    // Uses KellyLessonLoader but returns lightweight metadata
  },
  
  // Get lesson preview (for panel)
  async getPreview(dayNumber, options = {}) {
    // Returns: topic, emoji, visuals, copy preview, completeness
  },
  
  // Get full lesson (for player)
  async getFullLesson(dayNumber, options = {}) {
    // Uses KellyLessonLoader.loadLesson() directly
  },
  
  // Cache management
  cache: new Map(),
  clearCache(dayNumber) { ... }
}
```

#### Phase 2: Fix Audit Panel Learn Track Loading

**File:** `public/js/lesson-audit-panel.js`

**Changes:**
1. Use `KellyUnifiedLessonService.getPreview()` instead of manual LOCAL_PACKS lookup
2. Load Learn track data using `KellyLessonLoader.loadLesson()` if not in LOCAL_PACKS
3. Add visual previews (thumbnails, video previews)
4. Add full copy display (script preview, phase content)
5. Show visuals in both learner and educator views

**Key Fix:**
```javascript
async loadAudit(dayNumber) {
  // OLD: Only checked LOCAL_PACKS
  const pack = window.CURIOUS_KELLY?.LOCAL_PACKS?.[dayNumber];
  const learnTopic = pack?.lesson?.topic || 'Loading...'; // ❌ Shows "Loading..."
  
  // NEW: Use unified service
  const preview = await KellyUnifiedLessonService.getPreview(dayNumber);
  const learnTopic = preview.learn.topic || 'Lesson not available'; // ✅ Always has data
}
```

#### Phase 3: Add Visuals and Copy Display

**In `renderLearnerView()`:**
- Add visual preview section (thumbnails, video previews)
- Add copy preview (first 200 chars of each phase)
- Show full lesson structure

**In `renderEducatorView()`:**
- Show full copy for all phases
- Show all visuals with URLs
- Show asset inventory with previews

#### Phase 4: Ensure Data Coherency

**Calendar:**
- Use `KellyUnifiedLessonService.getMetadata()` for calendar dots
- Store metadata in `window.__lessonsData` for tooltips

**Panel:**
- Use `KellyUnifiedLessonService.getPreview()` when opened
- Pass same dayNumber to player

**Player:**
- Use `KellyUnifiedLessonService.getFullLesson()` 
- Accept dayNumber from panel's "Start Lesson" button

#### Phase 5: Seamless Flow

**Flow:** Calendar → Panel → Player

1. **Calendar Click:**
   ```javascript
   // Calendar dot click
   const day = parseInt(this.dataset.day);
   window.LessonAuditPanel.show(day);
   ```

2. **Panel Display:**
   ```javascript
   // Panel loads preview using unified service
   const preview = await KellyUnifiedLessonService.getPreview(day);
   // Shows: topic, visuals, copy preview, completeness
   ```

3. **Start Lesson:**
   ```javascript
   // "Start Lesson" button
   window.location.href = `/learn.html?day=${day}&track=learn`;
   // Player uses same unified service, gets same data
   ```

## Implementation Details

### Visual Display in Panel

**Learner View:**
- Hero visual (if available)
- Phase thumbnails
- Video preview thumbnails
- Copy preview (first 200 chars per phase)

**Educator View:**
- Full visual gallery
- All video URLs
- Full copy for all phases
- Asset inventory with previews

### Copy Display

**Format:**
```javascript
{
  hook: { script: "...", visual: "...", video: "..." },
  question: { script: "...", visual: "...", video: "..." },
  // ... all 7 phases
}
```

**Display:**
- Learner: Preview (first 200 chars) + "Read more" expand
- Educator: Full copy + all metadata

### Data Coherency Guarantees

1. **Same Source:** All components use `KellyUnifiedLessonService`
2. **Same Cache:** Shared cache ensures consistency
3. **Same Fallbacks:** All use KellyLessonLoader's bulletproof fallbacks
4. **Same Format:** Normalized data structure across all components

## Testing Checklist

- [ ] Calendar loads metadata correctly
- [ ] Panel loads Learn track (no "Loading...")
- [ ] Panel shows visuals
- [ ] Panel shows copy preview
- [ ] Panel shows full copy in educator view
- [ ] Player loads same lesson data
- [ ] Flow: Calendar → Panel → Player works seamlessly
- [ ] Data is consistent across all three components

## Files to Modify

1. **New:** `public/js/kelly-unified-lesson-service.js`
2. **Modify:** `public/js/lesson-audit-panel.js`
3. **Modify:** `public/index.html` (calendar data loading)
4. **Modify:** `public/learn.html` (player data loading - ensure it uses unified service)

## Next Steps

1. Create `kelly-unified-lesson-service.js`
2. Update audit panel to use unified service
3. Add visual and copy display to panel
4. Update calendar to use unified service
5. Test end-to-end flow
6. Deploy and verify

