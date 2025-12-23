# Homepage Calendar Redesign Proposal
**Date:** December 23, 2025  
**Issue:** Calendar display broken after adding audit panel  
**Goal:** Fix calendar + prepare for video trailer system

---

## Current Problems

### 1. Calendar Layout Issues
- **12-column grid** may be too cramped on smaller screens
- **Panel overlay** covers calendar when open (no proper z-index/layout management)
- **Day dots** too small to show meaningful information
- **No hover preview** of lesson content before clicking
- **No visual indication** of video trailer availability

### 2. Panel Integration Issues
- **Right-side panel** slides in but doesn't account for calendar width
- **No backdrop/overlay** - calendar remains visible but unusable
- **Panel width** may be too narrow for comprehensive audit view
- **Mobile experience** likely broken (panel takes full width)

### 3. Video Trailer Vision (Future)
- **Goal:** Each day dot should show a video trailer preview
- **Current state:** Not organized enough - assets scattered
- **Need:** Unified artifact system to generate trailers from

---

## Proposed Solutions

### Phase 1: Fix Calendar Layout (Immediate)

#### Option A: Responsive Grid System
```css
.calendar-demo {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
  gap: 8px;
  padding: 16px;
}

@media (min-width: 768px) {
  .calendar-demo {
    grid-template-columns: repeat(12, 1fr);
  }
}

@media (max-width: 767px) {
  .calendar-demo {
    grid-template-columns: repeat(6, 1fr);
  }
}
```

#### Option B: Month Cards (Better for trailers)
```html
<div class="calendar-months">
  <div class="month-card" data-month="january">
    <h3>January</h3>
    <div class="days-grid">
      <!-- 31 day dots -->
    </div>
  </div>
  <!-- ... 11 more months -->
</div>
```

**Benefits:**
- More space for day dots
- Can show month-level stats
- Better for video previews
- Easier to navigate

### Phase 2: Panel Integration Fix

#### Layout Strategy: Side-by-Side (Desktop)
```css
.app-preview {
  display: grid;
  grid-template-columns: 1fr 400px; /* Calendar | Panel */
  gap: 24px;
  align-items: start;
}

@media (max-width: 1024px) {
  .app-preview {
    grid-template-columns: 1fr;
  }
  
  .audit-panel {
    position: fixed;
    top: 0;
    right: -400px;
    width: 400px;
    height: 100vh;
    transition: right 0.3s ease;
  }
  
  .audit-panel.open {
    right: 0;
  }
}
```

#### Panel States:
1. **Closed:** Calendar full width
2. **Open:** Calendar shrinks, panel slides in
3. **Mobile:** Panel overlays (full-screen drawer)

### Phase 3: Video Trailer System (Future)

#### Artifact Organization Strategy

**Current Problem:**
- Videos scattered across Supabase, local files, CDN
- No unified "trailer" concept
- Hard to know what assets exist for a day

**Proposed Solution:**

1. **Trailer Definition:**
   ```typescript
   interface DayTrailer {
     day: number;
     thumbnail: string; // Best visual from day
     video: string; // 10-15 second preview
     duration: number; // Full lesson duration
     tracks: {
       learn: { title: string; preview: string };
       grow: { title: string; preview: string };
     };
     completeness: number; // 0-100
   }
   ```

2. **Trailer Generation Pipeline:**
   ```
   For each day:
   1. Collect all artifacts (videos, visuals, audio)
   2. Select best "hook" video (Hook phase, Explorer archetype)
   3. Create 10-15s preview (first 10s + fade)
   4. Generate thumbnail from best visual
   5. Store in unified location
   ```

3. **Calendar Display:**
   ```html
   <div class="day-dot" data-day="1">
     <video class="day-preview" preload="metadata" muted>
       <source src="/trailers/day-001-preview.mp4">
     </video>
     <div class="day-overlay">
       <span class="day-number">1</span>
       <span class="completeness-badge">95%</span>
     </div>
   </div>
   ```

---

## Recommended Implementation Order

### Step 1: Fix Calendar Layout (This Week)
- [ ] Implement responsive grid system
- [ ] Test on mobile/tablet/desktop
- [ ] Ensure day dots are clickable
- [ ] Add proper hover states

### Step 2: Fix Panel Integration (This Week)
- [ ] Implement side-by-side layout (desktop)
- [ ] Add overlay drawer (mobile)
- [ ] Test panel open/close transitions
- [ ] Ensure calendar remains usable

### Step 3: Add Hover Previews (Next Week)
- [ ] Show lesson title on hover
- [ ] Show completeness badge
- [ ] Show track indicators (Learn/Grow)
- [ ] Add tooltip with quick stats

### Step 4: Prepare for Trailers (Next Sprint)
- [ ] Create artifact inventory system
- [ ] Build trailer generation script
- [ ] Design trailer storage structure
- [ ] Test trailer playback in calendar

---

## Calendar Redesign Mockup

### Desktop View:
```
┌─────────────────────────────────────────────────────────┐
│  Calendar Grid (8 columns)    │  Audit Panel (400px)  │
│  ┌───┬───┬───┬───┬───┬───┬───┐│  ┌─────────────────┐  │
│  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 ││  │ Day 1 Details   │  │
│  ├───┼───┼───┼───┼───┼───┼───┤│  │                 │  │
│  │ 8 │ 9 │10 │11 │12 │13 │14 ││  │ [Learner View]  │  │
│  └───┴───┴───┴───┴───┴───┴───┘│  │                 │  │
│                                │  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Mobile View:
```
┌─────────────────────┐
│  Calendar Grid       │
│  ┌───┬───┬───┬───┐  │
│  │ 1 │ 2 │ 3 │ 4 │  │
│  ├───┼───┼───┼───┤  │
│  │ 5 │ 6 │ 7 │ 8 │  │
│  └───┴───┴───┴───┘  │
│                     │
│  [Panel slides over]│
└─────────────────────┘
```

---

## Technical Considerations

### Performance
- **Lazy load** trailers (only load on hover/click)
- **Thumbnail first** - show static image until hover
- **Progressive enhancement** - works without videos

### Accessibility
- **Keyboard navigation** - arrow keys to move between days
- **Screen reader** - announce day number, completeness, tracks
- **Focus indicators** - clear visual focus state

### Browser Support
- **Video codecs:** MP4 (H.264) + WebM fallback
- **Poster images:** Always show thumbnail
- **No autoplay:** Respect user preferences

---

## Next Steps

1. **Review this proposal** - Does this align with your vision?
2. **Choose layout approach** - Grid vs Month Cards?
3. **Implement fixes** - Start with calendar layout
4. **Test thoroughly** - Mobile, tablet, desktop
5. **Plan trailer system** - Once calendar is stable

---

**Priority:** 🔴 HIGH (Calendar is broken, needs immediate fix)  
**Complexity:** 🟡 MEDIUM (Layout fixes are straightforward)  
**Timeline:** 1-2 days for layout fixes, 1 week for hover previews

