# Journey Navigation — Guardrails & Architecture

## Overview

The Journey panel is the primary "jungle gym" for lesson exploration. It provides multiple views into the 365-day curriculum:

1. **📅 Calendar** — Year-at-a-glance with thumbnails
2. **📆 Week** — Current week with full lesson cards
3. **📚 Curriculum** — Dual-track browser with search
4. **🔖 Saved** — Bookmarked moments

## Architecture

### Panel System

```
┌─────────────────────────────────────────────────────┐
│  #right-panel                                       │
│  ├── .panel-header (📅 Journey title + close)       │
│  └── .panel-body                                    │
│       └── #panel-journey-slot                       │
│            └── .journey-scroll (moved from scene)   │
│                 ├── .stats-row (Streak/Completed)   │
│                 ├── .journey-tabs (4 tab buttons)   │
│                 └── .journey-content                │
│                      ├── #tab-calendar              │
│                      ├── #tab-week                  │
│                      ├── #tab-curriculum            │
│                      └── #tab-bookmarks             │
└─────────────────────────────────────────────────────┘
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `switchJourneyTab(tabId)` | Toggles tab visibility, builds content |
| `buildGridView()` | Renders 365 day cells with thumbnails |
| `buildWeekView()` | Renders 7-day cards with full lesson info |
| `buildCurriculumView()` | Initializes KellyCurriculumBrowser |
| `buildBookmarksView()` | Renders saved bookmarks |
| `showPhaseSelector(day)` | Opens modal to select phase |

### Event Listeners (setupEventListeners)

```javascript
// Journey tabs
document.querySelectorAll('.journey-tab').forEach(tab => {
  tab.addEventListener('click', () => switchJourneyTab(tab.dataset.tab));
});

// Calendar month navigation
document.getElementById('calendar-prev-month')?.addEventListener('click', ...);
document.getElementById('calendar-next-month')?.addEventListener('click', ...);

// Phase selector close
document.getElementById('phase-selector-close')?.addEventListener('click', hidePhaseSelector);
document.querySelector('.phase-selector-backdrop')?.addEventListener('click', hidePhaseSelector);

// Curriculum search (debounced 300ms)
curriculumSearchInput.addEventListener('input', (e) => {
  KellyCurriculumBrowser.search(e.target.value);
});
```

## Guardrails — DO NOT BREAK

### 1. Tab Content Visibility

The `.journey-tab-content` elements use `display: none` by default and `.active` class to show:

```css
.journey-tab-content { display: none; }
.journey-tab-content.active { display: block; }
```

⚠️ **DO NOT** add separate display rules for `.grid-view`, `.week-view` etc. They inherit visibility from their parent `.journey-tab-content`.

### 2. Panel Content Mounting

Content moves from `#scene-journey` to `#panel-journey-slot` when panel opens:

```javascript
// In mountSacredRightPanelContent() and syncSacredPanels()
journeySlot.appendChild(journeyScroll);
```

⚠️ **DO NOT** duplicate content — there's only ONE `.journey-scroll` element.

### 3. Curriculum Browser Integration

The `KellyCurriculumBrowser` module handles:
- Dual-track (Learn + Grow) display
- Month expansion/collapse
- Search functionality

```javascript
// Initialize curriculum tab
if (typeof KellyCurriculumBrowser !== 'undefined') {
  KellyCurriculumBrowser.init('curriculum-categories');
}

// Search
KellyCurriculumBrowser.search(query);
```

### 4. Day Cell Structure

Each calendar day cell has:

```html
<div class="day-cell" data-day="123">
  <div class="day-cell-bg">
    <img src="thumbnail.webp" />
  </div>
  <span class="day-cell-num">15</span>
  <div class="day-cell-tooltip">Lesson Topic</div>
</div>
```

### 5. Week Day Card Structure

Each week view card shows full lesson info:

```html
<div class="week-day">
  <div class="day-thumb">
    <img class="day-thumb-img" src="thumbnail.webp" />
  </div>
  <div class="day-num">15</div>
  <div class="day-info">
    <div class="day-topic">Lesson Title</div>
    <div class="day-preview">Marketing headline</div>
  </div>
  <div class="day-status">✓</div>
</div>
```

## Testing Checklist

Before deploying changes to Journey navigation:

- [ ] Calendar tab shows 12 months with day cells
- [ ] Thumbnails load for days that have them
- [ ] Click on day opens phase selector modal
- [ ] Phase selector shows topic and 7 phases
- [ ] Week tab shows 7 days with titles + descriptions
- [ ] Curriculum tab shows search input + categories
- [ ] Search filters lessons across tracks
- [ ] Saved tab shows bookmarks or empty state
- [ ] Tab switching preserves state
- [ ] Month navigation (Prev/Next) works
- [ ] Hover tooltips show on day cells

## Files Involved

| File | Purpose |
|------|---------|
| `public/learn.html` | Main app with all Journey UI |
| `public/js/kelly-curriculum-browser.js` | Curriculum search & display |
| `public/js/kelly-lesson-loader.js` | Lesson data fetching |
| `scripts/journey-audit.mjs` | Puppeteer test script |

## Future Enhancements (Out of Scope)

- **On-demand day generation** — Generate custom lessons when topic not in curriculum
- **Advanced search filters** — By track, month, difficulty
- **Lesson previews** — Video thumbnails on hover
- **Progress visualization** — Heat map of completed days
