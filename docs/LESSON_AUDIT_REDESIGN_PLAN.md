# Lesson Audit Redesign: Investigation & Implementation Plan

## Executive Summary

The current audit system (`kelly-lesson-audit.js`) is comprehensive but displays as a full-screen modal, blocking the calendar view. We need to:
1. **Redesign as right-side panel** (slide-in, non-blocking)
2. **Dual-view system**: Learner-first + Educator views
3. **Integrate Grow track** into homepage and audit
4. **Visual completeness indicators** for quick scanning
5. **Full artifact blueprint** showing all lesson components

---

## Current State Analysis

### Existing Audit System (`kelly-lesson-audit.js`)
- **Comprehensive**: Collects from JSON, Supabase, API, local files
- **Tracks**: Learn track only (Grow track partially supported)
- **Assets tracked**:
  - Videos (Supabase + local, all archetypes/ages/languages)
  - Audio (JSON + Supabase + API)
  - Images (thumbnails, infographics, visuals, option cards, Kelly responses)
  - Transcripts & alignments
  - Phases (7 phases per lesson)
- **Display**: Full-screen modal (blocks calendar)
- **View**: Technical/educator-focused

### Homepage (`index.html`)
- **Hero**: "One lesson a day. 365 days a year."
- **Calendar**: 12-column grid (months), clickable day dots
- **Features**: Mentions 365 lessons, ages, languages, personas
- **Missing**: No mention of Grow track / AI Fluency track
- **Interaction**: Single click = audit, double click = preview popup

### Lesson Preview Popup (`lesson-preview-popup.js`)
- **Current**: Compact modal showing completeness %
- **Shows**: Learn track completeness, Grow track completeness
- **Stats**: Phases, videos, visuals, archetypes
- **Status**: Production/Complete/Basic/Skeleton/Missing

---

## Artifact Inventory

### Learn Track Artifacts
1. **Base Content** (40% weight):
   - Topic (multilingual: EN/ES/FR)
   - 7 Phases: Hook, Question, Context, Choice, Reflection, Wisdom, Action
   - Phase scripts (multilingual)
   - Universal Truth

2. **Enhanced Content** (20% weight):
   - HD Videos (per phase, archetype, age, language)
   - Visuals (infographics, phase visuals)
   - Multiple archetypes (12 personas)
   - Option cards (512×512 images)

3. **Metadata**:
   - Category, emoji, headline, tagline
   - Fun facts, discussion questions
   - Date mapping

### Grow Track Artifacts
1. **Base Content** (30% weight):
   - Topic (AI fluency focus)
   - Learning Objective
   - Activity description

2. **Enhanced Content** (10% weight):
   - Full activity content
   - BYOK integration prompts
   - Practice scenarios

### Asset Sources
- **LOCAL_PACKS**: Static JavaScript files (`day-XXX-complete.js`)
- **JSON files**: `/lessons/day-XXX.json`
- **Supabase**: `core_lessons`, `lesson_atoms`, `grow_tracks`
- **API**: `/api/lessons/{day}`
- **Local files**: Generated assets in various directories

---

## Design Requirements

### 1. Right-Side Panel Architecture

**Layout**:
```
┌─────────────────────────────┬──────────────────────┐
│                             │                       │
│   Calendar (main content)    │   Audit Panel        │
│                             │   (slides in)        │
│                             │                       │
│                             │   [View Toggle]       │
│                             │   Learner | Educator  │
│                             │                       │
│                             │   [Content]           │
│                             │                       │
└─────────────────────────────┴──────────────────────┘
```

**Behavior**:
- Slide in from right (400-500px width)
- Overlay calendar (doesn't push it)
- Close button (X) or click outside
- Smooth animation (300ms ease-out)
- Responsive: Full-width on mobile, panel on desktop

### 2. Dual-View System

#### Learner-First View
**Goal**: Show what the learner will experience

**Sections**:
1. **Quick Preview**
   - Day number + date
   - Learn topic + emoji
   - Grow topic + emoji
   - Completeness badge (visual: Production/Complete/Basic/Skeleton)

2. **What You'll Learn**
   - 7 phases preview (icons + brief descriptions)
   - Visual indicators: ✓ Has video, 📊 Has visual, ⚠️ Text only
   - Estimated time per phase

3. **AI Fluency (Grow Track)**
   - Learning objective
   - Activity preview
   - BYOK status (if applicable)

4. **Richness Indicators**
   - Visual completeness meter (0-100%)
   - Asset count badges: "15 videos", "7 visuals", "3 archetypes"
   - Language availability: EN ✓ ES ✓ FR ✓

5. **Start Lesson** button

#### Educator View
**Goal**: Show full technical blueprint

**Sections**:
1. **Metadata**
   - Day number, date, tracks
   - Sources: JSON ✓, Supabase ✓, API ✓, Local ✓
   - Last updated timestamp

2. **Asset Inventory**
   - **Videos**: Table showing phase × archetype × age × language
   - **Visuals**: Infographics, phase visuals, option cards
   - **Audio**: All variants, sources
   - **Text**: All language variants, phase scripts
   - **Grow Track**: Full content breakdown

3. **Pipeline Status**
   - ElevenLabs: Generated/Not generated
   - Audio2Face: Rendered/Not rendered
   - Supabase: Synced/Not synced
   - Local files: Present/Missing

4. **Variant Matrix**
   - Languages: EN, ES, FR
   - Age buckets: Toddler, Child, Teen, Adult, Senior, Elder
   - Archetypes: All 12 personas
   - Phases: All 7 phases

5. **Errors & Warnings**
   - Missing assets
   - Pipeline failures
   - Sync issues

### 3. Visual Completeness Indicators

**Calendar Integration**:
- **Color coding**:
  - 🟢 Green (80-100%): Production ready
  - 🔵 Blue (60-79%): Complete
  - 🟡 Yellow (40-59%): Basic
  - ⚪ Gray (0-39%): Skeleton/Missing

- **Badge overlay** on day dots:
  - Small indicator showing completeness %
  - Track indicators: L (Learn) / G (Grow) badges
  - Hover: Quick preview tooltip

**Panel Indicators**:
- Progress bars per track
- Asset type icons with counts
- Status badges (Production/Complete/Basic/Skeleton)

### 4. Grow Track Integration

**Homepage Updates**:
1. **Hero section**: Update copy to mention "Two tracks: Learn + AI Fluency"
2. **Features section**: Add Grow track feature card
3. **Calendar**: Show Grow track completion badges
4. **Tooltips**: Include Grow track topic in hover

**Audit Panel**:
- Always show both tracks
- Side-by-side completeness comparison
- Grow track artifact breakdown
- BYOK integration status

---

## Implementation Plan

### Phase 1: Panel Architecture
1. Create `lesson-audit-panel.js` (new component)
2. CSS for right-side slide-in panel
3. Panel state management (open/close, view toggle)
4. Integration with homepage calendar clicks

### Phase 2: Data Collection Enhancement
1. Extend `kelly-lesson-audit.js` to include Grow track
2. Add completeness calculation (reuse from `lesson-preview-popup.js`)
3. Aggregate asset counts per track
4. Calculate visual completeness scores

### Phase 3: Learner-First View
1. Design learner-friendly layout
2. Phase preview cards with asset indicators
3. Completeness visualization
4. Quick start actions

### Phase 4: Educator View
1. Technical asset tables
2. Pipeline status indicators
3. Variant matrix display
4. Error/warning reporting

### Phase 5: Calendar Integration
1. Update day dot rendering with completeness colors
2. Add track badges (L/G)
3. Enhanced tooltips with track info
4. Visual completeness indicators

### Phase 6: Homepage Grow Track
1. Update hero copy
2. Add Grow track feature card
3. Update calendar tooltips
4. Add Grow track CTA

---

## Technical Specifications

### Panel Component Structure
```javascript
LessonAuditPanel = {
  // State
  currentDay: null,
  currentView: 'learner', // 'learner' | 'educator'
  auditData: null,
  
  // Methods
  show(dayNumber),
  close(),
  toggleView(),
  loadAudit(dayNumber),
  renderLearnerView(),
  renderEducatorView()
}
```

### CSS Classes
- `.audit-panel`: Main panel container
- `.audit-panel.open`: Open state
- `.audit-panel.learner-view`: Learner view active
- `.audit-panel.educator-view`: Educator view active
- `.completeness-badge`: Visual completeness indicator
- `.asset-indicator`: Asset type icons
- `.phase-preview`: Phase preview cards

### Data Structure
```javascript
{
  dayNumber: 1,
  date: Date,
  tracks: {
    learn: {
      completeness: 85,
      status: 'production',
      assets: {
        videos: 15,
        visuals: 7,
        phases: 7,
        archetypes: 3,
        languages: ['en', 'es', 'fr']
      },
      phases: [...]
    },
    grow: {
      completeness: 60,
      status: 'complete',
      assets: {
        topic: true,
        objective: true,
        activity: true
      }
    }
  },
  sources: {...},
  pipelines: {...}
}
```

---

## User Experience Flow

### Learner Journey
1. **Homepage**: See calendar with completeness colors
2. **Hover day**: See tooltip with Learn + Grow topics
3. **Click day**: Panel slides in with learner view
4. **See preview**: What they'll learn, completeness, time estimate
5. **Start lesson**: Click button → navigate to `/learn.html?day=X`

### Educator Journey
1. **Homepage**: See calendar with completeness colors
2. **Click day**: Panel slides in (defaults to learner view)
3. **Switch to educator**: Toggle button → see full blueprint
4. **Review assets**: Check videos, visuals, variants
5. **Identify gaps**: See missing assets, pipeline failures
6. **Take action**: Fix issues, regenerate assets

---

## Success Metrics

1. **Usability**: Panel opens/closes smoothly, doesn't block calendar
2. **Clarity**: Learners understand what they'll get
3. **Completeness**: Educators see full artifact inventory
4. **Visual**: Quick scanning via color coding and badges
5. **Integration**: Grow track visible throughout homepage

---

## Next Steps

1. **Approve design**: Review this plan
2. **Create panel component**: Build `lesson-audit-panel.js`
3. **Update audit system**: Enhance data collection
4. **Implement views**: Build learner + educator UIs
5. **Integrate homepage**: Add Grow track, update calendar
6. **Test & refine**: User testing, iteration

---

## Questions to Resolve

1. **Panel width**: 400px or 500px? (Consider content density)
2. **Default view**: Learner-first or remember last selection?
3. **Mobile behavior**: Full-screen or bottom sheet?
4. **Animation**: Slide-in or fade-in?
5. **Completeness threshold**: What % = "Production Ready"?

---

## Files to Create/Modify

### New Files
- `public/js/lesson-audit-panel.js` - Main panel component
- `public/css/audit-panel.css` - Panel styles (or add to existing)

### Modified Files
- `public/js/kelly-lesson-audit.js` - Enhance Grow track support
- `public/index.html` - Add panel HTML, update hero/features
- `public/js/lesson-preview-popup.js` - May integrate or replace

### Integration Points
- Calendar click handlers (`index.html` line ~1222)
- Tooltip system (`index.html` line ~1257)
- Completeness calculation (reuse from `lesson-preview-popup.js`)





