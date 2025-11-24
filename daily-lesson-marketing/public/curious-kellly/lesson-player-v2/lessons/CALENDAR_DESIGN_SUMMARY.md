# Calendar Design - Understanding & Recommendation

## ✅ Confirmed Understanding

### Layout Constraints:
1. **Kelly's 3D Avatar** - Always visible in center of screen
2. **Kelly IS the Lesson Player** - She lipsyncs lessons when playing
3. **Lesson Controls** - Middle bottom third of screen
4. **Side Rails** - UI and calendar on left/right side rails
5. **Interaction Model** - Clicking a day/topic/lesson makes Kelly play that lesson
6. **Always-On** - Kelly remains visible even when calendar is open

### Requirements:
- Display 365 days of lessons
- Show progress/completion status
- Indicate DNA lessons (🧬 badge)
- Multiple views: Yearly, Monthly, Weekly, Today's Lesson
- Non-intrusive to Kelly's presence
- Accessible and clear navigation

---

## 🎯 Recommended Framework: #2 (Left-Rail Nav + Right-Rail Details)

### Why This Framework?

**Primary Reasons:**
1. ✅ **Kelly-Centric:** Kelly remains fully visible and prominent
2. ✅ **Information Rich:** Can show lesson details without blocking Kelly
3. ✅ **Familiar Pattern:** Matches existing side-rail UI pattern
4. ✅ **Scalable:** Works perfectly for all 365 days
5. ✅ **Progressive Disclosure:** Navigation on left, details on right
6. ✅ **Educational Focus:** Right rail perfect for learning objectives

### Layout Structure:

```
┌─────────────────────────────────────────────────────────────┐
│ [Left Rail: 240px]  │  [Kelly: Center]  │  [Right Rail: 280px] │
│                      │                   │                      │
│ 📅 Calendar          │   👤 Kelly       │  [Selected Lesson]    │
│                      │   (3D Avatar)    │                      │
│ [Year] [Month]       │   Lipsyncing     │  Day 189              │
│ [Week] [Today]       │   when playing   │  🧬 Biochemistry      │
│                      │                   │                      │
│ Year View:           │                   │  Learning Objective: │
│ ┌──┐ ┌──┐ ┌──┐      │                   │  [Full text...]      │
│ │J │ │F │ │M │      │                   │                      │
│ │31│ │28│ │31│      │                   │  [▶ Play Lesson]     │
│ │🧬│ │🧬│ │🧬│      │                   │                      │
│ └──┘ └──┘ └──┘      │                   │  Progress: ▓▓▓░░ 60% │
│ ... (scrollable)     │                   │                      │
│                      │                   │                      │
│ [Controls: Bottom Third]                 │                      │
└─────────────────────────────────────────────────────────────┘
```

### Four Views Implemented:

#### 1. **Yearly View**
- 12 month cards in scrollable grid
- Shows: Days count, DNA count (🧬), Completed count (✓)
- Click month → Opens monthly view
- Year navigation: Previous/Next

#### 2. **Monthly View**
- Traditional 7-day calendar grid
- Day indicators: 🧬 (DNA), ✓ (completed), ● (in progress), ○ (upcoming)
- Today highlighted
- Click day → Shows in right rail
- Month navigation: Previous/Next

#### 3. **Weekly View**
- 7-day vertical list
- Each day: Full title, DNA badge, [Play] button
- Week navigation: Previous/Next
- Quick play from list

#### 4. **Today's Lesson**
- Large lesson card with full details
- DNA badge, learning objective, metadata
- [Play] or [Resume] buttons
- Previous/Next day navigation
- Yesterday/Tomorrow previews

---

## 🎨 Visual Design Principles

### Kelly-Centric Hierarchy:
1. **Kelly** (Primary - always visible, center)
2. **Selected Lesson** (Right rail - when day clicked)
3. **Calendar Navigation** (Left rail - always available)
4. **Controls** (Bottom - lesson playback)

### Color Coding:
- **Today:** Green border/highlight
- **Completed:** Checkmark (✓), darker background
- **In Progress:** Blue dot (●), blue border
- **DNA Lesson:** 🧬 badge, purple accent
- **Upcoming:** Gray (○), default state

### Interaction States:
- **Hover:** Light background, tooltip with full title
- **Selected:** Blue border, right rail shows details
- **Playing:** Kelly lipsyncing, progress bar active
- **Completed:** Checkmark visible, muted colors

---

## 📱 Responsive Behavior

### Desktop (1920x1080+):
- Full side rails visible
- Kelly center, full size
- All views fully functional

### Tablet (768-1920px):
- Collapsible side rails (icon mode)
- Kelly remains center
- Calendar expands on click

### Mobile (<768px):
- Bottom drawer calendar
- Kelly full screen when playing
- Simplified navigation

---

## 🔄 Navigation Flow

### Default:
```
App Opens → Today's Lesson View (Left Rail)
```

### User Journey:
```
Today's Lesson
  ↓ [View Calendar]
Monthly View
  ↓ [Year View]
Yearly View (12 months)
  ↓ Click Month Card
Monthly View (that month)
  ↓ Click Day
Right Rail Shows Details
  ↓ Click [Play]
Kelly Plays Lesson
```

### Quick Actions:
- **T** = Today's lesson
- **Y** = Year view
- **M** = Month view
- **W** = Week view
- **→** = Next day
- **←** = Previous day
- **Space** = Play/Pause

---

## 📊 Data Integration

### Calendar Data:
- Source: `365_day_calendar.json`
- 365 lessons with full metadata
- DNA lesson indicators
- Progress tracking (user-specific)

### Real-Time Updates:
- Progress synced from lesson player
- Completion status after lesson ends
- Streak tracking (consecutive days)
- Last played position saved

---

## ✅ Implementation Checklist

### Phase 1: Core Navigation
- [ ] Left rail calendar component
- [ ] View selector (Year/Month/Week/Today)
- [ ] Yearly view (12 month cards)
- [ ] Monthly view (calendar grid)
- [ ] Weekly view (7-day list)
- [ ] Today's lesson view

### Phase 2: Right Rail Details
- [ ] Selected lesson display
- [ ] DNA badge rendering
- [ ] Learning objective display
- [ ] Play/Resume buttons
- [ ] Progress indicators

### Phase 3: Integration
- [ ] Connect to 365_day_calendar.json
- [ ] Kelly lesson player integration
- [ ] Progress tracking
- [ ] Navigation state management

### Phase 4: Polish
- [ ] Animations and transitions
- [ ] Keyboard shortcuts
- [ ] Responsive behavior
- [ ] Accessibility features

---

## 📄 Documents Created

1. **CALENDAR_DESIGN_FRAMEWORKS.md** - 5 design options with rationale
2. **calendar-navigation-spec.md** - Detailed navigation specification
3. **CALENDAR_DESIGN_SUMMARY.md** - This summary (understanding + recommendation)

---

## 🎯 Final Recommendation

**Use Framework #2 (Left-Rail Nav + Right-Rail Details)**

This framework:
- ✅ Keeps Kelly as the star (always visible)
- ✅ Provides rich information without blocking
- ✅ Scales to 365 days beautifully
- ✅ Matches existing UI patterns
- ✅ Supports all four required views
- ✅ Educational focus (learning objectives visible)

**Alternative:** If Framework #2 feels too complex, use Framework #1 (Right-Rail Expandable) as a simpler starting point.

---

**Status:** ✅ Ready for implementation  
**Next Step:** Begin Phase 1 implementation

