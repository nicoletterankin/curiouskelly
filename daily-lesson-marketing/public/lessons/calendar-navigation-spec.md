# Calendar Navigation Specification

## Navigation Structure

### Four Views:

1. **Yearly View** - Overview of all 365 days
2. **Monthly View** - Traditional calendar month grid
3. **Weekly View** - 7-day detailed view
4. **Today's Lesson** - Focus on current day

---

## Yearly View

### Layout:
```
┌─────────────────────────────────────┐
│ [Year: 2025] [◀] [▶]                │
│                                     │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│ │Jan │ │Feb │ │Mar │ │Apr │       │
│ │ 31 │ │ 28 │ │ 31 │ │ 30 │       │
│ │🧬3 │ │🧬2 │ │🧬4 │ │🧬1 │       │
│ │✓15 │ │✓12 │ │✓18 │ │✓14 │       │
│ └────┘ └────┘ └────┘ └────┘       │
│                                     │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│ │May │ │Jun │ │Jul │ │Aug │       │
│ │ 31 │ │ 30 │ │ 31 │ │ 31 │       │
│ │🧬2 │ │🧬3 │ │🧬5 │ │🧬2 │       │
│ │✓16 │ │✓15 │ │✓19 │ │✓17 │       │
│ └────┘ └────┘ └────┘ └────┘       │
│                                     │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│ │Sep │ │Oct │ │Nov │ │Dec │       │
│ │ 30 │ │ 31 │ │ 30 │ │ 31 │       │
│ │🧬3 │ │🧬4 │ │🧬2 │ │🧬3 │       │
│ │✓13 │ │✓16 │ │✓14 │ │✓18 │       │
│ └────┘ └────┘ └────┘ └────┘       │
│                                     │
│ Stats: 365 lessons | 43 DNA | 189 completed │
└─────────────────────────────────────┘
```

### Features:
- **12 Month Cards:** Each shows:
  - Month name
  - Number of days
  - DNA lesson count (🧬 badge)
  - Completed lessons count (✓)
- **Click Month Card:** Opens monthly view
- **Year Navigation:** Previous/Next year buttons
- **Stats Bar:** Total lessons, DNA count, completed count
- **Scrollable:** If needed on smaller screens

### Interaction:
- Hover: Shows month preview
- Click: Opens monthly view for that month
- Today's month: Highlighted with border

---

## Monthly View

### Layout:
```
┌─────────────────────────────────────┐
│ [◀] January 2025 [▶]                │
│                                     │
│  S   M   T   W   T   F   S         │
│ ─── ─── ─── ─── ─── ─── ───        │
│     1   2   3   4   5   6          │
│  🧬 ✓   ○   ○   ○   ○   ○          │
│                                     │
│  7   8   9  10  11  12  13         │
│  ○   🧬  ○   ○   ○   ○   ○         │
│                                     │
│ 14  15  16  17  18  19  20         │
│  ○   ○   ○   ○   ○   ○   ○         │
│                                     │
│ 21  22  23  24  25  26  27         │
│  ○   ○   ○   ○   ○   ○   ○         │
│                                     │
│ 28  29  30  31                      │
│  ○   ○   ○   ○                      │
│                                     │
│ [Back to Year] [Today]              │
└─────────────────────────────────────┘
```

### Features:
- **Traditional Grid:** 7-day week layout
- **Day Indicators:**
  - 🧬 = DNA lesson
  - ✓ = Completed
  - ● = In progress
  - ○ = Not started
  - **Today:** Highlighted with border/color
- **Day Click:** Selects day, shows in right rail
- **Month Navigation:** Previous/Next month buttons
- **Quick Actions:** "Back to Year", "Jump to Today"

### Day Cell States:
- **Default:** Gray background, lesson title truncated
- **Hover:** Light blue background, full title tooltip
- **Selected:** Blue border, right rail shows details
- **Today:** Green border, "Today" badge
- **Completed:** Checkmark icon, darker background
- **DNA Lesson:** 🧬 badge in corner

---

## Weekly View

### Layout:
```
┌─────────────────────────────────────┐
│ [◀] Week of Jan 1, 2025 [▶]         │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Sunday, Jan 1                   │ │
│ │ 🧬 The Sun - Our Life-Giving...│ │
│ │ [Play] ▶                        │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Monday, Jan 2                   │ │
│ │ Habit Stacking for Productivity │ │
│ │ [Play] ▶                        │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Tuesday, Jan 3                  │ │
│ │ Our Amazing Planet Earth        │ │
│ │ [Play] ▶                        │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ... (7 days total)                  │
│                                     │
│ [Back to Month] [Today]             │
└─────────────────────────────────────┘
```

### Features:
- **7-Day List:** Vertical scrollable list
- **Each Day Shows:**
  - Day name and date
  - DNA badge if applicable
  - Full lesson title
  - [Play] button
  - Progress indicator (if started)
- **Week Navigation:** Previous/Next week buttons
- **Quick Play:** Click [Play] starts lesson immediately

### Interaction:
- **Click Day Card:** Selects day, shows in right rail
- **Click [Play]:** Starts lesson (Kelly begins)
- **Scroll:** Navigate through week
- **Today:** Highlighted, scrolls into view

---

## Today's Lesson View

### Layout:
```
┌─────────────────────────────────────┐
│ Today's Lesson                      │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Day 189 - July 8, 2025          │ │
│ │                                 │ │
│ │ 🧬 Biochemistry - The Chemistry│ │
│ │    of Life                      │ │
│ │                                 │ │
│ │ Learning Objective:             │ │
│ │ Explore molecular biology while │ │
│ │ understanding how biochemical...│ │
│ │                                 │ │
│ │ Age Variants: 2-5, 6-12, 13-17 │ │
│ │ Languages: EN, ES, FR           │ │
│ │                                 │ │
│ │ Progress: ▓▓▓▓▓░░░░░ 50%        │ │
│ │                                 │ │
│ │ [▶ Play Lesson] [Resume]        │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Yesterday: Day 188             │ │
│ │ [◀ Previous]                   │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Tomorrow: Day 190              │ │
│ │ [Next ▶]                       │ │
│ └─────────────────────────────────┘ │
│                                     │
│ [View Calendar]                     │
└─────────────────────────────────────┘
```

### Features:
- **Large Lesson Card:** Full lesson information
- **DNA Badge:** Prominent if DNA lesson
- **Learning Objective:** Full text, scrollable
- **Metadata:** Age variants, languages, progress
- **Actions:** Play, Resume, or Start buttons
- **Navigation:** Previous/Next day buttons
- **Quick Stats:** Yesterday/Tomorrow previews

### Interaction:
- **Click [Play]:** Starts lesson from beginning
- **Click [Resume]:** Continues from last position
- **Click Previous/Next:** Navigate to adjacent days
- **Click [View Calendar]:** Opens monthly view

---

## Navigation Flow

### Default State:
```
User opens app → Today's Lesson view shown
```

### Navigation Paths:
```
Today's Lesson
  ↓ [View Calendar]
Monthly View
  ↓ [Year View]
Yearly View
  ↓ Click Month
Monthly View
  ↓ Click Day
Weekly View (or Right Rail Details)
  ↓ Click [Play]
Kelly plays lesson
```

### Quick Actions:
- **Keyboard Shortcuts:**
  - `T` = Today's lesson
  - `Y` = Year view
  - `M` = Month view
  - `W` = Week view
  - `→` = Next day
  - `←` = Previous day
  - `Space` = Play/Pause

- **Bottom Controls:**
  - Calendar icon = Toggle calendar
  - Today button = Jump to today
  - Previous/Next = Navigate days

---

## Data Integration

### Calendar Data Source:
- `365_day_calendar.json`
- 365 lessons with metadata
- DNA lesson indicators
- Progress tracking (user-specific)

### Real-Time Updates:
- Progress synced from lesson player
- Completion status updated after lesson
- Streak tracking (consecutive days)
- Last played position saved

---

## Responsive Considerations

### Desktop (1920x1080+):
- Full side rails (240-320px each)
- Kelly center (remaining space)
- All views fully visible

### Tablet (768-1920px):
- Collapsible side rails
- Kelly remains center
- Calendar in overlay when needed

### Mobile (<768px):
- Bottom drawer calendar
- Kelly full screen when playing
- Simplified navigation

---

## Implementation Priority

1. **Today's Lesson View** (MVP)
2. **Monthly View** (Core navigation)
3. **Weekly View** (Detailed planning)
4. **Yearly View** (Overview)

---

**Status:** Ready for implementation
**Framework:** Framework 2 (Left-Rail Nav + Right-Rail Details)

