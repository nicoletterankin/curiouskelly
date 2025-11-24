# Calendar Layout Proposals - Two Panel Design

## Requirements Summary
- **Two panels**: Kelly (always visible) + Lesson Details/Calendar (expandable)
- **Interactive lessons**: No play button - users click options to progress
- **5 phases per lesson**: Welcome → Q1 → Q2 → Q3 → Wisdom
- **Two options per phase**: Each option has pre-recorded feedback
- **Age selector**: Change content age (2-5, 6-12, 13-17, 18-35, 36-60, 61-102)
- **Language selector**: Change Kelly's language (EN, ES, FR)
- **Navigation**: Forward/Back buttons for phase navigation

---

## Layout Proposal #1: Side Panel Slide-Over

### Description
Kelly occupies full screen by default. A side panel (left or right) slides in when user interacts with calendar/lesson. Panel can be toggled open/closed. Kelly smoothly shifts to accommodate panel.

### Layout Structure
```
┌─────────────────────────────────────┐
│ [Panel Toggle]                      │
├─────────────────────────────────────┤
│                                     │
│         KELLY (Full Screen)         │
│         (Shifts when panel open)    │
│                                     │
│                                     │
└─────────────────────────────────────┘

When Panel Open:
┌──────────┬──────────────────────────┐
│          │                          │
│  PANEL   │      KELLY (Shifted)     │
│  (300px) │      (Remaining space)   │
│          │                          │
│          │                          │
└──────────┴──────────────────────────┘
```

### Features
- **Panel width**: 300-350px
- **Smooth animation**: Kelly slides, panel slides in
- **Toggle button**: Always visible to open/close panel
- **Panel content**: Calendar navigation + Lesson details + Phase options
- **Controls**: Bottom bar with Forward/Back, Age selector, Language selector

### Pros
- Kelly always visible, never hidden
- Clean, uncluttered default state
- Familiar slide-over pattern
- Easy to dismiss when not needed

### Cons
- Kelly gets smaller when panel opens
- May feel cramped on smaller screens

---

## Layout Proposal #2: Split Screen Toggle

### Description
Two modes: "Full Kelly" (100% Kelly) and "Split View" (50/50 Kelly + Panel). User toggles between modes. In split view, both panels are always visible side-by-side.

### Layout Structure
```
Mode 1: Full Kelly
┌─────────────────────────────────────┐
│ [Split Toggle]                       │
├─────────────────────────────────────┤
│                                     │
│         KELLY (100%)                │
│                                     │
│                                     │
└─────────────────────────────────────┘

Mode 2: Split View
┌──────────────────┬──────────────────┐
│                  │                  │
│   KELLY (50%)    │  PANEL (50%)     │
│                  │                  │
│                  │                  │
└──────────────────┴──────────────────┘
```

### Features
- **Toggle button**: Switch between full Kelly and split view
- **Split ratio**: 50/50 or 60/40 (configurable)
- **Panel always visible in split mode**: Calendar + Lesson + Options
- **Controls**: Bottom bar with Forward/Back, Age, Language

### Pros
- Clear separation of modes
- Both panels visible in split mode
- No animation needed (instant toggle)
- Good for focused lesson interaction

### Cons
- Kelly smaller in split mode
- Binary choice (no partial view)

---

## Layout Proposal #3: Bottom Drawer

### Description
Kelly full screen. Lesson panel slides up from bottom (like mobile drawer). Can be expanded to 40% height or collapsed to just controls bar. Kelly scales down when drawer is expanded.

### Layout Structure
```
Collapsed (Controls Only):
┌─────────────────────────────────────┐
│                                     │
│         KELLY (Full Screen)         │
│                                     │
│                                     │
├─────────────────────────────────────┤
│ [◀] [▶] [Age] [Lang] [Expand ▲]    │
└─────────────────────────────────────┘

Expanded:
┌─────────────────────────────────────┐
│         KELLY (60% Height)           │
│                                     │
├─────────────────────────────────────┤
│  PANEL (40% Height)                 │
│  Calendar | Lesson | Options        │
│  [◀] [▶] [Age] [Lang] [Collapse ▼] │
└─────────────────────────────────────┘
```

### Features
- **Drawer height**: 40% when expanded, ~60px when collapsed
- **Smooth slide animation**: Up/down
- **Panel sections**: Tabs for Calendar / Lesson Details / Phase Options
- **Controls always visible**: Bottom bar

### Pros
- Kelly stays large even when expanded
- Familiar mobile pattern
- Good for touch interfaces
- Easy to collapse when not needed

### Cons
- Less horizontal space for lesson details
- May feel cramped for calendar view

---

## Layout Proposal #4: Overlay Panel

### Description
Kelly full screen always. Lesson panel overlays on top (like modal) but can be resized and repositioned. Can be minimized to corner or dismissed. Panel has semi-transparent background so Kelly is still visible behind it.

### Layout Structure
```
Default:
┌─────────────────────────────────────┐
│                                     │
│         KELLY (Full Screen)         │
│                                     │
│                                     │
└─────────────────────────────────────┘

Panel Open (Overlay):
┌─────────────────────────────────────┐
│         KELLY (Visible Behind)      │
│  ┌──────────────────────┐          │
│  │  PANEL (Overlay)     │ [X]      │
│  │  Calendar | Lesson   │          │
│  │  Options | Controls  │          │
│  └──────────────────────┘          │
│                                     │
└─────────────────────────────────────┘
```

### Features
- **Panel size**: 400px × 600px (resizable)
- **Position**: Draggable, can snap to corners
- **Background**: Semi-transparent (80% opacity)
- **Minimize**: Collapse to small icon in corner
- **Controls**: Inside panel + bottom bar

### Pros
- Kelly never moves or resizes
- Flexible positioning
- Can see Kelly while interacting with panel
- Modern overlay pattern

### Cons
- Panel may obscure Kelly
- More complex interaction model
- May feel cluttered

---

## Layout Proposal #5: Picture-in-Picture

### Description
Kelly full screen. Lesson panel appears as floating window (like PiP video). Can be moved, resized, minimized. Panel has calendar, lesson details, and phase options. Controls are in the floating panel.

### Layout Structure
```
Default:
┌─────────────────────────────────────┐
│                                     │
│         KELLY (Full Screen)         │
│                                     │
│                                     │
└─────────────────────────────────────┘

Panel Open (Floating):
┌─────────────────────────────────────┐
│         KELLY (Full Screen)         │
│  ┌──────────────┐                   │
│  │   PANEL      │ [─] [□] [×]      │
│  │   (Floating) │                   │
│  │   Calendar   │                   │
│  │   Lesson     │                   │
│  │   Options    │                   │
│  └──────────────┘                   │
│                                     │
└─────────────────────────────────────┘
```

### Features
- **Panel size**: 350px × 500px (resizable)
- **Position**: Draggable anywhere on screen
- **Window controls**: Minimize, maximize, close
- **Always on top**: Panel stays above Kelly
- **Controls**: Inside floating panel

### Pros
- Kelly never affected
- Very flexible
- Familiar windowing pattern
- Can have multiple panels (calendar + lesson)

### Cons
- May feel disconnected
- More complex UI
- Not ideal for touch

---

## Recommendation: **Layout #1 - Side Panel Slide-Over**

### Rationale
1. **Best balance**: Kelly stays visible, panel doesn't obscure
2. **Smooth UX**: Familiar slide-over pattern, intuitive toggle
3. **Flexible**: Panel can show calendar OR lesson details OR both (tabs)
4. **Touch-friendly**: Large touch targets, easy swipe gestures
5. **Scalable**: Works on all screen sizes
6. **Clean**: Uncluttered default state, panel appears when needed

### Implementation Details
- **Panel position**: Right side (or left, user preference)
- **Panel width**: 320px (optimal for lesson options)
- **Animation**: 300ms ease-in-out
- **Panel sections**: 
  - Tab 1: Calendar (Year/Month/Week/Today)
  - Tab 2: Lesson Details (Current phase, options, progress)
  - Tab 3: Settings (Age selector, Language selector)
- **Bottom controls**: Forward/Back buttons, current phase indicator
- **Panel toggle**: Button in top-right corner, or swipe gesture

---

## Interactive Lesson Flow

### Phase Structure
```
Welcome Phase
├─ Option A (pre-recorded feedback)
└─ Option B (pre-recorded feedback)
    ↓
Question 1 Phase
├─ Option A (pre-recorded feedback)
└─ Option B (pre-recorded feedback)
    ↓
Question 2 Phase
├─ Option A (pre-recorded feedback)
└─ Option B (pre-recorded feedback)
    ↓
Question 3 Phase
├─ Option A (pre-recorded feedback)
└─ Option B (pre-recorded feedback)
    ↓
Wisdom Phase
└─ Final message (no options)
```

### Controls
- **Forward/Back**: Navigate between phases (only forward available if not completed)
- **Option buttons**: Two large buttons per phase (except wisdom)
- **Age selector**: Dropdown (2-5, 6-12, 13-17, 18-35, 36-60, 61-102)
- **Language selector**: Dropdown (EN, ES, FR)
- **Progress indicator**: Shows current phase (Welcome, Q1, Q2, Q3, Wisdom)

---

## Next Steps
1. User selects preferred layout
2. Implement selected layout
3. Build interactive lesson phase system
4. Add age/language selectors
5. Integrate with Unity for Kelly's responses

