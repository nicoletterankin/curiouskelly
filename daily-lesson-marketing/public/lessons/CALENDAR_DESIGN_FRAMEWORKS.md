# Calendar Design Frameworks - Kelly-Centric Layout

## Design Constraints Confirmed ✅

### Layout Understanding:
- **Kelly's 3D Avatar:** Always visible in center of screen
- **Kelly's Role:** She IS the lesson player - lipsyncing lessons when playing
- **Lesson Controls:** Middle bottom third of screen
- **Side Rails:** UI and calendar on left/right side rails
- **Interaction:** Clicking a day/topic/lesson makes Kelly play that lesson
- **Always-On:** Kelly remains visible even when calendar is open

### Requirements:
- Display 365 days of lessons
- Show progress/completion status
- Indicate DNA lessons (🧬 badge)
- Multiple views: Yearly, Monthly, Weekly, Today's Lesson
- Non-intrusive to Kelly's presence
- Accessible and clear navigation

---

## Framework 1: Right-Rail Expandable Calendar

### Layout:
```
┌─────────────────────────────────────────────────────┐
│  [Left Rail: Settings/Search]  │  Kelly (Center)  │  [Right Rail: Calendar] │
│                                 │                  │                        │
│                                 │   👤 Kelly      │  📅 Calendar Icon      │
│                                 │   (3D Avatar)   │  ┌──────────────────┐  │
│                                 │                  │  │ View: [Year] ▼   │  │
│                                 │                  │  │                  │  │
│                                 │                  │  │ [Year View]      │  │
│                                 │                  │  │ 12 Month Cards  │  │
│                                 │                  │  │                  │  │
│                                 │                  │  │                  │  │
│  [Controls: Bottom Third]       │                  │  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Design Details:
- **Right rail:** Collapsed to icon (📅), expands on click
- **Expanded width:** 320-400px (doesn't block Kelly)
- **Year view:** 12 month cards in scrollable grid
- **Month view:** Traditional calendar grid (compact)
- **Week view:** 7-day list (vertical scroll)
- **Today:** Highlighted with "Play Today" button

### Rationale:
✅ **Pros:**
- Kelly remains fully visible
- Familiar calendar pattern
- Expandable = non-intrusive when closed
- Right rail matches existing right-rail pattern
- Can show progress badges on days

❌ **Cons:**
- Limited space for 365 days
- Requires scrolling for year view
- May feel cramped in month view

### Best For:
- Users who want traditional calendar feel
- Quick access to today's lesson
- Progress tracking at a glance

---

## Framework 2: Left-Rail Navigation + Right-Rail Details

### Layout:
```
┌─────────────────────────────────────────────────────┐
│  [Left Rail: Calendar Nav]  │  Kelly (Center)  │  [Right Rail: Lesson Details] │
│  ┌────────────────────┐     │                  │  ┌──────────────────────┐   │
│  │ 📅 Calendar        │     │   👤 Kelly      │  │ Selected: Day 189    │   │
│  │                    │     │   (3D Avatar)   │  │ "Biochemistry"       │   │
│  │ [Year] [Month]     │     │                  │  │                      │   │
│  │ [Week] [Today]     │     │                  │  │ 🧬 DNA Lesson        │   │
│  │                    │     │                  │  │                      │   │
│  │ Year View:         │     │                  │  │ Learning Objective: │   │
│  │ ┌──┐ ┌──┐ ┌──┐    │     │                  │  │ [Full text...]      │   │
│  │ │J │ │F │ │M │    │     │                  │  │                      │   │
│  │ │31│ │28│ │31│    │     │                  │  │ [Play Lesson] ▶️    │   │
│  │ └──┘ └──┘ └──┘    │     │                  │  │                      │   │
│  │ ... (scroll)       │     │                  │  │ Progress: ▓▓▓░░ 60% │   │
│  └────────────────────┘     │                  │  └──────────────────────┘   │
│  [Controls: Bottom Third]    │                  │                              │
└─────────────────────────────────────────────────────┘
```

### Design Details:
- **Left rail:** Calendar navigation (240-280px)
  - View selector tabs (Year/Month/Week/Today)
  - Compact calendar display
  - Scrollable month cards or grid
- **Right rail:** Selected lesson details (280-320px)
  - Shows full lesson info when day clicked
  - DNA badge, progress, play button
  - Learning objective preview

### Rationale:
✅ **Pros:**
- Separates navigation from details
- More space for lesson information
- Clear visual hierarchy
- Can show more context per lesson
- Two-panel approach = more information density

❌ **Cons:**
- Takes more screen space
- Kelly has less "breathing room"
- More complex navigation

### Best For:
- Users who want detailed lesson information
- Educational focus (showing learning objectives)
- Progress tracking and planning

---

## Framework 3: Floating Overlay Calendar

### Layout:
```
┌─────────────────────────────────────────────────────┐
│  [Left Rail: Settings]  │  Kelly (Center)  │  [Right Rail: Search] │
│                        │                  │                      │
│                        │   👤 Kelly      │                      │
│                        │   (3D Avatar)   │                      │
│                        │                  │                      │
│                        │                  │                      │
│                        │                  │                      │
│                        │                  │                      │
│  [Controls: Bottom]    │                  │                      │
│                        │                  │                      │
│                        │  ┌──────────────┐│                      │
│                        │  │ 📅 Calendar  ││                      │
│                        │  │ (Overlay)    ││                      │
│                        │  │              ││                      │
│                        │  │ [Year View]  ││                      │
│                        │  │              ││                      │
│                        │  └──────────────┘│                      │
└─────────────────────────────────────────────────────┘
```

### Design Details:
- **Trigger:** Calendar button in bottom controls or right rail
- **Overlay:** Semi-transparent modal (60-70% opacity)
- **Position:** Centered but offset to not block Kelly's face
- **Size:** 800-1000px wide, 600-700px tall
- **Backdrop:** Darkened background (Kelly dimmed but visible)
- **Views:** Tabs for Year/Month/Week/Today

### Rationale:
✅ **Pros:**
- Doesn't take permanent screen space
- Can show full calendar details
- Kelly remains visible (dimmed)
- Focused experience when calendar open
- Can be dismissed easily

❌ **Cons:**
- Blocks interaction with Kelly when open
- Requires modal management
- Less "always available" feeling
- May feel disconnected from main UI

### Best For:
- Users who want full-featured calendar
- Occasional calendar browsing
- Detailed planning sessions
- When screen space is premium

---

## Framework 4: Bottom Drawer Calendar

### Layout:
```
┌─────────────────────────────────────────────────────┐
│  [Left Rail]  │  Kelly (Center)  │  [Right Rail] │
│               │                  │                │
│               │   👤 Kelly      │                │
│               │   (3D Avatar)   │                │
│               │                  │                │
│               │                  │                │
│               │                  │                │
│               │                  │                │
│               │                  │                │
│ ────────────────────────────────────────────────── │
│ ┌─────────────────────────────────────────────────┐ │
│ │ 📅 Calendar Drawer (Slides up from bottom)     │ │
│ │                                                 │ │
│ │ [Year] [Month] [Week] [Today]                  │ │
│ │                                                 │ │
│ │ [Year View: 12 Month Cards]                    │ │
│ │                                                 │ │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Design Details:
- **Trigger:** Calendar button in bottom controls
- **Drawer:** Slides up from bottom (like mobile apps)
- **Height:** 40-60% of screen (Kelly's upper body still visible)
- **Views:** Tabs at top, content scrollable
- **Dismiss:** Swipe down or close button

### Rationale:
✅ **Pros:**
- Kelly's face remains visible
- Familiar mobile pattern
- Doesn't block side rails
- Easy to dismiss
- Good use of vertical space

❌ **Cons:**
- Covers bottom controls when open
- Less horizontal space for calendar
- May feel "mobile-first" (not desktop-optimized)
- Kelly's lower body hidden

### Best For:
- Mobile/touch interfaces
- Quick calendar access
- Users comfortable with drawer patterns
- When horizontal space is limited

---

## Framework 5: Split-Screen Calendar Mode

### Layout:
```
┌─────────────────────────────────────────────────────┐
│  [Calendar Mode Toggle]                             │
│ ┌──────────────────────┬─────────────────────────┐ │
│ │                      │                         │ │
│ │   Calendar View      │    Kelly (Smaller)      │ │
│ │   (60% width)        │    (40% width)          │ │
│ │                      │                         │ │
│ │  [Year/Month/Week]   │    👤 Kelly            │ │
│ │                      │    (3D Avatar)          │ │
│ │  Month Grid:         │                         │ │
│ │  ┌─┬─┬─┬─┬─┬─┬─┐     │    [Mini Controls]     │ │
│ │  │S│M│T│W│T│F│S│     │                         │ │
│ │  ├─┼─┼─┼─┼─┼─┼─┤     │                         │ │
│ │  │ │ │1│2│3│4│5│     │                         │ │
│ │  │6│7│8│9│10│11│12│   │                         │ │
│ │  ...                 │                         │ │
│ │                      │                         │ │
│ └──────────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Design Details:
- **Mode Toggle:** Button to switch between "Lesson Mode" and "Calendar Mode"
- **Calendar Mode:** 60% left (calendar), 40% right (Kelly)
- **Lesson Mode:** Normal layout (Kelly center, calendar in rail)
- **Views:** Full calendar views in calendar mode
- **Interaction:** Click day → switches to lesson mode + plays

### Rationale:
✅ **Pros:**
- Maximum calendar space when needed
- Kelly still visible (just smaller)
- Best of both worlds (dedicated modes)
- No compromises on calendar features
- Clear mode separation

❌ **Cons:**
- Requires mode switching
- Kelly smaller in calendar mode
- May feel like "two apps"
- Less seamless experience

### Best For:
- Users who do heavy calendar planning
- When calendar browsing is primary activity
- Desktop/large screen users
- Power users who want full features

---

## Recommendation: Framework 2 (Left-Rail Nav + Right-Rail Details)

### Why This Framework?

1. **Kelly-Centric:** Kelly remains fully visible and prominent
2. **Information Rich:** Can show lesson details without blocking Kelly
3. **Familiar Pattern:** Matches existing side-rail UI pattern
4. **Scalable:** Works for all 365 days
5. **Progressive Disclosure:** Navigation on left, details on right
6. **Non-Intrusive:** Can collapse/expand rails as needed
7. **Educational Focus:** Right rail perfect for learning objectives

### Implementation Details:

**Left Rail (240-280px):**
- Calendar icon/button at top
- View selector: [Year] [Month] [Week] [Today] tabs
- Compact calendar display:
  - **Year:** 12 month cards (2 columns, scrollable)
  - **Month:** Traditional 7-day grid (compact)
  - **Week:** 7-day vertical list
  - **Today:** Large "Today's Lesson" card with play button
- Progress indicators on days (✓ completed, ● in progress, ○ upcoming)
- DNA lesson badges (🧬) on days

**Right Rail (280-320px):**
- Shows when day is selected
- Lesson title and date
- DNA badge if applicable
- Learning objective (full text, scrollable)
- Age variants indicator (if DNA lesson)
- Languages available (EN/ES/FR)
- [Play Lesson] button (large, prominent)
- Progress bar if lesson started
- Related lessons (optional)

**Kelly (Center):**
- Always fully visible
- Lipsyncing when lesson plays
- Subtle animations when calendar day selected
- No blocking from calendar UI

**Bottom Controls (Middle Third):**
- Play/pause controls
- Progress bar
- Speed controls
- Language selector
- Calendar toggle (opens/closes calendar)

### Visual Hierarchy:
1. **Kelly** (primary focus - always visible)
2. **Selected Lesson** (right rail - when day clicked)
3. **Calendar Navigation** (left rail - always available)
4. **Controls** (bottom - lesson playback)

---

## Alternative: Hybrid Approach

If Framework 2 feels too complex, use **Framework 1 (Right-Rail Expandable)** as primary with **Framework 3 (Overlay)** for detailed views:

- **Default:** Right-rail collapsed calendar icon
- **Click icon:** Expands right rail with calendar
- **Click day:** Opens overlay with full lesson details
- **Best of both:** Simple default, detailed when needed

---

## Next Steps

1. ✅ Confirm framework selection
2. ⏳ Create detailed wireframes
3. ⏳ Design calendar component specifications
4. ⏳ Implement navigation system
5. ⏳ Integrate with Kelly's lesson player
6. ⏳ Test with 365-day calendar data

---

**Status:** Ready for review and approval

