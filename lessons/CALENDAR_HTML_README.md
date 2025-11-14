# Calendar Interface HTML - Usage Guide

## File Created
**`calendar-interface.html`** - Full-featured calendar interface implementing Framework #2

## Features Implemented

### ✅ Layout (Framework #2)
- **Left Rail (240px):** Calendar navigation with 4 views
- **Center:** Kelly placeholder (for Unity 3D avatar)
- **Right Rail (280px):** Selected lesson details
- **Bottom Controls:** Playback controls and navigation

### ✅ Four Views
1. **Today's Lesson** - Large card with current day's lesson
2. **Year View** - 12 month cards with stats
3. **Month View** - Traditional 7-day calendar grid
4. **Week View** - 7-day vertical list

### ✅ Real Data Integration
- Loads from `365_day_calendar.json`
- Shows all 365 lessons
- Displays DNA lesson badges (🧬)
- Shows learning objectives
- Progress indicators (ready for user data)

### ✅ Interactive Features
- Click day → Shows lesson in right rail
- Click [Play] → Triggers Kelly to play lesson
- Navigation: Previous/Next day, month, week
- Today button: Jump to today's lesson
- View switching: Toggle between 4 views

## How to Use

### 1. Open in Browser
```bash
# Make sure 365_day_calendar.json is in same directory
# Open calendar-interface.html in browser
```

### 2. Navigation
- **View Selector:** Click buttons in left rail to switch views
- **Month Navigation:** Use ◀ ▶ buttons in month view
- **Week Navigation:** Use ◀ ▶ buttons in week view
- **Day Selection:** Click any day to see lesson details
- **Today Button:** Bottom controls → "Today" button

### 3. Play Lesson
- Click [Play Lesson] button in right rail or today card
- Currently shows alert (replace with Unity integration)
- Kelly placeholder shows where 3D avatar will render

## Integration with Unity

### Kelly Avatar Integration
The center area has a placeholder:
```html
<div class="kelly-placeholder">
    <!-- Kelly 3D Avatar will be rendered here in Unity -->
</div>
```

**To integrate:**
1. Replace placeholder with Unity WebGL canvas
2. Connect `playLesson(day)` function to Unity
3. Pass lesson data to Unity for Kelly to play

### Lesson Playback
When user clicks [Play Lesson]:
```javascript
function playLesson(day) {
    // This triggers Kelly to play
    // Send lesson data to Unity
    // Unity renders Kelly lipsyncing the lesson
}
```

## Data Structure

The HTML expects `365_day_calendar.json` with:
- `lessons` array
- Each lesson has: `day`, `date`, `title`, `learning_objective`, `has_dna`, etc.

## Styling

- **Colors:** Purple gradient background (#667eea → #764ba2)
- **Kelly Area:** Centered, always visible
- **Side Rails:** Semi-transparent white with blur
- **Responsive:** Adapts to screen size

## Keyboard Shortcuts (Ready to Add)

- `T` = Today's lesson
- `Y` = Year view
- `M` = Month view
- `W` = Week view
- `→` = Next day
- `←` = Previous day
- `Space` = Play/Pause

## Next Steps

1. ✅ HTML interface complete
2. ⏳ Integrate with Unity (Kelly 3D avatar)
3. ⏳ Connect to lesson player
4. ⏳ Add progress tracking
5. ⏳ Add user preferences
6. ⏳ Add keyboard shortcuts

---

**Status:** ✅ Ready for Unity integration
**Framework:** #2 (Left-Rail Nav + Right-Rail Details)

