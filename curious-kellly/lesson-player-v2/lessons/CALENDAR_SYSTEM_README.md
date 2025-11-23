# 365-Day Learning Calendar System

## Overview

This system unifies all lesson files (curriculum JSONs and DNA lesson files) into a single 365-day calendar with multiple viewing options.

## Files Created

### 1. `365_day_calendar.json`
**Master calendar data file** that merges:
- All 12 month curriculum files (january_curriculum.json through december_curriculum.json)
- All DNA lesson files (the-sun, the-moon, puppies, leaves-change-color, molecular-biology, etc.)
- Metadata for each lesson (day number, date, title, learning objective, DNA status, etc.)

**Statistics:**
- Total lessons: 365
- DNA lessons: 20+ (automatically detected and merged)
- Curriculum lessons: 365

### 2. `generate_unified_calendar.py`
**Python script** that:
- Loads all month curriculum files
- Detects and merges DNA lesson files
- Maps DNA lessons to calendar days using:
  - Explicit day mappings (e.g., "the-sun" → Day 1)
  - Title keyword matching (e.g., "moon" in title → "the-moon" DNA)
- Generates the unified calendar JSON

**Usage:**
```bash
cd lessons
python generate_unified_calendar.py
```

### 3. Calendar Page Components

#### `calendar-page.html`
Main HTML structure with:
- Header with view selector and year navigation
- Four view panels (Year, Month, Week, List)
- Lesson detail modal

#### `calendar-page.css`
Complete styling for:
- Responsive design (mobile-friendly)
- Multiple calendar layouts
- DNA lesson highlighting
- Smooth transitions and hover effects

#### `calendar-page.js`
Interactive calendar application with:
- **Year View**: 12 month cards showing lesson counts and DNA badges
- **Month View**: Traditional calendar grid with lessons per day
- **Week View**: 7-day detailed view with full lesson information
- **List View**: Searchable, filterable list of all lessons
- **Lesson Detail Modal**: Full lesson information on click

## DNA Lesson Mappings

The system automatically detects and merges DNA lessons:

| DNA Lesson | Day | Detection Method |
|------------|-----|------------------|
| the-sun | 1 | Day mapping + title keywords |
| the-moon | 10 | Day mapping + title keywords |
| leaves-change-color | Various | Title keywords ("leaves", "autumn", "fall") |
| puppies | Various | Title keywords ("puppy", "puppies", "dog") |
| molecular-biology | 189 | Day mapping + title keywords |
| the-ocean | Various | Title keywords ("ocean", "marine", "sea") |
| aging-process | 210 | Day mapping + title keywords |

## Features

### Multiple Calendar Views

1. **Year View**
   - 12 month cards
   - Shows lesson count per month
   - Highlights months with DNA lessons
   - Click month to switch to month view

2. **Month View**
   - Traditional calendar grid
   - Shows lesson title for each day
   - DNA lessons marked with 🧬 badge
   - Click day to see lesson details

3. **Week View**
   - 7-day detailed view
   - Full lesson information per day
   - DNA lessons highlighted with blue border
   - Navigate weeks with arrow buttons

4. **List View**
   - All 365 lessons in scrollable list
   - Search by title or objective
   - Filter by DNA status (all/DNA only/curriculum only)
   - Click lesson for full details

### DNA Lesson Features

- **Automatic Detection**: DNA lessons are automatically detected and merged
- **Visual Highlighting**: DNA lessons marked with 🧬 badge and special styling
- **Rich Metadata**: DNA lessons include age variants, languages, universal concepts
- **Filtering**: Filter list view to show only DNA lessons

## Usage

### Viewing the Calendar

1. Open `calendar-page.html` in a web browser
2. Ensure `365_day_calendar.json` is in the same directory
3. Use view selector to switch between Year/Month/Week/List views
4. Click on any lesson to see full details

### Regenerating the Calendar

If you add new lessons or update existing ones:

```bash
cd lessons
python generate_unified_calendar.py
```

This will regenerate `365_day_calendar.json` with the latest data.

## Integration with Existing System

The calendar system integrates with:
- **Curriculum files**: All month_curriculum.json files
- **DNA lesson files**: Located in `curious-kellly/backend/config/lessons/`
- **Right-rail calendar**: The y/y/t format calendar in lesson-player

## Future Enhancements

Potential improvements:
- [ ] Add lesson completion tracking
- [ ] Add progress streaks
- [ ] Export calendar to iCal format
- [ ] Add lesson preview/playback
- [ ] Add category filtering in list view
- [ ] Add date range filtering
- [ ] Add lesson statistics dashboard

## Notes

- The calendar assumes a non-leap year (365 days)
- DNA lessons are automatically detected but can be manually mapped in `generate_unified_calendar.py`
- The system gracefully handles missing files or JSON errors
- All dates are formatted as "Month Day" (e.g., "January 1")

