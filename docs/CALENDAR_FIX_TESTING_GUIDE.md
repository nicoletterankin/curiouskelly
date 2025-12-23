# Calendar Fix - Local Testing Guide
**Date:** December 23, 2025  
**Server:** http://localhost:3000  
**Test Page:** http://localhost:3000/index.html

---

## Quick Start

1. **Server is running** at http://localhost:3000
2. **Open** http://localhost:3000/index.html in your browser
3. **Scroll down** to the "App Preview - Calendar Grid" section
4. **Test** the calendar and panel interaction

---

## What to Test

### ✅ Calendar Layout
1. **Desktop View:**
   - Calendar shows 12 columns (months)
   - Day dots are visible and evenly spaced
   - Hover over a day dot → It scales up
   - Colors show completeness (green/blue/yellow/gray)

2. **Panel Interaction:**
   - **Click any day dot** → Right-side panel slides in
   - **Calendar should shrink** (moves left, makes room for panel)
   - **Panel shows** learner view with lesson details
   - **Click overlay or X** → Panel closes, calendar expands back

3. **Mobile View** (resize browser to <768px):
   - Calendar shows fewer columns (6 or 4)
   - **Click day dot** → Panel overlays (full width)
   - Calendar stays in place (no layout shift)

---

## Expected Behavior

### Desktop (>1024px)
```
Before click:
┌─────────────────────────────────────┐
│  Calendar (full width, 900px)      │
└─────────────────────────────────────┘

After click:
┌──────────────────────────┬──────────┐
│  Calendar (shrunk)      │  Panel   │
│  (800px)                │  (500px) │
└──────────────────────────┴──────────┘
```

### Mobile (<768px)
```
Before click:
┌─────────────────────┐
│  Calendar (full)    │
└─────────────────────┘

After click:
┌─────────────────────┐
│  Panel (overlay)    │
│  (covers calendar)  │
└─────────────────────┘
```

---

## Things to Check

### ✅ Visual
- [ ] Calendar doesn't disappear when panel opens
- [ ] Calendar shrinks smoothly (no jump)
- [ ] Panel slides in smoothly
- [ ] Transitions are smooth (0.3s ease-out)
- [ ] Day dots remain visible and clickable

### ✅ Functional
- [ ] Single click opens panel
- [ ] Double click shows preview popup
- [ ] Panel closes when clicking overlay
- [ ] Panel closes when clicking X button
- [ ] Calendar returns to full width when panel closes

### ✅ Responsive
- [ ] Desktop: Calendar shrinks, panel side-by-side
- [ ] Tablet: Appropriate layout
- [ ] Mobile: Panel overlays, no layout shift

### ✅ Browser Console
- [ ] No JavaScript errors
- [ ] No CSS warnings
- [ ] Panel loads lesson data correctly

---

## Known Limitations

1. **`:has()` selector** - May not work in Safari <15.4
   - **Fallback:** JavaScript adds `audit-panel-open` class
   - Should work in all modern browsers

2. **Panel width** - Fixed at 500px
   - May be too narrow for some content
   - Can adjust in `lesson-audit-panel.js`

---

## Debugging Tips

### If calendar doesn't adjust:
1. Open browser DevTools (F12)
2. Check if `.audit-panel.open` class exists
3. Check if `body.audit-panel-open` class is added
4. Inspect `.app-preview` element for `margin-right: 500px`

### If panel doesn't open:
1. Check browser console for errors
2. Verify `LessonAuditPanel` is loaded
3. Check network tab for missing JS files

### If layout breaks:
1. Check viewport width
2. Verify CSS media queries are working
3. Test in different browser sizes

---

## Test Checklist

- [ ] **Desktop (>1024px):** Calendar shrinks when panel opens ✅
- [ ] **Desktop:** Calendar expands when panel closes ✅
- [ ] **Mobile (<768px):** Panel overlays calendar ✅
- [ ] **All sizes:** Transitions are smooth ✅
- [ ] **All sizes:** No layout jumps ✅
- [ ] **All sizes:** Day dots remain clickable ✅

---

**Test URL:** http://localhost:3000/index.html  
**Focus:** Scroll to calendar section, test panel interaction

