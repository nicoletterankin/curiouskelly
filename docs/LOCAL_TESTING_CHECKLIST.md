# Local Testing Checklist - Calendar Fixes
**Date:** December 23, 2025  
**Server:** http://localhost:3000

---

## Test Scenarios

### 1. Calendar Display ✅
- [ ] Calendar loads correctly (12 columns on desktop)
- [ ] Day dots are visible and clickable
- [ ] Completeness colors display correctly (green/blue/yellow/gray)
- [ ] Track badges show (Learn/Grow indicators)

### 2. Panel Integration ✅
- [ ] Click a day dot → Panel slides in from right
- [ ] Calendar adjusts (shrinks) when panel opens
- [ ] Panel shows learner view by default
- [ ] Can toggle to educator view
- [ ] Click overlay/close button → Panel closes
- [ ] Calendar returns to full width when panel closes

### 3. Responsive Behavior ✅
- [ ] Desktop (1024px+): Calendar shrinks, panel side-by-side
- [ ] Tablet (768-1024px): Calendar adjusts appropriately
- [ ] Mobile (<768px): Panel overlays (full width)

### 4. Interactions ✅
- [ ] Single click → Opens audit panel
- [ ] Double click → Shows preview popup
- [ ] Hover → Day dot scales up
- [ ] Tooltip shows lesson info

### 5. Visual Verification ✅
- [ ] Transitions are smooth
- [ ] No layout jumps
- [ ] Panel doesn't cover calendar completely
- [ ] Calendar remains usable when panel is open

---

## Known Issues to Watch For

1. **`:has()` selector support** - May not work in older browsers
   - Fallback: `body.audit-panel-open` class should work
   
2. **Panel state** - Ensure class is added/removed correctly
   - Check browser console for errors

3. **Mobile layout** - Panel should overlay, not push content

---

## Browser Testing

Test in:
- [ ] Chrome/Edge (latest)
- [ ] Firefox (latest)
- [ ] Safari (if available)
- [ ] Mobile browser (Chrome mobile)

---

## Performance

- [ ] Page loads quickly
- [ ] Calendar renders without delay
- [ ] Panel opens smoothly (no lag)
- [ ] No console errors

---

**Test URL:** http://localhost:3000/index.html  
**Focus Area:** Calendar section + Audit panel interaction

