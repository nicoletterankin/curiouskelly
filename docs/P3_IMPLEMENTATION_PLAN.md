# P3 Implementation Plan

**Status:** 📋 PLANNING  
**Created:** December 16, 2025  
**Depends on:** ONE_PAGE_KELLY_ARCHITECTURE.md (Phases 0-5 complete)

---

## Overview

P3 tasks extend the One-Page Kelly Architecture with additional content and UX polish.

### Scope

| Category | Tasks |
|----------|-------|
| **Home Scenes** | Compare, Gift, News, Impact, Values |
| **Journey Tab** | Commons (community) |
| **UX Polish** | Touch/swipe, keyboard nav, more redirects |

---

## Part 1: Additional Home Scenes

### Current State
- 3 scenes: Hero, Pricing, About
- Dot navigation working
- Deep links working (?tab=home&section=pricing)

### New Scenes to Add

| Scene # | Name | Source Page | Key Content |
|---------|------|-------------|-------------|
| 4 | Compare | compare-us.html | Honest comparison table vs Khan/Duolingo |
| 5 | Gift | gifts.html | Gift subscription options |
| 6 | News | newsroom.html | Press resources, awards |
| 7 | Impact | impact.html | SDG 4 mission, metrics |
| 8 | Values | diversity.html | Our values, inclusion |

### Implementation

```javascript
// Update HOME_SCENES array
const HOME_SCENES = ['hero', 'pricing', 'about', 'compare', 'gift', 'news', 'impact', 'values'];

// Add scenes to buildHomeOverlayHtml()
// Add deep link mappings to applyDeepLinkNavigation()
```

### Scene Designs (16:9 optimized)

#### Compare Scene
- **Header:** "How We Compare"
- **Content:** 3-column comparison grid (Kelly vs Khan vs Duolingo)
- **Key differentiators:** Daily habit, AI personalization, 16:9 immersive
- **CTA:** "Start Learning"

#### Gift Scene
- **Header:** "Give the Gift of Learning"
- **Content:** Gift card visual, 3 price tiers
- **Holiday theming** (optional seasonal variant)
- **CTA:** "Buy Gift Card"

#### News Scene
- **Header:** "Newsroom"
- **Content:** Latest 3 press items, awards badges
- **Press kit download link
- **CTA:** "Contact Press"

#### Impact Scene
- **Header:** "Our Impact"
- **Content:** SDG 4 badge, key metrics (learners, lessons, streaks)
- **Mission statement
- **CTA:** "Join the Mission"

#### Values Scene
- **Header:** "Our Values"
- **Content:** 4 value cards (Accessible, Transparent, Inclusive, Ethical)
- **Brief descriptions
- **CTA:** "Learn More" (links to about)

---

## Part 2: Journey Commons Tab

### Current State
- Journey tabs: Calendar, Week, Curriculum, Bookmarks
- switchJourneyModeTab() function exists

### New Tab: Commons

| Element | Description |
|---------|-------------|
| **Purpose** | Community features placeholder |
| **Content** | Coming soon message, waitlist signup |
| **Future features** | Discussion forums, study groups, leaderboards |

### Implementation

```javascript
// Add 'commons' to journey tabs in buildJourneyOverlayHtml()
// Create renderJourneyCommons() function
// Add placeholder content with waitlist form
```

---

## Part 3: UX Polish

### Touch/Swipe for Home Scenes

| Feature | Implementation |
|---------|----------------|
| **Swipe left/right** | Touch event handlers on home-scenes-container |
| **Velocity detection** | Snap to nearest scene |
| **Visual feedback** | Scene indicators during swipe |

```javascript
// Add to initHomeMode()
let touchStartX = 0;
container.addEventListener('touchstart', handleTouchStart);
container.addEventListener('touchmove', handleTouchMove);
container.addEventListener('touchend', handleTouchEnd);
```

### Keyboard Navigation for Settings

| Key | Action |
|-----|--------|
| **Tab** | Move between sections |
| **Enter** | Activate section |
| **Escape** | Close settings |
| **↑/↓** | Navigate within section list |

```javascript
// Add to initSettingsMode()
document.addEventListener('keydown', handleSettingsKeyboard);
```

### Additional Redirects

| Page | Redirect Target |
|------|-----------------|
| gifts.html | /learn.html?tab=home&section=gift |
| newsroom.html | /learn.html?tab=home&section=news |
| impact.html | /learn.html?tab=home&section=impact |
| diversity.html | /learn.html?tab=home&section=values |
| compare-us.html | /learn.html?tab=home&section=compare |
| commons.html | /learn.html?tab=journey&subtab=commons |

---

## Execution Order

### Sprint 1: Home Scenes (Est. 2-3 hours)
1. ✅ Read existing page content
2. [ ] Add 5 new scenes to buildHomeOverlayHtml()
3. [ ] Add CSS for new scene layouts
4. [ ] Update dot navigation
5. [ ] Add deep link mappings
6. [ ] Test all scenes

### Sprint 2: Journey Commons (Est. 30 min)
1. [ ] Add Commons tab to Journey
2. [ ] Create placeholder content
3. [ ] Wire up tab switching

### Sprint 3: UX Polish (Est. 1-2 hours)
1. [ ] Implement touch/swipe for Home
2. [ ] Add keyboard navigation to Settings
3. [ ] Add remaining redirects
4. [ ] Test on mobile

### Sprint 4: Redirects & Cleanup (Est. 30 min)
1. [ ] Add kelly-redirect.js to remaining pages
2. [ ] Test all redirects
3. [ ] Commit and deploy

---

## Dependencies

| Dependency | Status |
|------------|--------|
| ONE_PAGE_KELLY_ARCHITECTURE | ✅ Complete |
| kelly-redirect.js | ✅ Created |
| learn.html mode system | ✅ Working |
| Curriculum browser | ✅ Working |

---

## Success Metrics

- [ ] All 8 Home scenes render correctly
- [ ] Swipe navigation works on mobile
- [ ] Keyboard navigation works in Settings
- [ ] All P3 pages redirect properly
- [ ] No console errors
- [ ] Page load time unchanged

---

## Related Files

- `public/learn.html` — Main implementation
- `public/js/kelly-redirect.js` — Redirect system
- `docs/ONE_PAGE_KELLY_ARCHITECTURE.md` — Master architecture

---

*Ready for implementation.*
