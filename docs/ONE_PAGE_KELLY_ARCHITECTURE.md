# 🏠 One-Page Kelly Architecture

## The Vision

> "You open curiouskelly.com. You see Kelly. You're home. You explore. You learn. You adjust settings. You never leave. It just works. Forever."

**Status:** 🔒 LOCKED DIRECTION  
**Created:** December 16, 2025  
**Authority:** This document defines the architectural future of Curious Kelly.

---

## The Four Buckets

Everything lives in `learn.html`. Four tabs. One experience. 16:9 forever.

```
┌─────────────────────────────────────────────────────────────────┐
│                        learn.html                               │
│  ┌──────────┬──────────┬──────────┬──────────┐                 │
│  │   HOME   │ JOURNEY  │   LEARN  │ SETTINGS │                 │
│  └──────────┴──────────┴──────────┴──────────┘                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Kelly   │                                     │   Kelly    ││
│  │  (left)  │         16:9 Content Zone           │   (right)  ││
│  │          │                                     │            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Tab Definitions

### 🏠 HOME (Marketing/Welcome)
**Purpose:** First impression, value proposition, conversion  
**Sub-sections:**
- Hero / Welcome
- Plans (Pricing)
- Our Story (About)
- Compare
- Gift
- News
- Impact
- Values

**Design:** Horizontal scenes or vertical scroll within 16:9 frame.

---

### 🗺️ JOURNEY (Progress/Curriculum)
**Purpose:** Where you've been, where you're going  
**Sub-sections:**
- Calendar (daily view)
- Curriculum Browser (all 365 lessons)
- Weekly Missions
- Commons (community)
- Achievements/Streaks

**Design:** Card grids, Netflix-style horizontal scroll, progress visualization.

---

### 📚 LEARN (Active Lesson)
**Purpose:** The core learning experience  
**Sub-sections:**
- Lesson Player (video/audio/text)
- Phase Navigation
- Insights Bar
- Knowledge Nuggets
- Depth Card

**Design:** Already 16:9. Kelly speaks. Learner engages.

---

### ⚙️ SETTINGS (Account/Help/Legal)
**Purpose:** Everything else  
**Sub-sections:**
- Account (profile, preferences)
- Subscription (billing, plans)
- Help Center (FAQ, guides)
- Contact Us
- Legal
  - Terms of Service
  - Privacy Policy
  - Trust & Safety
  - Accessibility

**Design:** Master-detail layout. Sidebar nav left, content right. Long text scrolls inside styled panel.

---

## Page Migration Map

### Pages → HOME Tab
| Current File | Sub-section | Priority |
|--------------|-------------|----------|
| index.html | Hero | P1 |
| pricing.html | Plans | P1 |
| about.html | Our Story | P2 |
| compare-us.html | Compare | P2 |
| gifts.html | Gift | P2 |
| newsroom.html | News | P3 |
| impact.html | Impact | P3 |
| diversity.html | Values | P3 |

### Pages → JOURNEY Tab
| Current File | Sub-section | Priority |
|--------------|-------------|----------|
| curriculum.html | Curriculum Browser | P1 |
| missions.html | Weekly Missions | P2 |
| calendar.html | Calendar | P1 (already redirect) |
| commons.html | Commons | P3 |

### Pages → LEARN Tab
| Current File | Sub-section | Priority |
|--------------|-------------|----------|
| (lessons) | Lesson Player | ✅ Done |

### Pages → SETTINGS Tab
| Current File | Sub-section | Priority |
|--------------|-------------|----------|
| help.html | Help Center | P1 |
| contact.html | Contact Us | P1 |
| terms.html | Legal → Terms | P2 |
| privacy.html | Legal → Privacy | P2 |
| trust.html | Legal → Trust | P2 |
| accessibility.html | Accessibility | P2 |

### Stays Separate (B2B/Partner)
| File | Reason |
|------|--------|
| enterprise.html | Sales funnel, different audience |
| affiliates.html | Partner portal |
| ambassador.html | Partner program |
| careers.html | Recruiting |

---

## 16:9 Design Principles

### The Frame is Sacred
- Kelly sidebars are ALWAYS present (static or animated)
- Content zone is ALWAYS 16:9 aspect ratio
- Browser chrome is the only exception

### Content Adapts to Frame
- Content is DESIGNED for 16:9, not cropped into it
- Long content uses styled inner scroll (not browser scrollbar)
- OR content is paginated (scenes, chapters, cards)

### Scroll Behavior
```
WRONG: Browser scrolls, Kelly disappears
RIGHT: Content scrolls inside frame, Kelly stays fixed
```

### Mobile Adaptation
- Kelly sidebars collapse to accent bars (top/bottom)
- Content zone becomes full-width
- 16:9 ratio relaxes to screen ratio
- Core experience preserved

---

## URL Strategy

### Deep Links
```
curiouskelly.com                    → learn.html?tab=home
curiouskelly.com/pricing            → learn.html?tab=home&section=plans
curiouskelly.com/curriculum         → learn.html?tab=journey&section=curriculum
curiouskelly.com/day/17             → learn.html?tab=learn&day=17
curiouskelly.com/settings           → learn.html?tab=settings
curiouskelly.com/help               → learn.html?tab=settings&section=help
curiouskelly.com/terms              → learn.html?tab=settings&section=terms
```

### SEO Strategy
- Static HTML shells at original URLs
- Meta tags for crawlers
- JavaScript redirect to unified experience
- OR server-side detection (bot → static, human → app)

---

## Implementation Phases

### Phase 0: Foundation ✅
- [x] learn.html exists
- [x] Tab system exists (Home, Journey, Learn)
- [x] 16:9 lesson player works

### Phase 1: Four-Tab Structure ✅
- [x] Add Settings tab (full-screen mode)
- [x] Add Journey tab (full-screen mode)
- [x] Refine tab navigation (setUiMode system)
- [x] Ensure Kelly sidebars work for all tabs

### Phase 2: Settings Tab Build ✅
- [x] Master-detail layout
- [x] Help Center content
- [x] Contact section
- [x] Legal sections with styled scroll

### Phase 3: Journey Tab Enhancement ✅
- [x] Calendar view (month grid)
- [x] Week view (7-day grid)
- [x] Curriculum browser (all 365, Year 1 & Year 2)
- [x] Bookmarks view

### Phase 4: Home Tab Build
- [ ] Hero scene
- [ ] Pricing scene
- [ ] About scene
- [ ] Scene navigation

### Phase 5: Migration & Cleanup
- [ ] Old pages → redirects
- [ ] SEO verification
- [ ] Delete deprecated files

---

## Success Criteria

- [x] User can navigate entire site without leaving learn.html
- [x] 16:9 frame is consistent across all tabs
- [x] Kelly is always visible (desktop)
- [x] All legal content is accessible and readable
- [x] Deep links work for sharing
- [ ] SEO maintains or improves

---

## Related Documents

- [FUTURE_STATE_INDEX.md](./FUTURE_STATE_INDEX.md) — Strategic vision
- [FUTURE_STATE_VISION.md](./FUTURE_STATE_VISION.md) — Detailed blueprint
- [learn.html](../public/learn.html) — The One Page

---

*This is the way.*
