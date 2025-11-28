# 🎯 Kelly UX Redesign — Master Index

## Everything You Need to Transform the Experience

**Created:** November 28, 2025  
**Status:** Specifications & Mockups Complete — Ready for Implementation

---

## 🚀 Quick Start: View the Mockups

**Open in browser:** `mockups/index.html`

| Mockup          | Description                              | File                              |
| --------------- | ---------------------------------------- | --------------------------------- |
| **Kelly Frame** | Core lesson experience with all controls | `mockups/kelly-frame-mockup.html` |
| **Kelly Hub**   | Today's lesson + calendar + stats        | `mockups/kelly-hub-mockup.html`   |

---

## Document Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    KELLY REDESIGN DOCUMENTS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   SPECIFICATIONS                                                │
│   ──────────────                                                │
│   1. MOBILE_UX_AUDIT_REPORT.md                                 │
│      └── What's broken today (12 issues catalogued)            │
│                                                                 │
│   2. UNIFIED_CALENDAR_STRATEGY.md                              │
│      └── ONE view to rule navigation (Kelly Today Hub)         │
│                                                                 │
│   3. KELLY_EXPERIENCE_COMPLETE_SPEC.md  ← THE MAIN DOC        │
│      └── Complete TikTok-style learning experience             │
│          • Kelly Frame (2D/3D)                                 │
│          • Bottom navigation                                    │
│          • Side variant controls                               │
│          • Phase progression                                   │
│          • Hub integration                                     │
│          • 5-week implementation plan                          │
│                                                                 │
│   4. KELLY_REDESIGN_INDEX.md (this file)                       │
│      └── Master index tying everything together                │
│                                                                 │
│   DATA & ARCHITECTURE                                           │
│   ───────────────────                                          │
│   5. architecture/DATA_ARCHITECTURE.md                         │
│      └── Complete data schemas                                 │
│          • Supabase tables (SQL)                               │
│          • TypeScript interfaces                               │
│          • JSON schemas for lesson DNA                         │
│          • API contracts                                       │
│          • State management                                    │
│          • Difficulty feature (2 or 3 choices)                 │
│                                                                 │
│   MOCKUPS (Interactive HTML)                                   │
│   ──────────────────────────                                   │
│   6. mockups/index.html           ← START HERE                 │
│      └── Index page linking to all mockups                     │
│                                                                 │
│   7. mockups/kelly-frame-mockup.html                           │
│      └── Core lesson experience mockup                         │
│          • Phase indicator                                     │
│          • Side controls (age, language, tone, difficulty)     │
│          • Speech bubble with choices (2 or 3)                 │
│          • Bottom navigation                                   │
│          • Variant selector modals                             │
│          • 2D/3D toggle                                        │
│                                                                 │
│   8. mockups/kelly-hub-mockup.html                             │
│      └── Today Hub mockup                                      │
│          • Today's lesson hero card                            │
│          • Stats row (streak, progress)                        │
│          • Calendar with month tabs                            │
│          • Birthday feature                                    │
│          • Lesson preview modal                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Big Picture

### Before (Current State)

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  app.html   │  │curriculum   │  │ calendar    │  │  player     │
│  +sidebar   │  │   .html     │  │   .html     │  │   .html     │
│  (broken    │  │ (marketing  │  │ (completely │  │ (simple     │
│   mobile)   │  │  only)      │  │  broken)    │  │  player)    │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                              │
                    😵 User Confusion
                    "Where do I go?"
```

### After (New State)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    THE KELLY EXPERIENCE                         │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                         │  │
│   │                    KELLY FRAME                          │  │
│   │                 (2D or 3D Kelly)                        │  │
│   │                                                         │  │
│   │  ┌─────┐                                      ┌─────┐   │  │
│   │  │Phase│                                      │ Age │   │  │
│   │  │ ●○○ │                                      │Lang │   │  │
│   │  └─────┘                                      │Tone │   │  │
│   │                                               │2D/3D│   │  │
│   │              [Kelly Teaching]                 └─────┘   │  │
│   │                                                         │  │
│   │          ┌─────────────────────────────┐               │  │
│   │          │     Speech + Choices        │               │  │
│   │          └─────────────────────────────┘               │  │
│   │                                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  🏠 Home  │  📅 Cal  │  🎓 Learn  │  👤 Me  │  ⚙️ Set  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                    😊 Clear Experience
                    "Kelly teaches, I learn"
```

---

## Key Decisions Made

| Decision           | Choice               | Rationale                            |
| ------------------ | -------------------- | ------------------------------------ |
| Primary navigation | Bottom bar (5 icons) | TikTok-proven, thumb-friendly        |
| Variant controls   | Right side icons     | TikTok-proven, doesn't block content |
| Calendar approach  | ONE unified Hub      | Eliminates 4 competing views         |
| Default avatar     | 2D (user can toggle) | Fast load, works everywhere          |
| Phase UI           | Top pill indicator   | Subtle, doesn't obstruct Kelly       |
| Content layout     | Kelly = fullscreen   | She IS the experience                |
| Sidebar            | DELETED              | Replaced by Hub overlay              |

---

## Files Affected Summary

### Delete These Files

- `public/calendar.html` — Broken, replaced by Hub
- `public/css/calendar.css` — Not needed
- `public/js/calendar-page.js` — Not needed

### Redirect These URLs

- `/curriculum.html` → `/` (Hub in marketing mode)
- `/calendar.html` → `/app.html` (Calendar tab)

### Heavily Modify

- `public/app.html` — Strip sidebar, add new component system

### Create New

- `public/css/kelly-experience.css`
- `public/js/kelly-frame.js`
- `public/js/bottom-nav.js`
- `public/js/side-controls.js`
- `public/js/phase-controller.js`
- `public/js/hub-controller.js`
- `public/js/calendar-grid.js`

---

## Implementation Timeline

```
Week 1: Kelly Frame Foundation
├── Build <kelly-frame> with 2D
├── Build <bottom-nav>
├── Build <phase-indicator>
├── Build <speech-bubble>
└── Deliverable: Basic lesson playable

Week 2: Variant System
├── Build <side-controls>
├── Build <variant-selector>
├── Age/Language/Tone switching
└── Deliverable: Full variant support

Week 3: 3D Integration
├── Build <mode-toggle>
├── Integrate Unity canvas
├── 2D↔3D transitions
└── Deliverable: Dual mode working

Week 4: Hub & Calendar
├── Build <kelly-hub>
├── Build <calendar-grid>
├── Birthday feature
├── Lesson preview
└── Deliverable: Complete Hub

Week 5: Polish & Migration
├── Delete old files
├── Set up redirects
├── Performance optimization
├── Mobile testing
└── Deliverable: Production ready
```

---

## Quick Reference: The Core Experience Loop

```
1. USER OPENS APP
         │
         ▼
2. KELLY TODAY HUB SHOWS
   "Today: Citizenship"
   [Start Lesson] button
         │
         ▼
3. KELLY FRAME APPEARS
   - Kelly (2D/3D) fullscreen
   - Phase indicator top
   - Side controls right
   - Bottom nav below
         │
         ▼
4. LESSON PLAYS
   Welcome → Q1 → Q2 → Q3 → Wisdom
   (2 choices per Q phase)
         │
         ▼
5. LESSON COMPLETE
   - Celebration animation
   - Streak increments
   - Return to Hub
         │
         ▼
6. HUB SHOWS COMPLETION
   "Citizenship ✓"
   Tomorrow: [Preview]
```

---

## Success Looks Like

| What Users Say Now               | What They'll Say After                              |
| -------------------------------- | --------------------------------------------------- |
| "Where do I click?"              | "I just open and Kelly's ready"                     |
| "This sidebar covers everything" | "The controls are like TikTok"                      |
| "Which calendar should I use?"   | "I love finding my birthday lesson"                 |
| "It's broken on my phone"        | "It works perfectly on everything"                  |
| "What's today's lesson?"         | "Citizenship was today, we all learned it together" |

---

## Next Steps

1. **Review** all three specification documents
2. **Design in Figma** — Mobile-first mockups of Kelly Frame + Hub
3. **Approve** the visual design
4. **Implement** following the 5-week plan
5. **Test** on real devices
6. **Ship** 🚀

---

_All specifications are complete. The path forward is clear. Kelly is ready to teach the world._
