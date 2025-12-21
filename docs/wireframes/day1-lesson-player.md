## Day 1 Lesson Player Wireframes (Template)

This document freezes the target layout for the Day 1 “Starting Fresh” lesson player. Every other day must follow this scaffolding so we can swap Kelly media, infographics, and choice data without re‑designing the chrome.

---

### 1. Desktop View (1920×1080 reference)

```
┌─────────────────────────────── 100% viewport ────────────────────────────────┐
│ ┌──────── Left Padding (brand texture) ───────┐ ┌── Right Rails ───────────┐ │
│ │                                            │ │ Phase dots                │ │
│ │  ┌────────────── 9:16 Hero (Kelly Stage) ────────────────┐               │ │
│ │  │ ┌── Safe video area (16:9 content letterboxed) ────┐ │ │  Option rail │ │
│ │  │ │                                                │ │ │  • Card A     │ │
│ │  │ │  Kelly video w/ branded wings (object-fit:contain)│ │  • Card B     │ │
│ │  │ │                                                │ │ │  • Card C     │ │
│ │  │ └────────────────────────────────────────────────┘ │ │               │ │
│ │  │ ┌── Infographic panel (slides from left, 40%) ──┐  │ │               │ │
│ │  │ │  glass card + 16:9 graphic, respecting safe   │  │ │               │ │
│ │  │ │  zones. Kelly stays visible on right.         │  │ │               │ │
│ │  │ └───────────────────────────────────────────────┘  │ │               │ │
│ │  │  Caption rail (safe-zone aware)                    │ │               │ │
│ │  │  Bottom chat strip (max 25vh, right-aligned input) │ │               │ │
│ │  └────────────────────────────────────────────────────┘ │               │ │
│ │                                            │ │ Settings column (age/lang)│ │
│ └────────────────────────────────────────────┘ └──────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key rules**
- Kelly video never stretches; we letterbox it inside the 9:16 hero and fill the leftover area with a branded gradient.
- Infographic panel lives *inside* the hero. It’s hidden by default, opens via the “Lesson Visual” pill, and returns focus to Kelly when dismissed.
- Only one set of choices exists (right rail). Legacy inline buttons stay hidden for backward compatibility but receive no input focus.
- Chat rail is limited to 25 vh and docked to the right so speech bubbles + captions never collide.

---

### 2. Mobile View (430×932 reference)

```
┌───────────── Phone viewport ─────────────┐
│ Livestream badge                         │
│ ┌───── 9:16 Hero fills width ──────────┐ │
│ │ Kelly video letterboxed               │ │
│ │ Infographic panel transforms into     │ │
│ │ full-bleed sheet w/ Kelly PiP block   │ │
│ └──────────────────────────────────────┘ │
│ Caption rail (full width)                │
│ Option sheet (bottom drawer, expands)    │
│ Chat bubble (Floating FAB → full sheet)  │
└─────────────────────────────────────────┘
```

**Key rules**
- Phase navigation collapses into a pill that snaps to the right edge for thumb reach.
- Choice cards become a draggable sheet that covers the bottom 35% of the screen; cards stack vertically with larger tap targets.
- Infographic overlay takes over the hero but keeps Kelly visible via picture‑in‑picture so the learner never loses presence.

---

### 3. Interaction & Safe-Zone Requirements

| Element            | Positioning rule                                                                 | Notes                                               |
|--------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------|
| Kelly video        | `.kelly-video-stage` enforces 9:16 frame; `<video>` uses `object-fit: contain`.   | Safe-zone JSON (`public/kelly/videos/001/*`) guides overlay offsets. |
| Infographic panel  | Internal div (`.infographic-panel`) anchored to left. Slides in with CSS transform and traps focus. | Image is limited to 90% of panel width to maintain breathing room. |
| Caption rail       | Absolutely positioned using safe-zone `bottom` margin; auto-adjusts per phase.   | Prevents covering Kelly’s hands/props.              |
| Chat rail          | `max-height: 25vh` + `pointer-events: auto` only on inner elements; composer hugs right edge. | Always secondary to lesson controls.               |
| Choice rail        | Single source of truth. JS only attaches listeners to `#option-a|b|c`.            | On mobile, rail morphs into sheet but uses same markup. |

---

### 4. State Diagram (Phase vs. UI)

```
Phase change ──► loadLessonPhase()
                 ├─> Kelly video (HD) or fallback pose
                 ├─> CaptionSystem.render(text, safeZone)
                 ├─> OptionsSystem.setOptions(...)
                 └─> Infographic pill updates data-visual-url

Infographic pill click
  ├─ preload image
  ├─ open panel (hero overlay)
  └─ pause media (video + audio)

Panel close
  ├─ resume media
  └─ return focus to hero
```

---

### 5. Deliverables Bound to This Spec

1. CSS/HTML refactor in `public/learn.html` driven by these coordinates.
2. JS updates so `OptionsSystem`, caption placement, chat rail, and infographic overlay all respect the same layers.
3. Safe-zone manifest (JSON) referenced by the layout code for future days.

This document is the contract for the implementation that follows. Any deviations should be documented here first so every day after Day 1 inherits the exact same structure.


















