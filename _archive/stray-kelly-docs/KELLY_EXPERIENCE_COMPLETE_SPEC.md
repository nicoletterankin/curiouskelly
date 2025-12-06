# 🎬 The Complete Kelly Learning Experience

## Full System Specification — TikTok-Style Immersive Learning

**Version:** 1.0  
**Date:** November 28, 2025  
**Status:** Design Specification (Pre-Implementation)

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [The Kelly Frame (Core Experience)](#2-the-kelly-frame-core-experience)
3. [Bottom Navigation Bar](#3-bottom-navigation-bar)
4. [Right Side Controls (Variant Selector)](#4-right-side-controls-variant-selector)
5. [Phase Progression System](#5-phase-progression-system)
6. [2D vs 3D Kelly Toggle](#6-2d-vs-3d-kelly-toggle)
7. [The Kelly Today Hub](#7-the-kelly-today-hub)
8. [Screen-by-Screen Flows](#8-screen-by-screen-flows)
9. [Component Inventory](#9-component-inventory)
10. [Files to Create/Modify/Delete](#10-files-to-createmodifydelete)
11. [Implementation Phases](#11-implementation-phases)

---

## 1. System Architecture Overview

### The Core Principle

> **Kelly is ALWAYS the main content. Everything else is an overlay.**

Like TikTok where the video is fullscreen and controls float on top, Kelly (2D or 3D) fills the screen, and all UI elements are translucent overlays.

### The Three Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3: OVERLAYS (z-index: 100+)                              │
│   - Bottom nav bar                                              │
│   - Right side controls                                         │
│   - Phase indicator                                             │
│   - Speech bubble / choices                                     │
│   - Modals (Hub, Settings, Profile)                             │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 2: KELLY FRAME (z-index: 1)                              │
│   - 2D Kelly image OR 3D Unity canvas                          │
│   - Takes 100vw × 100vh                                         │
│   - Kelly changes expression based on phase                     │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 1: BACKGROUND (z-index: 0)                               │
│   - Subtle gradient or solid color                              │
│   - Only visible if Kelly doesn't fill frame                    │
└─────────────────────────────────────────────────────────────────┘
```

### State Machine

```
                    ┌─────────────┐
                    │   APP OPEN  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  HUB STATE  │ ← Default landing
                    │ (Today Hub) │
                    └──────┬──────┘
                           │ User taps "Start Today's Lesson"
                    ┌──────▼──────┐
                    │LESSON STATE │ ← Kelly teaches
                    │ (5 phases)  │
                    └──────┬──────┘
                           │ Lesson complete
                    ┌──────▼──────┐
                    │ DONE STATE  │ ← Celebration
                    │ (Streak++)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  HUB STATE  │ ← Return to Hub
                    └─────────────┘
```

---

## 2. The Kelly Frame (Core Experience)

### Visual Specification

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                                                                 │
│                                                                 │
│                                                                 │
│                    ┌─────────────────────┐                      │
│                    │                     │                      │
│                    │                     │                      │
│                    │      KELLY          │                      │
│                    │   (2D image or      │                      │
│                    │    3D avatar)       │                      │
│                    │                     │                      │
│                    │   Centered          │                      │
│                    │   Max 80% height    │                      │
│                    │                     │                      │
│                    └─────────────────────┘                      │
│                                                                 │
│                                                                 │
│                                                                 │
│                                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Background: Dark gradient (#0a0a0b → #111113)
Kelly: Centered, responds to phase
```

### 2D Kelly Expressions (Mapped to Phases)

| Phase                   | Expression  | Image File                    |
| ----------------------- | ----------- | ----------------------------- |
| Welcome                 | Curious     | `kelly-chair-curious.png`     |
| Q1, Q2, Q3 (presenting) | Explaining  | `kelly-chair-explaining.png`  |
| Q1, Q2, Q3 (waiting)    | Listening   | `kelly-chair-listening.png`   |
| Wisdom                  | Wisdom      | `kelly-chair-wisdom.png`      |
| Completion              | Celebrating | `kelly-chair-celebrating.png` |

### 3D Kelly Behaviors

| Phase              | Animation         | Lip Sync               |
| ------------------ | ----------------- | ---------------------- |
| Welcome            | Idle → Wave       | Yes, to welcome audio  |
| Q1-Q3 (presenting) | Talking gesture   | Yes, to question audio |
| Q1-Q3 (waiting)    | Head tilt, blink  | No                     |
| Wisdom             | Thoughtful pose   | Yes, to wisdom audio   |
| Completion         | Celebration dance | Yes, to congrats audio |

---

## 3. Bottom Navigation Bar

### Design (TikTok-Style)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                         (Kelly Frame)                           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│      ┌────┐    ┌────┐    ┌────┐    ┌────┐    ┌────┐           │
│      │ 🏠 │    │ 📅 │    │ 🎓 │    │ 👤 │    │ ⚙️ │           │
│      │    │    │    │    │    │    │    │    │    │           │
│      │Home│    │Cal │    │Learn│   │ Me │    │Set │           │
│      └────┘    └────┘    └────┘    └────┘    └────┘           │
│                          ▲                                      │
│                     (active)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Height: 60px (80px with safe area on iPhone)
Background: rgba(0, 0, 0, 0.85) with blur
Border-top: 1px solid rgba(255, 255, 255, 0.1)
```

### Navigation Items

| Icon | Label    | Action                                                         |
| ---- | -------- | -------------------------------------------------------------- |
| 🏠   | Home     | Opens Kelly Today Hub (slide up)                               |
| 📅   | Calendar | Opens 365-day calendar view                                    |
| 🎓   | Learn    | Current lesson (center, larger) — ALWAYS visible during lesson |
| 👤   | Me       | Profile, streak, progress, birthday setting                    |
| ⚙️   | Settings | App settings, 2D/3D toggle, audio, notifications               |

### Behavior

- **During Lesson:** Nav bar is semi-transparent, "Learn" icon is highlighted
- **Tap any other icon:** Lesson pauses, overlay slides up
- **Swipe down or tap outside:** Overlay closes, lesson resumes

---

## 4. Right Side Controls (Variant Selector)

### Design (TikTok-Style Side Icons)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                                                          ┌───┐ │
│                                                          │🎂 │ │
│                                                          │Age│ │
│                                                          └───┘ │
│                                                          ┌───┐ │
│                                                          │🌍 │ │
│                                                          │Lng│ │
│                         (Kelly Frame)                    └───┘ │
│                                                          ┌───┐ │
│                                                          │🎭 │ │
│                                                          │Tone│ │
│                                                          └───┘ │
│                                                          ┌───┐ │
│                                                          │🎯 │ │  ← NEW
│                                                          │Lvl │ │
│                                                          └───┘ │
│                                                          ┌───┐ │
│                                                          │🔄 │ │
│                                                          │2D/3D│ │
│                                                          └───┘ │
│                                                          ┌───┐ │
│                                                          │↗️ │ │
│                                                          │Share│ │
│                                                          └───┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Position: Right edge, vertically centered
Each icon: 44×44px touch target (6 controls total)
Background: rgba(0, 0, 0, 0.5) rounded
Gap: 16px between icons
Badge: Small badge (18×18px) on icon shows current value
```

### Control Definitions

#### 🎂 Age Control

**Tap opens radial/list selector:**

```
┌─────────────────┐
│   Your Age      │
├─────────────────┤
│ ○ 2-5   Tiny    │
│ ○ 6-12  Young   │
│ ● 13-17 Teen    │  ← Current
│ ○ 18-35 Adult   │
│ ○ 36-60 Grown   │
│ ○ 61+   Wise    │
└─────────────────┘
```

**Effect:** Instantly regenerates current phase content for new age. Kelly's language complexity changes. Same topic, different delivery.

#### 🌍 Language Control

**Tap opens selector:**

```
┌─────────────────┐
│   Language      │
├─────────────────┤
│ ● 🇺🇸 English   │  ← Current
│ ○ 🇪🇸 Español   │
│ ○ 🇫🇷 Français  │
└─────────────────┘
```

**Effect:** Restarts current phase in new language. Kelly speaks that language. Subtitles match.

#### 🎭 Tone Control

**Tap opens selector:**

```
┌─────────────────┐
│   Tone          │
├─────────────────┤
│ ● 🔬 Curious    │  ← Current
│ ○ 🎨 Playful    │
│ ○ 📚 Serious    │
└─────────────────┘
```

**Effect:** Changes Kelly's personality delivery. Same facts, different vibe.

#### 🎯 Level/Difficulty Control (NEW)

**Tap opens selector:**

```
┌─────────────────────────────────┐
│   Challenge Level               │
├─────────────────────────────────┤
│ ● 2 Choices — Standard          │  ← Current
│   Great for focused learning    │
│ ○ 3 Choices — Challenge Mode    │
│   More nuanced options          │
└─────────────────────────────────┘
```

**Effect:** Controls how many answer options appear per question phase:

- **2 Choices (Standard):** A and B only. Clear, binary thinking. Great for younger learners or quick sessions.
- **3 Choices (Challenge):** A, B, and C. The third option often combines A and B or introduces nuance. Promotes deeper critical thinking.

**Key Design Principle:** The third choice is NOT just another wrong answer. It's typically:

- A synthesis ("Both A and B")
- A nuanced perspective ("It depends on context")
- A deeper insight that requires understanding both simpler options

**Database Impact:** Each question phase stores 3 choices. When difficulty=2, only choices where `choice_order <= 2` are displayed. When difficulty=3, all 3 are shown.

**Independence:** This is ORTHOGONAL to age/language/tone. A 6-year-old can play challenge mode. An adult can use standard mode. The choice count doesn't change the content complexity—that's the age variant's job
**Effect:** Same content, different personality. Curious = questioning, Playful = fun analogies, Serious = factual.

#### 🔄 2D/3D Toggle

**Tap toggles:**

- 2D → 3D: Unity canvas loads, replaces image
- 3D → 2D: Unity hides, image shows

(See Section 6 for details)

#### ↗️ Share

**Tap opens share sheet:**

- "I just learned about Citizenship with @CuriousKelly!"
- Includes lesson card image
- Deep link to start same lesson

---

## 5. Phase Progression System

### The 5-Phase Structure

```
WELCOME → Q1 → Q2 → Q3 → WISDOM
   │       │     │     │      │
   ▼       ▼     ▼     ▼      ▼
 Intro  Choice Choice Choice Reflect
        A or B A or B A or B
```

### Phase Indicator UI

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    ┌───────────────────────────────────────────────────┐       │
│    │  ●────○────○────○────○                            │       │
│    │  W    Q1   Q2   Q3   ✨                           │       │
│    │  ▲                                                │       │
│    │  current                                          │       │
│    └───────────────────────────────────────────────────┘       │
│                                                                 │
│                         (Kelly Frame)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Position: Top center, below safe area
Style: Pills connected by lines
● = current/completed
○ = upcoming
Colors: Completed = Kelly Blue, Current = pulsing, Upcoming = gray
```

### Speech Bubble + Choices UI

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                         (Kelly)                                 │
│                                                                 │
│    ┌─────────────────────────────────────────────────────┐     │
│    │                                                     │     │
│    │  "What does it mean to be a good citizen?          │     │
│    │   Is it about following rules, or something        │     │
│    │   deeper?"                                          │     │
│    │                                                     │     │
│    │  ┌─────────────────────────────────────────────┐   │     │
│    │  │  A) Following laws and paying taxes         │   │     │
│    │  └─────────────────────────────────────────────┘   │     │
│    │                                                     │     │
│    │  ┌─────────────────────────────────────────────┐   │     │
│    │  │  B) Actively contributing to community      │   │     │
│    │  └─────────────────────────────────────────────┘   │     │
│    │                                                     │     │
│    └─────────────────────────────────────────────────────┘     │
│                                                                 │
│   [Bottom Nav Bar]                                              │
└─────────────────────────────────────────────────────────────────┘

Position: Bottom third of screen, above nav bar
Background: rgba(0, 0, 0, 0.85) with blur
Choices: Tap to select, then auto-advance (or "Continue" button)
```

### Phase Transition Animation

1. User selects choice
2. Choice highlights (Kelly Blue)
3. Kelly expression changes (Explaining → Listening → Explaining)
4. Speech bubble fades out
5. Phase indicator advances
6. New content fades in
7. Kelly speaks new phase (lip sync if 3D)

---

## 6. 2D vs 3D Kelly Toggle

### Toggle Location

- Primary: Right side controls (🔄 icon)
- Secondary: Settings page
- Initial: User prompted on first launch

### 2D Mode Specification

```
Component: <img> element
Source: /images/kelly/kelly-chair-{expression}.png
Expressions: curious, explaining, listening, wisdom, celebrating
Transitions: Crossfade (300ms) between expressions
Performance: Instant load, works offline
```

### 3D Mode Specification

```
Component: Unity WebGL canvas
Build: /unity/kelly-live/Build/
Features:
  - Real-time expression changes
  - Lip sync to TTS audio
  - Idle animations
  - Gesture animations
Performance: Requires GPU, ~10MB initial load
Fallback: If Unity fails, auto-switch to 2D with toast notification
```

### Toggle Behavior

```javascript
// Pseudocode
function toggle2D3D() {
  if (current === '2D') {
    showLoadingSpinner();
    loadUnityCanvas();
    onUnityReady(() => {
      fadeOut(kellyImage);
      fadeIn(unityCanvas);
      savePreference('3D');
    });
  } else {
    fadeOut(unityCanvas);
    fadeIn(kellyImage);
    unloadUnity(); // Free memory
    savePreference('2D');
  }
}
```

### User Prompt (First Launch)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│     Meet Kelly, Your AI Teacher                                 │
│                                                                 │
│     ┌───────────────┐    ┌───────────────┐                     │
│     │               │    │               │                     │
│     │   [2D Kelly]  │    │   [3D Kelly]  │                     │
│     │     image     │    │   animated    │                     │
│     │               │    │               │                     │
│     └───────────────┘    └───────────────┘                     │
│                                                                 │
│     ○ 2D Kelly             ○ 3D Kelly                          │
│       Fast & Simple          Animated & Immersive              │
│       Works everywhere       Needs good connection             │
│                                                                 │
│     [  Continue with 2D  ]  [  Try 3D Kelly  ]                 │
│                                                                 │
│     You can switch anytime in settings                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. The Kelly Today Hub

### Role in System

The Hub is the **default landing state** when not in a lesson. It's a full-screen overlay that shows:

- Today's lesson (hero)
- Calendar navigation
- Progress & streak

### When Hub Appears

- App opens (no active lesson)
- User completes lesson
- User taps Home in nav
- User taps Calendar in nav (filtered to calendar view)

### Hub Design (Updated for Full Integration)

```
┌─────────────────────────────────────────────────────────────────┐
│ ← Back                                    🔔                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                         │  │
│   │    TODAY · November 28, 2025                           │  │
│   │                                                         │  │
│   │    ╔═══════════════════════════════════════════════╗   │  │
│   │    ║                                               ║   │  │
│   │    ║           🏛️ CITIZENSHIP                      ║   │  │
│   │    ║                                               ║   │  │
│   │    ║   "Participating in and contributing         ║   │  │
│   │    ║    to community"                             ║   │  │
│   │    ║                                               ║   │  │
│   │    ╚═══════════════════════════════════════════════╝   │  │
│   │                                                         │  │
│   │    [      🎬 START TODAY'S LESSON      ]               │  │
│   │                                                         │  │
│   │    🔥 7 day streak                    91% complete      │  │
│   │                                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  NOV 2025                               < This Week >   │  │
│   │                                                         │  │
│   │   S     M     T     W     T     F     S                │  │
│   │  24    25    26    27   ★28    29    30                │  │
│   │   ✓     ✓     ✓     ✓    ●     ·     ·                │  │
│   │                                                         │  │
│   │  🎂 Your birthday: Mar 15 → "Creative Writing"         │  │
│   │                                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│   🏠      📅       🎓       👤      ⚙️                         │
│   Home   Calendar  Learn    Me    Settings                     │
└─────────────────────────────────────────────────────────────────┘
```

### Hub → Lesson Transition

1. User taps "START TODAY'S LESSON"
2. Hub slides down (or fades)
3. Kelly Frame becomes visible
4. Phase indicator appears
5. Welcome phase begins
6. Kelly speaks welcome

---

## 8. Screen-by-Screen Flows

### Flow 1: App Launch (Returning User)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│   Splash    │ ──▶ │  Hub shows  │ ──▶ │  Tap Start  │
│   (1 sec)   │     │  Today's    │     │             │
│             │     │   lesson    │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐     ┌──────▼──────┐
                    │             │     │             │
                    │   Lesson    │ ◀── │   Kelly     │
                    │  Complete   │     │   Teaches   │
                    │             │     │             │
                    └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │             │
                    │  Hub shows  │
                    │   Streak+1  │
                    │             │
                    └─────────────┘
```

### Flow 2: Change Variant Mid-Lesson

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│   In Q2     │ ──▶ │  Tap 🎂    │ ──▶ │   Age       │
│   Phase     │     │  (Age)      │     │  Selector   │
│             │     │             │     │   Opens     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │             │
                                        │  Select     │
                                        │  "Teen"     │
                                        │             │
                                        └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐     ┌──────▼──────┐
│             │     │             │     │             │
│  Continue   │ ◀── │   Q2 now    │ ◀── │  Content    │
│   Q2→Q3     │     │  Teen-level │     │  Reloads    │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Flow 3: Access Calendar Mid-Lesson

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│   In Q1     │ ──▶ │  Tap 📅    │ ──▶ │  Calendar   │
│   Phase     │     │  (Calendar) │     │  Overlay    │
│             │     │             │     │   Opens     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
        ┌──────────────────────────────────────┤
        │                                      │
        ▼                                      ▼
┌─────────────┐                        ┌─────────────┐
│             │                        │             │
│  Browse     │                        │  Find       │
│  Days       │                        │  Birthday   │
│             │                        │             │
└──────┬──────┘                        └──────┬──────┘
       │                                      │
       │         ┌─────────────┐              │
       └────────▶│             │◀─────────────┘
                 │  Tap Day    │
                 │  (e.g. Mar 15)│
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │             │
                 │  Preview    │
                 │  That Day's │
                 │  Lesson     │
                 └──────┬──────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ "Start      │  │ "Close      │  │ "Add to     │
│  This One"  │  │  Preview"   │  │  Calendar"  │
│ (if past)   │  │ (resume Q1) │  │ (future)    │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## 9. Component Inventory

### New Components to Build

| Component            | Type       | Purpose                                |
| -------------------- | ---------- | -------------------------------------- |
| `<kelly-frame>`      | Core       | Full-screen Kelly container (2D/3D)    |
| `<bottom-nav>`       | Navigation | 5-icon bottom navigation bar           |
| `<side-controls>`    | Controls   | Right-side variant selectors           |
| `<phase-indicator>`  | Progress   | 5-dot phase progress                   |
| `<speech-bubble>`    | Content    | Kelly's speech + choices               |
| `<kelly-hub>`        | Overlay    | Today Hub (hero + calendar + progress) |
| `<variant-selector>` | Modal      | Age/Language/Tone picker               |
| `<calendar-grid>`    | Display    | 365-day interactive calendar           |
| `<lesson-preview>`   | Modal      | Preview any day's lesson               |
| `<mode-toggle>`      | Control    | 2D/3D switch                           |

### Existing Components to Modify

| Component          | Location   | Changes Needed                        |
| ------------------ | ---------- | ------------------------------------- |
| Unity loader       | `app.html` | Extract to reusable, add show/hide    |
| Phase system       | `app.html` | Extract to component, add transitions |
| Lesson data loader | `app.html` | Support variant switching             |

### Components to Delete

| Component       | Location          | Reason                              |
| --------------- | ----------------- | ----------------------------------- |
| Sidebar         | `app.html`        | Replaced by Hub + bottom nav        |
| Calendar panel  | `calendar.html`   | Replaced by Hub calendar            |
| Curriculum grid | `curriculum.html` | Replaced by Hub (marketing version) |

---

## 10. Files to Create/Modify/Delete

### CREATE

```
public/
├── css/
│   └── kelly-experience.css      # All new styles
├── js/
│   ├── kelly-frame.js            # 2D/3D Kelly controller
│   ├── bottom-nav.js             # Navigation logic
│   ├── side-controls.js          # Variant controls
│   ├── phase-controller.js       # Phase state machine
│   ├── hub-controller.js         # Hub overlay logic
│   └── calendar-grid.js          # Calendar interactions
└── components/
    ├── speech-bubble.html        # Speech + choices template
    ├── variant-selector.html     # Age/Lang/Tone modals
    └── lesson-preview.html       # Preview modal
```

### MODIFY

```
public/
├── app.html                      # Strip sidebar, integrate new system
├── index.html                    # Add Hub for marketing (variant)
└── data/
    └── 365_day_calendar.json     # Add variant content paths
```

### DELETE

```
public/
├── calendar.html                 # Replaced by Hub
├── css/
│   └── calendar.css              # No longer needed
└── js/
    └── calendar-page.js          # No longer needed
```

### REDIRECT

```
/curriculum.html → / (with Hub open)
/calendar.html → /app.html (with Calendar tab active)
```

---

## 11. Implementation Phases

### Phase 1: Kelly Frame Foundation (Week 1)

- [ ] Build `<kelly-frame>` with 2D support
- [ ] Build `<bottom-nav>` with 5 icons
- [ ] Build `<phase-indicator>`
- [ ] Build `<speech-bubble>` with choices
- [ ] Wire up basic phase progression (no variants yet)
- [ ] **Deliverable:** Lesson playable with new UI, 2D only

### Phase 2: Variant System (Week 2)

- [ ] Build `<side-controls>` icons
- [ ] Build `<variant-selector>` modals
- [ ] Implement age variant switching
- [ ] Implement language switching
- [ ] Implement tone switching
- [ ] **Deliverable:** Full variant system working

### Phase 3: 3D Integration (Week 3)

- [ ] Build `<mode-toggle>`
- [ ] Integrate Unity canvas into `<kelly-frame>`
- [ ] Implement 2D↔3D transitions
- [ ] Add first-launch mode selection
- [ ] **Deliverable:** 2D/3D toggle fully working

### Phase 4: Hub & Calendar (Week 4)

- [ ] Build `<kelly-hub>` overlay
- [ ] Build `<calendar-grid>` component
- [ ] Implement birthday feature
- [ ] Build `<lesson-preview>` modal
- [ ] **Deliverable:** Full Hub experience

### Phase 5: Polish & Migration (Week 5)

- [ ] Delete old files
- [ ] Set up redirects
- [ ] Test all flows
- [ ] Performance optimization
- [ ] Mobile testing on real devices
- [ ] **Deliverable:** Production-ready experience

---

## Visual Summary: The Complete Experience

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Phase: ●───○───○───○───○                                      │
│         W   Q1  Q2  Q3  ✨                                      │
│                                                                 │
│                                                          ┌───┐ │
│                                                          │🎂 │ │
│                                                          ├───┤ │
│                                                          │🌍 │ │
│              ┌───────────────────────┐                   ├───┤ │
│              │                       │                   │🎭 │ │
│              │                       │                   ├───┤ │
│              │        KELLY          │                   │🔄 │ │
│              │      (2D or 3D)       │                   ├───┤ │
│              │                       │                   │↗️ │ │
│              │    [explaining]       │                   └───┘ │
│              │                       │                         │
│              └───────────────────────┘                         │
│                                                                 │
│    ┌───────────────────────────────────────────────────────┐   │
│    │ "What does citizenship mean to you? Is it about..."   │   │
│    │                                                       │   │
│    │  ┌─────────────────────────────────────────────────┐ │   │
│    │  │  A) Following laws and paying taxes             │ │   │
│    │  └─────────────────────────────────────────────────┘ │   │
│    │  ┌─────────────────────────────────────────────────┐ │   │
│    │  │  B) Actively contributing to community          │ │   │
│    │  └─────────────────────────────────────────────────┘ │   │
│    └───────────────────────────────────────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│     🏠        📅        🎓        👤        ⚙️                  │
│    Home    Calendar   Learn      Me     Settings               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

| Metric                 | Current | Target | How Measured                 |
| ---------------------- | ------- | ------ | ---------------------------- |
| Lesson completion rate | ~40%    | 80%    | Supabase analytics           |
| Daily return rate      | ~20%    | 50%    | DAU/MAU                      |
| Variant usage          | 0%      | 30%    | Track age/lang/tone changes  |
| 3D adoption            | 0%      | 40%    | Track mode preference        |
| Birthday engagement    | N/A     | 60%    | Track birthday preview views |
| Mobile usability       | Broken  | 100%   | Zero mobile bug reports      |

---

_This specification defines the complete Kelly Learning Experience. Implementation should follow the phases exactly, with each phase fully tested before proceeding to the next._
