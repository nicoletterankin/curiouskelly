# Unified VisionOS Shell (Phase 1)

**Status:** Complete  
**Scope:** Merge lesson player + calendar UI into a single VisionOS-style experience while keeping Kelly anchored at the center and wiring surface-level state for age, language, and calendar selections.

---

## Layout Overview

- **Left Stack**
  - Identity + streak capsule
  - Hamburger control (Calendar / Lesson / Settings), age selector, and language picker
  - Calendar panel with Today, Week, and Month views, backed by `lessons/365_day_calendar.json`
- **Center Column**
  - Kelly anchor (40–45% width) reserved for the Unity/iClone WebGL surface
  - Bridge hint card that links to the avatar pipeline (button currently stubbed)
- **Right Stack**
  - Lesson overview panel fed by the selected calendar day
  - Question card + choice chips (welcome vs. practice scaffolding)
  - Unified audio strip for Kelly’s off-screen script

Panels use the shared VisionOS UI kit (`lesson-player/ui-kit.css`) and retain micro-motion affordances (breathing, hover lift, parallax-ready layers).

---

## Calendar Surfaces

| View  | Description |
|-------|-------------|
| Today | Highlights the current day, lesson objective, tags, and duration. Includes CTA buttons for “Start today’s lesson” and “Full calendar.” |
| Week  | Monday-based strip showing 7 cards with 🧬 markers for DNA-backed days. Clicking any card selects the corresponding lesson. |
| Month | Mini grid with prev/next month controls, DNA indicators, and selected-day state. |

All views read from the canonical calendar JSON without duplicating data structures; month/week rendering recomputes when selection or month offset changes.

---

## Interaction Scaffolding

- **Age / Language**: Slider + bucket mapping keeps the UI ready for age-adapted DNA lookups. Language changes currently update status text and will later drive localization requests.
- **Choice Cards**: Provide two primary entry points (welcome or practice). Selecting a card updates the phase pill and audio strip copy, preparing the surface for Phase 2’s shared state manager.
- **Audio & Play Button**: Present but mocked; playback state toggles copy and the icon so we can later connect ElevenLabs audio + backend progress.
- **Hamburger Tabs**: Slots for calendar, lesson, and settings. Currently a lightweight menu; in later phases it can anchor contextual panels.

---

## Data Flow & Fallbacks

1. Load `lessons/365_day_calendar.json` once on init.
2. Derive today’s lesson (month/day match) or fallback to day 1.
3. Update: today card, lesson meta panel, question text, and action CTAs.
4. Render week + month surfaces with DNA badges + selection state.
5. Maintain local state for `age`, `ageBucket`, `language`, `currentView`, `selectedDay`, and `monthOffset` to keep UI responsive without backend calls.

When fetch fails, the Today card shows a server warning so that local server issues are obvious (mirrors the older calendar experience).

---

## Handoff to Upcoming Phases

- **Phase 2 (Shared State Manager)**: Wire question/choice rendering to real DNA interactions, hydrate age/language-specific content, and connect session progress to the calendar cards.
- **Phase 3 (Session Service Hooks)**: Replace the placeholder streak & status chips with real `/api/sessions/*` data, mark completions directly on the calendar, and sync the audio strip with backend progress events.
- **Phase 4 (Unity Bridge)**: Attach the WebGL/iFrame surface into `#kelly-viewport`, emit lesson/phase events over the bridge, and drive Kelly’s expressions based on the selected phase/choice.

This document should serve as the reference for anyone extending the unified shell or mapping backend work onto the new UI scaffolding.

---

## Phase 2: Shared State & DNA Integration

- Introduced `app/state-manager.js`, a lightweight observable store that drives age, language, calendar selection, DNA payloads, and playback state across the UI. Every major panel now reacts to `stateManager` updates instead of ad‑hoc mutations.
- `app/script.js` subscribes to state changes so that selecting a calendar day automatically fetches the associated `*-dna.json`, hydrates Kelly’s age/language variant, and advances phases through the same logic used in the legacy player (welcome → teaching → practice → wisdom).
- Question cards now render real DNA interactions, including age adaptations and translated prompts. Choice selections emit Kelly’s response, highlight the chosen tile, and progress the shared state to the next phase.
- Lesson meta, objectives, and tags switch live based on the loaded DNA metadata, keeping the right stack synchronized with the current age bucket and language.
- Calendar, streak chip, and today card all derive from the same store, so week/month grids stay in sync with lesson progress without fragile DOM queries.

---

## Phase 3: Session Service Hooks

- Added `app/session-client.js` to wrap `/api/sessions/*` endpoints (start, resume, progress, complete, and history) with localStorage persistence so the browser remembers active sessions across refreshes.
- `app/script.js` now starts or resumes a backend session whenever a DNA lesson is loaded, syncing the current phase with the server, pushing phase-completion updates after each choice, and marking the session complete when the learner reaches Wisdom.
- The streak chip pulls real completion streaks from `/api/sessions/history/:userId`, falling back gracefully when the backend is offline.
- Kelly’s status headline surfaces session state (“Session started”, “Session resumed”, “Lesson completed”), giving immediate visibility into whether backend tracking is active.
- The UI continues to function offline: if the session APIs are unreachable, the client falls back to local state and logs a friendly “offline mode” notice instead of breaking lesson flow.

