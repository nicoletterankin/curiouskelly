# 🚀 Prioritized Backlog: Curious Kelly Launch & Sprint

Based on `WHATS_NEXT.md` (Launch) and `ANTIGRAVITY_72_HOUR_SPRINT.md` (Content).

## 🔴 PRIORITY 1: The 72-Hour Content Sprint (Immediate)
**Goal:** Generate "Atoms" (content chunks) for 365 lessons.
**Status:** 0/21,900 Atoms generated. 305/365 Lessons missing.

1.  **[Script] Curriculum Draft (`src/data/curriculum_365.json`)**
    - Structure: ID, Topic, Universal Truth.
    - Action: Create JSON with placeholders for 365 days.
2.  **[Script] Bulk Insert (`src/scripts/bulk_insert_core.py`)**
    - Action: Script to read curriculum JSON and INSERT/UPSERT into `core_lessons` table.
3.  **[Script] Atom Generator (`src/scripts/generate_all_atoms.py`)**
    - Action: The "Factory" script. Loops through Lessons -> Archetypes -> Phases and calls generation API.
    - *Note: Requires `PersonaGenerator` class (likely needs implementation or location).*
4.  **[Execution] Run Generation**
    - Action: Execute the factory script to start generating content.

## 🟠 PRIORITY 2: Lesson Player V2 (Launch Blocker)
**Goal:** A production-ready web player for the lessons.
**Status:** Structure created, files missing.

1.  **Player Framework**
    - Create `curious-kellly/lesson-player-v2/index.html` (Skeleton).
    - Create `curious-kellly/lesson-player-v2/styles/player.css`.
2.  **Player Logic**
    - Implement `player-core.js` (State machine: Welcome -> Questions -> Wisdom).
    - Implement `age-adapter.js` (Variant switching).
    - Implement `calendar-panel.js` (Navigation).

## 🟡 PRIORITY 3: External Services & Config (User Action Required)
**Goal:** Connect the "plumbing" (Payments, Email, Domain).
**Status:** Config templates exist, need real keys.

1.  **Environment Setup**
    - User needs to populate `.env` with Stripe, SendGrid, Supabase keys.
2.  **Database Migration**
    - Run `npm run migrate` in `curious-kellly/backend`.
3.  **Domain & Email**
    - User to purchase domain and setup SendGrid/Stripe accounts.

## 🟢 PRIORITY 4: Integration & Polish
**Goal:** Connect Player to Backend and Polish.

1.  **Stripe Integration**
    - Connect Checkout flow to Landing Page.
2.  **Gift Redemption**
    - Build Gift Code redemption page.
3.  **Mobile Polish**
    - Responsive design checks.

---

## 📋 Immediate Next Steps (Proposed)
I recommend we start immediately with **Priority 1 (The Sprint)** because content generation takes time (CPU/API limits).

1.  I will create the directory structure for the scripts.
2.  I will draft the `curriculum_365.json`.
3.  I will write the `bulk_insert_core.py` script.












