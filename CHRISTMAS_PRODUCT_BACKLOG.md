# 🎄 Christmas Product Backlog: The "Gift" Experience

**Goal:** Launch the "365 Days with Kelly" gift experience.
**Focus:** Lesson Player V2 & Image Assets.
**Role:** Product Manager.

---

## 🔴 P0: Critical Visual Alignment (Immediate)
The Landing Page currently uses generic "Chair" images. We MUST use the specific "Gift" assets we have to sell the "365 Days" narrative.

### 1. Update Landing Page Hero
*   **Current:** Uses generic `kelly-directors-chair-curious.png`.
*   **Requirement:** Switch to `public/images/kelly/kelly-upperbody-panelopen-christmas.png`.
*   **Why:** This image (Kelly pointing) was specifically designed to say "Here is your year." It is the visual hook for the gift.

### 2. Update Player "Welcome" State
*   **Current:** Uses `kelly-directors-chair-curious.png` in the `kelly-welcome-overlay`.
*   **Requirement:** Use `kelly-closeup-fullscreen-christmas.png` (Needs verification/generation).
*   **Narrative:** When the learner opens their gift, Kelly should be looking *right at them* (Zoom Level 0), not sitting back in a chair.

---

## 🟠 P1: The "Gift Unboxing" Flow (Player V2)
The Player is the product. It needs to feel like unwrapping a present.

### 1. "Gift Mode" Toggle
*   **Feature:** A URL parameter (e.g., `?mode=gift_preview`) that triggers a special intro.
*   **Behavior:**
    1.  **Intro:** Kelly appears (Video/Anim or High-Res Image).
    2.  **Speech:** "Merry Christmas! I'm Kelly, your personal teacher for 2026."
    3.  **Reveal:** She gestures to the Calendar panel (opening it automatically).
    4.  **Call to Action:** "Let's start our first lesson together."

### 2. The "First 30 Days" Data Connection
*   **Problem:** Player loads dummy `applied-mathematics`.
*   **Fix:** Player must load `Day 1: The Sun` (from the `curriculum_365.json` data I analyzed earlier).
*   **Action:** Update `lesson-player-v2/js/app.js` to fetch from the real data structure (even if mocked locally for now) so it aligns with the marketing promise.

---

## 🟡 P2: Missing Asset Generation
We have the Hero, but we are missing the supporting cast from `CHRISTMAS_GIFT_VISUAL_PROMPTS.md`.

### 1. Generate "Closeup Fullscreen"
*   **Prompt:** `kelly-closeup-fullscreen-christmas.png`
*   **Usage:** Player Welcome Screen, Mobile Hero.

### 2. Generate "Full Body Panel Open"
*   **Prompt:** `kelly-fullbody-panelopen-christmas.png`
*   **Usage:** The "Calendar Showcase" section of the Landing Page.

---

## 🟢 P3: Polish & Consistency
*   **Font/Color:** Ensure the "Christmas Gift" red/gold accents are subtly introduced into the Blue/White Kelly brand for the landing page (e.g., "Gift" badges).
*   **Mobile Check:** Verify the Hero Image (Pointing) doesn't get cropped awkwardly on mobile.

---

## 📝 Execution Order
1.  **Edit `curiouskelly-landing-page.html`** to swap the Hero Image (Low effort, High impact).
2.  **Edit `lesson-player-v2/js/app.js`** to wire up "Day 1: The Sun" (Product truth).
3.  **Generate Missing Images** (Completes the set).






















