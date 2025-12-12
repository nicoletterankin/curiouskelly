# Vibe Interface & Archetype Mapping Requirements

## 1. Overview
This document defines the frontend requirements for the "Vibe Tuner" interface. This system allows users to personalize their learning experience by adjusting "Vibe" sliders, which map to one of the 12 pre-generated archetypes in our Supabase library.

**CRITICAL CONSTRAINT:** The frontend must **NEVER** attempt to generate new lesson content via LLM. It must only **SELECT** existing content from the `lesson_atoms` table.

---

## 2. The Library (Source of Truth)
The system is backed by 12 distinct archetypes stored in the `lesson_atoms` table. The frontend must map user inputs to exactly one of these keys:

| Archetype | Core Trait | Vibe Signature |
|:---|:---|:---|
| **The Survivor** | Survival / Utility | Practical + Serious |
| **The MacGyver** | Resourcefulness | Practical + Analytical |
| **The Provider** | Care / Community | Practical + Warm |
| **The Empath** | Connection / Feeling | Abstract + Warm |
| **The Storyteller** | Narrative / Meaning | Abstract + Expressive |
| **The Diplomat** | Harmony / Mediation | Practical + Social |
| **The Scientist** | Truth / Logic | Abstract + Analytical |
| **The Explorer** | Curiosity / Adventure | Abstract + Energetic |
| **The Mystic** | Spirit / Unknown | Abstract + Deep |
| **The Architect** | Structure / Vision | Abstract + Structured |
| **The Strategist** | Outcome / Planning | Practical + Structured |
| **The Rebel** | Change / Freedom | Practical + Chaotic |

---

## 3. The Frontend Controls (The Vibe Tuner)
The UI will introduce a "Vibe" button in the Identity Panel (left stack) that opens a floating glass panel containing two continuous sliders.

### Slider A: Perspective (X-Axis)
*   **Range:** 0 to 100
*   **Label 0:** "Concrete" (Practical, tangible, immediate)
*   **Label 100:** "Abstract" (Conceptual, theoretical, visionary)
*   **Default:** 50

### Slider B: Energy (Y-Axis)
*   **Range:** 0 to 100
*   **Label 0:** "Thinking" (Logic, structure, analysis)
*   **Label 100:** "Feeling" (Emotion, intuition, connection)
*   **Default:** 50

---

## 4. The Mapping Logic (The Matrix)
The system will use a "Nearest Neighbor" lookup to map the user's (X, Y) coordinates to the closest archetype centroid.

| Archetype | X (Perspective) | Y (Energy) |
|:---|:---:|:---:|
| **The Survivor** | 0 | 0 |
| **The Strategist** | 25 | 0 |
| **The MacGyver** | 25 | 25 |
| **The Scientist** | 100 | 0 |
| **The Architect** | 75 | 25 |
| **The Rebel** | 0 | 50 |
| **The Explorer** | 100 | 50 |
| **The Diplomat** | 25 | 75 |
| **The Provider** | 0 | 100 |
| **The Storyteller** | 75 | 75 |
| **The Mystic** | 100 | 100 |
| **The Empath** | 75 | 100 |

### Logic Implementation
```javascript
function getArchetype(x, y) {
    // Calculate Euclidean distance to all 12 centroids
    // Return archetype with minimum distance
}
```

---

## 5. Safety & Fallback Protocol

### 5.1 Missing Content Handling
If the selected archetype (e.g., "The Rebel") does not have an atom for the current lesson in the database:
1.  **DO NOT** trigger an LLM generation.
2.  **FALLBACK** to the "Safety Archetype": **The Scientist**.
3.  **LOG** the missing atom to the console/telemetry for batch correction.

### 5.2 Default State
*   Initial Load: Defaults to **The Scientist** (X=100, Y=0) or **The Explorer** (X=100, Y=50) to ensure high-quality "Curious Kelly" branding.

---

## 6. Implementation Plan
1.  **Update `app/index.html`**: Add "Vibe" button and Slider Panel HTML.
2.  **Update `app/styles.css`**: Add styles for the Vibe Panel (reusing glass-panel classes).
3.  **Update `app/script.js`**:
    *   Add `vibeCoords` to State Manager.
    *   Implement `getArchetype(x,y)` logic.
    *   Wire sliders to update state.
    *   Update `loadLessonDNA` to use dynamic archetype instead of hardcoded default.






















