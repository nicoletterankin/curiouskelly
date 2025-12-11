# Canonical IDs, Terms, and Data Model

> **STATUS:** DRAFT / PROPOSED
> **VERSION:** 1.0.0
> **DATE:** 2025-12-10

## 🚨 The Golden Rule: "One Thing, One Name"

This document establishes the **STRICT** naming conventions and ID formats for the Curious Kelly ecosystem. These rules apply to:
1.  Database Schemas
2.  TypeScript Interfaces
3.  File Naming
4.  Asset Generation Pipelines
5.  Human Communication

---

## 1. Core Identity Concepts

### 1.1 The Lesson
A single day's educational unit.
-   **Terms:** `Lesson`, `Day`, `Daily Lesson`.
-   **Canonical ID:** `day_number` (Integer: 1–365).
    -   *Why:* Humans say "Day 1", not "UUID 9f8a...".
    -   *Usage:* URLs, file paths, communication.
-   **Database ID:** `id` (UUID).
    -   *Usage:* Foreign keys, internal linking.
-   **Topic:** The subject matter (e.g., "Gravity"). Use `topic` (string).

### 1.2 The Avatar / Archetype
The persona presenting the lesson.
-   **Term:** `Archetype` (Primary), `Avatar` (Secondary/Visual).
-   **Canonical ID:** Exact String Enum (Case-Sensitive in Code, Title Case in DB).
-   **Approved Archetypes (12):**
    1.  `The Explorer`
    2.  `The Rebel`
    3.  `The Scientist`
    4.  `The Architect`
    5.  `The Diplomat`
    6.  `The Empath`
    7.  `The MacGyver`
    8.  `The Mystic`
    9.  `The Storyteller`
    10. `The Survivor`
    11. `The Provider`
    12. `The Strategist`
-   **File System Safe Name:** `explorer` (lowercase, "The " stripped).

### 1.3 The Phase
The specific segment of a lesson.
-   **Problem:** Mismatch between "Generation" (Factory) and "Runtime" (Player).
-   **Resolution:** We map them 1:1.

| Phase Order | Generation Term (Factory) | Player Term (Runtime) | Description |
| :--- | :--- | :--- | :--- |
| 1 | `Hook` | `welcome` | Intro & hook. No interaction. |
| 2 | `Fact1` | `q1` | First concept + Question. |
| 3 | `Fact2` | `q2` | Second concept + Question. |
| 4 | `Fact3` | `q3` | Third concept + Question. |
| 5 | `Wisdom` | `wisdom` | Synthesis & reflection. |

-   **Canonical ID (DB/Code):** Use **PascalCase** (`Hook`, `Fact1`) for asset generation. Use **lowercase** (`welcome`, `q1`) for player state.
-   **File System Safe Name:** `hook`, `fact1` (lowercase).

### 1.4 The Variant
The demographic adaptation.
-   **Components:** `Language` + `Age` + `Tone`.
-   **Language:** ISO code (`en`, `es`, `fr`).
-   **Age:** Bucket (`5-7`, `8-12`, `13-17`, `18-35`, `36-60`, `61+`).
-   **Tone:** Style (`playful`, `conversational`, `reflective`).

---

## 2. Universal ID Formats

### 2.1 Asset Frame ID (The "Golden Key")
Used to uniquely identify a single generated asset (video/image).

Format: `day-[N]-[archetype]-[phase]-[type]-[variant]`

Examples:
-   `day-001-scientist-hook-main-en` (Base video)
-   `day-001-scientist-fact1-option_a-en` (Response video)

### 2.2 Database IDs
-   **Core Lesson:** `core_lessons.id` (UUID).
-   **Lesson Atom:** `lesson_atoms.id` (UUID) = Unique combo of `(lesson_id, archetype, phase)`.
-   **Lesson Shard:** `lesson_shards.id` (UUID) = Variant data.

### 2.3 Runtime IDs
-   **Frame ID:** `[day_number]_[archetype_slug]_[phase_slug]`
    -   *Example:* `1_scientist_fact1`
-   **State ID:** The immediate player state.
    -   `playing` (Video running)
    -   `awaiting_choice` (Video paused, UI active)
    -   `responding` (Response video running)
    -   `transitioning` (Loading next phase)

---

## 3. Data Dictionary & Rosetta Stone

| Concept | Recommended Term | Forbidden Terms | ID Type |
| :--- | :--- | :--- | :--- |
| **1-365 Sequence** | `Day Number` | `Level`, `Stage`, `Date` | `Integer` |
| **Persona** | `Archetype` | `Character`, `Bot`, `Agent` | `String Enum` |
| **Segment** | `Phase` | `Step`, `Slide`, `Screen` | `String Enum` |
| **User** | `Learner` | `User`, `Player`, `Child` | `UUID` |
| **Visual Asset** | `Frame` | `Clip`, `Shot` | `Composite String` |

---

## 4. Implementation Rules (The "Lock")

1.  **Always** use `day_number` for ordering and human display.
2.  **Always** use UUIDs for database foreign keys.
3.  **Never** parse strings to get data (e.g., don't split filenames to find the day number in runtime code; pass the metadata).
4.  **Filesystem** is `lower_case` with "The " stripped (`scientist`).
5.  **Database** is `snake_case` (`the_scientist` or "The Scientist" as text).
6.  **Code** is `PascalCase` or `camelCase` depending on context (`TheScientist`, `archetype`).

## 5. Directory Structure (Canonical)

```
/generated-assets
  /day-[001-365]
    /[archetype-slug] (e.g., scientist)
      /hook.mp4
      /fact1.mp4
      /fact2.mp4
      /fact3.mp4
      /wisdom.mp4
      /responses
        /fact1_a.mp4
        /fact1_b.mp4
```

This structure is **MANDATORY** for the pipeline.

