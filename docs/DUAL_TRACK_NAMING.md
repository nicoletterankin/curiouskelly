# 🔒 Dual Track Naming — LOCKED DECISION

**Status:** 🔒 LOCKED  
**Decided:** December 16, 2025  
**Authority:** This document defines the naming convention for Kelly's two learning tracks.

---

## The Two Tracks

Every day, learners experience **two parallel tracks**:

| Track | Icon | Name | Purpose | Content |
|-------|------|------|---------|---------|
| **Track 1** | 🌟 | **Learn** | What the world IS | Science, history, nature, art, human achievement |
| **Track 2** | 🧠 | **Grow** | How to LEARN | Critical thinking, AI fluency, meta-learning, ethics |

---

## Key Principles

### 1. Parallel, Not Sequential
- Both tracks run **every day, simultaneously**
- There is no "finish Learn before starting Grow"
- Day 1 has both a Learn topic AND a Grow topic

### 2. Complementary Pair
- **Learn** gives you knowledge about the world
- **Grow** gives you skills to learn better
- Together: you become both knowledgeable AND wise

### 3. Simple Naming
- One word each: **Learn** and **Grow**
- No "Year 1" or "Year 2" in user-facing UI
- No "Foundations" or "AI Fluency" in user-facing UI
- Internal file paths may still use `year1-foundations` and `year2-ai-fluency`

---

## Display Examples

### Home View
```
TODAY'S LEARNING

🌟 Learn: The Sun (8 min)
🧠 Grow: What AI Can Do (5 min)

[Start 13-Minute Session]
```

### Curriculum Browser
```
Day 17
├── 🌟 Learn: The Sun
└── 🧠 Grow: What AI Can Do

Day 18
├── 🌟 Learn: Photosynthesis  
└── 🧠 Grow: What AI Can't Do
```

### Calendar View
```
┌─────────────────────────────────────────┐
│  17                                     │
│  🌟 The Sun                             │
│  🧠 What AI Can Do                      │
│  ○ ○ (not completed)                    │
└─────────────────────────────────────────┘
```

### Lesson Complete
```
✨ Great job!

You completed:
🌟 Learn: The Sun
🧠 Grow: What AI Can Do

Tomorrow:
🌟 Learn: Photosynthesis
🧠 Grow: What AI Can't Do
```

---

## Internal Naming (Files & Code)

| User-Facing | Internal/Technical |
|-------------|-------------------|
| Learn | `learn-track`, `year1-foundations` |
| Grow | `grow-track`, `year2-ai-fluency` |

The internal paths remain unchanged for backward compatibility.

---

## Color & Icon Standards

| Track | Icon | Primary Color | Hex |
|-------|------|---------------|-----|
| Learn | 🌟 | Gold/Amber | `#f59e0b` |
| Grow | 🧠 | Purple/Violet | `#8b5cf6` |

---

## What This Replaces

| Old Term | New Term |
|----------|----------|
| Year 1 | Learn |
| Year 2 | Grow |
| Foundations of Knowledge | Learn Track |
| AI Fluency & Meta-Learning | Grow Track |
| Program Year | Track |

---

## Usage Guidelines

### ✅ DO
- "Today's Learn topic is The Sun"
- "Your Grow lesson today is about AI"
- "Complete both Learn and Grow to finish the day"
- "365 days of Learn + Grow"

### ❌ DON'T
- "Year 1 lesson" or "Year 2 lesson"
- "Foundations track" or "AI Fluency track"
- Imply one must be done before the other
- Use these terms interchangeably

---

## Related Documents

- [ONE_PAGE_KELLY_ARCHITECTURE.md](./ONE_PAGE_KELLY_ARCHITECTURE.md) — 16:9 unified experience
- [MULTI_YEAR_PROGRAM_ARCHITECTURE.md](./MULTI_YEAR_PROGRAM_ARCHITECTURE.md) — Technical structure
- [FUTURE_STATE_INDEX.md](./FUTURE_STATE_INDEX.md) — Strategic vision

---

*This decision is LOCKED. Changes require explicit approval.*
