# 🔮 Stealth Assessment System Index

**Philosophy:** Assessment that feels like play, not testing.

> "The best test is one you never know you're taking."

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [STEALTH_ASSESSMENT_ARCHITECTURE.md](./STEALTH_ASSESSMENT_ARCHITECTURE.md) | Complete design specification |
| [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) | How to wire into lesson player |

---

## What This System Does

### ✅ Collects (Invisibly)
- **Response patterns**: First-try accuracy, hint usage, redirect recovery
- **Timing signals**: Choice speed, phase durations, exploration depth
- **Engagement metrics**: Replays, pauses, completion, abandonment
- **Session patterns**: Streaks, time of day, device type

### ✅ Computes (In Background)
- **Engagement style**: Explorer, Deliberator, Speedrunner, Reflector
- **Learning velocity**: Accelerating, Steady, Warming Up
- **Overall mastery**: 0-100% based on first-try accuracy
- **Strengths & growth areas**: Positive, encouraging framing

### ✅ Displays (In Settings)
- "Your Learning Journey" dashboard
- Visual progress cards
- Strength badges
- Growth area suggestions
- Privacy controls (toggle, export, delete)

---

## What This System Does NOT Do

❌ Create a separate "placement test"  
❌ Show grades or scores that feel judgmental  
❌ Compare learners to each other  
❌ Use data for lesson selection (forbidden per CLAUDE.md)  
❌ Classify "learning styles" (forbidden per CLAUDE.md)  
❌ Share data externally  
❌ Create anxiety or test-taking pressure  

---

## File Structure

```
docs/assessment/
├── ASSESSMENT_INDEX.md           # This file
├── STEALTH_ASSESSMENT_ARCHITECTURE.md  # Full design spec
└── INTEGRATION_GUIDE.md          # Implementation guide

public/
├── js/
│   └── learner-observer.js       # JavaScript observation class
└── components/
    └── learning-journey.html     # Settings UI component

supabase/migrations/
└── 20241216_stealth_assessment.sql  # Database schema
```

---

## Implementation Status

### Phase 1: Foundation ✅ Complete (Dec 17, 2025)
- [x] Run database migration in Supabase
- [x] Created `learner_observations` table
- [x] Created `learner_insights` table  
- [x] Created `user_learning_journey` view
- [x] RLS policies enabled
- [x] Indexes created

### Phase 2: Collection ✅ Complete (Dec 16, 2025)
- [x] Include observer script in learn.html
- [x] Wire up phase tracking (`updatePhaseProgress`)
- [x] Wire up options shown tracking (`enterPhaseWithChoices`)
- [x] Wire up choice tracking (`handleUniversalChoice`)
- [x] Wire up save on complete (`completeLesson`)
- [x] Wire up save on abandon (`beforeunload`)
- [x] Test observation saving ✓ VERIFIED

### Phase 3: Insights ✅ Complete (Dec 17, 2025)
- [x] `compute_learner_insights()` function deployed
- [x] Auto-trigger on completion working
- [x] Insight accuracy validated (test showed "Deliberator" detection)

### Phase 4: Display ✅ Complete (Dec 17, 2025)
- [x] Add Learning Journey section to Settings
- [x] Professional SVG icons (replaced emojis)
- [x] Settings panel moved to right side
- [x] Journey cards show engagement style, mastery, momentum
- [x] Strengths and growth areas displayed
- [x] Loading states and empty states
- [ ] User experience testing

---

## Key Principles

### 1. Never Feel Like a Test
All observation happens during normal lesson flow. No special "test mode," no "quiz" language, no separate assessment screens.

### 2. Behavioral Signals > Correct Answers
*How* they learn matters more than *what* they get right:
- Do they rush or deliberate?
- Do they recover well from redirects?
- Do they replay content for deeper understanding?

### 3. Growth-Oriented Display
Instead of: "Score: 72/100"
We show: "📈 Growing Strong! Your curiosity is building foundations."

### 4. Complete Privacy Control
- Toggle to disable observation entirely
- Export all data as JSON
- Delete all history permanently
- Clear explanation of what we collect

### 5. Placement Through Normal Lessons
The first 3 lessons calibrate our understanding:
- Lesson 1: Baseline behavior patterns
- Lesson 2: Response to varied difficulty
- Lesson 3: Pattern confirmation

No separate "placement test" needed.

---

## Database Schema Summary

### `learner_observations`
Per-lesson behavioral data:
- Response quality (accuracy, hints, redirects)
- Timing metrics (choice speed, phase durations)
- Engagement signals (replays, pauses, completion)

### `learner_insights`
Computed profile:
- Engagement style
- Learning velocity
- Overall mastery (0-100)
- Strengths array
- Growth areas array
- Confidence level (builds over lessons)

### `user_learning_journey` (View)
Simplified read for UI display with human-friendly labels.

---

## Privacy Compliance

✅ **Data Minimization**: Only collect what's needed for insights  
✅ **Purpose Limitation**: Used only to improve learner's experience  
✅ **User Control**: Toggle off, export, delete at any time  
✅ **Transparency**: Clear explanation in Settings  
✅ **Row-Level Security**: Users only access their own data  

---

## Questions?

This system was designed to help Kelly be a better teacher by understanding each learner's unique style—without ever making them feel tested.

For implementation help, see [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md).

---

*"The best learning happens when you forget you're learning."*
