# Lesson Player & Calendar System - Expert Understanding

**Date:** December 2024  
**Purpose:** Comprehensive understanding of lesson-player and calendar systems for working on lesson interactions

---

## 🎯 System Overview

### Architecture Summary

The system consists of **three main components** that need integration:

1. **Lesson Player** (`lesson-player/`) - Interactive lesson playback with age-adaptive content
2. **Calendar System** (`lessons/`) - 365-day calendar with lesson navigation
3. **Backend Session Service** (`curious-kellly/backend/`) - Progress tracking and persistence

**Current State:** These are **separate systems** that need to be unified for a cohesive daily learning experience.

---

## 📚 Lesson Player System

### Core Components

**Location:** `lesson-player/index.html`, `script.js`

**Main Class:** `LessonPlayer`

#### Key Features:
- ✅ Age slider (2-102) with 6 age buckets: `2-5`, `6-12`, `13-17`, `18-35`, `36-60`, `61-102`
- ✅ Age-adaptive content loading from DNA files
- ✅ Audio playback (ElevenLabs-generated MP3s)
- ✅ Interactive choices/questions system
- ✅ Phase progression: `welcome` → `teaching` → `practice` → `wisdom`
- ✅ Progress bar and time display
- ✅ Teaching moments system (timestamp-based highlights)
- ✅ Kelly image system with expression selection

#### Current State:
```javascript
class LessonPlayer {
  constructor() {
    this.currentAge = 25;
    this.currentAgeBucket = '18-35';
    this.currentLanguage = 'en';
    this.currentStep = 'welcome';
    this.currentPhase = 'welcome';
    this.isPlaying = false;
    this.lessonData = null;
    this.lessonManifest = null;
  }
}
```

#### Key Methods:
- `loadTodayLesson()` - Attempts to load from calendar, falls back to sample
- `loadLessonById(lessonId)` - Loads DNA file and manifest
- `loadAgeAppropriateContent()` - Updates content based on age bucket
- `showInteraction()` - Displays questions and choices
- `handleChoice(choice)` - Processes user selection and moves to next phase
- `updateProgress()` - Updates progress bar and time display

#### Limitations:
- ❌ No persistence (progress lost on refresh)
- ❌ Hardcoded sample lesson ("leaves-change-color")
- ❌ No connection to calendar
- ❌ No resume functionality
- ❌ No completion tracking

---

## 📅 Calendar System

### Core Components

**Location:** `lessons/calendar-page.html`, `calendar-page.js`

**Main Class:** `CalendarApp`

#### Key Features:
- ✅ 365-day calendar data (`365_day_calendar.json`)
- ✅ Multiple views: `today`, `year`, `month`, `week`
- ✅ DNA lesson detection and badges (🧬)
- ✅ Lesson detail panel with phase navigation
- ✅ Age selector (6 buckets)
- ✅ Language selector (EN, ES, FR)
- ✅ Kelly zoom system (7 zoom levels: 0-6)
- ✅ Side panel navigation (slides in from right)

#### Current State:
```javascript
class CalendarApp {
  constructor() {
    this.calendarData = null;
    this.currentYear = 2025;
    this.currentMonth = 0;
    this.currentView = 'today';
    this.selectedDay = null;
    this.currentLesson = null;
    this.currentPhase = 'welcome';
    this.currentAge = '6-12';
    this.currentLanguage = 'en';
    this.phaseHistory = [];
  }
}
```

#### Key Methods:
- `loadCalendarData()` - Loads `365_day_calendar.json`
- `loadLesson(day)` - Loads DNA file for selected day
- `renderPhase()` - Renders current phase content (welcome/Q1/Q2/Q3/wisdom)
- `selectOption(choiceIndex)` - Handles user choice selection
- `goToNextPhase()` / `goToPreviousPhase()` - Phase navigation
- `updatePhaseProgress()` - Updates phase indicators

#### Phase System:
The calendar uses a **5-phase system**:
1. `welcome` - Welcome message
2. `q1` - First question
3. `q2` - Second question
4. `q3` - Third question
5. `wisdom` - Wisdom moment

#### Limitations:
- ❌ No connection to lesson player
- ❌ No progress tracking (completion status)
- ❌ No streak display
- ❌ No resume functionality
- ❌ Calendar and player are separate apps

---

## 🧬 DNA Lesson Structure

### File Format

**Location:** `lessons/*-dna.json`

**Example:** `lessons/the-sun-dna.json`

#### Structure:
```json
{
  "id": "the-sun",
  "title": "Stellar physics enables life...",
  "version": "1.0.0",
  "calendar": {
    "day": 1,
    "date": "January 1"
  },
  "ageVariants": {
    "2-5": { /* age-specific content */ },
    "6-12": { /* age-specific content */ },
    // ... other age buckets
  },
  "interactions": [
    {
      "step": "welcome",
      "question": "What do you think...",
      "choices": [
        {
          "text": "Option A",
          "response": "Great thinking!",
          "nextStep": "teaching"
        }
      ],
      "ageAdaptations": {
        "2-5": { /* age-specific question/choices */ },
        // ... other age buckets
      }
    }
  ]
}
```

#### Key Properties:

**Age Variants:**
- Each age bucket (2-5, 6-12, etc.) has:
  - `title`, `description`, `script`
  - `language` object with `en`, `es`, `fr` translations
  - `welcome`, `mainContent`, `wisdomMoment` text
  - `objectives`, `vocabulary`, `abstract_concepts`
  - `teachingMoments` array with timestamps

**Interactions:**
- Each interaction has:
  - `step` or `phase`: `"welcome"`, `"teaching"`, `"practice"`, `"wisdom"`
  - `question`: The question Kelly asks
  - `choices`: Array of 2-4 choice objects
  - `ageAdaptations`: Age-specific question/choices per bucket

**Choices:**
- Each choice has:
  - `text`: Display text
  - `response`: Kelly's response when selected
  - `nextStep`: Next phase to move to
  - `learningValue`: Optional (e.g., "moderate", "high")

---

## 🔄 Interaction Flow

### Lesson Player Flow

```
1. User opens lesson player
   ↓
2. loadTodayLesson() called
   ↓
3. Attempts to load from calendar (365_day_calendar.json)
   ↓
4. If found, loads DNA file via loadLessonById()
   ↓
5. displayLesson() shows age-appropriate content
   ↓
6. showInteraction() displays current step's question/choices
   ↓
7. User selects choice
   ↓
8. handleChoice() processes selection
   ↓
9. Shows Kelly's response
   ↓
10. Moves to nextStep (next phase)
   ↓
11. Repeats from step 6 until lesson complete
```

### Calendar Flow

```
1. User opens calendar page
   ↓
2. loadCalendarData() loads 365_day_calendar.json
   ↓
3. Auto-selects today's lesson
   ↓
4. loadLesson(day) loads DNA file
   ↓
5. renderLessonTab() displays lesson info
   ↓
6. renderPhase() shows current phase content
   ↓
7. User selects option via selectOption()
   ↓
8. Updates phaseHistory
   ↓
9. goToNextPhase() moves to next phase
   ↓
10. Repeats until wisdom phase complete
```

### Phase Progression

**Lesson Player phases:**
- `welcome` → `teaching` → `practice` → `wisdom`

**Calendar phases:**
- `welcome` → `q1` → `q2` → `q3` → `wisdom`

**Note:** These are **different phase systems** that need alignment!

---

## 💾 Backend Session Service

### Location

`curious-kellly/backend/src/services/session.js`

### API Endpoints

```
POST   /api/sessions/start              # Create new session
GET    /api/sessions/active             # Get all active sessions
GET    /api/sessions/:sessionId         # Get session status
POST   /api/sessions/:sessionId/progress # Update progress
POST   /api/sessions/:sessionId/complete # Mark complete
POST   /api/sessions/:sessionId/toggle-pause # Pause/Resume
GET    /api/sessions/:sessionId/stats   # Get statistics
```

### Session Data Structure

```javascript
{
  sessionId: "uuid",
  userId: "optional",
  age: 35,
  lessonId: "the-sun",
  startedAt: "2025-11-11T...",
  lastActivity: timestamp,
  progress: {
    currentPhase: "teaching",
    completedPhases: ["welcome"],
    interactionsCompleted: [
      { interactionId: "...", completedAt: "..." }
    ],
    teachingMomentsViewed: [
      { timestamp: 15, viewedAt: "..." }
    ]
  },
  state: {
    isActive: true,
    isPaused: false,
    isCompleted: false
  },
  durationMs: 480000,
  durationMin: 8
}
```

### Key Methods

- `createSession(lessonId, age, userId)` - Creates new session
- `getSession(sessionId)` - Retrieves session
- `updateProgress(sessionId, updates)` - Updates progress
  - `currentPhase`: Current phase name
  - `completedPhase`: Phase to mark as complete
  - `interactionCompleted`: Interaction ID completed
  - `teachingMomentViewed`: Timestamp viewed
- `completeSession(sessionId)` - Marks session complete
- `togglePause(sessionId)` - Pauses/resumes session

### Current Status

- ✅ Fully implemented
- ✅ Redis storage with in-memory fallback
- ✅ Session timeout (30 minutes)
- ❌ **Not integrated** with lesson player
- ❌ **Not integrated** with calendar

---

## 🔗 Integration Gaps

### Critical Gaps (P0)

#### 1. **No Connection Between Calendar and Player**
- **Problem:** Calendar and lesson player are separate apps
- **Impact:** User can't navigate from calendar to player
- **Solution:** Integrate calendar into player or create unified entry point

#### 2. **No Progress Persistence**
- **Problem:** Progress lost on refresh
- **Impact:** Can't resume lessons
- **Solution:** Connect player to backend session service

#### 3. **Different Phase Systems**
- **Problem:** Player uses `welcome/teaching/practice/wisdom`, Calendar uses `welcome/q1/q2/q3/wisdom`
- **Impact:** Confusion, mapping issues
- **Solution:** Standardize on one phase system or create mapping

#### 4. **No Completion Tracking**
- **Problem:** Can't mark lessons complete
- **Impact:** No streaks, no progress indicators
- **Solution:** Mark complete in backend, display in calendar

### Important Gaps (P1)

#### 5. **No Resume Functionality**
- **Problem:** Always starts from beginning
- **Solution:** Check for active session, resume from last phase

#### 6. **No Calendar Progress Indicators**
- **Problem:** Can't see which lessons completed
- **Solution:** Add visual indicators (✓, ●, ○)

#### 7. **Hardcoded Sample Lesson**
- **Problem:** Player loads "leaves-change-color" instead of today's lesson
- **Solution:** Load from calendar data

---

## 🎯 Interaction System Details

### How Interactions Work

#### In Lesson Player:

```javascript
showInteraction() {
  // Find interaction for current step
  const interaction = this.lessonData.interactions.find(
    i => i.step === this.currentStep
  );
  
  // Display question
  questionEl.textContent = interaction.question;
  
  // Display choices
  interaction.choices.forEach((choice, index) => {
    button.textContent = choice.text;
    button.addEventListener('click', () => {
      this.handleChoice(choice);
    });
  });
}

handleChoice(choice) {
  // Show Kelly's response
  this.showKellyResponse(choice.response);
  
  // Move to next step
  this.currentStep = choice.nextStep || 'teaching';
  this.currentPhase = choice.nextStep || 'mainContent';
  
  // Show next interaction
  setTimeout(() => {
    this.showInteraction();
  }, 2000);
}
```

#### In Calendar:

```javascript
renderPhase() {
  // Find interaction for current phase
  let interaction = null;
  
  if (this.currentPhase === 'welcome') {
    interaction = this.lessonDNA.interactions.find(
      i => i.step === 'welcome' || i.phase === 'welcome'
    );
  } else if (this.currentPhase === 'wisdom') {
    interaction = this.lessonDNA.interactions.find(
      i => i.step === 'wisdom' || i.phase === 'wisdom'
    );
  } else {
    // Find question interactions (q1, q2, q3)
    const questionInteractions = this.lessonDNA.interactions.filter(
      i => i.step !== 'welcome' && i.step !== 'wisdom'
    );
    const qIndex = ['q1', 'q2', 'q3'].indexOf(this.currentPhase);
    interaction = questionInteractions[qIndex];
  }
  
  // Get age-adapted content
  const ageAdaptation = interaction.ageAdaptations?.[this.currentAge];
  const question = ageAdaptation?.question || interaction.question;
  const choices = ageAdaptation?.choices || interaction.choices;
  
  // Render question and choices
}

selectOption(choiceIndex) {
  const choice = choices[choiceIndex];
  
  // Show response
  kellyStatus.textContent = 'Responding...';
  
  // Update phase history
  this.phaseHistory.push({
    phase: this.currentPhase,
    choice: choiceIndex,
    response: choice.response
  });
  
  // Move to next phase
  this.updateControls();
}
```

### Age Adaptation

Both systems support age-adaptive content:

```javascript
// Get age-specific content
const ageAdaptation = interaction.ageAdaptations?.[this.currentAge];
const question = ageAdaptation?.question || interaction.question;
const choices = ageAdaptation?.choices || interaction.choices;
```

If `ageAdaptations` exists for the current age bucket, use that. Otherwise, fall back to default `question` and `choices`.

### Language Support

DNA files include multilingual content:

```javascript
const ageVariant = this.lessonDNA.ageVariants?.[this.currentAge];
const languageContent = ageVariant?.language?.[this.currentLanguage] || 
                        ageVariant?.language?.en;
```

Supports: `en`, `es`, `fr`

---

## 📊 Data Flow

### Calendar Data

**File:** `lessons/365_day_calendar.json`

**Structure:**
```json
{
  "lessons": [
    {
      "day": 1,
      "date": "January 1",
      "title": "The Sun",
      "learning_objective": "...",
      "has_dna": true,
      "dna_file": "the-sun"
    }
  ]
}
```

**Generated by:** `lessons/generate_unified_calendar.py`

### DNA Files

**Location:** `lessons/*-dna.json`

**Loaded by:**
- Calendar: `loadLesson(day)` → fetches `${dna_file}-dna.json`
- Player: `loadLessonById(lessonId)` → fetches `${lessonId}-dna.json`

### Manifests

**Location:** `lessons/manifests/*-manifest.json`

**Contains:**
- Audio file paths per age/language/phase
- Image file paths per expression
- Asset metadata

**Used by:** Lesson player for asset loading

---

## 🚀 Next Steps for Integration

### Phase 1: Basic Integration

1. **Connect Player to Calendar**
   - Load today's lesson from calendar data
   - Add "Open in Player" button in calendar
   - Share lesson state between systems

2. **Connect Player to Backend**
   - Create session on lesson start
   - Save progress on phase completion
   - Load session on lesson open (resume)

3. **Standardize Phase System**
   - Map calendar phases (q1/q2/q3) to player phases
   - Or standardize on one system

### Phase 2: Progress Tracking

4. **Completion Tracking**
   - Mark lessons complete in backend
   - Display completion status in calendar
   - Calculate and display streaks

5. **Visual Indicators**
   - Add ✓ (completed), ● (in progress), ○ (not started)
   - Color-code calendar days
   - Show progress percentage

### Phase 3: Enhanced Experience

6. **Resume Functionality**
   - Check for active session on open
   - Show "Resume" vs "Start" button
   - Auto-resume from last phase

7. **Daily Ritual**
   - Auto-highlight today's lesson
   - Completion celebration
   - "Come back tomorrow" message

---

## 🔍 Key Files Reference

### Lesson Player
- `lesson-player/index.html` - Main UI
- `lesson-player/script.js` - Player logic (682 lines)
- `lesson-player/styles.css` - Styling
- `lesson-player/components/right-rail.js` - Right sidebar
- `lesson-player/components/read-along.js` - Read-along component

### Calendar
- `lessons/calendar-page.html` - Calendar UI
- `lessons/calendar-page.js` - Calendar logic (848 lines)
- `lessons/calendar-page.css` - Styling
- `lessons/365_day_calendar.json` - Master calendar data
- `lessons/generate_unified_calendar.py` - Calendar generator

### Backend
- `curious-kellly/backend/src/services/session.js` - Session service
- `curious-kellly/backend/src/api/sessions.js` - API routes

### DNA Files
- `lessons/*-dna.json` - Lesson DNA files
- `lessons/manifests/*-manifest.json` - Asset manifests
- `lesson-player/lesson-dna-schema.json` - Schema definition

### Documentation
- `LESSON_PLAYER_AND_CALENDAR_PRODUCT_PLAN.md` - Product plan
- `lessons/CALENDAR_SYSTEM_README.md` - Calendar docs
- `lesson-player/README.md` - Player docs

---

## ✅ Understanding Checklist

- [x] Lesson player architecture and flow
- [x] Calendar system architecture and flow
- [x] DNA file structure and age variants
- [x] Interaction system (questions, choices, responses)
- [x] Phase progression systems
- [x] Backend session service API
- [x] Integration gaps and requirements
- [x] Data flow (calendar → DNA → player)
- [x] Age adaptation mechanism
- [x] Language support (EN/ES/FR)
- [x] Key files and their purposes

---

**Status:** Ready to work on lesson interactions  
**Last Updated:** December 2024




