# Lesson Player & Calendar - Product Plan
**Date:** November 2025  
**Status:** Current State Analysis & Product Planning

---

## 📊 Executive Summary

This document analyzes the current lesson player and calendar system, identifies gaps, maps the user journey, and provides product planning recommendations for "The Daily Lesson" experience.

**Key Finding:** We have solid foundations (lesson player, calendar UI, DNA structure) but need integration, state management, and user persistence to create a cohesive daily learning experience.

---

## 🎯 What We Have (Current State)

### ✅ **Lesson Player** (`lesson-player/`)

**Location:** `lesson-player/index.html`, `script.js`

**Features Implemented:**
- ✅ Age slider (2-102) with 6 age buckets
- ✅ Age-adaptive content loading
- ✅ Audio playback (ElevenLabs-generated MP3s)
- ✅ Video placeholder system
- ✅ Interactive choices/questions
- ✅ Phase progression (Welcome → Teaching → Practice → Wisdom)
- ✅ Progress bar and time display
- ✅ Teaching moments system (timestamp-based)
- ✅ Right-rail UI components (Live, Find, Settings, Calendar)
- ✅ Read-along component

**Current Limitations:**
- ❌ No persistence (no resume from last position)
- ❌ No user accounts or progress tracking
- ❌ Sample lesson only (hardcoded "leaves-change-color")
- ❌ No connection to calendar
- ❌ No streak tracking
- ❌ No completion state saving

**Files:**
```
lesson-player/
├── index.html              # Main player UI
├── script.js               # Player logic (550 lines)
├── styles.css              # Main styles
├── components/
│   ├── right-rail.js      # Right sidebar (Live, Find, Settings, Calendar)
│   └── read-along.js      # Read-along component
└── videos/audio/          # Generated audio files
```

---

### ✅ **Calendar System** (`lessons/`)

**Location:** `lessons/calendar-page.html`, `calendar-page.js`

**Features Implemented:**
- ✅ 365-day calendar data (`365_day_calendar.json`)
- ✅ Multiple views: Today, Year, Month, Week
- ✅ DNA lesson detection and badges (🧬)
- ✅ Lesson detail panel
- ✅ Phase navigation (Welcome → Q1 → Q2 → Q3 → Wisdom)
- ✅ Age selector (6 buckets)
- ✅ Language selector (EN, ES, FR)
- ✅ Kelly zoom system (7 zoom levels with images)
- ✅ Side panel navigation

**Current Limitations:**
- ❌ No connection to lesson player
- ❌ No progress tracking (completion status)
- ❌ No streak display
- ❌ No resume functionality
- ❌ No user-specific data
- ❌ Calendar and player are separate apps

**Files:**
```
lessons/
├── calendar-page.html      # Calendar UI
├── calendar-page.js        # Calendar logic (848 lines)
├── calendar-page.css       # Calendar styles
├── 365_day_calendar.json   # Master calendar data
└── [DNA lesson files]      # Individual lesson DNA files
```

---

### ✅ **Backend Services** (`curious-kellly/backend/`)

**Location:** `curious-kellly/backend/src/services/`

**Features Implemented:**
- ✅ Session management service
- ✅ Progress tracking (phase-based)
- ✅ Pause/Resume functionality
- ✅ Session history
- ✅ Redis storage (with in-memory fallback)

**API Endpoints:**
```
POST   /api/sessions/start              # Create session
GET    /api/sessions/active             # Get active sessions
GET    /api/sessions/:sessionId         # Get session status
POST   /api/sessions/:sessionId/progress # Update progress
POST   /api/sessions/:sessionId/complete # Mark complete
POST   /api/sessions/:sessionId/toggle-pause # Pause/Resume
```

**Current Limitations:**
- ❌ Not integrated with lesson player
- ❌ Not integrated with calendar
- ❌ No user authentication
- ❌ No streak calculation
- ❌ No daily lesson completion tracking

---

### ✅ **Right-Rail Calendar Component**

**Location:** `lesson-player/components/right-rail.js`

**Features:**
- ✅ y/y/t format (Yesterday/Yesterday/Tomorrow)
- ✅ Streak counter UI (hardcoded "7 days")
- ✅ Progress summary UI
- ✅ Calendar panel in right rail

**Current Limitations:**
- ❌ No real data (hardcoded values)
- ❌ Not connected to backend
- ❌ Not connected to calendar system

---

## 🚧 What We Need (Gaps & Requirements)

### **P0: Critical Integration**

#### 1. **Unified User Experience**
**Problem:** Calendar and lesson player are separate apps with no connection.

**Solution:**
- Integrate calendar into lesson player (or vice versa)
- Single entry point: Calendar shows today's lesson → Click → Opens lesson player
- Shared state between calendar and player

**Implementation:**
- Add calendar view to lesson player (use right-rail calendar component)
- Add "Open Lesson" button in calendar that launches player
- Share session state between components

---

#### 2. **Progress Persistence**
**Problem:** No way to resume lessons or track completion.

**Solution:**
- Connect lesson player to backend session service
- Save progress on each phase completion
- Load last position on lesson open
- Mark lessons as complete

**Implementation:**
```javascript
// In lesson-player/script.js
async loadLesson(lessonId) {
  // Check for existing session
  const session = await fetch(`/api/sessions/active?lessonId=${lessonId}`);
  if (session.exists) {
    // Resume from last phase
    this.currentPhase = session.progress.currentPhase;
    this.loadPhase(session.progress.currentPhase);
  } else {
    // Start new session
    await fetch('/api/sessions/start', {
      method: 'POST',
      body: JSON.stringify({ lessonId, age: this.currentAge })
    });
  }
}

async completePhase(phase) {
  // Save progress
  await fetch(`/api/sessions/${this.sessionId}/progress`, {
    method: 'POST',
    body: JSON.stringify({ phase, completed: true })
  });
}
```

---

#### 3. **Daily Lesson Integration**
**Problem:** Lesson player loads hardcoded sample lesson, not today's lesson from calendar.

**Solution:**
- Load today's lesson from `365_day_calendar.json`
- Auto-select today's lesson on app open
- Support navigation to any day's lesson

**Implementation:**
```javascript
// In lesson-player/script.js
async loadTodayLesson() {
  // Fetch calendar data
  const calendar = await fetch('../lessons/365_day_calendar.json');
  const data = await calendar.json();
  
  // Find today's lesson
  const today = new Date();
  const todayLesson = data.lessons.find(l => {
    const lessonDate = new Date(l.date + ', ' + today.getFullYear());
    return lessonDate.getDate() === today.getDate() &&
           lessonDate.getMonth() === today.getMonth();
  });
  
  if (todayLesson) {
    await this.loadLesson(todayLesson.day, todayLesson.dna_file);
  }
}
```

---

#### 4. **Streak & Completion Tracking**
**Problem:** No way to track daily completion or streaks.

**Solution:**
- Track lesson completion per day
- Calculate consecutive days (streak)
- Display in calendar and right-rail
- Store in backend (user-specific when auth added)

**Implementation:**
```javascript
// Backend: Add streak calculation
async function calculateStreak(userId) {
  const completions = await getDailyCompletions(userId);
  let streak = 0;
  const today = new Date();
  
  for (let i = 0; i < completions.length; i++) {
    const date = new Date(completions[i].date);
    const daysDiff = (today - date) / (1000 * 60 * 60 * 24);
    
    if (daysDiff === streak) {
      streak++;
    } else {
      break;
    }
  }
  
  return streak;
}
```

---

### **P1: Enhanced Features**

#### 5. **Resume from Last Position**
**Problem:** Users can't resume where they left off.

**Solution:**
- Save current phase and timestamp
- Show "Resume" button if session exists
- Auto-resume on lesson open if session active

**Status:** Backend supports this, needs frontend integration.

---

#### 6. **Calendar Progress Indicators**
**Problem:** Calendar doesn't show which lessons are completed.

**Solution:**
- Add visual indicators: ✓ (completed), ● (in progress), ○ (not started)
- Color-code days in calendar
- Show progress percentage per lesson

**Implementation:**
```javascript
// In calendar-page.js
renderMonthView() {
  // ... existing code ...
  const classes = ['day-cell'];
  if (isToday) classes.push('today');
  if (lesson?.has_dna) classes.push('has-dna');
  
  // Add progress indicators
  const progress = await getLessonProgress(lesson.day);
  if (progress.completed) classes.push('completed');
  else if (progress.inProgress) classes.push('in-progress');
}
```

---

#### 7. **Right-Rail Calendar Data**
**Problem:** Right-rail calendar shows hardcoded values.

**Solution:**
- Connect to backend for real progress data
- Update y/y/t display with actual completion status
- Show real streak count

**Implementation:**
```javascript
// In right-rail.js
async updateCalendar() {
  const progress = await fetch('/api/progress/daily');
  const data = await progress.json();
  
  // Update y/y/t
  this.updateCalendar(
    data.yesterday.completed,
    data.today.inProgress,
    data.streak
  );
}
```

---

### **P2: Nice-to-Have**

#### 8. **User Authentication**
- User accounts
- Cross-device sync
- Personal progress history

#### 9. **Lesson Recommendations**
- Suggest lessons based on completion
- Highlight upcoming lessons
- Show related topics

#### 10. **Social Features**
- Share completion
- Family progress tracking
- Community streaks

---

## 🗺️ User Journey

### **Current Journey (Broken)**

```
1. User opens lesson-player/index.html
   ↓
2. Sees hardcoded "Leaves Change Color" lesson
   ↓
3. Adjusts age slider
   ↓
4. Plays lesson
   ↓
5. Answers questions
   ↓
6. Lesson ends → No completion saved
   ↓
7. User closes browser → Progress lost
   ↓
8. User returns → Starts from beginning again
```

**Problems:**
- No connection to calendar
- No persistence
- No sense of daily ritual
- No progress tracking

---

### **Target Journey (Ideal)**

```
1. User opens app (lesson-player or calendar)
   ↓
2. Sees today's lesson highlighted
   ↓
3. Clicks "Start Today's Lesson" or "Resume"
   ↓
4. Lesson player opens with today's lesson
   ↓
5. Adjusts age slider (saved to session)
   ↓
6. Plays lesson, answers questions
   ↓
7. Progress auto-saves after each phase
   ↓
8. User pauses/closes → Progress saved
   ↓
9. User returns → Sees "Resume" button
   ↓
10. Completes lesson → Marked complete in calendar
   ↓
11. Streak counter updates
   ↓
12. Tomorrow → New lesson appears
```

**Key Moments:**
- **Entry:** Calendar shows today's lesson prominently
- **Resume:** Clear "Resume" vs "Start" distinction
- **Progress:** Visual feedback (progress bar, phase indicators)
- **Completion:** Celebration, streak update, calendar update
- **Return:** Easy access to today's lesson

---

### **Detailed User Flows**

#### **Flow 1: First-Time User**
```
1. Opens app → Sees calendar with today highlighted
2. Clicks today's lesson card
3. Lesson player opens → Welcome phase
4. Adjusts age slider → Content adapts
5. Clicks play → Kelly speaks
6. Answers questions → Progresses through phases
7. Completes lesson → "Great job! Come back tomorrow"
8. Calendar shows ✓ on today
9. Streak: 1 day
```

#### **Flow 2: Returning User (Same Day)**
```
1. Opens app → Sees calendar
2. Today's lesson shows "Resume" button (not "Start")
3. Clicks "Resume"
4. Lesson player opens at last phase
5. Continues from where left off
6. Completes lesson → Marked complete
```

#### **Flow 3: Returning User (Next Day)**
```
1. Opens app → Sees calendar
2. Today's lesson is new (different from yesterday)
3. Yesterday shows ✓ (completed)
4. Streak: 2 days (if yesterday completed)
5. Clicks today's lesson → Starts fresh
```

#### **Flow 4: Browsing Past/Future Lessons**
```
1. Opens calendar → Switches to Month view
2. Sees past lessons with ✓ (completed)
3. Sees future lessons with ○ (locked/upcoming)
4. Clicks past lesson → Can review (read-only or replay)
5. Clicks future lesson → "This lesson unlocks on [date]"
```

---

## 📋 Product Planning Recommendations

### **Phase 1: Foundation (Week 1-2)**

**Goal:** Connect calendar and lesson player, add basic persistence.

**Tasks:**
1. ✅ Integrate calendar into lesson player (or create unified entry point)
2. ✅ Connect lesson player to backend session service
3. ✅ Load today's lesson from calendar data
4. ✅ Save progress on phase completion
5. ✅ Add "Resume" button when session exists

**Deliverable:** User can start today's lesson, pause, and resume.

---

### **Phase 2: Progress Tracking (Week 3)**

**Goal:** Visual progress indicators and completion tracking.

**Tasks:**
1. ✅ Mark lessons as complete in backend
2. ✅ Display completion status in calendar (✓, ●, ○)
3. ✅ Calculate and display streak
4. ✅ Update right-rail calendar with real data
5. ✅ Show progress percentage per lesson

**Deliverable:** Calendar shows completion status, streak works.

---

### **Phase 3: Daily Ritual (Week 4)**

**Goal:** Reinforce daily learning habit.

**Tasks:**
1. ✅ Auto-highlight today's lesson on app open
2. ✅ Show "Today's Lesson" card prominently
3. ✅ Add completion celebration
4. ✅ Show "Come back tomorrow" message
5. ✅ Add reminder notifications (optional)

**Deliverable:** Clear daily ritual, easy return flow.

---

### **Phase 4: Enhanced Experience (Week 5+)**

**Goal:** Polish and additional features.

**Tasks:**
1. ✅ User authentication (optional for MVP)
2. ✅ Cross-device sync
3. ✅ Lesson history/review
4. ✅ Social sharing
5. ✅ Analytics dashboard

**Deliverable:** Production-ready daily learning experience.

---

## 🎯 Success Metrics

### **Engagement Metrics**
- **Daily Active Users (DAU):** Target: 40% of registered users
- **Lesson Completion Rate:** Target: 70% of started lessons
- **Streak Length:** Target: Average 7+ days
- **Return Rate (D1, D7, D30):** Target: D1≥45%, D7≥30%, D30≥20%

### **Product Metrics**
- **Time to First Lesson:** Target: <30 seconds from app open
- **Resume Usage:** Target: 50% of sessions are resumes
- **Calendar Engagement:** Target: 60% of users browse calendar
- **Age Slider Usage:** Target: 80% of users adjust age

### **Technical Metrics**
- **Session Save Success Rate:** Target: >99%
- **Progress Load Time:** Target: <500ms
- **Calendar Load Time:** Target: <1s

---

## 🔧 Technical Architecture Recommendations

### **Unified App Structure**

```
app/
├── index.html              # Entry point (calendar + player)
├── calendar/
│   ├── calendar-view.js    # Calendar UI component
│   └── calendar-state.js   # Calendar state management
├── player/
│   ├── lesson-player.js    # Player UI component
│   └── player-state.js     # Player state management
├── shared/
│   ├── session-service.js  # Backend API client
│   ├── progress-service.js # Progress tracking
│   └── calendar-data.js    # Calendar data loader
└── components/
    ├── right-rail.js       # Right sidebar
    └── read-along.js       # Read-along
```

### **State Management**

**Recommended:** Simple event-based state sharing

```javascript
// shared/state-manager.js
class StateManager {
  constructor() {
    this.state = {
      currentLesson: null,
      currentPhase: 'welcome',
      progress: {},
      sessionId: null
    };
    this.listeners = [];
  }
  
  setState(updates) {
    this.state = { ...this.state, ...updates };
    this.notify();
  }
  
  subscribe(callback) {
    this.listeners.push(callback);
  }
  
  notify() {
    this.listeners.forEach(cb => cb(this.state));
  }
}
```

### **Backend Integration**

**API Client:**
```javascript
// shared/session-service.js
class SessionService {
  async startSession(lessonId, age) {
    const response = await fetch('/api/sessions/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lessonId, age })
    });
    return response.json();
  }
  
  async getSession(sessionId) {
    const response = await fetch(`/api/sessions/${sessionId}`);
    return response.json();
  }
  
  async updateProgress(sessionId, phase, data) {
    const response = await fetch(`/api/sessions/${sessionId}/progress`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phase, ...data })
    });
    return response.json();
  }
  
  async completeLesson(sessionId) {
    const response = await fetch(`/api/sessions/${sessionId}/complete`, {
      method: 'POST'
    });
    return response.json();
  }
}
```

---

## 📝 Implementation Checklist

### **Week 1: Integration**
- [ ] Create unified entry point (index.html)
- [ ] Integrate calendar component into lesson player
- [ ] Connect lesson player to backend session API
- [ ] Load today's lesson from calendar data
- [ ] Test session creation and retrieval

### **Week 2: Persistence**
- [ ] Save progress on phase completion
- [ ] Load last position on lesson open
- [ ] Add "Resume" button when session exists
- [ ] Handle session expiration
- [ ] Test pause/resume flow

### **Week 3: Progress Tracking**
- [ ] Mark lessons as complete
- [ ] Calculate streak (backend)
- [ ] Display completion status in calendar
- [ ] Update right-rail calendar with real data
- [ ] Show progress indicators (✓, ●, ○)

### **Week 4: Daily Ritual**
- [ ] Auto-highlight today's lesson
- [ ] Add completion celebration
- [ ] Show "Come back tomorrow" message
- [ ] Add yesterday/tomorrow navigation
- [ ] Test full daily flow

---

## 🎨 UI/UX Recommendations

### **Calendar Integration**

**Option A: Calendar as Sidebar**
- Calendar in left sidebar (collapsible)
- Lesson player in center
- Right rail for settings/calendar (y/y/t)

**Option B: Calendar as Main View**
- Calendar as default view
- Click lesson → Modal or slide-over player
- Player can be fullscreen or embedded

**Recommendation:** Option A (Calendar as Sidebar) - keeps lesson player prominent while allowing easy navigation.

### **Progress Indicators**

**Visual Design:**
- ✓ Green checkmark = Completed
- ● Blue dot = In progress
- ○ Gray circle = Not started
- 🧬 Purple badge = DNA lesson available

**Calendar Day Styling:**
```css
.day-cell.completed {
  background: #10b981; /* Green */
  color: white;
}

.day-cell.in-progress {
  border: 2px solid #3b82f6; /* Blue */
  background: #dbeafe;
}

.day-cell.has-dna {
  position: relative;
}

.day-cell.has-dna::after {
  content: '🧬';
  position: absolute;
  top: 2px;
  right: 2px;
  font-size: 10px;
}
```

### **Resume Button**

**Design:**
- Show "Resume" button if session exists and not complete
- Show "Start" button if no session or completed
- Show progress: "Resume from Phase 2 (60% complete)"

**Implementation:**
```html
<button class="resume-btn" v-if="hasActiveSession">
  <span class="resume-icon">▶️</span>
  <span class="resume-text">Resume</span>
  <span class="resume-progress">Phase 2 • 60%</span>
</button>
```

---

## 🚀 Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize features** based on user needs
3. **Create detailed tickets** for Phase 1 tasks
4. **Set up development environment** (if needed)
5. **Begin Phase 1 implementation**

---

## 📚 Related Documents

- `CURIOUS_KELLLY_EXECUTION_PLAN.md` - Overall execution plan
- `lessons/CALENDAR_SYSTEM_README.md` - Calendar system docs
- `lesson-player/README.md` - Lesson player docs
- `CLAUDE.md` - Operating rules and constraints

---

**Document Status:** Draft for review  
**Last Updated:** November 2025  
**Owner:** Product Team








