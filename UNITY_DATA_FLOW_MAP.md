# Unity Data Flow: Supabase → Player → Unity

Complete mapping of how lesson content flows from database to Unity avatar.

---

## 1. SUPABASE QUERY LOGIC

### Query Location
**File:** `app/supabase-service.js`

### Query Methods

#### A. Calendar Lessons (All Lessons)
```javascript
// Method: getAllCoreLessons()
// Location: app/supabase-service.js:35-46
// Query: SELECT id, day_number, topic, universal_truth FROM core_lessons ORDER BY day_number
// Filters: NONE (fetches all lessons)
// Returns: Array of lesson summaries
```

**Query Parameters:**
- ❌ **Day number**: NOT used in query (fetches ALL lessons)
- ❌ **Age range**: NOT queried (handled client-side)
- ❌ **Language**: NOT queried (handled client-side)
- ❌ **Tone**: NOT queried (handled client-side)
- ✅ **Archetype**: Used in atom queries (see below)

**Usage:** Called on app init to populate calendar (`app/script.js:340`)

---

#### B. Lesson Content (Atoms)
```javascript
// Method: getAtomsForLesson(coreLessonId, archetype)
// Location: app/supabase-service.js:77-94
// Query: SELECT phase, content FROM lesson_atoms 
//        WHERE core_lesson_id = ? AND archetype = ?
// Returns: Object map { "Hook": content, "Fact1": content, ... }
```

**Query Parameters:**
- ✅ **coreLessonId**: Required (from selected lesson)
- ✅ **Archetype**: Required (from vibe tuner: "The Scientist", "The Explorer", etc.)
- ❌ **Age range**: NOT queried (embedded in content JSON)
- ❌ **Language**: NOT queried (embedded in content JSON)

**Fallback Logic:** (`app/script.js:383-398`)
- If archetype atoms missing → fallback to "The Scientist"
- If still missing → show error

---

### Query Flow Summary

```
User Action → State Change → Query Trigger
─────────────────────────────────────────
1. App Init → loadCalendarData() → getAllCoreLessons()
2. Lesson Selected → loadLessonDNA() → getAtomsForLesson(id, archetype)
3. Archetype Changed → loadLessonDNA() → getAtomsForLesson(id, newArchetype)
```

**Key Insight:** Age and language are NOT database filters - they're used client-side to select variants from the atom's content JSON.

---

## 2. LESSON CONTENT STRUCTURE

### A. Core Lesson Object (from `core_lessons` table)

```javascript
{
  id: "uuid-string",
  day_number: 1,                    // 1-365
  topic: "The Sun",                 // Lesson title
  universal_truth: "Every ray of sunlight...", // Learning essence
  // Optional fields may exist but not consistently queried
}
```

**Location:** `app/supabase-service.js:17-29`

---

### B. Lesson Atom Object (from `lesson_atoms` table)

```javascript
{
  id: "uuid-string",
  core_lesson_id: "uuid-reference",
  archetype: "The Scientist",       // One of 12 archetypes
  phase: "Hook",                    // Hook | Fact1 | Fact2 | Fact3 | Wisdom
  content: {                        // JSON object with variants
    // Age variants (embedded in content)
    "2-5": { ... },
    "6-12": { ... },
    "13-17": { ... },
    "18-35": { ... },
    "36-60": { ... },
    "61-102": { ... },
    
    // Language variants (nested in age variants)
    "18-35": {
      en: { script: "...", options: [...] },
      es: { script: "...", options: [...] },
      fr: { script: "...", options: [...] }
    }
  }
}
```

**Location:** `app/supabase-service.js:55-69`

---

### C. Phase Mapping (Atomic → App Phases)

```javascript
// Location: app/script.js:400-410
const lessonData = {
  phases: {
    welcome: phases['Hook'],      // Atomic "Hook" → App "welcome"
    teaching: phases['Fact1'],     // Atomic "Fact1" → App "teaching"
    practice: phases['Fact2'],     // Atomic "Fact2" → App "practice"
    wisdom: phases['Wisdom']       // Atomic "Wisdom" → App "wisdom"
  }
};
```

**Note:** Fact3 is currently unused in the app.

**Phase Aliases:** (`app/script.js:6-15`)
```javascript
const PHASE_ALIASES = {
  welcome: 'welcome',
  teaching: 'teaching',
  practice: 'practice',
  wisdom: 'wisdom',
  q1: 'teaching',    // Legacy support
  q2: 'practice',    // Legacy support
  q3: 'practice',    // Legacy support
  q4: 'wisdom'       // Legacy support
};
```

---

### D. Variant Selection Logic

**Location:** `app/script.js:952-964`

```javascript
// Step 1: Get age variant
getVariant(state) {
  return state.lessonData?.ageVariants?.[state.ageBucket] || null;
}

// Step 2: Get language from variant
getVariantLanguage(state, variant) {
  return source?.language?.[state.language] || 
         source?.language?.en || 
         source?.language?.es || 
         null;
}
```

**Age Buckets:** (`app/script.js:58-65`)
```javascript
{
  '2-5': [2, 5],
  '6-12': [6, 12],
  '13-17': [13, 17],
  '18-35': [18, 35],
  '36-60': [36, 60],
  '61-102': [61, 102]
}
```

---

## 3. UI CONTROLS & STATE CHANGES

### A. Age Slider (2-102)

**Location:** `app/index.html:80` (input range)
**Handler:** `app/script.js:163-169`

**On Change:**
```javascript
1. Calculate age bucket from value
2. Update state: { age: value, ageBucket: bucket }
3. Update UI: age display, bucket highlight
4. Trigger: State subscription → updateLessonMetaFromVariant()
5. Re-render: Phase content with new age variant
```

**What Updates:**
- ✅ Age display number
- ✅ Active bucket highlight
- ✅ Lesson content (via variant selection)
- ✅ Phase text (age-adapted)
- ❌ **NOT Unity** (missing - should update avatar age)

---

### B. Language Selector (EN/ES/FR)

**Location:** `app/index.html:92-96` (select dropdown)
**Handler:** `app/script.js:184-187`

**On Change:**
```javascript
1. Update state: { language: 'en' | 'es' | 'fr' }
2. Update UI: session status text
3. Trigger: State subscription → updateLessonMetaFromVariant()
4. Re-render: Phase content with new language variant
```

**What Updates:**
- ✅ Session status text
- ✅ Lesson content (via variant selection)
- ✅ Phase text (language-adapted)
- ✅ Audio script text
- ❌ **NOT Unity** (missing - should update audio/lip-sync)

---

### C. Calendar Day Selection

**Location:** `app/script.js:820-829` (selectLessonByDay)

**On Click:**
```javascript
1. Find lesson by day number
2. Update state: { 
     selectedLesson: lesson,
     selectedDay: day,
     currentPhase: 'welcome'
   }
3. Trigger: State subscription → loadLessonDNA()
4. Query: getAtomsForLesson(lesson.id, archetype)
5. Map: Atomic phases → App phases
6. Establish: New session via SessionClient
7. Emit: Unity 'session-start' event
```

**What Updates:**
- ✅ Calendar highlights
- ✅ Lesson overview panel
- ✅ Phase content (new lesson)
- ✅ Unity session-start event ✅
- ✅ Session tracking

---

### D. Vibe Tuner (Archetype Selection)

**Location:** `app/index.html:54-72` (slider inputs)
**Handler:** `app/script.js:154-161, 271-289`

**On Change:**
```javascript
1. Calculate nearest archetype from X/Y coords
2. Update state: { vibeCoords: {x, y}, currentArchetype: name }
3. Update UI: Archetype name/traits display
4. If archetype changed AND lesson selected:
   → loadLessonDNA() with new archetype
   → Query: getAtomsForLesson(lesson.id, newArchetype)
   → Re-render: All phases with new archetype content
```

**What Updates:**
- ✅ Archetype display
- ✅ Lesson content (new archetype atoms)
- ✅ All phases re-rendered
- ❌ **NOT Unity** (missing - should update avatar personality/expressions)

---

## 4. UNITY UPDATE TRIGGERS

### Current Unity Events (Implemented)

**Location:** `app/unity-bridge.js` (emit method)

#### A. Session Start
**Trigger:** Lesson selected, session established
**Location:** `app/script.js:1019-1024, 1040-1045`

```javascript
unityBridge.emit('session-start', {
  mode: 'new' | 'resume',
  sessionId: 'uuid',
  lessonId: 'lesson-id',
  phase: 'welcome' | 'teaching' | 'practice' | 'wisdom'
});
```

**When:**
- New lesson selected (`establishSession()`)
- Session resumed from storage

---

#### B. Phase Progress
**Trigger:** Phase changes, choice selected
**Location:** `app/script.js:499-503, 1064-1068`

```javascript
unityBridge.emit('phase-progress', {
  phase: 'welcome' | 'teaching' | 'practice' | 'wisdom',
  sessionId: 'uuid',
  lessonId: 'lesson-id',
  completedPhase: 'previous-phase' // optional
});
```

**When:**
- Phase transition (welcome → teaching → practice → wisdom)
- Choice selection advances phase
- Manual phase navigation (if implemented)

---

#### C. Choice Selected
**Trigger:** User clicks choice button
**Location:** `app/script.js:669-674`

```javascript
unityBridge.emit('choice-selected', {
  choiceId: 'choice-id-or-text',
  currentPhase: 'practice',
  nextPhase: 'wisdom',
  sessionId: 'uuid'
});
```

**When:**
- User selects a choice card
- Choice triggers phase transition

---

#### D. Session Complete
**Trigger:** Wisdom phase reached, session marked complete
**Location:** `app/script.js:1083-1086`

```javascript
unityBridge.emit('session-complete', {
  lessonId: 'lesson-id',
  durationMin: 5
});
```

**When:**
- Wisdom phase rendered
- Session completion confirmed

---

### Missing Unity Updates (Should Trigger But Don't)

#### ❌ Age Change → Unity
**Should:** Update avatar age model, voice pitch, animations
**Current:** No Unity event emitted
**Fix Needed:** Add to `app/script.js:163-169`

```javascript
// ADD THIS:
this.unityBridge.emit('age-changed', {
  age: value,
  ageBucket: bucket,
  sessionId: state.sessionId
});
```

---

#### ❌ Language Change → Unity
**Should:** Update audio file, lip-sync data, language-specific animations
**Current:** No Unity event emitted
**Fix Needed:** Add to `app/script.js:184-187`

```javascript
// ADD THIS:
this.unityBridge.emit('language-changed', {
  language: event.target.value,
  sessionId: state.sessionId,
  currentPhase: state.currentPhase
});
```

---

#### ❌ Archetype Change → Unity
**Should:** Update avatar personality, expressions, animation style
**Current:** No Unity event emitted
**Fix Needed:** Add to `app/script.js:282-288`

```javascript
// ADD THIS:
if (archetype.name !== state.currentArchetype) {
  this.unityBridge.emit('archetype-changed', {
    archetype: archetype.name,
    traits: archetype.traits,
    sessionId: state.sessionId
  });
}
```

---

#### ❌ Audio URL → Unity
**Should:** Send audio file path for current phase/age/language
**Current:** No audio URL transmission
**Fix Needed:** Add audio URL calculation and emission

```javascript
// NEEDED: Calculate audio URL
const audioUrl = `/lessons/audio/${lessonSlug}/${ageBucket}-${language}-${phase}.mp3`;

// EMIT:
this.unityBridge.emit('audio-load', {
  url: audioUrl,
  phase: currentPhase,
  ageBucket: state.ageBucket,
  language: state.language
});
```

---

## 5. COMPLETE DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│ USER ACTION                                                 │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ UI CONTROL EVENT                                            │
│ • Age slider change                                         │
│ • Language selector change                                  │
│ • Calendar day click                                        │
│ • Vibe tuner change                                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STATE MANAGER UPDATE                                        │
│ • state.age, state.ageBucket                               │
│ • state.language                                            │
│ • state.selectedLesson, state.selectedDay                   │
│ • state.vibeCoords, state.currentArchetype                 │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ STATE SUBSCRIPTION TRIGGERS                                 │
│ • loadLessonDNA() → Supabase query                         │
│ • updateLessonMetaFromVariant() → Client-side filtering    │
│ • renderPhase() → UI update                                │
└─────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│ SUPABASE      │      │ VARIANT        │
│ QUERY         │      │ SELECTION      │
│               │      │                │
│ getAtomsFor   │      │ getVariant()   │
│ Lesson(id,    │      │ getVariant     │
│ archetype)    │      │ Language()     │
└───────────────┘      └───────────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ LESSON DATA STRUCTURE                                       │
│ {                                                            │
│   phases: {                                                 │
│     welcome: { script, options },                           │
│     teaching: { script, options },                          │
│     practice: { script, options },                           │
│     wisdom: { script }                                      │
│   }                                                          │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ UI RENDERING                                                │
│ • Question text                                             │
│ • Choice buttons                                            │
│ • Audio script                                              │
│ • Phase pill                                                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ UNITY BRIDGE EMIT                                           │
│ • session-start                                             │
│ • phase-progress                                            │
│ • choice-selected                                           │
│ • session-complete                                          │
│                                                              │
│ ❌ MISSING:                                                 │
│ • age-changed                                               │
│ • language-changed                                          │
│ • archetype-changed                                         │
│ • audio-load                                                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ UNITY WEBGL / NATIVE CLIENT                                 │
│ • Receives events via postMessage or WebSocket             │
│ • Updates avatar animations, expressions, audio              │
│ • Sends telemetry back (fps, pose, latency)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. SUMMARY: WHAT EXISTS VS. MISSING

### ✅ Implemented
1. **Supabase Queries:**
   - Calendar lessons (all)
   - Lesson atoms by archetype
   - Fallback to "The Scientist"

2. **State Management:**
   - Age, language, archetype tracking
   - Lesson selection
   - Phase progression

3. **UI Updates:**
   - Age slider → content variant
   - Language selector → content variant
   - Calendar → lesson load
   - Vibe tuner → archetype change → content reload

4. **Unity Events:**
   - session-start ✅
   - phase-progress ✅
   - choice-selected ✅
   - session-complete ✅

### ❌ Missing
1. **Unity Age Updates:**
   - No event when age changes
   - Avatar doesn't update age model

2. **Unity Language Updates:**
   - No event when language changes
   - No audio URL transmission
   - No lip-sync data update

3. **Unity Archetype Updates:**
   - No event when archetype changes
   - Avatar doesn't update personality/expressions

4. **Audio Integration:**
   - No audio URL calculation
   - No audio file path sent to Unity
   - No coordination between HTML5 audio and Unity

5. **Phase Audio Mapping:**
   - No mapping of phase → audio file
   - Audio files exist but not referenced

---

## 7. RECOMMENDED FIXES

### Priority 1: Add Missing Unity Events

**File:** `app/script.js`

1. **Age Change Handler** (line ~169):
```javascript
this.elements.ageSlider?.addEventListener('input', (event) => {
  const value = Number(event.target.value);
  const bucket = this.getBucketForAge(value);
  this.stateManager.setState({ age: value, ageBucket: bucket });
  this.updateAgeDisplay(value);
  this.highlightBucket(bucket);
  
  // ADD:
  const state = this.stateManager.getState();
  if (state.sessionId) {
    this.unityBridge.emit('age-changed', {
      age: value,
      ageBucket: bucket,
      sessionId: state.sessionId
    });
  }
});
```

2. **Language Change Handler** (line ~187):
```javascript
this.elements.languageSelector?.addEventListener('change', (event) => {
  const language = event.target.value;
  this.stateManager.setState({ language });
  this.elements.sessionStatus.textContent = `Language set to ${language.toUpperCase()}`;
  
  // ADD:
  const state = this.stateManager.getState();
  if (state.sessionId && state.selectedLesson) {
    this.unityBridge.emit('language-changed', {
      language,
      sessionId: state.sessionId,
      currentPhase: state.currentPhase
    });
    // Trigger audio URL update
    this.updateUnityAudio(state);
  }
});
```

3. **Archetype Change Handler** (line ~282):
```javascript
if (archetype.name !== state.currentArchetype) {
  this.stateManager.setState({ currentArchetype: archetype.name });
  if (state.selectedLesson) {
    this.loadLessonDNA(state.selectedLesson);
    
    // ADD:
    if (state.sessionId) {
      this.unityBridge.emit('archetype-changed', {
        archetype: archetype.name,
        traits: archetype.traits,
        sessionId: state.sessionId
      });
    }
  }
}
```

### Priority 2: Add Audio URL Calculation

**File:** `app/script.js`

Add method:
```javascript
getAudioUrl(state, phase) {
  if (!state.selectedLesson) return null;
  
  // Map phase to audio phase name
  const phaseMap = {
    welcome: 'welcome',
    teaching: 'mainContent',
    practice: 'mainContent',
    wisdom: 'wisdomMoment'
  };
  
  const audioPhase = phaseMap[phase] || 'mainContent';
  const lessonSlug = state.selectedLesson.slug || 
                     state.selectedLesson.topic?.toLowerCase().replace(/\s+/g, '-');
  
  return `/lessons/audio/${lessonSlug}/${state.ageBucket}-${state.language}-${audioPhase}.mp3`;
}

updateUnityAudio(state) {
  const audioUrl = this.getAudioUrl(state, state.currentPhase);
  if (audioUrl && state.sessionId) {
    this.unityBridge.emit('audio-load', {
      url: audioUrl,
      phase: state.currentPhase,
      ageBucket: state.ageBucket,
      language: state.language
    });
  }
}
```

Call `updateUnityAudio()` in:
- `renderPhase()` after phase content loaded
- Language change handler
- Age change handler (if phase active)

---

## END OF DOCUMENT













