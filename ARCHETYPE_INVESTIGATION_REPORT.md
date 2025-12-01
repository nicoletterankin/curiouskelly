# 🔍 ARCHETYPE INVESTIGATION REPORT

## Executive Summary

**Status:** ✅ Archetype system is **fully implemented in the database** but **NOT connected to the UI**

**Current State:**

- ✅ Database has 12 archetypes per lesson (60 atoms per day: 5 phases × 12 archetypes)
- ✅ 1,800 atoms generated for days 1-30 with all archetype variants
- ❌ `learn.html` does NOT use archetype at all
- ❌ NO UI selector for archetype
- ❌ NO localStorage storage for archetype preference

---

## 1. WHERE IS ARCHETYPE CURRENTLY STORED/USED?

### ✅ Database Schema (`lesson_atoms` table)

```sql
CREATE TABLE lesson_atoms (
    id UUID PRIMARY KEY,
    core_lesson_id UUID REFERENCES core_lessons(id),
    day_number INTEGER,
    archetype TEXT NOT NULL,  -- ← STORED HERE
    phase TEXT NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**12 Archetypes in Database:**

1. Survivor
2. Caregiver
3. Explorer
4. Rebel
5. Lover
6. Creator
7. Jester
8. Sage
9. Magician
10. Hero
11. Everyman
12. Ruler

### ✅ Other Files Using Archetype

**Files that QUERY archetype:**

- `public/app.html` (lines 1715, 1745, 2531, 2534, 2539) - Has archetype selector
- `public/player.html` (line 478, 551, 562) - Maps tone → archetype
- `app/supabase-service.js` (lines 96, 100, 105, 117, 122) - Fetches atoms by archetype
- `app/elevenlabs-voice-engine.js` (lines 43+) - Has ARCHETYPE_VOICE_SETTINGS for all 12
- `scripts/test-server.js` (lines 248, 252, 253) - Mock expressions/gestures by archetype

**Files that STORE archetype preference:**

- `kelly-data.js` - **DOES NOT include archetype** (only age, language, tone, difficulty, mode)

---

## 2. IS ARCHETYPE BEING USED IN `learn.html`?

### ❌ NO - Current Status

**What `learn.html` DOES use:**

```javascript
state.variants = {
  age: localStorage.getItem('kelly_age') || '18-35',
  language: localStorage.getItem('kelly_language') || 'en',
  difficulty: parseInt(localStorage.getItem('kelly_difficulty') || '2')
};
```

**What's MISSING:**

- No `archetype` in state.variants
- No archetype localStorage key
- No archetype in getUserPreferences()
- No archetype modal
- No archetype button in sidebar

**Current Lesson Loading (line 1013-1034):**

```javascript
async function loadLesson(dayNumber) {
  // Uses golden lesson or placeholder
  // NO Supabase query
  // NO archetype filtering
}
```

### 🔴 CRITICAL GAP

`learn.html` is using **hardcoded golden lesson data** (Day 333 only) and **placeholder lessons** for other days. It's NOT querying Supabase `lesson_atoms` table at all!

---

## 3. HOW OTHER FILES USE ARCHETYPE

### `public/app.html` - Full Implementation

```javascript
// Line 1745: Maps tone → archetype
const archetypeMap = {
  curious: 'The Scientist',
  playful: 'The Jester',
  serious: 'The Sage'
};

// Line 2531: Loads atoms filtered by archetype
const { data } = await supabase
  .from('lesson_atoms')
  .select('phase, content, archetype')
  .eq('core_lesson_id', lessonId)
  .eq('archetype', targetArchetype);
```

### `public/player.html` - Tone → Archetype Mapping

```javascript
// Line 478: Internal mapping (not user-selectable)
const TONE_MAP = {
  curious: 'The Scientist',
  playful: 'The Jester',
  serious: 'The Sage'
};

// Uses tone to determine which archetype atoms to load
```

### `app/elevenlabs-voice-engine.js` - Voice Settings Per Archetype

```javascript
// Line 43: Each archetype has unique voice characteristics
const ARCHETYPE_VOICE_SETTINGS = {
  'The Scientist': { stability: 0.5, similarity_boost: 0.75, style: 0.3 },
  'The Explorer': { stability: 0.6, similarity_boost: 0.7, style: 0.4 },
  'The Sage': { stability: 0.7, similarity_boost: 0.8, style: 0.2 }
  // ... all 12 archetypes
};
```

---

## 4. SHOULD ARCHETYPE BE USER-SELECTABLE OR AUTO-DETECTED?

### Current Implementations Show 3 Different Approaches:

#### Approach A: **Tone → Archetype Mapping** (player.html)

- User selects TONE (curious/playful/serious)
- System maps tone to archetype internally
- **Pros:** Simpler UX, only 3 choices
- **Cons:** Only uses 3 of 12 archetypes

#### Approach B: **Direct Archetype Selection** (app.html)

- User selects archetype directly
- Full access to all 12 archetypes
- **Pros:** Maximum personalization
- **Cons:** More complex UX, requires explanation

#### Approach C: **Auto-Detection** (not implemented anywhere)

- System detects archetype from user behavior/preferences
- **Pros:** Zero user effort
- **Cons:** Complex ML, may not match user preference

---

## 5. RECOMMENDED APPROACH FOR `learn.html`

### ✅ **Use Approach A: Tone → Archetype Mapping**

**Rationale:**

1. `learn.html` already has TONE selector in the requirements (GAP 2)
2. Matches `player.html` pattern (consistency)
3. Simpler UX - users understand "curious/playful/serious" better than archetype names
4. Still leverages all archetype content in database

**Implementation:**

```javascript
// Tone → Archetype mapping (internal, not exposed to user)
const TONE_TO_ARCHETYPE = {
  curious: 'Sage', // Wisdom-seeking, thoughtful
  playful: 'Jester', // Fun, lighthearted
  serious: 'Ruler' // Structured, authoritative
};

// When loading lesson atoms from Supabase:
const tone = state.variants.tone || 'curious';
const archetype = TONE_TO_ARCHETYPE[tone];

const { data } = await supabase
  .from('lesson_atoms')
  .eq('day_number', dayNumber)
  .eq('archetype', archetype)
  .order('phase');
```

---

## 6. WHERE WOULD ARCHETYPE GO IN UI?

### If User-Selectable (NOT RECOMMENDED for launch):

**Option 1: Onboarding Flow**

- Add archetype quiz during first-time user setup
- Store result in localStorage
- Allow changing in Settings later
- **Best for:** Long-term personalization

**Option 2: Sidebar (like Age/Language)**

- Add archetype button to action-buttons
- Modal with 12 archetype options
- **Best for:** Frequent switching

**Option 3: Hidden (Mapped from Tone) ← RECOMMENDED**

- User selects Tone (curious/playful/serious)
- System maps to archetype internally
- No direct archetype UI needed
- **Best for:** Launch simplicity

---

## 7. CURRENT SIDEBAR STRUCTURE (learn.html)

```html
<!-- Lines 712-743: Right side action buttons -->
<div class="action-buttons">
  <button class="action-btn" id="btn-age">
    <div class="icon-wrap">
      🎂
      <span class="badge" id="badge-age">18</span>
    </div>
    <span class="label">Age</span>
  </button>

  <button class="action-btn" id="btn-language">
    <div class="icon-wrap">
      🌍
      <span class="badge" id="badge-language">EN</span>
    </div>
    <span class="label">Lang</span>
  </button>

  <button class="action-btn" id="btn-difficulty">
    <div class="icon-wrap">
      🎯
      <span class="badge" id="badge-difficulty">2</span>
    </div>
    <span class="label">Level</span>
  </button>

  <button class="action-btn" id="btn-share">
    <div class="icon-wrap">↗️</div>
    <span class="label">Share</span>
  </button>

  <button class="sound-btn" id="btn-sound">🔊</button>
</div>
```

**Pattern for new buttons:**

- `.action-btn` wrapper
- `.icon-wrap` with icon + optional `.badge`
- `.label` text below icon
- Modal overlay for selection

---

## 8. ANSWERS TO YOUR QUESTIONS

### Q1: Where is archetype currently stored/used in the codebase?

**A:** Stored in `lesson_atoms` table (database). Used in `app.html`, `player.html`, `supabase-service.js`, `elevenlabs-voice-engine.js`. **NOT used in `learn.html`**.

### Q2: Is archetype being used to select which content variant to show?

**A:** Yes in `app.html` and `player.html`. **NO in `learn.html`** (it's not loading from database at all).

### Q3: Should archetype be user-selectable in UI, or auto-detected/assigned?

**A:** **Recommended: Map from Tone** (curious → Sage, playful → Jester, serious → Ruler). User selects tone, system uses archetype internally. Simpler UX, still leverages full archetype system.

### Q4: If user-selectable, where would it go in the UI flow?

**A:** **Don't make it user-selectable for launch.** Use tone mapping. If needed later, add to onboarding quiz or Settings (not sidebar - too many buttons already).

---

## 9. CRITICAL FINDING: `learn.html` NOT USING DATABASE

### 🔴 MAJOR ISSUE DISCOVERED

`learn.html` is currently:

- ✅ Loading Day 333 from `golden-lesson-citizenship.js` (hardcoded)
- ✅ Generating placeholder lessons for other days (lines 1036-1072)
- ❌ **NOT querying Supabase at all**
- ❌ **NOT using the 1,800 atoms we just generated**

### Required Fix:

```javascript
// Replace lines 1013-1034 with:
async function loadLesson(dayNumber) {
  state.dayNumber = dayNumber;
  state.currentPhase = 1;
  state.choicesMade = {};

  // Get user preferences
  const prefs = getUserPreferences();
  const archetype = TONE_TO_ARCHETYPE[prefs.tone] || 'Sage';

  // Query Supabase for lesson atoms
  const { data: coreLesson } = await supabase
    .from('core_lessons')
    .select('id, topic, universal_truth')
    .eq('day_number', dayNumber)
    .single();

  if (!coreLesson) {
    showToast('Lesson not available yet');
    return;
  }

  const { data: atoms } = await supabase
    .from('lesson_atoms')
    .select('phase, content')
    .eq('core_lesson_id', coreLesson.id)
    .eq('archetype', archetype)
    .order('phase');

  // Build lesson from atoms
  state.lesson = {
    dayNumber,
    topic: coreLesson.topic,
    topicEmoji: '📚',
    phases: atoms.map((atom) => ({
      type: atom.phase,
      text: atom.content.script,
      choices: atom.content.options
    }))
  };

  updateUI();
  renderPhase(state.lesson.phases[0]);
}
```

---

## 10. NEXT STEPS

### Immediate (for GAP 2: Tone Selector):

1. ✅ Add tone to state.variants
2. ✅ Add tone to localStorage (STORAGE_KEYS.TONE)
3. ✅ Add tone button to sidebar
4. ✅ Create tone modal (3 options: curious/playful/serious)
5. ✅ Map tone → archetype internally
6. ✅ **FIX: Connect learn.html to Supabase** (load atoms by archetype)

### Future (Post-Launch):

- Consider adding archetype quiz to onboarding
- Allow advanced users to select archetype directly in Settings
- Track which archetypes users prefer most
- Generate analytics on archetype usage

---

## SUMMARY

**Archetype is fully built in the database but completely disconnected from learn.html.**

**Recommended fix:**

1. Add Tone selector to UI (curious/playful/serious)
2. Map tone → archetype internally (Sage/Jester/Ruler)
3. Connect learn.html to Supabase to load atoms by archetype
4. Don't expose archetype names to users (keep it simple)

**This approach:**

- ✅ Leverages all 12 archetypes in database
- ✅ Keeps UX simple (3 tone choices vs 12 archetype names)
- ✅ Matches pattern used in player.html
- ✅ Allows future expansion to full archetype selection





