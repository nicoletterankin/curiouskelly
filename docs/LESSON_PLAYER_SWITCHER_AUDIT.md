# Lesson Player Switcher Audit - Age, Tone, Language

**Date:** December 5, 2025  
**Status:** 🔴 CRITICAL GAPS IDENTIFIED

## Executive Summary

The age/tone/language switchers in the lesson player (`app/script.js`) **do not work** because:
1. The UI updates state but **no database queries are triggered** on change
2. The code expects an old DNA file format (`ageVariants`) but receives a different structure from the database (`lesson_atoms`)
3. The `lesson_shards` table (38,700 age/region/tone variants) is **never queried**
4. The `lesson_age_hooks` table is **never queried**

---

## Database Tables Available

| Table | Records | Purpose | Used in Player? |
|-------|---------|---------|-----------------|
| `core_lessons` | 365 | Main lesson metadata | ✅ Yes |
| `lesson_atoms` | 21,915 | Content by archetype + phase | ✅ Yes (partial) |
| `lesson_shards` | 38,700 | Age/region/tone variants | ❌ **NO** |
| `lesson_age_hooks` | ~2,190 | Age-specific hooks by day | ❌ **NO** |

### Table Schemas

#### `lesson_atoms`
```
core_lesson_id, archetype, phase, content
```
- **archetype**: "The Scientist", "The Explorer", "The Survivor", etc.
- **phase**: "Hook", "Fact1", "Fact2", "Fact3", "Wisdom"
- **content**: JSON with `{script, options, responses}`

#### `lesson_shards`
```
core_lesson_id, age, region, tone, birth_year, script_content
```
- **age**: Integer (e.g., 5, 10, 25, 60)
- **region**: "en", "es", "fr"
- **tone**: "playful", "curious", "serious"
- **script_content**: JSON with personalized script

#### `lesson_age_hooks`
```
day_number, age_bucket, hook
```
- **age_bucket**: "2-5", "6-12", "13-17", "18-29", "30-54", "55+"
- **hook**: Personalized intro text for that age group

---

## Current Data Flow (BROKEN)

### What Happens Now

```
┌─────────────────────────────────────────────────────────────────┐
│                     app/script.js                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Age Slider Event Handler (lines 163-169)                    │
│     ┌──────────────────────────────────────────────────────────┐│
│     │ const value = Number(event.target.value);                ││
│     │ const bucket = this.getBucketForAge(value);              ││
│     │ this.stateManager.setState({ age: value, ageBucket: bucket }); ││
│     │ this.updateAgeDisplay(value);  // ✅ UI updates          ││
│     │ this.highlightBucket(bucket);  // ✅ UI updates          ││
│     │ // ❌ NO DATABASE QUERY                                  ││
│     └──────────────────────────────────────────────────────────┘│
│                                                                  │
│  2. State Subscription (lines 298-307)                          │
│     ┌──────────────────────────────────────────────────────────┐│
│     │ if (state.ageBucket !== prev.ageBucket ||                ││
│     │     state.language !== prev.language) {                  ││
│     │   this.updateLessonMetaFromVariant(state);               ││
│     │   this.renderPhase(state);  // ❌ Uses STALE data        ││
│     │   // ❌ NO DATABASE RE-FETCH                             ││
│     │ }                                                        ││
│     └──────────────────────────────────────────────────────────┘│
│                                                                  │
│  3. getVariant() Method (line 952-954)                          │
│     ┌──────────────────────────────────────────────────────────┐│
│     │ return state.lessonData?.ageVariants?.[state.ageBucket]; ││
│     │ // ❌ ALWAYS RETURNS NULL because lessonData has:        ││
│     │ //    { phases: { welcome, teaching, ... } }             ││
│     │ //    NOT { ageVariants: { '18-35': {...} } }            ││
│     └──────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Data Structure Mismatch

**What the code EXPECTS (from old DNA files):**
```javascript
{
  "ageVariants": {
    "2-5": {
      "title": "The Big Ball of Fire!",
      "welcome": "Hi there little friend!",
      "language": {
        "en": { "title": "...", "mainContent": "..." },
        "es": { "title": "...", "mainContent": "..." }
      }
    },
    "18-35": { ... }
  },
  "interactions": [
    { "step": "teaching", "ageAdaptations": { "2-5": {...}, "18-35": {...} } }
  ]
}
```

**What the database RETURNS (from lesson_atoms):**
```javascript
{
  "phases": {
    "welcome": { "script": "...", "options": [...] },
    "teaching": { "script": "...", "options": [...] },
    "practice": { "script": "...", "options": [...] },
    "wisdom": { "script": "...", "options": [...] }
  }
}
```

**❌ There's NO `ageVariants` key! The data is flat per-archetype, not per-age.**

---

## Age Bucket Mapping Inconsistency

### In `app/script.js` (lines 58-65):
```javascript
this.bucketRanges = {
  '2-5': [2, 5],
  '6-12': [6, 12],
  '13-17': [13, 17],
  '18-35': [18, 35],  // ⚠️
  '36-60': [36, 60],  // ⚠️
  '61-102': [61, 102],
};
```

### In `lesson_age_hooks` table / `index.html`:
```javascript
if (age <= 5) ageBucket = '2-5';
else if (age <= 12) ageBucket = '6-12';
else if (age <= 17) ageBucket = '13-17';
else if (age <= 29) ageBucket = '18-29';   // ⚠️ DIFFERENT
else if (age <= 54) ageBucket = '30-54';   // ⚠️ DIFFERENT
else ageBucket = '55+';                     // ⚠️ DIFFERENT
```

**⚠️ The bucket ranges don't match! Player uses `18-35`, database has `18-29`.**

---

## Working Implementations (For Reference)

### learn.html - Tone → Archetype Mapping (WORKS)
```javascript
const TONE_TO_ARCHETYPE = {
  curious: 'The Scientist',
  playful: 'The Explorer',
  serious: 'The Survivor'
};

async function selectVariant(type, value) {
  if (type === 'tone') {
    await loadLesson(state.dayNumber);  // ✅ RE-FETCHES FROM DB
    return;
  }
  // ...
}
```

### app.html - Lesson Shards Loading (WORKS)
```javascript
async function loadLessonShards(lessonId) {
  const { data, error } = await supabase
    .from('lesson_shards')
    .select('age, region, tone, script_content')
    .eq('core_lesson_id', lessonId);
  return data;
}

function getShardForSettings(shards, age, tone, region) {
  // Try exact match first
  let shard = shards.find(s => 
    s.age === age && 
    s.tone === tone && 
    (s.region === region || !s.region)
  );
  // Fall back to just age match
  if (!shard) shard = shards.find(s => s.age === age);
  return shard;
}
```

### index.html - Age Hooks Loading (WORKS)
```javascript
async function getHookForAge(age, dayNumber) {
  let ageBucket;
  if (age <= 5) ageBucket = '2-5';
  else if (age <= 12) ageBucket = '6-12';
  // ...

  const { data } = await supabase
    .from('lesson_age_hooks')
    .select('hook')
    .eq('day_number', dayNumber)
    .eq('age_bucket', ageBucket)
    .single();

  return data?.hook;
}
```

---

## What's Missing in `app/script.js`

### 1. No Lesson Shards Integration
**Required:** Query `lesson_shards` for age/region/tone specific content

```javascript
// Add to SupabaseService.js:
async getLessonShards(coreLessonId) {
  const { data, error } = await this.client
    .from('lesson_shards')
    .select('age, region, tone, script_content')
    .eq('core_lesson_id', coreLessonId);
  return data || [];
}

// Add to app/script.js:
async loadLessonShards() {
  const state = this.stateManager.getState();
  if (!state.selectedLesson?.id) return;
  
  const shards = await SupabaseService.getLessonShards(state.selectedLesson.id);
  const matchedShard = this.getShardForSettings(shards, state.age, state.language);
  
  if (matchedShard) {
    // Merge shard content into lesson display
    this.applyShardContent(matchedShard);
  }
}
```

### 2. No Age Hooks Integration
**Required:** Query `lesson_age_hooks` for personalized intro

```javascript
// Add to SupabaseService.js:
async getAgeHook(dayNumber, ageBucket) {
  const { data } = await this.client
    .from('lesson_age_hooks')
    .select('hook')
    .eq('day_number', dayNumber)
    .eq('age_bucket', ageBucket)
    .single();
  return data?.hook;
}

// Add to app/script.js:
async loadAgeHook() {
  const state = this.stateManager.getState();
  const dayNumber = state.selectedLesson?.day;
  if (!dayNumber) return;
  
  const hook = await SupabaseService.getAgeHook(dayNumber, state.ageBucket);
  if (hook) {
    // Display personalized hook in welcome phase
    this.updateWelcomeHook(hook);
  }
}
```

### 3. Age Slider Event Missing Database Fetch
**Required:** Trigger re-fetch when age changes

```javascript
// CURRENT (line 163-169):
this.elements.ageSlider?.addEventListener('input', (event) => {
  const value = Number(event.target.value);
  const bucket = this.getBucketForAge(value);
  this.stateManager.setState({ age: value, ageBucket: bucket });
  this.updateAgeDisplay(value);
  this.highlightBucket(bucket);
  // ❌ MISSING: re-fetch age-specific content
});

// REQUIRED:
this.elements.ageSlider?.addEventListener('input', async (event) => {
  const value = Number(event.target.value);
  const bucket = this.getBucketForAge(value);
  this.stateManager.setState({ age: value, ageBucket: bucket });
  this.updateAgeDisplay(value);
  this.highlightBucket(bucket);
  
  // ✅ RE-FETCH AGE-SPECIFIC CONTENT
  if (this.stateManager.getState().selectedLesson) {
    await this.loadAgeHook();
    await this.loadLessonShards();
  }
});
```

### 4. Language Change Missing Database Fetch
**Required:** Trigger re-fetch when language changes

```javascript
// CURRENT (line 184-187):
this.elements.languageSelector?.addEventListener('change', (event) => {
  this.stateManager.setState({ language: event.target.value });
  this.elements.sessionStatus.textContent = `Language set to ${event.target.value.toUpperCase()}`;
  // ❌ MISSING: re-fetch language-specific content
});

// REQUIRED:
this.elements.languageSelector?.addEventListener('change', async (event) => {
  const lang = event.target.value;
  this.stateManager.setState({ language: lang });
  this.elements.sessionStatus.textContent = `Language set to ${lang.toUpperCase()}`;
  
  // ✅ RE-FETCH LANGUAGE-SPECIFIC CONTENT
  if (this.stateManager.getState().selectedLesson) {
    await this.loadLessonShards();  // shards have region field
  }
});
```

### 5. Tone/Archetype Mapping Missing
**Required:** Map Vibe Tuner to database archetypes and re-fetch

The Vibe Tuner already maps to archetypes, and it DOES re-fetch:
```javascript
if (archetype.name !== state.currentArchetype) {
  this.stateManager.setState({ currentArchetype: archetype.name });
  if (state.selectedLesson) {
    this.loadLessonDNA(state.selectedLesson);  // ✅ This works
  }
}
```

But the archetype names need to match what's in the database. Current matrix:
- "The Scientist" ✅ (matches DB)
- "The Explorer" ✅ (matches DB)
- "The Survivor" ✅ (matches DB)
- Others may not match exactly

---

## Recommended Fix Order

### Priority 1: Make Age Slider Functional
1. Add `getAgeHook()` to SupabaseService
2. Add `getLessonShards()` to SupabaseService
3. Modify age slider event to call these functions
4. Update `renderPhase()` to use shard content when available

### Priority 2: Make Language Selector Functional
1. Modify language change event to trigger shard re-fetch
2. Update content display to use region-specific script_content

### Priority 3: Make Tone Functional  
1. Already partially working via Vibe Tuner → Archetype
2. Add simple tone selector (curious/playful/serious) 
3. Map to correct archetypes as learn.html does

### Priority 4: Align Age Buckets
1. Update `bucketRanges` in app/script.js to match database
2. Or create migration to update database to match code

---

## Files That Need Changes

| File | Changes Required |
|------|------------------|
| `app/supabase-service.js` | Add `getAgeHook()`, `getLessonShards()` methods |
| `app/script.js` | Add age/language event handlers that trigger DB queries |
| `app/script.js` | Update `renderPhase()` to use shard content |
| `app/script.js` | Fix age bucket ranges to match database |
| `app/index.html` | Consider adding simple tone selector |

---

## Data Requirements Check

Before implementing fixes, verify database has content:

```sql
-- Check age hooks exist
SELECT COUNT(*), age_bucket 
FROM lesson_age_hooks 
GROUP BY age_bucket;

-- Check shards exist
SELECT COUNT(*), region, tone 
FROM lesson_shards 
GROUP BY region, tone;

-- Check atoms per archetype
SELECT COUNT(*), archetype 
FROM lesson_atoms 
GROUP BY archetype;
```

---

## Summary

| Feature | UI Exists | Event Handler | DB Query | Works? |
|---------|-----------|---------------|----------|--------|
| Age Slider | ✅ | ✅ Updates state | ❌ None | ❌ NO |
| Language Selector | ✅ | ✅ Updates state | ❌ None | ❌ NO |
| Vibe Tuner (Tone) | ✅ | ✅ Maps archetype | ✅ Re-fetches atoms | ⚠️ Partial |

**The lesson player looks beautiful but the age/language controls are cosmetic only. They update UI state but don't fetch the actual age/language-specific content from the database.**


