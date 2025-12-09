# Lesson Player Switcher - Implementation Guide

**Reference:** See `LESSON_PLAYER_SWITCHER_AUDIT.md` for full analysis

## Quick Start: What's Broken & Why

When you change Kelly's age from 25 to 5 years old:
1. ✅ The slider moves
2. ✅ The age number updates
3. ✅ The bucket highlights
4. ❌ **The script stays the same**
5. ❌ **The questions stay the same**

**Root Cause:** The event handler updates UI state but never queries the database for age-specific content.

---

## Fix 1: Add Database Methods to SupabaseService

Add these methods to `app/supabase-service.js`:

```javascript
/**
 * Fetch age-specific hook for a lesson day
 * @param {number} dayNumber 
 * @param {string} ageBucket - e.g., "2-5", "6-12", "13-17", "18-29", "30-54", "55+"
 * @returns {Promise<string|null>}
 */
async getAgeHook(dayNumber, ageBucket) {
  const { data, error } = await this.client
    .from('lesson_age_hooks')
    .select('hook')
    .eq('day_number', dayNumber)
    .eq('age_bucket', ageBucket)
    .single();

  if (error) {
    console.warn(`[Supabase] No age hook for day ${dayNumber}, bucket ${ageBucket}`);
    return null;
  }
  return data?.hook;
}

/**
 * Fetch lesson shards (age/region/tone variants) for a lesson
 * @param {string} coreLessonId 
 * @returns {Promise<Array>}
 */
async getLessonShards(coreLessonId) {
  const { data, error } = await this.client
    .from('lesson_shards')
    .select('age, region, tone, script_content')
    .eq('core_lesson_id', coreLessonId);

  if (error) {
    console.warn(`[Supabase] Error loading shards:`, error);
    return [];
  }
  return data || [];
}
```

---

## Fix 2: Add Shard Matching Logic to UnifiedLessonApp

Add this method to `app/script.js`:

```javascript
/**
 * Find the best matching shard for current settings
 * @param {Array} shards - All shards for this lesson
 * @param {number} age - User's age (e.g., 25)
 * @param {string} tone - Tone preference (e.g., "curious")
 * @param {string} region - Language/region (e.g., "en", "es")
 * @returns {Object|null}
 */
getShardForSettings(shards, age, tone, region) {
  if (!shards || shards.length === 0) return null;

  // Try exact match: age + tone + region
  let shard = shards.find(s => 
    s.age === age && 
    s.tone === tone && 
    s.region === region
  );

  // Fallback: age + region (any tone)
  if (!shard) {
    shard = shards.find(s => s.age === age && s.region === region);
  }

  // Fallback: just age
  if (!shard) {
    shard = shards.find(s => s.age === age);
  }

  // Fallback: closest age
  if (!shard && shards.length > 0) {
    shard = shards.reduce((closest, current) => {
      const closestDiff = Math.abs(closest.age - age);
      const currentDiff = Math.abs(current.age - age);
      return currentDiff < closestDiff ? current : closest;
    });
  }

  return shard;
}
```

---

## Fix 3: Add Content Loaders

Add these methods to `app/script.js`:

```javascript
/**
 * Load age-specific hook and update welcome phase
 */
async loadAgeHook() {
  const state = this.stateManager.getState();
  const dayNumber = state.selectedLesson?.day;
  
  if (!dayNumber) return;

  // Map app bucket to database bucket format
  const dbAgeBucket = this.mapAgeBucketToDb(state.ageBucket);
  
  const hook = await SupabaseService.getAgeHook(dayNumber, dbAgeBucket);
  
  if (hook) {
    this.stateManager.setState({ ageHook: hook });
    this.elements.sessionStatus.textContent = `✨ Personalized for ages ${state.ageBucket}`;
  }
}

/**
 * Load lesson shards and find match for current settings
 */
async loadLessonShards() {
  const state = this.stateManager.getState();
  
  if (!state.selectedLesson?.id) return;

  const shards = await SupabaseService.getLessonShards(state.selectedLesson.id);
  
  if (shards.length === 0) {
    console.log('[App] No shards available for this lesson');
    return;
  }

  // Map tone from archetype if needed
  const tone = this.archetypeToTone(state.currentArchetype);
  
  const shard = this.getShardForSettings(
    shards, 
    state.age, 
    tone,
    state.language
  );

  if (shard && shard.script_content) {
    this.stateManager.setState({ currentShard: shard });
    console.log(`[App] ✅ Loaded shard: age=${shard.age}, tone=${shard.tone}, region=${shard.region}`);
  }
}

/**
 * Map app age bucket to database format
 * App uses: '18-35', '36-60', '61-102'
 * DB uses: '18-29', '30-54', '55+'
 */
mapAgeBucketToDb(appBucket) {
  const mapping = {
    '2-5': '2-5',
    '6-12': '6-12',
    '13-17': '13-17',
    '18-35': '18-29',  // Approximate
    '36-60': '30-54',  // Approximate
    '61-102': '55+'
  };
  return mapping[appBucket] || '18-29';
}

/**
 * Map archetype to tone
 */
archetypeToTone(archetype) {
  const mapping = {
    'The Scientist': 'curious',
    'The Explorer': 'playful',
    'The Survivor': 'serious',
    // Add more as needed
  };
  return mapping[archetype] || 'curious';
}
```

---

## Fix 4: Update Event Handlers

### Age Slider Handler (replace lines 163-169):

```javascript
this.elements.ageSlider?.addEventListener('input', async (event) => {
  const value = Number(event.target.value);
  const bucket = this.getBucketForAge(value);
  
  // Update state
  this.stateManager.setState({ age: value, ageBucket: bucket });
  
  // Update UI immediately
  this.updateAgeDisplay(value);
  this.highlightBucket(bucket);
  
  // Debounce database fetch (don't hammer DB while sliding)
  if (this.ageDebounceTimer) {
    clearTimeout(this.ageDebounceTimer);
  }
  
  this.ageDebounceTimer = setTimeout(async () => {
    if (this.stateManager.getState().selectedLesson) {
      this.elements.sessionStatus.textContent = 'Loading age-specific content...';
      await this.loadAgeHook();
      await this.loadLessonShards();
      this.renderPhase(this.stateManager.getState());
    }
  }, 300);  // Wait 300ms after last slider move
});
```

### Language Selector Handler (replace lines 184-187):

```javascript
this.elements.languageSelector?.addEventListener('change', async (event) => {
  const lang = event.target.value;
  
  // Update state
  this.stateManager.setState({ language: lang });
  this.elements.sessionStatus.textContent = `Loading ${lang.toUpperCase()} content...`;
  
  // Re-fetch content for new language
  if (this.stateManager.getState().selectedLesson) {
    await this.loadLessonShards();
    this.renderPhase(this.stateManager.getState());
  }
  
  this.elements.sessionStatus.textContent = `Language set to ${lang.toUpperCase()}`;
});
```

---

## Fix 5: Update renderPhase to Use New Data

Modify `renderAtomicPhase()` to use shard content when available:

```javascript
renderAtomicPhase(state, phase) {
  const atom = state.lessonData.phases[phase];
  if (!atom) {
    this.elements.questionText.textContent = 'Kelly is formulating thoughts...';
    this.hideChoices();
    return;
  }

  // ✨ NEW: Use shard content if available for this language/age
  let script = atom.script || atom.content || "Kelly is listening...";
  
  if (state.currentShard?.script_content) {
    const shardContent = state.currentShard.script_content;
    
    // Use phase-specific content from shard if available
    if (shardContent[phase]) {
      script = shardContent[phase].script || shardContent[phase].text || script;
    } else if (shardContent.script) {
      // Some shards have a single script field
      script = shardContent.script;
    }
  }

  // ✨ NEW: Inject age hook in welcome phase
  if (phase === 'welcome' && state.ageHook) {
    script = state.ageHook;
  }

  this.setAudioScript(`Kelly: ${script}`);
  this.elements.questionText.textContent = script;

  // Render Options (existing code)
  const options = atom.options || [];
  if (options.length > 0) {
    this.renderChoiceButtons(options);
  } else {
    if (phase === 'wisdom') {
      this.renderWisdomAction();
    } else {
      this.hideChoices();
    }
  }
  
  if (phase === 'wisdom') {
    this.completeSessionIfNeeded();
  }
}
```

---

## Fix 6: Add State Properties

Add to the state object in constructor:

```javascript
this.stateManager = new StateManager({
  age: 25,
  ageBucket: '18-35',
  language: 'en',
  currentView: 'today',
  calendarLessons: [],
  todayLesson: null,
  selectedLesson: null,
  selectedDay: null,
  monthOffset: 0,
  lessonData: null,
  currentPhase: 'welcome',
  isPlaying: false,
  streak: 0,
  sessionId: null,
  sessionLessonId: null,
  sessionState: null,
  sessionCompleted: false,
  vibeCoords: { x: 100, y: 0 },
  currentArchetype: 'The Scientist',
  // ✨ NEW properties for age/language content
  ageHook: null,
  currentShard: null,
});
```

---

## Testing the Fix

1. **Age Test:**
   - Load a lesson
   - Move age slider from 25 to 5
   - Expected: Script changes to child-friendly language
   - Check console for `[App] ✅ Loaded shard: age=5...`

2. **Language Test:**
   - Load a lesson
   - Change language from EN to ES
   - Expected: Script changes to Spanish
   - Check console for `[Supabase] Loaded shards...`

3. **Tone Test (via Vibe Tuner):**
   - Load a lesson
   - Drag vibe sliders to change archetype
   - Expected: Content reloads with new archetype
   - Already working ✅

---

## Database Requirements

Make sure these tables have data:

```sql
-- Should return > 0
SELECT COUNT(*) FROM lesson_age_hooks;

-- Should show distribution
SELECT age_bucket, COUNT(*) FROM lesson_age_hooks GROUP BY age_bucket;

-- Should return > 0
SELECT COUNT(*) FROM lesson_shards;

-- Should show distribution
SELECT region, tone, COUNT(*) FROM lesson_shards GROUP BY region, tone;
```

If tables are empty, content generation scripts need to be run first.

---

## Summary Checklist

- [ ] Add `getAgeHook()` to `app/supabase-service.js`
- [ ] Add `getLessonShards()` to `app/supabase-service.js`
- [ ] Add `getShardForSettings()` to `app/script.js`
- [ ] Add `loadAgeHook()` to `app/script.js`
- [ ] Add `loadLessonShards()` to `app/script.js`
- [ ] Add `mapAgeBucketToDb()` to `app/script.js`
- [ ] Add `archetypeToTone()` to `app/script.js`
- [ ] Update age slider event handler with debounced DB fetch
- [ ] Update language selector event handler with DB fetch
- [ ] Update `renderAtomicPhase()` to use shard/hook content
- [ ] Add `ageHook` and `currentShard` to state
- [ ] Verify database has content in `lesson_age_hooks` and `lesson_shards`


