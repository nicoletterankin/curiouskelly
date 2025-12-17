# Stealth Assessment Integration Guide

**Purpose:** Wire the observation system into the lesson player  
**Dependencies:** `learner-observer.js`, Supabase, `learn.html`

---

## Quick Start

### 1. Run Database Migration

```bash
# In Supabase SQL Editor, run:
# supabase/migrations/20241216_stealth_assessment.sql
```

### 2. Include the Observer Script

Add to `public/learn.html` before the closing `</body>` tag:

```html
<script src="/js/learner-observer.js"></script>
```

### 3. Initialize Observer on Lesson Load

In the `loadLesson()` function (around line 8012), add initialization:

```javascript
// Initialize learner observer for this lesson
if (window.LearnerObserver) {
  window.lessonObserver = new window.LearnerObserver();
  window.lessonObserver.startLesson(
    dayNumber,
    currentLesson?.id || null,
    state.kellyId,
    state.ageBucket
  );
}
```

### 4. Hook Phase Tracking

In the `renderPhase()` function, add at the start:

```javascript
// Track phase start
window.lessonObserver?.onPhaseStart(phaseName);
```

When options appear:

```javascript
// Track when options are shown
window.lessonObserver?.onOptionsShown();
```

### 5. Hook Choice Tracking

In the `selectChoice()` function:

```javascript
// Track the choice quality
const quality = selectedOption.quality || 'good';
window.lessonObserver?.onChoice(quality);
window.lessonObserver?.onPhaseEnd(currentPhase);
```

### 6. Hook Hint Tracking

In `showStuckHint()`:

```javascript
window.lessonObserver?.onHintShown();
```

### 7. Save on Lesson Complete

In `completeLesson()` function:

```javascript
// Save observation data
if (window.lessonObserver && isLoggedIn()) {
  const userId = getUserId(); // Get from Supabase auth
  await window.lessonObserver.save(supabase, userId, true, null);
}
```

### 8. Save on Lesson Abandon

Add a beforeunload handler:

```javascript
window.addEventListener('beforeunload', async () => {
  if (window.lessonObserver && isLoggedIn() && !lessonCompleted) {
    const userId = getUserId();
    await window.lessonObserver.save(
      supabase, 
      userId, 
      false, 
      getCurrentPhase()
    );
  }
});
```

---

## Integration Points Reference

| Event | Function to Hook | Observer Method |
|-------|------------------|-----------------|
| Lesson starts | `loadLesson()` | `startLesson(day, id, arch, age)` |
| Phase renders | `renderPhase()` | `onPhaseStart(phaseName)` |
| Options appear | after options HTML | `onOptionsShown()` |
| User chooses | `selectChoice()` | `onChoice(quality)` |
| Hint shown | `showStuckHint()` | `onHintShown()` |
| Audio replays | audio replay button | `onAudioReplay()` |
| Video replays | video replay | `onVideoReplay()` |
| User pauses | pause button | `onPause()` |
| Lesson complete | `completeLesson()` | `save(supabase, userId, true)` |
| User abandons | `beforeunload` | `save(supabase, userId, false, phase)` |

---

## Code Snippet: Full Integration

Add this block inside `learn.html`:

```javascript
// ============================================
// STEALTH OBSERVATION SYSTEM
// ============================================

let lessonObserver = null;

function initObserver(dayNumber, lessonId) {
  if (!window.LearnerObserver) return;
  
  lessonObserver = new window.LearnerObserver();
  lessonObserver.startLesson(
    dayNumber,
    lessonId,
    state.kellyId || 'explorer',
    state.ageBucket || 'all'
  );
  
  console.debug('[Stealth] Observer initialized for day', dayNumber);
}

function trackPhaseStart(phaseName) {
  lessonObserver?.onPhaseStart(phaseName);
}

function trackOptionsShown() {
  lessonObserver?.onOptionsShown();
}

function trackChoice(quality) {
  lessonObserver?.onChoice(quality || 'good');
}

function trackHint() {
  lessonObserver?.onHintShown();
}

function trackAudioReplay() {
  lessonObserver?.onAudioReplay();
}

async function saveObservation(completed = true) {
  if (!lessonObserver) return;
  
  try {
    const { data: session } = await supabase.auth.getSession();
    const userId = session?.session?.user?.id;
    
    if (!userId) {
      console.debug('[Stealth] Not logged in, observation not saved');
      return;
    }
    
    const abandonedPhase = completed ? null : getCurrentPhase();
    const result = await lessonObserver.save(supabase, userId, completed, abandonedPhase);
    
    if (result.success) {
      console.debug('[Stealth] Observation saved:', result.sessionId);
    }
  } catch (err) {
    console.error('[Stealth] Save error:', err);
  }
}

// Hook into page unload
let observationSaved = false;
window.addEventListener('beforeunload', () => {
  if (!observationSaved && lessonObserver) {
    saveObservation(false);
  }
});
```

---

## Settings Panel Integration

### Add Learning Journey to Settings

In the settings modal HTML:

```html
<!-- Learning Journey Section -->
<div id="settings-learning-journey">
  <!-- Component loads here -->
</div>
```

Include the component:

```html
<link rel="import" href="/components/learning-journey.html">
<!-- Or just include the HTML directly -->
```

Load insights when settings open:

```javascript
async function openSettings() {
  // ... existing settings code ...
  
  // Load learning journey
  if (window.LearningJourneyComponent && supabase) {
    const container = document.getElementById('settings-learning-journey');
    const journey = new window.LearningJourneyComponent(container, supabase);
    
    const { data } = await supabase.auth.getSession();
    if (data?.session?.user?.id) {
      journey.load(data.session.user.id);
    }
  }
}
```

---

## Testing Checklist

### Local Testing

1. [ ] Observer initializes on lesson load (check console)
2. [ ] Phase starts tracked (check console logs)
3. [ ] Options shown tracked
4. [ ] Choices tracked with quality
5. [ ] Hints tracked
6. [ ] Completion saves to Supabase
7. [ ] Abandonment saves partial data

### Supabase Verification

```sql
-- Check observations being saved
SELECT * FROM learner_observations 
ORDER BY started_at DESC 
LIMIT 10;

-- Check insights computed
SELECT * FROM learner_insights;

-- Manually trigger insight computation
SELECT compute_learner_insights('user-uuid-here');
```

### Privacy Controls

1. [ ] Toggle disables observation (check localStorage)
2. [ ] Export downloads complete JSON
3. [ ] Delete removes all observations and insights

---

## Troubleshooting

### Observations Not Saving

1. Check user is logged in
2. Check RLS policies allow insert
3. Check for console errors
4. Verify session_id uniqueness

### Insights Not Computing

1. Verify at least 3 completed lessons exist
2. Check trigger is active
3. Run manual computation to debug
4. Check function permissions

### Component Not Loading

1. Check Supabase client available
2. Verify user authenticated
3. Check view exists and accessible
4. Look for console errors

---

## Production Checklist

Before deploying:

- [ ] Migration run in production Supabase
- [ ] RLS policies verified
- [ ] Observer script minified and cached
- [ ] Privacy controls working
- [ ] Export/delete functions tested
- [ ] No debug logs in production
- [ ] Analytics tracking added (optional)

---

## Future Enhancements

### Phase 2: Subject Proficiency

Track performance by lesson tags to build subject-specific insights:

```javascript
// After lesson complete, tag analysis
const tags = currentLesson.tags || [];
tags.forEach(tag => {
  updateSubjectProficiency(tag, observationSummary);
});
```

### Phase 3: Adaptive Difficulty

Use insights to adjust content difficulty:

```javascript
const insights = await fetchUserInsights(userId);
if (insights.difficulty_comfort === 'advanced') {
  // Show more challenging variants
}
```

### Phase 4: Kelly Memory

Reference past performance in responses:

```javascript
const patterns = await getUserPatterns(userId);
if (patterns.needsEncouragement) {
  kellyResponse += " You're doing great—I've seen you tackle tough ones before!";
}
```

---

**End of Integration Guide**

*"The best assessments are invisible. The learner just feels understood."*
