# Dual-Track Lesson Architecture - Complete Directive
**Last Updated:** December 22, 2025  
**Purpose:** Complete understanding of Learn + Grow dual-track system for expert-level work

---

## 🎯 Core Architecture: One Lesson, Two Tracks

**Every lesson has TWO independent tracks:**

1. **Learn Track** (`lesson.topic`)
   - Traditional educational content
   - Universal topics (science, history, nature, etc.)
   - 7 phases: Hook → Cliff → Fact1 → Fact2 → Fact3 → Wisdom → Outro
   - Stored in: `lesson.topic`, `lesson.atoms[]`

2. **Grow Track** (`grow.topic`)
   - AI fluency & digital literacy
   - AI-specific topics (understanding AI, communicating with AI, etc.)
   - Same 7-phase structure
   - Stored in: `grow.topic`, `grow.objective`, `grow.activity`

**Both tracks share:**
- Same day number (1-365)
- Same date
- Same completion tracking
- Same user progress

---

## 📦 Storage Structure

### Static Files (`day-XXX-complete.js`)

```javascript
window.CURIOUS_KELLY.DAY_001 = {
  meta: {
    day_number: 1,
    version: "v3.0-skeleton"
  },
  
  // LEARN TRACK
  lesson: {
    day_number: 1,
    topic: "Starting Fresh",           // Learn topic
    headline: "New beginnings...",
    universal_truth: "New beginnings...",
    emoji: "🍁",
    category: "general"
  },
  atoms: [
    {
      phase: "Hook",
      content: { script: "..." }       // Learn track content
    },
    // ... 7 phases for Learn track
  ],
  
  // GROW TRACK
  grow: {
    topic: "I'm an AI - Understanding Your Digital Learning Partner",  // Grow topic
    objective: "Develop foundational AI literacy...",
    activity: "Practice asking AI questions..."
  },
  
  // Age variants (applies to both tracks)
  ageVariants: {
    "2-5": { persona: "Playful Friend", phases: {...} },
    // ... 6 age buckets
  }
}
```

**Key Points:**
- `lesson.topic` = Learn track topic
- `grow.topic` = Grow track topic
- Both tracks have same day number
- Both tracks have same phase structure
- Age variants apply to both tracks

---

## 🔄 Loading Priority

### For Learn Track:
1. `LOCAL_PACKS[dayNum].lesson` → Learn topic + atoms
2. Supabase `core_lessons` + `lesson_atoms` (track='learn')
3. Fallbacks

### For Grow Track:
1. `LOCAL_PACKS[dayNum].grow` → Grow topic + objective
2. Supabase `core_lessons` + `lesson_atoms` (track='grow')
3. Fallbacks

**Function:** `KellyLessonLoader.loadLesson(dayNumber, { track: 'learn' | 'grow' })`

---

## 📊 Completeness Metrics

### Learn Track Completeness:
- **Base Content** (Required):
  - ✅ Topic present (`lesson.topic`)
  - ✅ 7 phases with scripts (`atoms[]` with 7 phases)
  - ✅ Universal truth (`lesson.universal_truth`)
  
- **Enhanced Content** (Optional):
  - ✅ HD videos (`atoms[].hd_video_url`)
  - ✅ Visuals (`atoms[].visual_url`)
  - ✅ Age variants (`ageVariants`)
  - ✅ Multiple archetypes (The Scientist, The Explorer, etc.)

### Grow Track Completeness:
- **Base Content** (Required):
  - ✅ Topic present (`grow.topic`)
  - ✅ Objective (`grow.objective`)
  - ✅ Activity (`grow.activity`)
  
- **Enhanced Content** (Optional):
  - ✅ Full phase content (if available)
  - ✅ BYOK prompts
  - ✅ Practice exercises

### Overall Lesson Completeness:
```
Completeness = (
  (Learn base content ? 40% : 0%) +
  (Learn enhanced content ? 20% : 0%) +
  (Grow base content ? 30% : 0%) +
  (Grow enhanced content ? 10% : 0%)
)
```

**Status Levels:**
- **Skeleton** (0-40%): Base Learn content only
- **Basic** (40-60%): Learn + Grow base content
- **Complete** (60-80%): Base + some enhanced content
- **Production** (80-100%): All content + videos + visuals

---

## 🎨 Display Requirements

### Homepage Calendar Double-Click Popup:

**Current Problem:** Full-screen audit covers everything

**New Design:** Compact preview card showing:
1. **Header:**
   - Day number + date
   - Learn topic (with emoji)
   - Grow topic (with emoji)

2. **Completeness Indicators:**
   - Progress bar (0-100%)
   - Status badge (Skeleton/Basic/Complete/Production)
   - Quick stats:
     - ✅ Learn: Base | Enhanced
     - ✅ Grow: Base | Enhanced
     - ✅ Videos: X/7 phases
     - ✅ Visuals: X/7 phases

3. **Quick Actions:**
   - "Start Learn Track →"
   - "Start Grow Track →"
   - "View Full Details" (opens full audit)

**Size:** Max 600px wide, scrollable if needed

---

## 🔧 Implementation Functions

### Calculate Completeness:
```javascript
function calculateLessonCompleteness(dayNumber) {
  const pack = window.CURIOUS_KELLY?.LOCAL_PACKS?.[dayNumber];
  if (!pack) return { completeness: 0, status: 'missing' };
  
  let score = 0;
  const checks = {
    learnBase: false,
    learnEnhanced: false,
    growBase: false,
    growEnhanced: false
  };
  
  // Learn base (40%)
  if (pack.lesson?.topic && pack.atoms?.length >= 7) {
    checks.learnBase = true;
    score += 40;
  }
  
  // Learn enhanced (20%)
  if (pack.atoms?.some(a => a.hd_video_url || a.visual_url)) {
    checks.learnEnhanced = true;
    score += 20;
  }
  
  // Grow base (30%)
  if (pack.grow?.topic && pack.grow?.objective) {
    checks.growBase = true;
    score += 30;
  }
  
  // Grow enhanced (10%)
  if (pack.grow?.activity) {
    checks.growEnhanced = true;
    score += 10;
  }
  
  const status = score >= 80 ? 'production' :
                 score >= 60 ? 'complete' :
                 score >= 40 ? 'basic' : 'skeleton';
  
  return { completeness: score, status, checks };
}
```

### Show Preview Popup:
```javascript
function showLessonPreview(dayNumber) {
  const pack = window.CURIOUS_KELLY?.LOCAL_PACKS?.[dayNumber];
  const completeness = calculateLessonCompleteness(dayNumber);
  
  // Build compact preview card
  // Display Learn + Grow topics
  // Show completeness indicators
  // Provide quick actions
}
```

---

## ✅ Best Practices

### When Loading Lessons:
1. **Always check both tracks:**
   ```javascript
   const learnLesson = await loadLesson(dayNum, { track: 'learn' });
   const growLesson = await loadLesson(dayNum, { track: 'grow' });
   ```

2. **Display both topics:**
   ```javascript
   const learnTopic = pack.lesson?.topic || 'Loading...';
   const growTopic = pack.grow?.topic || 'Loading...';
   ```

3. **Track completion separately:**
   ```javascript
   state.completedLessons.learn = [...];
   state.completedLessons.grow = [...];
   ```

### When Displaying Lessons:
1. **Show dual-track preview:**
   - Learn topic (primary)
   - Grow topic (secondary)
   - Completeness for both

2. **Provide track selection:**
   - "Start Learn Track"
   - "Start Grow Track"
   - Track toggle in lesson player

3. **Calculate completeness accurately:**
   - Check base content first
   - Then enhanced content
   - Show clear status

---

## 🎯 Success Criteria

- [x] Understand dual-track structure
- [x] Know where Learn/Grow topics are stored
- [x] Can calculate completeness
- [x] Can display both tracks
- [x] Can show compact preview
- [x] Can provide track selection

---

**Status:** ✅ Architecture understood and documented





