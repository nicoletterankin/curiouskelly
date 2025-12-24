# Curious Kelly Lesson Architecture - Expert Guide
**Last Updated:** December 22, 2025  
**Purpose:** Complete understanding of how lessons are stored, loaded, and displayed

---

## 🎯 Core Principle: Offline-First Architecture

**The lesson system is designed to work 100% offline.** Static lessons are hardcoded base parts that ensure every lesson plays, even without internet.

---

## 📦 Lesson Storage Architecture

### 1. Static Base Lessons (Hardcoded - Priority 1)

**Location:** `/public/data/day-XXX-complete.js` (365 files)

**Structure:**
```javascript
window.CURIOUS_KELLY = window.CURIOUS_KELLY || {};
window.CURIOUS_KELLY.LOCAL_PACKS = window.CURIOUS_KELLY.LOCAL_PACKS || {};
window.CURIOUS_KELLY.DAY_001 = {
  meta: {
    day_number: 1,
    version: "v3.0-skeleton",
    is_skeleton: true
  },
  lesson: {
    day_number: 1,
    topic: "Starting Fresh",
    headline: "New beginnings offer opportunities...",
    universal_truth: "New beginnings offer opportunities...",
    emoji: "🍁",
    category: "general"
  },
  atoms: [
    {
      id: "day001-hook-001",
      phase: "Hook",
      content: {
        script: "Welcome to Day 1!...",
        kellyPose: "welcome",
        kellyEmotion: "curious"
      }
    },
    // ... more phases
  ],
  grow: {
    topic: "I'm an AI - Understanding Your Digital Learning Partner",
    objective: "Develop foundational AI literacy..."
  },
  ageVariants: {
    "2-5": { persona: "Playful Friend", phases: {...} },
    "6-12": { persona: "Cool Big Sister", phases: {...} },
    // ... more age variants
  }
}
```

**Why Hardcoded:**
- ✅ **Offline-first**: Works without internet
- ✅ **Deterministic**: Same content every time
- ✅ **Fast**: No network latency
- ✅ **Bulletproof**: Never fails to load
- ✅ **Base layer**: Supabase variants enrich, don't replace

**Access Pattern:**
```javascript
const localPacks = window.CURIOUS_KELLY?.LOCAL_PACKS || {};
const pack = localPacks[`day-${String(dayNum).padStart(3, '0')}`] || 
             localPacks[dayNum] || 
             localPacks[String(dayNum)];
```

---

### 2. Curriculum Metadata (JSON Files - Priority 2)

**Location:** `/public/data/curriculum/year1-foundations/` and `/year2-ai-fluency/`

**Structure:**
```json
{
  "month": "january",
  "days": [
    {
      "day": 1,
      "title": "Starting Fresh",
      "category": "general"
    },
    // ... more days
  ]
}
```

**Purpose:**
- Provides Learn/Grow track titles for calendar views
- Used when LOCAL_PACKS doesn't have topic
- Faster than loading full lesson packs for lists

**Access Pattern:**
```javascript
fetch(`/data/curriculum/year1-foundations/${month}_curriculum.json`)
```

---

### 3. Supabase Dynamic Variants (Priority 3+)

**Tables:**
- `core_lessons`: Base lesson metadata (365 rows)
- `lesson_atoms`: Archetype-specific content (21,915 rows)
- `lesson_shards`: Age/region personalization (38,700 rows)

**Purpose:**
- **Enrichment**: Adds HD videos, visuals, enhanced scripts
- **Variants**: Different archetypes (The Scientist, The Explorer, etc.)
- **Personalization**: Age-specific content (2-5, 6-12, 13-17, 18-35, 36-60, 61-102)
- **Optional**: System works without it

**Access Pattern:**
```javascript
// Via KellyLessonLoader
const payload = await window.KellyLessonLoader.loadLesson(dayNumber, {
  archetype: 'The Scientist',
  age: 30,
  region: 'adult'
});
```

---

## 🔄 Lesson Loading Priority Chain

### KellyLessonLoader.loadLesson() Priority:

1. **On-Demand Generation** (if URL param `?topic=...`)
   - Client-side generated lesson
   - Used for custom topics

2. **LOCAL_PACKS** (Static - Offline-First)
   ```javascript
   const localPack = window.CURIOUS_KELLY?.LOCAL_PACKS?.[packKey];
   if (localPack && (localPack.lesson || localPack.atoms)) {
     return { lesson: localPack.lesson, atoms: localPack.atoms, source: 'local_pack' };
   }
   ```
   - ✅ Always checked first for Learn track
   - ✅ Works offline
   - ✅ Deterministic

3. **Supabase** (Dynamic - Network)
   ```javascript
   const { data: lesson } = await supabase
     .from('core_lessons')
     .select('*')
     .eq('day_number', dayNumber)
     .single();
   ```
   - Adds HD videos, visuals
   - Provides archetype variants
   - Enriches static base

4. **Cloudflare D1** (Mirror - Network)
   - Backup Supabase mirror
   - Currently disabled

5. **Static JSON** (Pre-exported - Network)
   - Fallback export files

6. **Emergency Fallback** (Hardcoded)
   - Last resort
   - Always works

---

## 📊 How Lessons Are Displayed

### Journey Panel (Calendar View)

**Function:** `populateJourneyPanel()` (Line 14660)

**Data Sources (Priority):**
1. **LOCAL_PACKS** - Static hardcoded lessons
   ```javascript
   const pack = localPacks[`day-${String(d).padStart(3, '0')}`];
   if (pack?.lesson) {
     lessons.push({
       day_number: d,
       learnTopic: pack.lesson.topic,
       growTopic: pack.grow?.topic,
       topic: pack.lesson.topic,
       emoji: pack.lesson.emoji
     });
   }
   ```

2. **Curriculum JSON** - Month-based metadata
   ```javascript
   const curriculumLessons = await loadWeekLessons(startDay);
   ```

3. **Supabase** - Asset previews (thumbnails, videos)
   ```javascript
   const { data: assets } = await supabase
     .from('lesson_atoms')
     .select('visual_url, hd_video_url')
     .in('core_lesson_id', lessonIds);
   ```

**Display:**
- Day number + date
- Learn track topic
- Grow track topic
- Preview thumbnail (if available)
- Completion status
- Asset count badge

---

### Lesson Player

**Function:** `loadLessonRuntime()` (Line 12515)

**Flow:**
1. Calls `KellyLessonLoader.loadLesson(dayNumber)`
2. Priority chain executes (LOCAL_PACKS → Supabase → ...)
3. `applyLoadedLesson()` processes payload
4. `renderPhase()` displays content

**Key Variables:**
- `currentLesson`: Current lesson metadata
- `lessonAtoms`: Array of phase content (7 phases)
- `state.currentDay`: Day number (1-365)
- `state.currentPhase`: Phase index (0-6)

---

## 🎨 Why We Hardcode Base Parts

### 1. Offline-First Philosophy
- **Every lesson must play offline**
- Static files ensure this
- Network is optional enrichment

### 2. Performance
- **No network latency** for base content
- Instant lesson loading
- Better user experience

### 3. Reliability
- **Never fails** to load base lesson
- Supabase can be down, lessons still work
- Bulletproof architecture

### 4. Deterministic Content
- **Same content every time** (base layer)
- No surprises
- Predictable experience

### 5. Base Layer Pattern
- **Static = Base**: Universal content
- **Supabase = Variants**: Archetype/age personalization
- **Clear separation**: Base vs. enrichment

---

## 🔍 Key Functions Reference

### Loading Functions

**`loadLessonRuntime(dayNumber)`** (Line 12515)
- Main entry point for loading a lesson
- Calls `KellyLessonLoader.loadLesson()`
- Processes result with `applyLoadedLesson()`

**`loadWeekLessons(startDay)`** (Line 12770)
- Loads lesson titles from curriculum JSON
- Used for calendar/week views
- Returns array of `{ day_number, learnTopic, growTopic }`

**`populateJourneyPanel()`** (Line 14660)
- Populates journey panel with lessons
- Checks LOCAL_PACKS first, then curriculum JSON
- Displays Learn/Grow topics, thumbnails, status

**`KellyLessonLoader.loadLesson(dayNumber, options)`** (`kelly-lesson-loader.js`)
- Canonical lesson loader
- Implements priority chain
- Returns `{ lesson, atoms, shards, source }`

### Display Functions

**`renderPhase(phaseIndex)`**
- Renders a specific phase
- Uses `lessonAtoms[phaseIndex]`
- Displays script, visuals, choices

**`showPhaseSelector(dayNumber)`** (Line 19378)
- Shows phase selection modal
- Loads lesson from cache or `loadWeekLessons()`
- Allows jumping to specific phase

**`buildGridView()`** (Line 19117)
- Builds calendar grid view
- Uses `lessonsCache` and LOCAL_PACKS
- Displays thumbnails, topics, completion status

---

## 📁 File Structure

```
public/
├── data/
│   ├── day-001-complete.js      # Static lesson 1
│   ├── day-002-complete.js      # Static lesson 2
│   ├── ...                      # ... 365 total
│   ├── day-365-complete.js      # Static lesson 365
│   └── curriculum/
│       ├── year1-foundations/   # Learn track metadata
│       │   ├── january_curriculum.json
│       │   └── ... (12 months)
│       └── year2-ai-fluency/    # Grow track metadata
│           ├── january_curriculum.json
│           └── ... (12 months)
├── js/
│   └── kelly-lesson-loader.js   # Canonical loader
└── learn.html                   # Main lesson player
```

---

## 🎯 Best Practices

### When Adding New Lessons

1. **Create static pack** (`day-XXX-complete.js`)
   - Base content (topic, atoms, phases)
   - Age variants (2-5, 6-12, etc.)
   - Grow track content

2. **Add to curriculum JSON** (optional)
   - For calendar metadata
   - Faster list loading

3. **Enrich in Supabase** (optional)
   - HD videos
   - Visuals
   - Enhanced scripts

### When Loading Lessons

1. **Always check LOCAL_PACKS first**
   ```javascript
   const pack = window.CURIOUS_KELLY?.LOCAL_PACKS?.[packKey];
   ```

2. **Use KellyLessonLoader for runtime**
   ```javascript
   const payload = await window.KellyLessonLoader.loadLesson(dayNumber);
   ```

3. **Cache results**
   ```javascript
   lessonsCache[dayNumber] = lesson;
   ```

### When Displaying Lessons

1. **Journey Panel**: Check LOCAL_PACKS → curriculum JSON → Supabase
2. **Lesson Player**: Use `loadLessonRuntime()` → `KellyLessonLoader`
3. **Calendar View**: Use `lessonsCache` + LOCAL_PACKS

---

## ✅ Architecture Checklist

- [x] Static lessons hardcoded in day-XXX-complete.js
- [x] LOCAL_PACKS checked first (offline-first)
- [x] Supabase enriches, doesn't replace
- [x] Journey panel displays static lessons
- [x] Calendar view uses LOCAL_PACKS
- [x] Lesson player works offline
- [x] Priority chain documented
- [x] Functions reference complete

---

**Status:** ✅ Architecture understood and documented


