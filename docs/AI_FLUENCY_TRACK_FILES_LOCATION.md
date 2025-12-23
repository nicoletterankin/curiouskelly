# AI Fluency (Grow) Track - File Locations

**Date:** December 23, 2025  
**Status:** ✅ Found all files

---

## 📁 File Locations

### 1. **Monthly Curriculum JSON Files** (12 files)
**Location:** `lessons/year2-ai-fluency/`

**Files:**
- `january_curriculum.json` (Days 1-31)
- `february_curriculum.json` (Days 32-59)
- `march_curriculum.json` (Days 60-90)
- `april_curriculum.json` (Days 91-120)
- `may_curriculum.json` (Days 121-151)
- `june_curriculum.json` (Days 152-181)
- `july_curriculum.json` (Days 182-212)
- `august_curriculum.json` (Days 213-243)
- `september_curriculum.json` (Days 244-273)
- `october_curriculum.json` (Days 274-304)
- `november_curriculum.json` (Days 305-334)
- `december_curriculum.json` (Days 335-365)

**Structure:**
```json
{
  "year": 2,
  "program": "AI Fluency & Meta-Learning",
  "month": "January",
  "theme": "Foundations",
  "themeDescription": "What is AI? What am I? What makes humans special?",
  "days": [
    {
      "day": 1,
      "date": "January 1",
      "title": "I'm an AI - Understanding Your Digital Learning Partner",
      "learning_objective": "Develop foundational AI literacy..."
    },
    // ... more days
  ]
}
```

**Total:** 365 days of curriculum topics

---

### 2. **Individual Detailed Lesson JSON** (1 file)
**Location:** `lessons/year2-ai-fluency/day-001-im-an-ai.json`

**Content:**
- Complete lesson structure with phases
- Age variants (2-5, 6-12, 13-17, 18-35, 36-60, 61-102)
- Socratic questions
- Interactive choices
- Parent companion guide
- Teacher guide
- Recall prompts

**Status:** ✅ Day 1 complete, detailed format

---

### 3. **Complete Lesson Packs** (365 JavaScript files)
**Location:** `public/data/day-XXX-complete.js`

**Example:** `public/data/day-001-complete.js`

**Structure:**
```javascript
window.CURIOUS_KELLY.DAY_001 = {
  "meta": { "day_number": 1, ... },
  
  // LEARN TRACK
  "lesson": {
    "topic": "Starting Fresh",
    "headline": "...",
    ...
  },
  
  // GROW TRACK (AI Fluency)
  "grow": {
    "topic": "I'm an AI - Understanding Your Digital Learning Partner",
    "objective": "Develop foundational AI literacy..."
  },
  
  "ageVariants": { ... }
};
```

**Status:** ✅ All 365 files exist  
**Grow Track:** Included in `grow` object in each file

---

### 4. **Overview Document**
**Location:** `lessons/year2-ai-fluency/YEAR2_AI_FLUENCY_OVERVIEW.md`

**Content:**
- Complete 365-day curriculum roadmap
- Monthly themes
- Daily topics with universal truths
- Teacher integration guide
- Connection to Year 1

---

## 📊 Current Status

### ✅ Complete
- **12 monthly curriculum JSON files** - All topics defined
- **365 complete lesson packs** - Grow track data included
- **1 detailed lesson** - Day 1 fully developed
- **Overview document** - Complete roadmap

### ⚠️ Partial
- **Individual detailed lessons** - Only Day 1 has full detail
- **Grow track content** - Topics defined, but detailed phases/content only for Day 1

### ❌ Missing
- **HTML files** - No dedicated HTML files for Grow track lessons
- **Detailed lesson JSON** - Days 2-365 need detailed lesson structure (like Day 1)

---

## 🔍 How to Access Grow Track Data

### From Complete Lesson Packs:
```javascript
// Access Grow track for any day
const day = window.CURIOUS_KELLY.LOCAL_PACKS[1]; // Day 1
const growTopic = day.grow.topic;
const growObjective = day.grow.objective;
```

### From Monthly Curriculum Files:
```javascript
// Load monthly curriculum
const january = await fetch('/lessons/year2-ai-fluency/january_curriculum.json');
const data = await january.json();
const day1 = data.days.find(d => d.day === 1);
```

### From Detailed Lesson (Day 1 only):
```javascript
// Load detailed Day 1 lesson
const day1Detailed = await fetch('/lessons/year2-ai-fluency/day-001-im-an-ai.json');
const lesson = await day1Detailed.json();
```

---

## 📝 File Summary

| Type | Location | Count | Status |
|------|----------|-------|--------|
| Monthly Curriculum JSON | `lessons/year2-ai-fluency/*.json` | 12 | ✅ Complete |
| Detailed Lesson JSON | `lessons/year2-ai-fluency/day-001-im-an-ai.json` | 1 | ✅ Day 1 only |
| Complete Lesson Packs | `public/data/day-XXX-complete.js` | 365 | ✅ All days |
| Overview Document | `lessons/year2-ai-fluency/YEAR2_AI_FLUENCY_OVERVIEW.md` | 1 | ✅ Complete |
| HTML Files | N/A | 0 | ❌ None |

---

## 🎯 Next Steps

1. **Generate detailed lesson JSON** for Days 2-365 (like Day 1)
2. **Create HTML templates** for Grow track lessons (if needed)
3. **Populate Grow track content** in complete lesson packs (currently only topic/objective)

---

**Status:** ✅ All curriculum files found  
**Grow Track Topics:** 365/365 defined  
**Detailed Lessons:** 1/365 complete

