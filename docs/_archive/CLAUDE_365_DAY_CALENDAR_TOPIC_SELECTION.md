# 365-Day Calendar Topic Selection Guide for Claude

## ⚠️ CRITICAL CLARIFICATION

**The 30-day curriculum in `lesson-authoring-guide.md` is EXAMPLE/DUMMY topics only.**

- ❌ **DO NOT** use the 30-day curriculum as your source of truth
- ❌ **DO NOT** assume those topics need to be in the 365-day calendar
- ✅ **DO** use the **365-day calendar** (`lessons/365_day_calendar.json`) as your source of truth
- ✅ **DO** check the calendar before creating a lesson to see if the topic already exists

---

## 📅 The 365-Day Calendar is Your Source of Truth

**File:** `lessons/365_day_calendar.json`

This file contains **all 365 lessons** for the entire year. Each lesson entry includes:
- `day`: Day number (1-365)
- `date`: Calendar date (e.g., "January 1", "November 15")
- `title`: Lesson title
- `lesson_id`: URL-friendly identifier
- `learning_objective`: What learners will understand
- `has_dna`: Whether a DNA file exists yet (`true`/`false`)
- `dna_file`: Name of DNA file if it exists
- `category`: Topic category
- `tags`: Related tags

**When creating a lesson:**
1. **First, check** `lessons/365_day_calendar.json` to see if the topic already exists
2. **If it exists:** Use that entry's `lesson_id` and `dna_file` name
3. **If it doesn't exist:** The topic may not be in the calendar yet (check with the team)

---

## 🎯 Universal Topic Selection Criteria

### ✅ **EXCELLENT Universal Topics** (Use These)

**High-quality universal topics MUST:**

1. **Be Age-Less** (meaningful to ages 2-102)
   - ✅ Observable by toddlers AND relatable to elders
   - ✅ Has depth that scales from simple to profound
   - ❌ NOT age-specific (e.g., "retirement planning", "dating", "homework")

2. **Be Observable/Experiential**
   - ✅ Can be seen, felt, or experienced in daily life
   - ✅ Natural phenomena everyone encounters
   - ❌ NOT abstract concepts without real-world connection

3. **Have Substantive Educational Value**
   - ✅ Teaches valuable principles, skills, or understanding
   - ✅ Connects to deeper concepts (science, philosophy, human nature)
   - ❌ NOT too simple or shallow (e.g., "play" without depth)

4. **Be Culturally Universal**
   - ✅ Meaningful across cultures and backgrounds
   - ✅ Not tied to specific holidays, traditions, or regions
   - ❌ NOT culturally specific (unless explained universally)

5. **Inspire Wonder and Curiosity**
   - ✅ Creates "aha!" moments
   - ✅ Connects to universal human experiences
   - ❌ NOT purely technical or dry

### 📋 **Topic Categories (Prioritize These)**

#### 1. **Natural Phenomena** (Highest Priority)
Observable, educational, universally experienced:

- ✅ **Clouds** - The sky's changing shapes (weather, water cycle)
- ✅ **Rain** - Water's journey from sky to earth (precipitation, life-giving)
- ✅ **Rainbow** - Light's beautiful display (refraction, color spectrum)
- ✅ **Wind** - The invisible force we feel (air pressure, movement)
- ✅ **Trees** - Nature's living pillars (growth, seasons, ecosystems)
- ✅ **Flowers** - Nature's beautiful invitations (pollination, color, fragrance)
- ✅ **Shadows** - Light's absence creates shape (geometry, time, perspective)
- ✅ **Reflections** - When surfaces mirror the world (light, symmetry, perception)
- ✅ **Stars** - What stars are and why they twinkle (astronomy, light)
- ✅ **The Sun** - Our magnificent life-giving star (energy, life, physics)

**Why these work:** Everyone sees/experiences these, regardless of age or culture.

#### 2. **Universal Human Experiences** (High Priority)
Deep, meaningful, educational:

- ✅ **Cooperation** - How we work together (collaboration, mutual benefit, systems thinking)
- ✅ **Curiosity** - The engine of discovery (questioning, exploration, scientific thinking)
- ✅ **Observation** - Learning by paying attention (mindfulness, scientific method, awareness)
- ✅ **Reflection** - Understanding what we've learned (metacognition, self-awareness, growth)
- ✅ **Connection** - How we share ideas and feelings (communication, empathy, relationships)
- ✅ **Community** - How groups create together (social dynamics, belonging, collective action)
- ✅ **Persistence** - Continuing when it's hard (resilience, growth mindset, determination)
- ✅ **Legacy** - What we leave behind (intergenerational thinking, values, impact)
- ✅ **Compassion** - Understanding and helping others (empathy, care, social responsibility)
- ✅ **Trust** - The foundation of relationships (reliability, safety, social bonds)

**Why these work:** Universal human capacities that deepen with age and experience.

#### 3. **Time & Cycles** (Medium Priority)
Observable patterns everyone experiences:

- ✅ **Seasons** - Nature's repeating patterns (cycles, change, adaptation)
- ✅ **Day and Night** - Earth's daily rhythm (rotation, light, circadian rhythms)
- ✅ **Growth** - How living things change over time (development, transformation)
- ✅ **Cycles** - Patterns that repeat (predictability, systems, nature)

**Why these work:** Fundamental experiences of time and change.

#### 4. **Perception & Understanding** (Medium Priority)
Deep concepts accessible at all levels:

- ✅ **Perspective** - Seeing things from different views (empathy, critical thinking)
- ✅ **Pattern** - Recognizing order in the world (mathematics, observation, prediction)
- ✅ **Change** - How nothing stays the same (impermanence, adaptation, acceptance)
- ✅ **Transformation** - Becoming something new (metamorphosis, growth, potential)

**Why these work:** Concepts that can be understood simply or deeply.

---

## ❌ **AVOID These Types of Topics**

### 1. **Age-Specific Content**
- ❌ Retirement planning
- ❌ Dating/relationships (too specific)
- ❌ Homework/studying
- ❌ Career-specific skills
- ❌ Age-restricted activities

### 2. **Overly Technical/Specialized**
- ❌ Advanced mathematics (calculus, linear algebra)
- ❌ Specialized professional skills
- ❌ Complex scientific theories without observable basis
- ❌ Technology-specific implementations

### 3. **Culturally Specific**
- ❌ Specific holidays (unless explained universally)
- ❌ Regional traditions (unless universal principles)
- ❌ Religious practices (unless universal spiritual concepts)
- ❌ National/ethnic-specific customs

### 4. **Too Simple/Shallow**
- ❌ Basic concepts without depth (e.g., "play" without exploration)
- ❌ Single-word topics without substance
- ❌ Topics that can't scale from toddler to elder understanding

### 5. **Controversial Topics**
- ❌ Politics (unless universal democratic principles)
- ❌ Religion (unless universal spiritual concepts)
- ❌ Divisive social issues
- ❌ Commercial products/brands

---

## 🔍 How to Check if a Topic Exists

### Method 1: Search the Calendar File
```bash
# Search for topic keywords in 365_day_calendar.json
grep -i "your-topic-keyword" lessons/365_day_calendar.json
```

### Method 2: Check by Day/Date
- If you know the date (e.g., "November 15"), find that day in the calendar
- The calendar is organized by day number (1-365)

### Method 3: Check by Category
- Topics are categorized (science, general, etc.)
- Search within categories if you know the type

---

## 📊 Current Calendar Status

**As of latest update:**
- **Total lessons:** 365 (complete calendar)
- **Lessons with DNA:** ~43 lessons (11.8% coverage)
- **Target:** 100+ DNA lessons (27%+ coverage)

**Categories in calendar:**
- Science
- General
- Technology (being reduced)
- Arts
- Health/Wellness
- Social Sciences
- Mathematics
- Philosophy/Ethics

---

## ✅ Quality Checklist Before Creating a Lesson

Before selecting a topic, verify:

- [ ] **Age-less:** Can a 3-year-old experience/observe this?
- [ ] **Age-less:** Can an 80-year-old relate to this?
- [ ] **Observable:** Is this something people can see, feel, or experience?
- [ ] **Educational:** Does this teach valuable principles or skills?
- [ ] **Deep:** Can this be explored at multiple complexity levels?
- [ ] **Universal:** Is this culturally universal?
- [ ] **Wonder:** Does this inspire curiosity and wonder?
- [ ] **Not duplicate:** Does this topic already exist in the calendar?
- [ ] **Not too technical:** Can this be understood without specialized knowledge?
- [ ] **Not controversial:** Is this safe and appropriate for all ages?

---

## 🎯 Topic Selection Process

### Step 1: Identify Need
- Check `lessons/365_day_calendar.json` for gaps
- Look for lessons with `has_dna: false` that need content
- Identify high-priority universal topics missing from calendar

### Step 2: Verify Universal Quality
- Run through quality checklist above
- Ensure topic works for ages 2-102
- Verify it's observable/experiential

### Step 3: Check Calendar
- Search calendar for existing topic
- If exists, use that entry's metadata
- If doesn't exist, verify it should be added

### Step 4: Create Lesson DNA
- Use existing DNA files as templates
- Follow schema in `lesson-dna-schema.json`
- Include all 6 age variants and 3 languages

---

## 📚 Reference Files

**Primary Sources:**
- `lessons/365_day_calendar.json` - **YOUR SOURCE OF TRUTH** for all 365 lessons
- `lessons/HIGH_QUALITY_REPLACEMENT_TOPICS.md` - Examples of excellent universal topics
- `lessons/TOPIC_IMPROVEMENT_PLAN.md` - Strategy for improving topic quality

**Schema & Templates:**
- `content-agent-base/lesson-dna-schema.json` - JSON schema for lessons
- `content-agent-base/lesson-template.json` - Starting template
- `lessons/the-sun-dna.json` - Complete example lesson

**Guides:**
- `content-agent-base/CONTENT_AGENT_ONBOARDING.md` - Full onboarding guide
- `content-agent-base/lesson-authoring-guide.md` - Writing guide (ignore 30-day curriculum section)

---

## 🚫 Common Mistakes to Avoid

1. ❌ **Using 30-day curriculum as source** - It's just examples!
2. ❌ **Creating duplicate topics** - Always check calendar first
3. ❌ **Selecting age-specific topics** - Must work for 2-102
4. ❌ **Choosing overly technical topics** - Must be accessible
5. ❌ **Picking culturally specific topics** - Must be universal
6. ❌ **Selecting shallow topics** - Must have educational depth

---

## ✅ Summary: Your Topic Selection Rules

1. **Source of Truth:** `lessons/365_day_calendar.json` (NOT the 30-day curriculum)
2. **Check First:** Always verify topic doesn't already exist
3. **Quality Criteria:** Age-less, observable, educational, universal, inspiring
4. **Priority Categories:** Natural phenomena > Universal experiences > Time/cycles > Perception
5. **Avoid:** Age-specific, overly technical, culturally specific, controversial, shallow topics

---

**Remember:** The 30-day curriculum in `lesson-authoring-guide.md` is **EXAMPLE TOPICS ONLY**. Your real source of truth is the **365-day calendar**. Always check there first!








