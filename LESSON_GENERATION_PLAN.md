# 365 Lesson Generation Plan

## Target: December 17, 2025 Launch

---

## Executive Summary

| Item                | Value                       |
| ------------------- | --------------------------- |
| Lessons to generate | 347                         |
| Days available      | 19                          |
| Strategy            | AI-powered batch generation |
| Estimated cost      | $200-500 API calls          |

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  365_day_calendar.json                                          │
│  (Topics, titles, learning objectives)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  LESSON GENERATOR SCRIPT                                        │
│  - Reads calendar data                                          │
│  - Loads DNA schema template                                    │
│  - Calls AI API for each lesson                                 │
│  - Validates output against schema                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: 365 DNA JSON files                                     │
│  /public/data/lessons/day-001-the-sun.dna.json                  │
│  /public/data/lessons/day-002-habit-stacking.dna.json           │
│  ...                                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Build Generator (Days 1-2)

### Script: `scripts/generate-lessons.js`

```javascript
// Pseudo-code for lesson generation

const BATCH_SIZE = 10; // Generate 10 lessons per API call
const TOTAL_LESSONS = 365;

async function generateAllLessons() {
  const calendar = loadCalendar();
  const schema = loadDNASchema();
  const existingDNA = loadExistingDNA(); // 18 complete lessons

  for (let i = 0; i < TOTAL_LESSONS; i += BATCH_SIZE) {
    const batch = calendar.lessons.slice(i, i + BATCH_SIZE);

    for (const lesson of batch) {
      if (existingDNA[lesson.day]) {
        console.log(`Day ${lesson.day}: Using existing DNA`);
        continue;
      }

      const dna = await generateLessonDNA(lesson, schema);
      await validateDNA(dna, schema);
      await saveDNA(dna, `day-${lesson.day.toString().padStart(3, '0')}.dna.json`);

      // Rate limiting
      await sleep(1000);
    }
  }
}
```

### AI Prompt Template

```markdown
You are generating educational content for Curious Kelly, an AI teacher.

## Lesson Details

- Day: {day}
- Date: {date}
- Topic: {title}
- Learning Objective: {learning_objective}
- Category: {category}

## Required Output Structure

Generate a complete lesson DNA with:

### 1. Age Variants (6 versions)

For each age group, create age-appropriate content:

- 2-5 years: Simple words, wonder, safety
- 6-12 years: Curious explorer, connections
- 13-17 years: Practical applications, careers
- 18-35 years: Professional depth
- 36-60 years: Family, community, legacy
- 61-102 years: Wisdom, reflection, sharing

### 2. Phases (5 per age variant)

- Welcome: Introduction to topic (no choices)
- Q1: First question with 2-3 choices
- Q2: Deeper question with 2-3 choices
- Q3: Application question with 2-3 choices
- Wisdom: Closing wisdom moment (no choices)

### 3. Languages (3 versions per age variant)

- English (en)
- Spanish (es)
- French (fr)

### 4. Required Fields

- core_metaphor for each age
- abstract_concepts translations
- vocabulary (keyTerms)
- tone guidelines
- expression cues for Kelly avatar

Output valid JSON following this exact schema:
{schema}
```

---

## Phase 2: Generation Run (Days 3-12)

### Daily Targets

| Day | Lessons | Cumulative | Notes         |
| --- | ------- | ---------- | ------------- |
| 1   | 35      | 35         | Initial batch |
| 2   | 35      | 70         |               |
| 3   | 35      | 105        |               |
| 4   | 35      | 140        |               |
| 5   | 35      | 175        | Halfway       |
| 6   | 35      | 210        |               |
| 7   | 35      | 245        |               |
| 8   | 35      | 280        |               |
| 9   | 35      | 315        |               |
| 10  | 32      | 347        | Complete      |

### Automation Script

```bash
# Run generation in batches
node scripts/generate-lessons.js --start 1 --end 50 --output ./public/data/lessons/
node scripts/generate-lessons.js --start 51 --end 100 --output ./public/data/lessons/
# ... continue for all batches
```

### Progress Tracking

```javascript
// Progress dashboard
{
  "total": 365,
  "complete": 18,
  "generated": 0,
  "validated": 0,
  "failed": [],
  "lastRun": "2025-11-28T00:00:00Z"
}
```

---

## Phase 3: Validation & QA (Days 13-15)

### Automated Validation

```javascript
function validateLessonDNA(dna) {
  const errors = [];

  // Check all age variants exist
  const requiredAges = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];
  for (const age of requiredAges) {
    if (!dna.ageVariants[age]) {
      errors.push(`Missing age variant: ${age}`);
    }
  }

  // Check all languages exist
  const requiredLangs = ['en', 'es', 'fr'];
  for (const age of requiredAges) {
    for (const lang of requiredLangs) {
      if (!dna.ageVariants[age]?.language?.[lang]) {
        errors.push(`Missing language ${lang} for age ${age}`);
      }
    }
  }

  // Check phases exist
  if (!dna.interactions || dna.interactions.length < 3) {
    errors.push('Missing interaction phases');
  }

  return errors;
}
```

### Sample Review (20% of lessons)

- Randomly select ~70 lessons
- Human review for quality
- Fix systematic issues

---

## Phase 4: Integration (Days 16-17)

### Update learn.html to Load DNA

```javascript
async function loadLessonDNA(dayNumber) {
  const paddedDay = dayNumber.toString().padStart(3, '0');
  const response = await fetch(`/data/lessons/day-${paddedDay}.dna.json`);
  return await response.json();
}
```

### Fallback Strategy

```javascript
// If DNA not found, use simplified content
async function getLesson(dayNumber) {
  try {
    return await loadLessonDNA(dayNumber);
  } catch {
    // Fallback to calendar basic info
    return generateSimplifiedLesson(calendar.lessons[dayNumber - 1]);
  }
}
```

---

## Cost Breakdown

### AI API Costs (Claude/GPT-4)

| Item               | Calculation                  | Cost         |
| ------------------ | ---------------------------- | ------------ |
| Input tokens       | ~2000 tokens × 347 lessons   | ~694K tokens |
| Output tokens      | ~8000 tokens × 347 lessons   | ~2.8M tokens |
| **Total (Claude)** | $3/1M input + $15/1M output  | ~$45         |
| **Total (GPT-4)**  | $30/1M input + $60/1M output | ~$190        |

**Buffer for retries:** 2x = **$100-400 total**

### Time Investment

| Task                       | Hours           |
| -------------------------- | --------------- |
| Build generator script     | 4-6             |
| Run generation (monitored) | 10-15           |
| Validation & fixes         | 8-10            |
| Integration & testing      | 4-6             |
| **Total**                  | **26-37 hours** |

---

## Risk Mitigation

### 1. API Rate Limits

- Use exponential backoff
- Batch requests appropriately
- Have backup API key

### 2. Content Quality Issues

- Validate JSON schema on every output
- Log all failures for manual review
- Have human review high-stakes lessons

### 3. Translation Quality

- Use native language examples in prompts
- Review sample of each language
- Mark for post-launch improvement

### 4. Time Overrun

- Priority order: Dec lessons first (Days 335-365)
- Then Nov (Days 305-334)
- Then work backwards

---

## Files to Create

```
scripts/
├── generate-lessons.js       # Main generator
├── validate-lessons.js       # Validation script
├── generation-progress.json  # Progress tracker
└── prompts/
    └── lesson-generator.md   # AI prompt template

public/data/
├── lessons/                  # Generated DNA files
│   ├── day-001.dna.json
│   ├── day-002.dna.json
│   └── ...
└── schema/
    └── lesson-dna-schema.json
```

---

## Decision Points for You

1. **API Choice**: Claude (cheaper) or GPT-4 (more consistent)?
2. **Priority Order**: December first or start from Day 1?
3. **Quality Bar**: Full review or spot-check only?
4. **Languages**: All 3 from start or English-first?

---

## Next Steps

If you approve this plan:

1. **Today**: I build the generator script
2. **Tomorrow**: Test with 10 lessons, you review quality
3. **Days 3-12**: Full generation run
4. **Days 13-17**: Validation & integration

**Ready to start?**
