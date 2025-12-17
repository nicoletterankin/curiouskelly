# Lesson Content Generation Plan

## Objective
Generate lesson JSON files for Days 352-365 (Dec 18-31) to enable HeyGen video production.

---

## Topics from 365-Day Calendar

| Day | Date | Topic | Category |
|-----|------|-------|----------|
| 352 | Dec 18 | Quieting Your Thoughts | Mindfulness |
| 353 | Dec 19 | Being Where You Are | Presence |
| 354 | Dec 20 | What Makes You Come Alive | Purpose |
| 355 | Dec 21 | Why You Get Up | Motivation |
| 356 | Dec 22 | What Makes Life Matter | Meaning |
| 357 | Dec 23 | Deciding Right From Wrong | Ethics |
| 358 | Dec 24 | What You Care About Most | Values |
| 359 | Dec 25 | Who You Are When No One's Looking | Identity |
| 360 | Dec 26 | What You Leave Behind | Legacy |
| 361 | Dec 27 | Looking Back to Learn | Reflection |
| 362 | Dec 28 | Marking What Matters | Celebration |
| 363 | Dec 29 | Appreciating What You Have | Gratitude |
| 364 | Dec 30 | Starting Fresh | New Beginnings |
| 365 | Dec 31 | 365 Days of Growing | Growth/Completion |

---

## JSON Structure Required

Based on `day-351.json`, each lesson needs:

```json
{
  "meta": {
    "day": NUMBER,
    "date": "2025-12-XX",
    "topic": "Topic Title",
    "emoji": "🎯",
    "category": "Category Name",
    "version": "v4.0-launch-locked",
    "age_groups": 1,
    "target_audience": "adult",
    "voice_id": "wAdymQH5YucAkXwmrdL0"
  },
  "headline": "Attention-grabbing one-liner about the topic",
  "universal_truth": "Timeless wisdom statement",
  "fun_facts": [
    "Fact 1 with surprising statistic or research",
    "Fact 2 with real-world example",
    "Fact 3 with practical application"
  ],
  "discussion_questions": [
    "Personal reflection question",
    "Experience-based question",
    "Application question"
  ],
  "phases": {
    "hook": {
      "script": "Opening hook (2-3 sentences, ~13s)",
      "duration": 13
    },
    "cliff": {
      "script": "Cliffhanger setup (2 sentences, ~11s)",
      "prompt": "Question for the learner",
      "options": [
        {
          "text": "First answer option",
          "letter": "A",
          "quality": "good",
          "response": "Kelly's response to this choice"
        },
        {
          "text": "Second answer option",
          "letter": "B",
          "quality": "best",
          "response": "Kelly's response to this choice"
        }
      ],
      "duration": 11
    },
    "fact1": {
      "title": "Short Title",
      "script": "First teaching point (3-4 sentences, ~16s)",
      "duration": 16
    },
    "fact2": {
      "title": "Short Title",
      "script": "Second teaching point (3-4 sentences, ~15s)",
      "duration": 15
    },
    "fact3": {
      "title": "Short Title",
      "script": "Third teaching point (3-4 sentences, ~15s)",
      "duration": 15
    },
    "wisdom": {
      "script": "Wisdom moment with actionable takeaway (3-4 sentences, ~14s)",
      "duration": 14
    },
    "outro": {
      "script": "Closing and preview of tomorrow (2-3 sentences, ~9s)",
      "duration": 9
    }
  },
  "phaseOrder": ["hook", "cliff", "fact1", "fact2", "fact3", "wisdom", "outro"],
  "totalDuration": 93,
  "kelly_images": {
    "hook": "/kelly/phases/{DAY}/hook.png",
    "q1": "/kelly/phases/{DAY}/q1.png",
    "q2": "/kelly/phases/{DAY}/q2.png",
    "q3": "/kelly/phases/{DAY}/q3.png",
    "wisdom": "/kelly/phases/{DAY}/wisdom.png"
  },
  "growTrack": {
    "title": "Related Skill - Action Focus",
    "emoji": "🎯",
    "learning_objective": "What the learner will practice",
    "activity": "Specific action to take today"
  }
}
```

---

## Content Requirements Per Lesson

### Phase Durations & Word Counts
| Phase | Duration | Words (~2.5 w/s) | Purpose |
|-------|----------|------------------|---------|
| hook | 13s | ~33 words | Grab attention, introduce topic |
| cliff | 11s | ~28 words | Create curiosity, pose question |
| fact1 | 16s | ~40 words | First key teaching point |
| fact2 | 15s | ~38 words | Second key teaching point |
| fact3 | 15s | ~38 words | Third key teaching point |
| wisdom | 14s | ~35 words | Synthesize into actionable insight |
| outro | 9s | ~23 words | Close warmly, preview tomorrow |
| **Total** | **93s** | **~235 words** | |

### Content Quality Standards

1. **Hook**: Start with question, surprising fact, or relatable scenario
2. **Cliff**: Build tension, make learner want to know the answer
3. **Facts**: Evidence-based, cite research/examples when possible
4. **Wisdom**: Personal, actionable, memorable
5. **Outro**: Warm, forward-looking, encouraging

### Voice & Tone
- Conversational, not lecturing
- Curious, not condescending
- Warm, like talking to a friend
- Empowering, not prescriptive
- "We" and "you" over "I"

---

## Generation Approach

### Option A: Manual Creation (High Quality, Slow)
- Research each topic
- Write scripts manually
- Review for voice/tone
- ~30-45 min per lesson
- Total: 7-10 hours

### Option B: AI-Assisted with Template (Medium Quality, Medium Speed)
- Use Claude to draft based on template
- Human review and polish
- ~15-20 min per lesson
- Total: 3-5 hours

### Option C: Batch Generation (Lower Quality, Fast)
- Generate all 14 at once using structured prompts
- Quick review pass
- ~5-10 min per lesson for review
- Total: 1-2 hours

**Recommendation**: Option B - AI-assisted with template. Balance of quality and speed.

---

## Execution Plan

### Phase 1: Template & Prompt Setup (30 min)
1. Create generation prompt with day-351 as example
2. Define category-specific guidance for each topic
3. Set up validation checklist

### Phase 2: Batch Generation (2 hours)
1. Generate Days 352-358 (first batch)
2. Quick review for obvious issues
3. Generate Days 359-365 (second batch)
4. Quick review for obvious issues

### Phase 3: Quality Pass (1 hour)
1. Review each lesson for:
   - Word count per phase
   - Tone consistency
   - Factual accuracy
   - Flow between phases
2. Fix any issues

### Phase 4: Validation & Save (30 min)
1. Validate JSON structure
2. Save to `public/lessons/day-{N}.json`
3. Verify HeyGen generator can read them

---

## Immediate Next Step

Create a lesson generation script that:
1. Takes a topic, day number, and category as input
2. Uses Claude API to generate content
3. Validates against the schema
4. Saves to the correct location

OR

Generate the first lesson (Day 352) manually as a quality reference, then batch the rest.

---

## Files to Create

```
public/lessons/
├── day-351.json ✅ (exists)
├── day-352.json (Quieting Your Thoughts)
├── day-353.json (Being Where You Are)
├── day-354.json (What Makes You Come Alive)
├── day-355.json (Why You Get Up)
├── day-356.json (What Makes Life Matter)
├── day-357.json (Deciding Right From Wrong)
├── day-358.json (What You Care About Most)
├── day-359.json (Who You Are When No One's Looking)
├── day-360.json (What You Leave Behind)
├── day-361.json (Looking Back to Learn)
├── day-362.json (Marking What Matters)
├── day-363.json (Appreciating What You Have)
├── day-364.json (Starting Fresh)
└── day-365.json (365 Days of Growing)
```

---

## Success Criteria

- [ ] 14 lesson JSON files created
- [ ] All validate against schema
- [ ] All work with HeyGen generator (dry-run test)
- [ ] Scripts total ~235 words each
- [ ] Tone is consistent with Day 351
- [ ] Topics flow naturally day-to-day

---

*Ready to begin execution upon approval.*
