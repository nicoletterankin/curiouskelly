# Curious Kelly: Content Management Agent Onboarding Guide

**Version:** 1.0  
**Date:** December 2024  
**Purpose:** Essential knowledge for AI content agents building lessons using Claude Code

---

## 🎯 Project Mission & Overview

**Curious Kelly** is an AI-powered learning companion that delivers age-adaptive daily lessons to learners aged 2-102. The system uses a single universal topic per day, but Kelly (the avatar teacher) adapts her age, language complexity, examples, and teaching style to match each learner's developmental stage.

**Core Innovation:** One topic, six age personas, three languages (EN/ES/FR), all precomputed in a structured JSON format called "DNA."

**Your Role:** Create complete lesson DNA files that work across all age groups, ensuring universal topics are accessible to toddlers and profound for elders.

---

## 🧬 DNA Lesson Architecture

### Universal Lesson Structure

Every lesson is a JSON file (`*-dna.json`) containing:

1. **Universal Concept Framework:**
   - `universal_concept`: Timeless principle (e.g., "stellar_physics_enables_life")
   - `core_principle`: Deeper learning truth (e.g., "scientific_observation_creates_shared_knowledge")
   - `learning_essence`: Practical understanding to convey
   - All three must be translatable to ES/FR

2. **Six Age Variants (Required):**
   - `2-5`: Playful toddler Kelly (age 3) - 3-4 min, simple language
   - `6-12`: Curious kid Kelly (age 9) - 5-6 min, enthusiastic
   - `13-17`: Teen mentor Kelly (age 15) - 8-9 min, relatable
   - `18-35`: Knowledgeable adult Kelly (age 27) - 10 min, sophisticated
   - `36-60`: Wise mentor Kelly (age 48) - 11-12 min, perspective-rich
   - `61-102`: Reflective elder Kelly (age 82) - 13 min, profound

3. **Multilingual Content (Required):**
   - Each age variant includes `language` object with `en`, `es`, `fr`
   - All content sections must be translated (no runtime generation)

4. **Interactive Phases:**
   - `welcome`: Hook and introduction
   - `mainContent`: Core teaching
   - `interactions`: Questions with choices (2-4 options per question)
   - `wisdomMoment`: Memorable insight
   - `teachingMoments`: Timestamped highlights

### Key DNA File Properties

```json
{
  "id": "topic-slug",
  "title": "Universal title",
  "ageVariants": {
    "2-5": {
      "kellyAge": 3,
      "kellyPersona": "playful-toddler",
      "voiceProfile": { "speechRate": 0.85, "pitch": 2, "energy": "bright" },
      "language": { "en": {...}, "es": {...}, "fr": {...} },
      "pacing": { "welcome": "30s", "teaching": "2min", ... }
    }
  },
  "interactions": [
    {
      "step": "welcome",
      "question": "Age-appropriate question",
      "choices": [
        { "text": "Option", "response": "Kelly's reply", "nextStep": "teaching" }
      ],
      "ageAdaptations": { "2-5": {...}, ... }
    }
  ]
}
```

**Critical Rule:** Languages are **precomputed** in every DNA file. Never generate translations at runtime.

---

## 👤 Kelly Avatar System

Kelly is a 3D avatar rendered in Unity, with six distinct personas:

| Age Group | Kelly Age | Persona | Voice Speed | Pitch | Energy | Attention Span |
|-----------|-----------|---------|-------------|-------|--------|----------------|
| 2-5 | 3 | playful-toddler | 0.85 | +2 | bright | 3-4 min |
| 6-12 | 9 | curious-kid | 0.80 | 0 | bright | 5-6 min |
| 13-17 | 15 | teen-mentor | 0.75 | -1 | moderate | 8-9 min |
| 18-35 | 27 | knowledgeable-adult | 0.70 | 0 | moderate | 10 min |
| 36-60 | 48 | experienced-guide | 0.65 | -1 | calm | 11-12 min |
| 61-102 | 82 | wise-elder | 0.60 | -2 | calm | 13 min |

**Voice Synthesis:** ElevenLabs (primary), OpenAI TTS (fallback). Never use browser TTS.

**Audio Requirements:** Minimum 60 minutes training audio per voice model. Never compress or trim datasets.

---

## 📝 Content Creation Workflow

### Step 1: Topic Selection
- ✅ **Good:** Observable phenomena (leaves, clouds), universal experiences (friendship, gratitude), natural wonders (rain, stars)
- ❌ **Avoid:** Age-specific content (retirement planning), controversial topics, commercial products, culturally specific holidays

### Step 2: Write Universal Framework (30 min)
- Define `universal_concept`, `core_principle`, `learning_essence`
- Ensure translatable to ES/FR
- Verify topic works for ages 2-102

### Step 3: Create Age Variants (2-3 hours per age group)
**Recommended order:**
1. Start with **18-35** (baseline, easiest)
2. Simplify for **2-5, 6-12, 13-17** (younger)
3. Add depth for **36-60, 61-102** (older)

**For each age variant, write:**
- `welcome`: Hook (30s-1min)
- `mainContent`: Core teaching (2-4.5min based on age)
- `keyPoints`: 3-5 takeaways
- `interactionPrompts`: 3-5 engaging questions
- `wisdomMoment`: Memorable insight (30s-2min)
- `teachingMoments`: 2-3 timestamped highlights

### Step 4: Add Interactions
- Each interaction has `question`, `choices` (2-4 options), `ageAdaptations`
- Choices include `text`, `response` (Kelly's reply), `nextStep` (phase progression)
- Age-adapt questions and choices per bucket

### Step 5: Translate to ES/FR
- Translate all `language.en` content to `language.es` and `language.fr`
- Maintain age-appropriate complexity in each language
- Verify cultural sensitivity

### Step 6: Validate & Test
```bash
# Validate against schema
node curious-kellly/content-tools/validate-lesson.js your-lesson.json

# Preview for specific age
node curious-kellly/content-tools/preview-lesson.js your-lesson.json --age 35

# Generate audio (optional)
node curious-kellly/content-tools/generate-audio.js your-lesson.json
```

**Time per lesson:** ~12-15 hours (6 age variants × 3 languages)

---

## 🛠️ Technical Requirements

### File Locations
- **Template:** `curious-kellly/content-tools/lesson-template.json`
- **Output:** `curious-kellly/backend/config/lessons/*-dna.json` or `lessons/*-dna.json`
- **Schema:** `curious-kellly/backend/config/lesson-dna-schema.json`
- **Guide:** `curious-kellly/content-tools/lesson-authoring-guide.md`

### Required Tools
- Node.js (for validation/preview scripts)
- JSON editor (VS Code recommended)
- ElevenLabs API key (for audio generation, optional)

### Validation Rules
- Must pass JSON schema validation
- All 6 age variants required
- All 3 languages (EN/ES/FR) required per variant
- Unique lesson ID (kebab-case: `topic-name`)
- Word count guidelines per age group
- No syntax errors, all required fields present

---

## ✅ Quality Standards

### Universal Topic Checklist
- [ ] Works for ages 2-102
- [ ] Not age-specific or controversial
- [ ] Observable or experiential
- [ ] Has depth for exploration

### Age Variant Checklist
- [ ] Language appropriate for age
- [ ] Pacing matches attention span
- [ ] Kelly age and persona set correctly
- [ ] Voice profile parameters correct
- [ ] Examples relevant to age group

### Content Quality Checklist
- [ ] Accurate information (fact-checked)
- [ ] Engaging and accessible
- [ ] Actionable insights
- [ ] Memorable wisdom moment
- [ ] Safe and age-appropriate
- [ ] Culturally sensitive

### Technical Checklist
- [ ] Valid JSON (no syntax errors)
- [ ] Passes schema validation
- [ ] All required fields present
- [ ] Unique lesson ID
- [ ] Multilingual completeness (EN/ES/FR)

---

## 📚 Key Resources & Examples

### Documentation
- `CURIOUS_KELLLY_EXECUTION_PLAN.md` - Full project roadmap
- `LESSON_SYSTEM_EXPERTISE.md` - Technical system details
- `curious-kellly/content-tools/lesson-authoring-guide.md` - Complete writing guide
- `CLAUDE.md` - Operating rules and constraints

### Example Lessons
- `lessons/the-sun-dna.json` - Complete example with all variants
- `curious-kellly/content-tools/lesson-template.json` - Starting template

### 30-Lesson Curriculum (Target)
**Week 1:** Leaves, Water, Clouds, Light, Sound, Seeds, Stars  
**Week 2:** Friendship, Kindness, Listening, Patience, Gratitude, Courage, Curiosity  
**Week 3:** Balance, Breathing, Movement, Rest, Energy, Senses, Growth  
**Week 4:** Colors, Patterns, Stories, Music, Questions, Imagination, Memory  
**Week 5:** Time, Change

**Goal:** 2 lessons/day = 30 lessons in 15 days

---

## 🚨 Critical Constraints

### Never Do This
- ❌ Generate translations at runtime (must precompute)
- ❌ Use browser TTS (use ElevenLabs/OpenAI)
- ❌ Compress or trim training audio datasets
- ❌ Create age-specific topics (must be universal)
- ❌ Skip age variants or languages
- ❌ Bypass validation

### Always Do This
- ✅ Precompute all languages in DNA files
- ✅ Validate before submitting
- ✅ Test with preview tool
- ✅ Follow schema exactly
- ✅ Maintain age-appropriate language
- ✅ Include wisdom moments

---

## 🎯 Success Metrics

**Per Lesson:**
- Passes all validation checks
- Works across all 6 age groups
- All 3 languages complete
- Audio generation successful (if applicable)
- Preview shows correct content per age

**Project Goal:**
- 30 production-ready lessons
- Universal topics that engage ages 2-102
- Consistent quality across all variants
- Ready for daily lesson calendar integration

---

**You're ready to create amazing universal lessons! Start with the template, follow the guide, validate often, and remember: one topic, six ages, three languages, infinite learning.** 🌍








