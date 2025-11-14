# PhaseDNA v2 Migration Guide

**Purpose:** Step-by-step guide for migrating lessons to PhaseDNA v2 format

---

## Quick Start

### Option 1: Use Migration Script (Automated)

```bash
# Migrate alternative schema lesson to PhaseDNA v2
node curious-kellly/content-tools/migrate-to-phasedna-v2.js \
  lessons/molecular_biology_dna.json \
  curious-kellly/backend/config/lessons/molecular-biology-v2.json
```

**Note:** The script creates structural migration. You'll need to:
1. Add actual `welcome`/`mainContent`/`wisdomMoment` text
2. Generate video files
3. Add proper teaching moments with timestamps
4. Add expression cues aligned to teaching moments

---

### Option 2: Manual Migration (Recommended for Quality)

Follow the steps below to manually migrate with full control.

---

## Step-by-Step Migration Process

### Step 1: Update Top-Level Fields

**Add version control:**
```json
{
  "version": "2.0.0",
  "createdAt": "2025-01-XXT00:00:00.000Z",
  "updatedAt": "2025-01-XXT00:00:00.000Z",
  "author": "Your Name"
}
```

**Add calendar integration (if applicable):**
```json
{
  "calendar": {
    "day": 189,
    "date": "July 8"
  }
}
```

**Add universal concepts:**
```json
{
  "universal_concept": "collaborative_molecular_systems_enable_life",
  "universal_concept_translations": {
    "en": "...",
    "es": "...",
    "fr": "..."
  },
  "core_principle": "...",
  "core_principle_translations": { ... },
  "learning_essence": "...",
  "learning_essence_translations": { ... }
}
```

---

### Step 2: Map Age Buckets

**Old Schema → PhaseDNA v2:**
- `early_childhood` → `2-5`
- `youth` → `6-12`
- `young_adult` → `13-17`
- `midlife` → `36-60`
- `wisdom_years` → `61-102`

**Note:** PhaseDNA v2 requires all 6 age buckets. If migrating from 5-bucket schema, you may need to create content for `18-35` bucket.

---

### Step 3: Enhance Age Variants

For each age variant, add:

**1. Execution Elements (Required):**
```json
{
  "video": "kelly_lesson-id_2-5.mp4",
  "script": "Opening script text...",
  "kellyAge": 3,
  "kellyPersona": "playful-toddler",
  "voiceProfile": {
    "provider": "elevenlabs",
    "voiceId": "wAdymQH5YucAkXwmrdL0",
    "speechRate": 0.85,
    "pitch": 2,
    "energy": "bright",
    "language": "en-US"
  }
}
```

**2. Pedagogical Fields (Optional but Recommended):**
```json
{
  "core_metaphor": "body_city_with_helper_workers",
  "core_metaphor_translations": {
    "en": "Body city with helper workers",
    "es": "...",
    "fr": "..."
  },
  "complexity_level": "concrete_observable_actions",
  "attention_span": "3-4_minutes",
  "cognitive_focus": "simple_cause_and_effect_in_body",
  "examples": [
    "eating_food_gives_energy",
    "breathing_helps_body_work"
  ]
}
```

**3. Language Structure (Required):**
Consolidate translations into unified `language.en/es/fr` structure:
```json
{
  "language": {
    "en": {
      "title": "...",
      "welcome": "...",
      "mainContent": "...",
      "keyPoints": [ ... ],
      "interactionPrompts": [ ... ],
      "wisdomMoment": "...",
      "cta": "...",
      "summary": "...",
      "core_metaphor": "...",  // NEW: Optional
      "abstract_concepts": { ... },  // NEW: Optional
      "cultural_notes": "..."  // NEW: Optional
    },
    "es": { ... },
    "fr": { ... }
  }
}
```

**4. Tone Patterns (Optional but Recommended):**
```json
{
  "tone": {
    "voice_character": "loving_elder_sharing_wonder_about_life",
    "emotional_temperature": "warm_amazed_nurturing",
    "language_patterns": {
      "openings": [ ... ],
      "transitions": [ ... ],
      "encouragements": [ ... ],
      "closings": [ ... ]
    },
    "metaphor_style": "...",
    "question_approach": "...",
    "validation_style": "..."
  },
  "tone_translations": {
    "en": { "language_patterns": { ... } },
    "es": { "language_patterns": { ... } },
    "fr": { "language_patterns": { ... } }
  }
}
```

**5. Teaching Moments & Expression Cues (Required):**
```json
{
  "teachingMoments": [
    {
      "id": "tm1-2-5",
      "timestamp": 15,
      "type": "explanation",
      "content": "Show picture of..."
    }
  ],
  "expressionCues": [
    {
      "id": "ec1-2-5",
      "momentRef": "tm1-2-5",
      "type": "macro-gesture",
      "offset": 0,
      "duration": 2,
      "intensity": "emphatic",
      "gazeTarget": "camera"
    }
  ]
}
```

---

### Step 4: Migrate Interactions

**Convert `core_lesson_structure` to `interactions` array:**

**Old Format:**
```json
{
  "core_lesson_structure": {
    "question_1": {
      "concept_focus": "...",
      "choice_architecture": {
        "option_a": "...",
        "option_b": "..."
      }
    }
  }
}
```

**New Format:**
```json
{
  "interactions": [
    {
      "step": "welcome",
      "question": "Have you ever wondered...?",
      "concept_focus": "...",  // NEW: Optional
      "universal_principle": "...",  // NEW: Optional
      "cognitive_target": "...",  // NEW: Optional
      "choices": [
        {
          "text": "Option A",
          "nextStep": "teaching",
          "response": "Great thinking!",
          "learningValue": "high"
        }
      ],
      "ageAdaptations": {
        "2-5": {
          "question": "...",
          "choices": [ ... ],
          "hints": [ ... ],
          "scenario": "..."  // NEW: Optional
        }
      }
    }
  ]
}
```

---

### Step 5: Add Optional Frameworks

**Add at top level (optional but valuable):**
```json
{
  "example_selector_data": { ... },
  "daily_fortune_elements": { ... },
  "daily_fortune_elements_translations": { ... },
  "language_adaptation_framework": { ... },
  "quality_validation_targets": { ... }
}
```

---

## Validation

After migration, validate your lesson:

```bash
# Validate against PhaseDNA v2 schema
node curious-kellly/content-tools/validate-lesson-v2.js \
  curious-kellly/backend/config/lessons/molecular-biology-v2.json
```

---

## Common Migration Issues & Solutions

### Issue 1: Missing Age Bucket (18-35)
**Problem:** Alternative schema has 5 buckets, PhaseDNA v2 requires 6.

**Solution:** 
- Create content for `18-35` bucket
- Use `young_adult` content as starting point
- Adjust `kellyAge` to 27, `kellyPersona` to `knowledgeable-adult`

### Issue 2: Fragmented Translations
**Problem:** Translations scattered across multiple fields.

**Solution:**
- Consolidate into `language.en/es/fr` structure
- Move `concept_name_translations` → `language.XX.title`
- Move `core_metaphor_translations` → `language.XX.core_metaphor`
- Move `abstract_concepts_translations` → `language.XX.abstract_concepts`

### Issue 3: Missing Execution Elements
**Problem:** No `video`, `script`, `voiceProfile`, `teachingMoments`, `expressionCues`.

**Solution:**
- Generate video filenames: `kelly_{lesson-id}_{age-bucket}.mp4`
- Create opening scripts based on `welcome` text
- Use default voice profiles (see migration script)
- Add placeholder teaching moments (update with actual timestamps later)
- Add placeholder expression cues (align to teaching moments later)

### Issue 4: Missing Interaction Flow
**Problem:** `core_lesson_structure` doesn't have `step` or `nextStep`.

**Solution:**
- Map question order to steps: `question_1` → `welcome`, `question_2` → `teaching`, `question_3` → `practice`
- Add `nextStep` to choices based on question position
- Convert `choice_architecture` to `choices` array with `text`, `nextStep`, `response`, `learningValue`

---

## Example: Complete Migration

See `curious-kellly/backend/config/lessons/molecular-biology-v2-example.json` for a complete example showing:
- ✅ All PhaseDNA v1 required fields
- ✅ PhaseDNA v2 optional pedagogical fields
- ✅ Complete language structure (EN/ES/FR)
- ✅ Tone patterns
- ✅ Teaching moments with expression cues
- ✅ Enhanced interactions with pedagogical metadata
- ✅ Optional frameworks

---

## Next Steps After Migration

1. **Review Content** - Ensure all text is age-appropriate
2. **Generate Videos** - Create video files for each age variant
3. **Time Teaching Moments** - Add accurate timestamps based on video
4. **Align Expression Cues** - Match expression cues to teaching moments
5. **Test in Lesson Player** - Verify lesson works correctly
6. **Validate** - Run validator to check quality

---

**Status:** ✅ Migration tools ready  
**Example:** ✅ Complete example lesson available  
**Validator:** ✅ v2 validator available


