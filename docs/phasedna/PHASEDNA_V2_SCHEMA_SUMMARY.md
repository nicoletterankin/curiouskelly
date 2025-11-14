# PhaseDNA v2 Schema Summary

**Created:** 2025-01-XX  
**Purpose:** Extended PhaseDNA v1 schema with optional pedagogical richness from alternative schema

---

## Overview

PhaseDNA v2 extends PhaseDNA v1 with optional fields that add pedagogical depth, cultural awareness, and quality validation capabilities while maintaining **100% backward compatibility** with PhaseDNA v1 lessons.

---

## Key Additions

### 1. Top-Level Enhancements

**New Optional Fields:**
- `version` - Schema version (e.g., "1.0.0")
- `createdAt` / `updatedAt` - ISO 8601 timestamps
- `author` - Author attribution
- `calendar` - Daily Lesson pipeline integration (`day`, `date`)
- `universal_concept` / `universal_concept_translations` - Universal concept across all ages
- `core_principle` / `core_principle_translations` - Core principle underlying lesson
- `learning_essence` / `learning_essence_translations` - Learning essence description
- `example_selector_data` - Age-specific example scenarios for interactions
- `daily_fortune_elements` / `daily_fortune_elements_translations` - Identity shift elements
- `language_adaptation_framework` - Cultural and linguistic adaptation framework
- `quality_validation_targets` - Quality validation criteria

---

### 2. Age Variant Enhancements

**New Optional Fields per Age Variant:**
- `core_metaphor` / `core_metaphor_translations` - Core metaphor for this age
- `complexity_level` - Complexity descriptor (enum: concrete_observable_actions, systems_thinking_with_concrete_examples, etc.)
- `attention_span` - Expected attention span (e.g., "3-4_minutes")
- `cognitive_focus` - Cognitive focus descriptor
- `examples` - Array of age-appropriate example scenarios
- `abstract_concepts` / `abstract_concepts_translations` - Abstract concept mappings
- `tone` - Tone and delivery patterns:
  - `voice_character` - Voice character descriptor
  - `emotional_temperature` - Emotional temperature descriptor
  - `language_patterns` - Pre-defined openings, transitions, encouragements, closings
  - `metaphor_style` - Metaphor style descriptor
  - `question_approach` - Question approach descriptor
  - `validation_style` - Validation style descriptor
- `tone_translations` - Translations of tone language patterns

**Enhanced Language Variant:**
- Added optional `core_metaphor` field
- Added optional `abstract_concepts` object
- Added optional `cultural_notes` field

**Enhanced Teaching Moments:**
- Added new types: `visual`, `reflection`, `application`, `legacy` (in addition to existing: explanation, question, demonstration, story, wisdom)

---

### 3. Interaction Enhancements

**New Optional Fields per Interaction:**
- `concept_focus` - Concept focus for this interaction
- `universal_principle` - Universal principle this interaction addresses
- `cognitive_target` - Cognitive target for this interaction

**Enhanced Age Adaptations:**
- Added optional `scenario` field for age-specific scenarios

---

## Backward Compatibility

✅ **100% Backward Compatible**

All PhaseDNA v1 lessons will validate against PhaseDNA v2 schema because:
- All new fields are **optional**
- All PhaseDNA v1 required fields remain required
- No breaking changes to existing field structures
- Existing lessons can be gradually enhanced with new fields

---

## Migration Path

### Phase 1: Use PhaseDNA v1 (Current)
- Existing lessons continue to work
- No changes required

### Phase 2: Gradual Enhancement (Optional)
- Add optional fields as needed
- Enhance lessons with pedagogical richness
- Add cultural adaptation frameworks

### Phase 3: Full PhaseDNA v2 (Future)
- All lessons include optional fields
- Full pedagogical richness
- Complete cultural adaptation support

---

## Usage Examples

### Minimal PhaseDNA v2 Lesson (Backward Compatible)
```json
{
  "id": "the-sun",
  "title": "Our Amazing Sun",
  "description": "Discover the incredible star...",
  "metadata": { ... },
  "ageVariants": { ... },
  "interactions": [ ... ]
}
```
✅ This validates against PhaseDNA v2 (same as PhaseDNA v1)

### Enhanced PhaseDNA v2 Lesson (With Optional Fields)
```json
{
  "id": "molecular-biology",
  "title": "Collaborative Molecular Systems",
  "version": "1.0.0",
  "createdAt": "2025-01-XXT00:00:00.000Z",
  "updatedAt": "2025-01-XXT00:00:00.000Z",
  "author": "UI-TARS Team",
  "description": "...",
  "calendar": {
    "day": 189,
    "date": "July 8"
  },
  "universal_concept": "collaborative_molecular_systems_enable_life",
  "universal_concept_translations": {
    "en": "Collaborative molecular systems enable life",
    "es": "...",
    "fr": "..."
  },
  "core_principle": "...",
  "learning_essence": "...",
  "metadata": { ... },
  "ageVariants": {
    "2-5": {
      // All PhaseDNA v1 required fields
      "title": "...",
      "video": "...",
      "script": "...",
      // New optional fields
      "core_metaphor": "body_city_with_helper_workers",
      "complexity_level": "concrete_observable_actions",
      "attention_span": "3-4_minutes",
      "examples": [ ... ],
      "tone": {
        "language_patterns": {
          "openings": [ ... ],
          "transitions": [ ... ]
        }
      }
    }
  },
  "interactions": [
    {
      "step": "welcome",
      "question": "...",
      "concept_focus": "...",  // NEW: Optional
      "universal_principle": "...",  // NEW: Optional
      "choices": [ ... ]
    }
  ],
  "language_adaptation_framework": { ... },
  "quality_validation_targets": { ... }
}
```

---

## Schema Files

- **PhaseDNA v1:** `lesson-player/lesson-dna-schema.json` (Original, production-ready)
- **PhaseDNA v2:** `lesson-player/lesson-dna-schema-v2.json` (Extended, backward-compatible)

---

## Validation

Both schemas can be used for validation:

```javascript
// Validate PhaseDNA v1 lesson
const v1Schema = require('./lesson-dna-schema.json');
const v1Valid = ajv.compile(v1Schema);
v1Valid(lessonData);

// Validate PhaseDNA v2 lesson (also validates v1 lessons)
const v2Schema = require('./lesson-dna-schema-v2.json');
const v2Valid = ajv.compile(v2Schema);
v2Valid(lessonData);
```

---

## Next Steps

1. ✅ **Schema Created** - PhaseDNA v2 schema file created
2. ⏳ **Update Validator** - Extend validator to support v2 optional fields
3. ⏳ **Migrate Example** - Convert `molecular_biology_dna.json` to PhaseDNA v2 format
4. ⏳ **Documentation** - Create authoring guide for PhaseDNA v2
5. ⏳ **Tooling** - Update lesson authoring tools to support new fields

---

## Benefits

### For Authors
- ✅ Richer pedagogical metadata
- ✅ Better cultural adaptation support
- ✅ Quality validation targets
- ✅ More expressive tone control

### For System
- ✅ Backward compatible (no breaking changes)
- ✅ Gradual migration path
- ✅ Enhanced lesson quality
- ✅ Better global reach

### For Learners
- ✅ More age-appropriate content
- ✅ Better cultural sensitivity
- ✅ Richer learning experiences
- ✅ More engaging interactions

---

**Status:** ✅ Schema Complete  
**Backward Compatibility:** ✅ 100%  
**Ready for Use:** ✅ Yes (optional fields can be added gradually)


