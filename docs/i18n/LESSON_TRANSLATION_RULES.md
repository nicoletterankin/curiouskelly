# 📚 LESSON TRANSLATION RULES

## System Operating Rules for AI-Assisted Lesson Translation

**Created:** December 17, 2025  
**Status:** DRAFT — Awaiting Approval  
**Scope:** Days 1-365 × 3 Languages (EN/ES/PT)

---

## 🎯 Mission

Translate 365 daily lessons from English to Spanish and Portuguese while:
- Preserving Kelly's warm, curious, intelligent personality
- Maintaining educational accuracy and cultural relevance
- Ensuring consistent terminology and voice across all lessons
- Creating production-ready files that work with existing infrastructure

---

## 📊 Content Inventory

### What Exists Today

| Asset Type | Location | Count | Translate? |
|------------|----------|-------|------------|
| Lesson JSON | `public/lessons/day-{N}.json` | ~200 | ✅ Yes |
| Watch pages | `public/watch/day-{N}.html` | ~215 | ✅ Yes (meta only) |
| Email HTML | `generated-emails/day-{N}-email.html` | ~111 | ✅ Yes |
| Video manifests | `content/email-summary-video/day-{NNN}-*.json` | ~154 | ✅ Yes |

### Fields to Translate per Lesson JSON (v5.0-full-choices)

> **IMPORTANT:** As of v5.0, ALL phases have choices (not just cliff).
> Phase names changed: fact1→q1, fact2→q2, fact3→q3

```
meta.topic                    ← Topic name (e.g., "Starting Fresh" → "Empezando de Nuevo")
headline                      ← Main hook
universal_truth              ← Core wisdom message
fun_facts[]                  ← Array of 3 educational facts
discussion_questions[]       ← Array of 3 reflection questions

# ALL phases now have the same structure:
phases.hook.title            ← Phase title
phases.hook.script           ← Opening narration
phases.hook.prompt           ← Question text (NEW - all phases have this)
phases.hook.options[].text   ← Answer choices (NEW - all phases have this)
phases.hook.options[].response ← Feedback per choice (NEW - all phases have this)

phases.cliff.title           ← Phase title
phases.cliff.script          ← Cliffhanger setup
phases.cliff.prompt          ← Question text
phases.cliff.options[].text  ← Answer choices
phases.cliff.options[].response ← Feedback per choice

phases.q1.title              ← (was fact1) Fact section title
phases.q1.script             ← Fact narration
phases.q1.prompt             ← Question text (NEW)
phases.q1.options[].text     ← Answer choices (NEW)
phases.q1.options[].response ← Feedback per choice (NEW)

phases.q2.title              ← (was fact2)
phases.q2.script             ←
phases.q2.prompt             ← (NEW)
phases.q2.options[].text     ← (NEW)
phases.q2.options[].response ← (NEW)

phases.q3.title              ← (was fact3)
phases.q3.script             ←
phases.q3.prompt             ← (NEW)
phases.q3.options[].text     ← (NEW)
phases.q3.options[].response ← (NEW)

phases.wisdom.title          ← Phase title
phases.wisdom.script         ← Wisdom message
phases.wisdom.prompt         ← Question text (NEW)
phases.wisdom.options[].text ← Answer choices (NEW)
phases.wisdom.options[].response ← Feedback per choice (NEW)

phases.outro.title           ← Phase title
phases.outro.script          ← Closing narration
phases.outro.prompt          ← Question text (NEW)
phases.outro.options[].text  ← Answer choices (NEW)
phases.outro.options[].response ← Feedback per choice (NEW)

growTrack.title              ← Growth activity title
growTrack.learning_objective ← What learner achieves
growTrack.activity           ← Activity instructions
```

### Total Translation Fields per Lesson

| Component | EN Fields | ES/PT Translations |
|-----------|-----------|-------------------|
| Meta/headline | 3 | 6 |
| Fun facts | 3 | 6 |
| Discussion questions | 3 | 6 |
| Phases (7 × 5 fields each) | 35 | 70 |
| Phase options (7 × 2 × 2 fields) | 28 | 56 |
| GrowTrack | 3 | 6 |
| **TOTAL** | **75** | **150** |

### Fields to NEVER Translate

```
meta.day                     ← Number
meta.date                    ← Keep ISO format
meta.emoji                   ← Universal
meta.category                ← Keep English (used as key)
meta.version                 ← Technical
meta.target_audience         ← Technical
meta.voice_id               ← ElevenLabs ID

phases.*.duration           ← Seconds (will change for translated audio)
phases.cliff.options[].letter ← A, B, C
phases.cliff.options[].quality ← "best", "good"

phaseOrder[]                ← Technical
totalDuration               ← Will recalculate
```

---

## 🏗️ Output File Structure

Per `CLAUDE.md`: *"Languages are precomputed in every DNA/content file (EN + ES/FR)."*

### Recommended: Embedded Multilingual Structure

```json
{
  "meta": {
    "day": 1,
    "date": "2025-01-01",
    "topic": {
      "en": "Starting Fresh",
      "es": "Empezando de Nuevo",
      "pt": "Começando de Novo"
    },
    "emoji": "🍁",
    "category": "Beginnings",
    "version": "v4.0-i18n",
    "languages": ["en", "es", "pt"]
  },
  "headline": {
    "en": "Every ending holds the seed of a new beginning",
    "es": "Cada final contiene la semilla de un nuevo comienzo",
    "pt": "Todo fim contém a semente de um novo começo"
  },
  "phases": {
    "hook": {
      "script": {
        "en": "Welcome to Day One...",
        "es": "Bienvenidos al Día Uno...",
        "pt": "Bem-vindos ao Dia Um..."
      },
      "duration": {
        "en": 12,
        "es": 14,
        "pt": 13
      }
    }
  }
}
```

### Alternative: Separate Files (for phased rollout)

```
public/lessons/
├── en/
│   ├── day-1.json
│   └── day-2.json
├── es/
│   ├── day-1.json
│   └── day-2.json
└── pt/
    ├── day-1.json
    └── day-2.json
```

**Decision needed:** Which structure? Embedded is per CLAUDE.md but requires API changes.

---

## 🗣️ Voice & Personality Rules

### Kelly's Core Traits (Preserve in All Languages)

| Trait | English | Spanish | Portuguese |
|-------|---------|---------|------------|
| Warm greeting | "Welcome!" | "¡Bienvenidos!" | "Bem-vindos!" |
| Curious tone | "Have you ever wondered..." | "¿Alguna vez te has preguntado..." | "Você já se perguntou..." |
| Encouraging | "You're doing great!" | "¡Lo estás haciendo genial!" | "Você está indo muito bem!" |
| Wisdom close | "Here's today's wisdom:" | "Esta es la sabiduría de hoy:" | "Eis a sabedoria de hoje:" |

### Voice Consistency Rules

1. **Use informal "you":**
   - Spanish: "tú" (not "usted")
   - Portuguese: "você" (Brazilian Portuguese)

2. **Kelly speaks TO the learner:**
   - Always second person singular
   - Direct, personal, not lecturing

3. **Preserve rhetorical structure:**
   - If English asks a question, translation asks a question
   - If English uses a metaphor, translation uses equivalent metaphor
   - If English has a pause (em-dash), translation has a pause

4. **Maintain wonder and curiosity:**
   - "Fascinating" → "Fascinante" (ES) / "Fascinante" (PT)
   - "Amazing" → "Increíble" (ES) / "Incrível" (PT)
   - Avoid clinical/academic tone

### Cultural Adaptation (Not Just Translation)

| English Reference | Spanish Adaptation | Portuguese Adaptation |
|-------------------|--------------------|-----------------------|
| "Super Bowl" | "la final de la Copa" | "a final da Copa" |
| "Thanksgiving" | Keep + explain OR omit | Keep + explain OR omit |
| "High school" | "secundaria" | "ensino médio" |
| Fahrenheit temps | Convert to Celsius | Convert to Celsius |
| Imperial units | Convert to metric | Convert to metric |

---

## ✅ Quality Rules

### Accuracy Preservation

1. **Scientific facts must remain accurate:**
   - "66 days to form a habit" → same number in all languages
   - Research citations preserved (e.g., "University College London")
   - Statistics unchanged

2. **Proper nouns stay in original:**
   - Person names (Einstein, Marie Curie)
   - Place names (unless commonly localized: "London" → "Londres")
   - Brand names

3. **Technical terms:**
   - "Neuroplasticity" → "Neuroplasticidad" (ES) / "Neuroplasticidade" (PT)
   - Scientific terms should be the accepted local term, not invented

### Length Considerations

| Phase | English ~Duration | Spanish ~Duration | Portuguese ~Duration |
|-------|-------------------|-------------------|----------------------|
| hook | 12-15s | +10-15% longer | +5-10% longer |
| fact | 14-18s | +10-15% longer | +5-10% longer |
| wisdom | 12-14s | +10-15% longer | +5-10% longer |

**Note:** Spanish is typically 10-15% longer than English for same content. Portuguese closer to English length. Duration fields will need recalculation after TTS generation.

---

## 🔄 Translation Workflow

### Phase 1: Machine Translation (AI-Assisted)

```
For each day-{N}.json:
  1. Load English source
  2. Translate each field per rules above
  3. Maintain JSON structure exactly
  4. Output to staging location
  5. Validate JSON syntax
```

### Phase 2: Human Review (Future)

```
  6. Native speaker review
  7. Kelly voice/personality check
  8. Educational accuracy check
  9. Cultural appropriateness check
  10. Approve or revise
```

### Phase 3: Audio Generation (Future)

```
  11. Generate TTS via ElevenLabs (Spanish voice)
  12. Generate TTS via ElevenLabs (Portuguese voice)
  13. Calculate actual durations
  14. Update duration fields
  15. Sync with video/animation pipeline
```

---

## 🚫 Forbidden Actions

1. **Never delete English content** — Always preserve original
2. **Never invent facts** — If unsure, flag for review
3. **Never change meaning** — Accuracy over fluency
4. **Never use browser TTS** — Per CLAUDE.md
5. **Never guess cultural references** — Flag for review
6. **Never translate code/IDs** — voice_id, file paths, etc.
7. **Never skip validation** — JSON must be valid

---

## 📁 Working Directories

```
Input:
  public/lessons/day-{N}.json          ← English source

Output (Phase 1 - Staging):
  content/translations/es/day-{N}.json  ← Spanish drafts
  content/translations/pt/day-{N}.json  ← Portuguese drafts

Output (Phase 2 - Production):
  public/lessons/day-{N}.json           ← Multilingual embedded
  OR
  public/lessons/es/day-{N}.json        ← Language-specific files
  public/lessons/pt/day-{N}.json
```

---

## 📊 Progress Tracking

### Translation Status Schema

```json
{
  "day": 1,
  "en": { "status": "complete", "version": "v4.0" },
  "es": { "status": "draft", "version": "v1.0", "reviewed": false },
  "pt": { "status": "draft", "version": "v1.0", "reviewed": false }
}
```

### Batch Processing

- Process in batches of 10-20 lessons
- Commit after each batch
- Track progress in `content/translations/PROGRESS.json`

---

## 🎯 Success Criteria

A lesson translation is COMPLETE when:

- [ ] All translatable fields have ES and PT versions
- [ ] JSON is valid and parseable
- [ ] Kelly's personality is preserved
- [ ] Educational facts are accurate
- [ ] Cultural references are appropriate
- [ ] No English text remains in translated fields
- [ ] Proper nouns are handled correctly
- [ ] Length is reasonable (not 2x original)

---

## 📋 Approval Checklist

Before starting translation work, confirm:

- [ ] File structure decision (embedded vs. separate)
- [ ] API changes needed for multilingual loading
- [ ] Voice model availability (ES/PT ElevenLabs voices)
- [ ] Review process for human QA
- [ ] Budget for TTS generation (~365 lessons × 2 languages)

---

## 🔧 Tooling Needed

1. **Validation script** — Check JSON structure and required fields
2. **Progress tracker** — Which lessons are translated, reviewed, approved
3. **Diff viewer** — Compare EN source to ES/PT translations
4. **Length estimator** — Predict audio duration from text
5. **Batch translator** — Process multiple lessons efficiently

---

## 📝 Example Translation

### English Source (Day 1, hook)

```json
{
  "hook": {
    "script": "Welcome to Day One. Not just of this journey, but of something bigger. Every moment offers a chance to begin again. Today, we explore why fresh starts are more available than we think.",
    "duration": 12
  }
}
```

### Spanish Translation

```json
{
  "hook": {
    "script": "Bienvenidos al Día Uno. No solo de este viaje, sino de algo más grande. Cada momento ofrece la oportunidad de empezar de nuevo. Hoy exploramos por qué los nuevos comienzos están más disponibles de lo que creemos.",
    "duration": 14
  }
}
```

### Portuguese Translation

```json
{
  "hook": {
    "script": "Bem-vindos ao Dia Um. Não apenas desta jornada, mas de algo maior. Cada momento oferece a chance de recomeçar. Hoje exploramos por que novos começos estão mais disponíveis do que pensamos.",
    "duration": 13
  }
}
```

---

## 🚀 Ready to Begin?

Before executing, I need your decision on:

1. **File structure:** Embedded multilingual OR separate language directories?
2. **Staging location:** `content/translations/` OR directly to `public/lessons/`?
3. **Batch size:** How many lessons per commit?
4. **Priority order:** Days 1-50 first, or most recent (Days 161+)?

---

*These rules ensure consistent, high-quality translations that preserve Kelly's voice while serving learners worldwide.*
