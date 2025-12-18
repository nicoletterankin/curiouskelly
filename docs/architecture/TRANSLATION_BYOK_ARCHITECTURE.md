# 🌍 TRANSLATION BYOK ARCHITECTURE

## Ultra-Think: Scalable Multilingual Content via Community Contribution

**Created:** December 17, 2025  
**Status:** ARCHITECTURE PROPOSAL  
**Scope:** 365 lessons × 3+ languages × 75 fields = 82,000+ strings

---

## 📊 The Problem Scale

| Metric | Count |
|--------|-------|
| Lessons | 365 |
| Fields per lesson | ~75 |
| Target languages (Phase 1) | 2 (ES, PT) |
| Target languages (Phase 2) | +4 (FR, DE, HI, ZH) |
| **Total translations needed** | **54,750** (Phase 1) |
| **Estimated characters** | **~5.5M** |

### Cost Comparison (If We Pay Directly)

| Provider | Rate | Phase 1 Cost |
|----------|------|--------------|
| Google Translate API | $20/1M chars | ~$110 |
| DeepL API Pro | $25/1M chars | ~$138 |
| OpenAI GPT-4o | ~$5/1M tokens | ~$200 |
| Claude Sonnet | ~$3/1M tokens | ~$120 |
| **Human Translation** | $0.10/word | **$55,000+** |

**Key Insight:** API translation is cheap. The expensive part is **quality assurance** and **voice preservation**.

---

## 🎯 Design Goals

1. **Zero-cost to us** — Users contribute translations using their own API credits
2. **Kelly voice preserved** — Not just translation, but personality localization
3. **Quality enforced** — Bad translations never reach production
4. **Community contribution** — Translations benefit everyone
5. **Incremental progress** — Any user can translate any lesson
6. **Provider agnostic** — Support Google, DeepL, OpenAI, Anthropic, etc.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRANSLATION BYOK SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   USER       │   │   PROVIDER   │   │   QUALITY    │   │   COMMONS    │ │
│  │   LAYER      │   │   LAYER      │   │   LAYER      │   │   LAYER      │ │
│  ├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤ │
│  │ • API Key    │   │ • Google     │   │ • Length     │   │ • Cache      │ │
│  │ • Selection  │   │ • DeepL      │   │ • Format     │   │ • Versioning │ │
│  │ • Progress   │   │ • OpenAI     │   │ • Voice      │   │ • Attribution│ │
│  │ • Credits    │   │ • Anthropic  │   │ • Accuracy   │   │ • Fallback   │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│         │                 │                 │                 │             │
│         └─────────────────┼─────────────────┼─────────────────┘             │
│                           ▼                 ▼                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     TRANSLATION PIPELINE                             │   │
│  │                                                                      │   │
│  │   [Source EN] → [Translate] → [Validate] → [Review] → [Publish]    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 BYOK Key Management

### Supported Providers

| Provider | Key Format | Free Tier | Notes |
|----------|------------|-----------|-------|
| **Google Cloud Translation** | `AIza...` | 500K chars/mo | Best for bulk |
| **DeepL** | `deadbeef-...` | 500K chars/mo | Best quality ES/PT |
| **OpenAI** | `sk-...` | None | Best for voice preservation |
| **Anthropic** | `sk-ant-...` | None | Best for personality |
| **Azure Translator** | GUID | 2M chars/mo | Enterprise option |

### Key Storage Strategy

```
NEVER store raw API keys in database.

Option A: Browser-only (keys in localStorage)
  ✅ Zero server storage
  ✅ User controls keys
  ❌ Can't batch overnight
  ❌ Lost on device change

Option B: Encrypted server storage (user's choice)
  ✅ Batch processing possible
  ✅ Cross-device
  ❌ We hold encrypted keys
  ❌ Trust requirement

Option C: Hybrid (Recommended)
  - Keys stay in browser by default
  - User can opt-in to server storage
  - Keys encrypted with user's password
  - Revocation at any time
```

---

## 📋 Translation Unit Schema

### What Gets Translated

```json
{
  "translation_unit": {
    "id": "day-1.phases.hook.script",
    "source_text": "Welcome to Day One...",
    "source_lang": "en",
    "target_lang": "es",
    "context": {
      "field_type": "script",
      "phase": "hook",
      "lesson_topic": "Starting Fresh",
      "kelly_mode": "warm",
      "character_limit": null,
      "preserve_formatting": true
    },
    "instructions": "Kelly is a warm, curious AI teacher. Use informal 'tú'. Preserve rhetorical questions and pauses."
  }
}
```

### Translation Result

```json
{
  "translation_result": {
    "unit_id": "day-1.phases.hook.script",
    "translated_text": "Bienvenidos al Día Uno...",
    "target_lang": "es",
    "provider": "deepl",
    "provider_model": "deepl-pro",
    "confidence": 0.95,
    "quality_scores": {
      "length_ratio": 1.12,
      "format_preserved": true,
      "voice_score": 0.88,
      "fluency_score": 0.92
    },
    "contributed_by": "user_abc123",
    "contributed_at": "2025-12-17T10:30:00Z",
    "status": "pending_review"
  }
}
```

---

## 🔄 Translation Pipeline

### Stage 1: Selection
```
User selects:
├── Language: Spanish (es)
├── Scope: Day 1-10
├── Provider: DeepL (their key)
└── Quality: Standard / High (affects prompting)
```

### Stage 2: Batching
```
System creates batches:
├── Group by field type (scripts together, prompts together)
├── Include context for each batch
├── Estimate cost before running
└── User confirms
```

### Stage 3: Translation
```
For each batch:
├── Send to provider with context prompt
├── Handle rate limits gracefully
├── Store raw response
└── Track progress
```

### Stage 4: Quality Validation

```javascript
function validateTranslation(source, translation, context) {
  const checks = {
    // Length check (translations shouldn't be 2x longer)
    lengthRatio: translation.length / source.length,
    lengthValid: ratio > 0.8 && ratio < 1.5,
    
    // Format preservation
    hasMatchingPunctuation: checkPunctuation(source, translation),
    preservesLineBreaks: checkLineBreaks(source, translation),
    
    // Placeholder preservation (if any)
    preservesPlaceholders: checkPlaceholders(source, translation),
    
    // Kelly voice markers
    hasWarmGreeting: /bienvenid|bem-vind/i.test(translation),
    usesInformalYou: /\btú\b|\bvocê\b/i.test(translation),
    
    // No English remnants
    noEnglishWords: !containsCommonEnglishWords(translation)
  };
  
  return {
    passed: Object.values(checks).every(v => v === true || v > 0.8),
    scores: checks,
    issues: Object.entries(checks).filter(([k, v]) => !v).map(([k]) => k)
  };
}
```

### Stage 5: Review Queue

```
Translation statuses:
├── draft        → Just translated, not validated
├── validated    → Passed auto-checks
├── flagged      → Failed some checks, needs human review
├── approved     → Human approved
├── published    → Live in production
└── rejected     → Bad translation, try again
```

---

## 🗄️ Database Schema

### translation_commons

```sql
CREATE TABLE translation_commons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- What's being translated
  unit_id TEXT NOT NULL,           -- "day-1.phases.hook.script"
  source_lang TEXT NOT NULL DEFAULT 'en',
  target_lang TEXT NOT NULL,       -- "es", "pt", etc.
  
  -- The translation
  source_text TEXT NOT NULL,
  translated_text TEXT NOT NULL,
  
  -- Quality & provenance
  provider TEXT NOT NULL,          -- "deepl", "openai", etc.
  provider_model TEXT,
  quality_scores JSONB,
  
  -- Attribution
  contributed_by UUID REFERENCES users(id),
  contributed_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Lifecycle
  status TEXT DEFAULT 'draft',     -- draft, validated, flagged, approved, published, rejected
  reviewed_by UUID REFERENCES users(id),
  reviewed_at TIMESTAMPTZ,
  
  -- Versioning
  source_version TEXT,             -- Hash of source text
  version INTEGER DEFAULT 1,
  replaced_by UUID REFERENCES translation_commons(id),
  
  -- Indexes
  UNIQUE(unit_id, target_lang, version)
);

-- Enable RLS
ALTER TABLE translation_commons ENABLE ROW LEVEL SECURITY;

-- Anyone can read published translations
CREATE POLICY "Anyone can read published"
  ON translation_commons FOR SELECT
  USING (status = 'published');

-- Contributors can see their own
CREATE POLICY "Users can see own contributions"
  ON translation_commons FOR SELECT
  USING (contributed_by = auth.uid());
```

### translation_progress

```sql
CREATE TABLE translation_progress (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  day_number INTEGER NOT NULL,
  target_lang TEXT NOT NULL,
  
  -- Progress tracking
  total_units INTEGER NOT NULL,
  translated_count INTEGER DEFAULT 0,
  validated_count INTEGER DEFAULT 0,
  published_count INTEGER DEFAULT 0,
  
  -- Completion
  is_complete BOOLEAN DEFAULT FALSE,
  completed_at TIMESTAMPTZ,
  
  UNIQUE(day_number, target_lang)
);
```

### user_translation_stats

```sql
CREATE TABLE user_translation_stats (
  user_id UUID PRIMARY KEY REFERENCES users(id),
  
  -- Contribution counts
  translations_contributed INTEGER DEFAULT 0,
  translations_approved INTEGER DEFAULT 0,
  translations_rejected INTEGER DEFAULT 0,
  
  -- Languages
  languages_contributed TEXT[] DEFAULT '{}',
  
  -- Recognition
  contribution_rank TEXT,  -- "Bronze", "Silver", "Gold", "Platinum"
  badges JSONB DEFAULT '[]',
  
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 🎨 User Interface

### Translation Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│  🌍 Translation Center                              [Your API Keys] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Your Contribution: 127 translations | Rank: Silver Contributor 🥈  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Language Progress                                                ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ 🇪🇸 Spanish   ████████████░░░░░░░░  58% (212/365 lessons)       ││
│  │ 🇧🇷 Portuguese ████████░░░░░░░░░░░░  42% (154/365 lessons)       ││
│  │ 🇫🇷 French     ██░░░░░░░░░░░░░░░░░░   8%  (29/365 lessons)       ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Quick Actions                                                    ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ [Translate Day 1-10 to Spanish]  Est. 2,400 chars | ~$0.05      ││
│  │ [Help Review Flagged]            12 translations need review     ││
│  │ [Improve Low-Rated]              8 translations scored < 0.7     ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Leaderboard This Week                                            ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ 1. Maria G.     342 translations  🇪🇸 🇧🇷                         ││
│  │ 2. João P.      289 translations  🇧🇷                             ││
│  │ 3. Pierre L.    156 translations  🇫🇷                             ││
│  │ 4. You          127 translations  🇪🇸                             ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Translation Editor

```
┌─────────────────────────────────────────────────────────────────────┐
│  Day 1: Starting Fresh | Phase: Hook | Field: script               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  English (Source)                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Welcome to Day One. Not just of this journey, but of something  ││
│  │ bigger. Every moment offers a chance to begin again.            ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  Spanish (Translation)                              [Auto-Translate]│
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Bienvenidos al Día Uno. No solo de este viaje, sino de algo    ││
│  │ más grande. Cada momento ofrece la oportunidad de empezar       ││
│  │ de nuevo.                                                        ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  Quality Checks                                                      │
│  ✅ Length ratio: 1.12 (acceptable)                                 │
│  ✅ Uses informal "tú"                                              │
│  ✅ Warm greeting preserved                                         │
│  ⚠️ Consider: "Bienvenido" (singular) vs "Bienvenidos" (plural)    │
│                                                                      │
│  Context: Kelly is warmly welcoming the learner to their first     │
│  lesson. The tone should be inviting, personal, and exciting.       │
│                                                                      │
│  [Previous Field]                    [Save & Next]    [Skip]        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create translation_commons table
- [ ] Create translation_progress tracking
- [ ] Build BYOK key management UI
- [ ] Implement Google Translate connector
- [ ] Basic quality validation

### Phase 2: Pipeline (Week 2)
- [ ] Add DeepL connector
- [ ] Add OpenAI connector with Kelly voice prompting
- [ ] Batch translation system
- [ ] Progress dashboard
- [ ] Cost estimation

### Phase 3: Quality (Week 3)
- [ ] Review queue UI
- [ ] Voice consistency scoring
- [ ] Human review workflow
- [ ] Flagging and improvement system

### Phase 4: Community (Week 4)
- [ ] Leaderboard
- [ ] Contribution badges
- [ ] Translation attribution
- [ ] Quality ratings by users

---

## 🎤 Kelly Voice Preservation

### The Challenge

Machine translation loses personality. "Welcome!" becomes "Bienvenidos!" but loses the warmth.

### The Solution: Voice-Aware Prompting

```javascript
function buildTranslationPrompt(unit, targetLang) {
  return `
You are translating content for Kelly, a warm, curious, intelligent AI teacher.

## Kelly's Voice Characteristics
- Warm and welcoming, never cold or clinical
- Uses informal address (tú in Spanish, você in Portuguese)
- Asks rhetorical questions to engage
- Celebrates learning moments with genuine enthusiasm
- Uses pauses (em-dashes) for dramatic effect
- Never lectures; always explores together

## Translation Guidelines for ${targetLang}
- Preserve rhetorical structure (questions stay questions)
- Keep the same level of enthusiasm
- Maintain pauses and emphasis markers
- Use culturally appropriate idioms
- Never add or remove meaning

## Context
- Field type: ${unit.context.field_type}
- Phase: ${unit.context.phase}
- Topic: ${unit.context.lesson_topic}
- Kelly's mode: ${unit.context.kelly_mode}

## Source Text (English)
${unit.source_text}

## Your Translation (${targetLang})
Translate the above, preserving Kelly's warm, curious voice:
`;
}
```

---

## 💡 Smart Features

### 1. Translation Memory

```javascript
// Before translating, check if similar text was already translated
async function checkTranslationMemory(sourceText, targetLang) {
  const similar = await supabase
    .from('translation_commons')
    .select('source_text, translated_text, quality_scores')
    .eq('target_lang', targetLang)
    .eq('status', 'published')
    .textSearch('source_text', sourceText.substring(0, 50));
  
  if (similar.data?.length > 0) {
    // Suggest existing translation or use as reference
    return similar.data[0];
  }
  return null;
}
```

### 2. Consistency Checking

```javascript
// Ensure terminology consistency across lessons
const KELLY_TERMS = {
  en: {
    "Here's today's wisdom": true,
    "Let's explore": true,
    "Have you ever wondered": true
  },
  es: {
    "Esta es la sabiduría de hoy": true,
    "Exploremos": true,
    "¿Alguna vez te has preguntado": true
  }
};

function checkTerminologyConsistency(source, translation, targetLang) {
  // Flag if standard phrases aren't translated consistently
}
```

### 3. Automatic Backfill

```javascript
// When source text changes, mark translations as needing update
async function onSourceTextChange(unitId, newSourceText) {
  const oldHash = await getStoredHash(unitId);
  const newHash = hashText(newSourceText);
  
  if (oldHash !== newHash) {
    // Mark all translations as potentially outdated
    await supabase
      .from('translation_commons')
      .update({ status: 'needs_update', source_version: newHash })
      .eq('unit_id', unitId);
  }
}
```

---

## 📊 Success Metrics

| Metric | Target |
|--------|--------|
| Spanish coverage | 100% of 365 lessons by Q1 2026 |
| Portuguese coverage | 100% of 365 lessons by Q1 2026 |
| Average quality score | > 0.85 |
| User contribution rate | > 50% of translations from users |
| Review turnaround | < 24 hours for flagged items |
| Zero cost to company | 100% BYOK funded |

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Low user contribution | Gamification, leaderboards, badges |
| Poor quality translations | Strict validation, human review queue |
| API key security | Browser-only storage, no server keys |
| Vandalism/spam | Rate limiting, quality gates, IP banning |
| Voice drift | Terminology database, consistency checks |
| Provider API changes | Abstract provider layer, multiple options |

---

## 🔧 API Endpoints

```
POST /api/translation/translate
  - body: { unit_id, target_lang, provider, api_key }
  - returns: { translated_text, quality_scores, status }

GET /api/translation/progress
  - query: { lang }
  - returns: { lessons: [{ day, status, percent_complete }] }

POST /api/translation/review
  - body: { translation_id, action: 'approve' | 'reject' | 'edit', new_text? }
  - returns: { status }

GET /api/translation/leaderboard
  - returns: { users: [{ name, count, languages }] }
```

---

## 🎯 Decision Points for You

1. **Key Storage:** Browser-only (safer) or optional server storage (more features)?

2. **Quality Bar:** How strict? Auto-publish validated, or always human review?

3. **Gamification Level:** Simple counts, or full badges/ranks/rewards?

4. **Priority Languages:** Just ES/PT first, or open to all from start?

5. **Contribution Model:** Anyone can contribute, or require account/subscription?

---

*This architecture enables community-powered translation at zero cost while maintaining Kelly's unique voice across all languages.*
