# Commons-Governed Content Architecture

**Created:** December 17, 2025  
**Status:** Ultra-Think Design Document  
**Goal:** All post-launch content changes flow through Learner Commons

---

## Executive Summary

After launch, **the only way to change what Kelly says is through Learner Commons**. This creates:
- Democratic governance of educational content
- Community-driven quality improvement
- Audit trail for every change
- Version history for every phase of every lesson

---

## The Problem

### Current State
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Developer      │────▶│  Static Files   │────▶│  User Sees      │
│  edits JSON     │     │  /public/data/  │     │  Lesson         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ↑
        │ Manual code deploy
        │ No governance
        │ No audit trail
```

### Desired State
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Community      │────▶│  Commons        │────▶│  User Sees      │
│  Proposals      │     │  (Source of     │     │  Lesson         │
│  & Votes        │     │   Truth)        │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ↑                       │
        │                       ├──▶ Static Files (cache/fallback)
        │                       ├──▶ API responses
        │                       └──▶ Audio/Video regeneration queue
        │
        └── Staff review gate for safety
```

---

## Core Concept: Content Addressable by Phase

### Every Piece of Content Has a Unique ID

```typescript
// Content Address Structure
interface ContentAddress {
  day: number;           // 1-365
  phase: Phase;          // Hook, Cliff, Fact1, Fact2, Fact3, Wisdom, Outro
  type: ContentType;     // talk, question, option, response, comment
  variant?: string;      // A, B, or specific variant key
  age?: AgeBucket;       // If age-specific content
  language?: Language;   // en, es, fr
}

// Examples:
// day=17, phase=hook, type=talk                    → Main hook script
// day=17, phase=cliff, type=option, variant=A     → Option A text
// day=17, phase=cliff, type=response, variant=A   → Response to option A
// day=17, phase=fact1, type=talk, age=2-5         → Age-specific variant
```

### Content Address String Format
```
{day}.{phase}.{type}[.{variant}][.{age}][.{lang}]

Examples:
  017.hook.talk              → Day 17 Hook talk script
  017.cliff.option.A         → Day 17 Cliff option A
  017.cliff.response.A       → Day 17 Cliff response to A
  017.fact1.talk.2-5         → Day 17 Fact1 for ages 2-5
  017.hook.talk.en           → Day 17 Hook in English
  351.wisdom.talk            → Day 351 Wisdom talk script
```

---

## Database Schema: Commons-Governed Content

### Table: `content_atoms` (Source of Truth)

```sql
-- THE SINGLE SOURCE OF TRUTH FOR ALL LESSON CONTENT
CREATE TABLE content_atoms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Address (unique composite key)
    day_number INTEGER NOT NULL CHECK (day_number >= 1 AND day_number <= 365),
    phase TEXT NOT NULL CHECK (phase IN (
        'hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro'
    )),
    content_type TEXT NOT NULL CHECK (content_type IN (
        'talk', 'question', 'option', 'response', 'comment', 'fun_fact'
    )),
    variant TEXT,                    -- A, B, or null for main content
    age_bucket TEXT,                 -- 2-5, 6-12, 13-17, 18-35, 36-60, 61+
    language TEXT DEFAULT 'en',      -- en, es, fr
    
    -- The actual content
    text_content TEXT NOT NULL,      -- Script/text
    metadata JSONB DEFAULT '{}',     -- kellyPose, kellyEmotion, duration, etc.
    
    -- Governance
    version INTEGER DEFAULT 1,
    is_live BOOLEAN DEFAULT true,    -- Currently active version
    change_source TEXT NOT NULL CHECK (change_source IN (
        'initial_seed',              -- Launch content
        'commons_proposal',          -- Community proposal
        'staff_direct',              -- Staff edit (with justification)
        'automated_translation',     -- Machine translation
        'audio_regeneration'         -- Content updated for TTS compatibility
    )),
    change_reference UUID,           -- proposal_id if from commons
    change_reason TEXT,              -- Required for staff_direct
    changed_by UUID REFERENCES auth.users(id),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint: one live version per address
    UNIQUE(day_number, phase, content_type, COALESCE(variant, ''), 
           COALESCE(age_bucket, ''), COALESCE(language, 'en'), is_live)
);

-- Indexes for fast lookup
CREATE INDEX idx_content_day ON content_atoms(day_number);
CREATE INDEX idx_content_phase ON content_atoms(day_number, phase);
CREATE INDEX idx_content_live ON content_atoms(is_live) WHERE is_live = true;
CREATE INDEX idx_content_address ON content_atoms(
    day_number, phase, content_type, variant, age_bucket, language
);
```

### Table: `content_history` (Audit Trail)

```sql
-- EVERY VERSION OF EVERY PIECE OF CONTENT
CREATE TABLE content_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_atom_id UUID NOT NULL REFERENCES content_atoms(id),
    
    -- Snapshot
    version INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    metadata JSONB,
    
    -- Change tracking
    change_source TEXT NOT NULL,
    change_reference UUID,
    change_reason TEXT,
    changed_by UUID REFERENCES auth.users(id),
    
    -- When
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- For rollback capability
    is_rollback_target BOOLEAN DEFAULT false
);

CREATE INDEX idx_history_atom ON content_history(content_atom_id);
CREATE INDEX idx_history_version ON content_history(content_atom_id, version DESC);
```

### Enhanced `commons_proposals` Table

```sql
-- Add these fields to the existing commons_proposals table
ALTER TABLE commons_proposals ADD COLUMN IF NOT EXISTS
    target_atoms TEXT[] NOT NULL DEFAULT '{}';  -- Content addresses affected

ALTER TABLE commons_proposals ADD COLUMN IF NOT EXISTS
    proposed_changes JSONB NOT NULL DEFAULT '{}';
    -- Format: { "017.hook.talk": { "old": "...", "new": "..." } }

ALTER TABLE commons_proposals ADD COLUMN IF NOT EXISTS
    requires_audio_regen BOOLEAN DEFAULT false;
    
ALTER TABLE commons_proposals ADD COLUMN IF NOT EXISTS
    requires_video_regen BOOLEAN DEFAULT false;

-- Example proposal targeting specific content:
-- {
--   "target_atoms": ["017.fact2.talk"],
--   "proposed_changes": {
--     "017.fact2.talk": {
--       "current": "Dreams aren't just entertainment...",
--       "proposed": "Dreams aren't just nighttime entertainment..."
--     }
--   },
--   "requires_audio_regen": true,
--   "rationale": "Added 'nighttime' for clarity about when dreams occur"
-- }
```

---

## Content Lifecycle

### 1. Initial Seed (Launch)

```
Static Files → Migration Script → content_atoms table
                                     │
                                     └─▶ change_source: 'initial_seed'
                                         version: 1
                                         is_live: true
```

### 2. Community Proposal

```
User submits proposal
        │
        ▼
┌───────────────────┐
│ commons_proposals │
│ status: 'open'    │
│ target_atoms: [...│
└───────────────────┘
        │
        ▼ (14 days voting + review)
        │
┌───────────────────┐
│ status: 'approved'│
└───────────────────┘
        │
        ▼ (automated pipeline)
        │
┌───────────────────────────────────────┐
│ For each target_atom:                 │
│   1. Archive current → content_history│
│   2. Update content_atoms             │
│   3. Mark old version is_live=false   │
│   4. Queue audio regeneration         │
│   5. Queue static file rebuild        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│ status: 'implemented' │
└───────────────────┘
```

### 3. Static File Generation

```typescript
// Nightly job: regenerate static files from content_atoms
async function regenerateStaticFiles() {
  for (let day = 1; day <= 365; day++) {
    const atoms = await getContentAtomsForDay(day);
    const staticPack = buildStaticPack(atoms);
    await writeFile(`public/data/day-${pad(day)}-complete.js`, staticPack);
  }
}
```

---

## API Design

### GET /api/content/:address

Returns the current live content for an address.

```typescript
// GET /api/content/017.hook.talk
{
  "address": "017.hook.talk",
  "content": "Have you ever woken up from a dream...",
  "metadata": {
    "kellyPose": "welcome",
    "kellyEmotion": "curious",
    "duration": 18
  },
  "version": 3,
  "lastUpdated": "2025-12-20T14:30:00Z",
  "source": "commons_proposal",
  "proposalId": "abc-123"
}
```

### GET /api/content/:day/full

Returns all content for a day (for lesson player).

```typescript
// GET /api/content/17/full
{
  "day": 17,
  "lesson": { ... },
  "phases": {
    "hook": {
      "talk": { "content": "...", "version": 3 },
      "question": { "content": "...", "version": 1 },
      "options": {
        "A": { "content": "...", "version": 2 },
        "B": { "content": "...", "version": 1 }
      },
      "responses": {
        "A": { "content": "...", "version": 2 },
        "B": { "content": "...", "version": 1 }
      }
    },
    // ... other phases
  }
}
```

### POST /api/content/propose

Submit a content change proposal.

```typescript
// POST /api/content/propose
{
  "targetAtoms": ["017.fact2.talk"],
  "proposedChanges": {
    "017.fact2.talk": {
      "proposed": "Dreams aren't just nighttime entertainment..."
    }
  },
  "type": "enhance",
  "title": "Clarify when dreams occur",
  "rationale": "Adding 'nighttime' makes it clearer for younger learners"
}
```

---

## Per-Phase Commons UI

### In Lesson Player (`/learn.html`)

Each phase shows a Commons icon that opens phase-specific discussions:

```
┌────────────────────────────────────────────────────────┐
│  🌙 FACT 2: Memory Consolidation          💬 Commons  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  "Dreams aren't just entertainment—they're workers.   │
│   While you dream, your brain is doing something      │
│   called memory consolidation..."                     │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │ 📝 Suggest Improvement │ 💡 3 proposals │ 📊 12%  │ │
│  │    for this phase        open           changed  │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### Phase Commons Modal

```
┌─────────────────────────────────────────────────────────┐
│  💬 Commons: Day 17, Fact 2                        ✕    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📝 OPEN PROPOSALS (3)                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 🔼 15  Add clarification about REM timing        │   │
│  │        by @curious_learner • 3 days ago          │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 🔼 8   Simplify "memory consolidation" term      │   │
│  │        by @educator_mom • 5 days ago             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  📚 COMMUNITY NOTES (5)                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 💡 Expert note: Memory consolidation research... │   │
│  │    by @neuroscience_prof ✓ verified              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  📜 VERSION HISTORY                                     │
│  ├─ v3 (Dec 20, 2025) - Added "nighttime"              │
│  ├─ v2 (Dec 18, 2025) - Clarified memory concept       │
│  └─ v1 (Dec 17, 2025) - Initial launch                 │
│                                                         │
│  [+ Suggest Improvement] [+ Add Note]                   │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Content Migration (Week 1)

1. Create `content_atoms` and `content_history` tables
2. Write migration script to seed from static files
3. Generate unique IDs for all existing content
4. Verify data integrity

### Phase 2: API Layer (Week 2)

1. Build `/api/content/:address` endpoint
2. Build `/api/content/:day/full` endpoint
3. Update lesson loader to try API before static files
4. Add fallback to static files if API fails

### Phase 3: Proposal Enhancement (Week 3)

1. Add `target_atoms` to proposals
2. Build UI for targeting specific phases
3. Show current content alongside proposed changes
4. Diff view for reviewers

### Phase 4: Implementation Pipeline (Week 4)

1. Automated content update on approval
2. Audio regeneration queue
3. Static file rebuild job
4. Notification to affected users

### Phase 5: Per-Phase UI (Week 5-6)

1. Commons icon per phase in lesson player
2. Phase-specific proposal modal
3. Version history view
4. Community notes per phase

---

## Governance Rules

### Who Can Propose Changes

| User Level | Can Propose | Requires Review |
|------------|-------------|-----------------|
| Guest | ❌ | N/A |
| Registered | ✅ | Always |
| Contributor | ✅ | >10 votes OR staff |
| Trusted | ✅ | >25 votes OR staff |
| Expert | ✅ | >50 votes OR auto-approve |
| Staff | ✅ | Optional |

### Voting Thresholds

| Proposal Type | Votes to Review | Auto-Approve |
|--------------|-----------------|--------------|
| Typo fix | 5 | 25+ with 90% up |
| Enhance | 10 | Never |
| Expand | 15 | Never |
| Remove | 25 | Never |
| Correct (factual) | 5 | Never (requires expert) |

### Safety Gates

1. **Profanity filter** - Auto-reject inappropriate content
2. **Factual claims** - Require source citation
3. **Age-appropriateness** - Content flagged for review
4. **Staff veto** - Any staff can block for 48h review
5. **Rollback capability** - Any version can be restored

---

## Metrics & Monitoring

### Content Health Dashboard

```
┌─────────────────────────────────────────────────────────┐
│  📊 Content Governance Dashboard                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Total Content Atoms: 25,550 (365 days × 70 avg atoms) │
│  Changed Since Launch: 847 (3.3%)                       │
│  Pending Proposals: 42                                  │
│  Avg Time to Implementation: 8.3 days                   │
│                                                         │
│  Most Active Days:                                      │
│  1. Day 17 (Why We Dream) - 23 changes                 │
│  2. Day 1 (The Sun) - 18 changes                       │
│  3. Day 42 (Black Holes) - 15 changes                  │
│                                                         │
│  Top Contributors:                                      │
│  1. @science_teacher - 45 proposals, 38 implemented    │
│  2. @curious_parent - 32 proposals, 28 implemented     │
│  3. @neuroscience_prof - 12 proposals, 12 implemented  │
└─────────────────────────────────────────────────────────┘
```

---

## Migration Strategy

### Day 1: Database Setup
```sql
-- Run in Supabase SQL editor
\i docs/backend/migrations/002_content_atoms.sql
```

### Day 2-3: Content Seed
```bash
# Migrate all static files to content_atoms
npx tsx scripts/migrate-to-content-atoms.ts --all

# Verify
npx tsx scripts/verify-content-atoms.ts
```

### Day 4: API Deployment
```bash
# Deploy new API endpoints
git push  # Triggers Vercel deploy
```

### Day 5: Feature Flag Rollout
```typescript
// In config.js
KELLY_CONFIG.USE_CONTENT_ATOMS = true;  // Enable API-first loading
KELLY_CONFIG.STATIC_FALLBACK = true;    // Keep static files as backup
```

---

## The Ultimate Goal

**After this is implemented:**

1. ✅ All lesson content lives in `content_atoms` table
2. ✅ Every piece of content has version history
3. ✅ Community can propose changes to any phase
4. ✅ Staff reviews ensure quality and safety
5. ✅ Approved changes auto-propagate to all formats
6. ✅ Static files are regenerated as cache/fallback
7. ✅ Audio/video regeneration is queued automatically
8. ✅ Full audit trail of every change ever made

**The principle:**
> "If it's not in Commons, it's not in the lesson."

---

## Related Documents

- [LEARNER_COMMONS.md](../features/LEARNER_COMMONS.md) - Original Commons design
- [SUPABASE_INDEPENDENCE_ROADMAP.md](../SUPABASE_INDEPENDENCE_ROADMAP.md) - Database strategy
- [PHASEDNA_V2_COMPLETE.md](../phasedna/PHASEDNA_V2_COMPLETE.md) - Lesson structure

---

*Last updated: December 17, 2025*
