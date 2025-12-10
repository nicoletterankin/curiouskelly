# Learner Commons - Democratic Lesson Governance

> *"How we create and manage every little thing Kelly says is our product and our promise."*

---

## Executive Summary

The **Learner Commons** is a community-governed knowledge base where anyone can vote for lessons to be enhanced, removed, changed, or expanded. It's the wiki for "Lesson of the Day" — the democratic authority to improve how Kelly teaches.

**Key principle**: The authority to change Kelly comes from making it easy to contribute.

---

## Vision

Every lesson has a mini knowledge base where learners can:
- 🗳️ **Vote** on proposed changes
- 💡 **Propose** enhancements, corrections, or new content
- 💬 **Discuss** teaching approaches and perspectives
- 📊 **Track** the status of community suggestions
- ✨ **Celebrate** when their contributions ship

---

## System Architecture

### Core Entities

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LEARNER COMMONS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   LESSONS    │───▶│  PROPOSALS   │───▶│    VOTES     │          │
│  │ (core_lessons)│   │              │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         │                   ▼                   │                   │
│         │            ┌──────────────┐          │                   │
│         │            │ DISCUSSIONS  │          │                   │
│         │            │  (threads)   │          │                   │
│         │            └──────────────┘          │                   │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────────────────────────────────────────────┐          │
│  │              LESSON KNOWLEDGE BASE                    │          │
│  │  • Expert notes    • Historical context               │          │
│  │  • Age adaptations • Source citations                 │          │
│  │  • Related topics  • Community insights               │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Supabase Schema

### Table: `commons_proposals`

Proposed changes to lessons from the community.

```sql
CREATE TABLE commons_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lesson_id UUID NOT NULL REFERENCES core_lessons(id),
    user_id UUID REFERENCES auth.users(id),
    
    -- Proposal type
    type TEXT NOT NULL CHECK (type IN (
        'enhance',      -- Make existing content better
        'expand',       -- Add new perspectives/content
        'correct',      -- Fix factual errors
        'simplify',     -- Make easier to understand
        'remove',       -- Remove problematic content
        'translate',    -- Add/improve translations
        'accessibility' -- Improve accessibility
    )),
    
    -- Content
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    rationale TEXT,                    -- Why this change matters
    proposed_content JSONB,            -- The actual proposed changes
    affected_phases TEXT[],            -- Which lesson phases affected
    affected_age_groups TEXT[],        -- Which age groups affected
    
    -- Status tracking
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN (
        'open',           -- Accepting votes
        'under_review',   -- Being evaluated by team
        'approved',       -- Accepted, pending implementation
        'implemented',    -- Live in production
        'declined',       -- Not accepted (with reason)
        'withdrawn'       -- Author withdrew
    )),
    status_reason TEXT,                -- Why status changed
    
    -- Voting summary (denormalized for performance)
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    vote_score INTEGER GENERATED ALWAYS AS (upvotes - downvotes) STORED,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    implemented_at TIMESTAMPTZ,
    
    -- Moderation
    is_flagged BOOLEAN DEFAULT FALSE,
    flag_reason TEXT
);

-- Indexes for performance
CREATE INDEX idx_proposals_lesson ON commons_proposals(lesson_id);
CREATE INDEX idx_proposals_status ON commons_proposals(status);
CREATE INDEX idx_proposals_type ON commons_proposals(type);
CREATE INDEX idx_proposals_score ON commons_proposals(vote_score DESC);
CREATE INDEX idx_proposals_user ON commons_proposals(user_id);
```

### Table: `commons_votes`

User votes on proposals.

```sql
CREATE TABLE commons_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id UUID NOT NULL REFERENCES commons_proposals(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id),
    
    vote_type TEXT NOT NULL CHECK (vote_type IN ('up', 'down')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- One vote per user per proposal
    UNIQUE(proposal_id, user_id)
);

-- Trigger to update vote counts on proposals
CREATE OR REPLACE FUNCTION update_proposal_votes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.vote_type = 'up' THEN
            UPDATE commons_proposals SET upvotes = upvotes + 1 WHERE id = NEW.proposal_id;
        ELSE
            UPDATE commons_proposals SET downvotes = downvotes + 1 WHERE id = NEW.proposal_id;
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        IF OLD.vote_type = 'up' THEN
            UPDATE commons_proposals SET upvotes = upvotes - 1 WHERE id = OLD.proposal_id;
        ELSE
            UPDATE commons_proposals SET downvotes = downvotes - 1 WHERE id = OLD.proposal_id;
        END IF;
    ELSIF TG_OP = 'UPDATE' THEN
        -- Handle vote change
        IF OLD.vote_type = 'up' THEN
            UPDATE commons_proposals SET upvotes = upvotes - 1 WHERE id = OLD.proposal_id;
        ELSE
            UPDATE commons_proposals SET downvotes = downvotes - 1 WHERE id = OLD.proposal_id;
        END IF;
        IF NEW.vote_type = 'up' THEN
            UPDATE commons_proposals SET upvotes = upvotes + 1 WHERE id = NEW.proposal_id;
        ELSE
            UPDATE commons_proposals SET downvotes = downvotes + 1 WHERE id = NEW.proposal_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_proposal_votes
AFTER INSERT OR UPDATE OR DELETE ON commons_votes
FOR EACH ROW EXECUTE FUNCTION update_proposal_votes();
```

### Table: `commons_discussions`

Discussion threads on proposals or lessons.

```sql
CREATE TABLE commons_discussions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Can be attached to a proposal OR directly to a lesson
    proposal_id UUID REFERENCES commons_proposals(id) ON DELETE CASCADE,
    lesson_id UUID REFERENCES core_lessons(id),
    parent_id UUID REFERENCES commons_discussions(id),  -- For replies
    
    user_id UUID REFERENCES auth.users(id),
    
    -- Content
    content TEXT NOT NULL,
    
    -- Reactions (simplified)
    reaction_counts JSONB DEFAULT '{"helpful": 0, "insightful": 0, "question": 0}',
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_edited BOOLEAN DEFAULT FALSE,
    
    -- Moderation
    is_hidden BOOLEAN DEFAULT FALSE,
    hidden_reason TEXT,
    
    -- At least one context required
    CONSTRAINT discussion_context CHECK (proposal_id IS NOT NULL OR lesson_id IS NOT NULL)
);

CREATE INDEX idx_discussions_proposal ON commons_discussions(proposal_id);
CREATE INDEX idx_discussions_lesson ON commons_discussions(lesson_id);
CREATE INDEX idx_discussions_parent ON commons_discussions(parent_id);
```

### Table: `commons_lesson_notes`

Community-contributed knowledge for each lesson.

```sql
CREATE TABLE commons_lesson_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lesson_id UUID NOT NULL REFERENCES core_lessons(id),
    user_id UUID REFERENCES auth.users(id),
    
    -- Note type
    type TEXT NOT NULL CHECK (type IN (
        'expert_context',      -- Deep domain knowledge
        'historical_note',     -- Historical connections
        'source_citation',     -- Academic/credible sources
        'teaching_tip',        -- Pedagogical advice
        'age_adaptation',      -- Age-specific suggestions
        'cultural_context',    -- Cultural considerations
        'common_misconception',-- What learners often misunderstand
        'real_world_example',  -- Practical applications
        'discussion_prompt',   -- Questions to spark thinking
        'related_topic'        -- Connections to other lessons
    )),
    
    -- Content
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT[],            -- URLs or citations
    related_lessons INTEGER[], -- day_numbers of related lessons
    
    -- Quality signals
    upvotes INTEGER DEFAULT 0,
    is_verified BOOLEAN DEFAULT FALSE,  -- Staff-verified accuracy
    is_featured BOOLEAN DEFAULT FALSE,  -- Highlighted on lesson page
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Moderation
    is_hidden BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_notes_lesson ON commons_lesson_notes(lesson_id);
CREATE INDEX idx_notes_type ON commons_lesson_notes(type);
CREATE INDEX idx_notes_featured ON commons_lesson_notes(is_featured) WHERE is_featured = TRUE;
```

### Table: `commons_user_contributions`

Track user contribution stats for recognition.

```sql
CREATE TABLE commons_user_contributions (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id),
    
    -- Contribution counts
    proposals_submitted INTEGER DEFAULT 0,
    proposals_approved INTEGER DEFAULT 0,
    proposals_implemented INTEGER DEFAULT 0,
    votes_cast INTEGER DEFAULT 0,
    discussions_posted INTEGER DEFAULT 0,
    notes_contributed INTEGER DEFAULT 0,
    
    -- Recognition
    contributor_level TEXT DEFAULT 'newcomer' CHECK (contributor_level IN (
        'newcomer',     -- Just starting
        'contributor',  -- 5+ accepted contributions
        'trusted',      -- 25+ accepted contributions
        'expert',       -- 100+ accepted contributions
        'steward'       -- Community moderator
    )),
    
    -- Activity
    first_contribution_at TIMESTAMPTZ,
    last_contribution_at TIMESTAMPTZ,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Table: `commons_activity_log`

Track all commons activity for transparency.

```sql
CREATE TABLE commons_activity_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Context
    user_id UUID REFERENCES auth.users(id),
    lesson_id UUID REFERENCES core_lessons(id),
    proposal_id UUID REFERENCES commons_proposals(id),
    
    -- Activity
    action TEXT NOT NULL CHECK (action IN (
        'proposal_created',
        'proposal_updated',
        'proposal_approved',
        'proposal_declined',
        'proposal_implemented',
        'vote_cast',
        'discussion_posted',
        'note_added',
        'note_verified',
        'content_flagged',
        'content_moderated'
    )),
    
    -- Details
    details JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- For public activity feed
    is_public BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_activity_lesson ON commons_activity_log(lesson_id);
CREATE INDEX idx_activity_user ON commons_activity_log(user_id);
CREATE INDEX idx_activity_created ON commons_activity_log(created_at DESC);
```

---

## Row Level Security (RLS) Policies

```sql
-- Enable RLS on all commons tables
ALTER TABLE commons_proposals ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_votes ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_discussions ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_lesson_notes ENABLE ROW LEVEL SECURITY;

-- Proposals: Anyone can read, authenticated users can create/update own
CREATE POLICY "Anyone can view proposals" ON commons_proposals
    FOR SELECT USING (NOT is_flagged OR auth.uid() = user_id);

CREATE POLICY "Authenticated users can create proposals" ON commons_proposals
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own proposals" ON commons_proposals
    FOR UPDATE USING (auth.uid() = user_id AND status = 'open');

-- Votes: Users can manage own votes
CREATE POLICY "Anyone can view votes" ON commons_votes
    FOR SELECT USING (TRUE);

CREATE POLICY "Authenticated users can vote" ON commons_votes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can change own vote" ON commons_votes
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can remove own vote" ON commons_votes
    FOR DELETE USING (auth.uid() = user_id);

-- Discussions: Anyone can read non-hidden, authenticated can post
CREATE POLICY "Anyone can view discussions" ON commons_discussions
    FOR SELECT USING (NOT is_hidden OR auth.uid() = user_id);

CREATE POLICY "Authenticated users can post" ON commons_discussions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can edit own posts" ON commons_discussions
    FOR UPDATE USING (auth.uid() = user_id);

-- Notes: Anyone can read non-hidden, authenticated can contribute
CREATE POLICY "Anyone can view notes" ON commons_lesson_notes
    FOR SELECT USING (NOT is_hidden);

CREATE POLICY "Authenticated users can add notes" ON commons_lesson_notes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can edit own notes" ON commons_lesson_notes
    FOR UPDATE USING (auth.uid() = user_id);
```

---

## User Experience

### 1. Lesson Commons Page (`/commons.html`)

**Main views:**
- **Activity Feed**: Recent proposals, discussions, implementations
- **Trending Proposals**: Sorted by vote score + recency
- **By Lesson**: Browse proposals/notes for specific lessons
- **My Contributions**: Personal dashboard

### 2. Per-Lesson Commons (`/learn.html?day=X` → Commons tab)

Each lesson has a Commons section:
- Current proposals for this lesson
- Community notes and context
- Discussion thread
- "Suggest Improvement" button

### 3. Proposal Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   OPEN      │────▶│UNDER REVIEW │────▶│  APPROVED   │────▶│IMPLEMENTED  │
│             │     │             │     │             │     │             │
│ Voting open │     │ Team eval   │     │ Queued for  │     │ Live!       │
│ 14 days     │     │             │     │ deployment  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                                        │
       │ Not enough votes                       │ Declined
       ▼                                        ▼
┌─────────────┐                         ┌─────────────┐
│  ARCHIVED   │                         │  DECLINED   │
│             │                         │             │
│ Auto-close  │                         │ With reason │
└─────────────┘                         └─────────────┘
```

### 4. Recognition System

| Level | Requirements | Badges |
|-------|--------------|--------|
| Newcomer | 0 contributions | — |
| Contributor | 5+ accepted | 🌱 |
| Trusted | 25+ accepted | 🌿 |
| Expert | 100+ accepted | 🌳 |
| Steward | Mod privileges | 👑 |

---

## API Endpoints

### Proposals

```javascript
// Get proposals for a lesson
GET /rest/v1/commons_proposals?lesson_id=eq.{lesson_id}&order=vote_score.desc

// Create proposal
POST /rest/v1/commons_proposals
{ lesson_id, type, title, description, rationale, proposed_content }

// Vote on proposal
POST /rest/v1/commons_votes
{ proposal_id, vote_type: 'up' | 'down' }
```

### Discussions

```javascript
// Get discussions for a lesson
GET /rest/v1/commons_discussions?lesson_id=eq.{lesson_id}&parent_id=is.null&order=created_at.desc

// Post discussion
POST /rest/v1/commons_discussions
{ lesson_id, content }
```

### Notes

```javascript
// Get notes for a lesson
GET /rest/v1/commons_lesson_notes?lesson_id=eq.{lesson_id}&is_hidden=eq.false

// Add note
POST /rest/v1/commons_lesson_notes
{ lesson_id, type, title, content, sources }
```

---

## Moderation

### Community Guidelines

1. **Be constructive**: Proposals should improve learning
2. **Be specific**: Explain what and why
3. **Be respectful**: Disagree thoughtfully
4. **Be factual**: Cite sources when possible
5. **Be inclusive**: Consider all ages and backgrounds

### Moderation Actions

- **Flag content**: Any user can flag
- **Hide content**: Moderators can hide flagged content
- **Ban user**: Repeated violations
- **Feature content**: Highlight exceptional contributions

### Automated Safeguards

- Rate limiting on submissions (5 proposals/day, 50 votes/day)
- Content filtering for inappropriate language
- Spam detection
- Duplicate detection

---

## Integration Points

### With Lesson Player

```javascript
// In learn.html, add Commons tab
<div class="tab" data-tab="commons">
    <h3>Community</h3>
    <div id="lesson-proposals"></div>
    <div id="lesson-notes"></div>
    <button onclick="openProposalModal()">Suggest Improvement</button>
</div>
```

### With Analytics

Track:
- Proposal submission rate
- Vote participation
- Implementation rate
- Time from proposal to implementation
- User retention correlation with contribution

### With Content Pipeline

When proposal is approved:
1. Create Jira/GitHub issue
2. Notify content team
3. Track implementation status
4. Auto-update proposal status when deployed

---

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Monthly active contributors | 1% of MAU | People submitting proposals/notes |
| Proposal → Implementation rate | >20% | Of non-declined proposals |
| Average time to implementation | <30 days | For approved proposals |
| Community satisfaction | >4.5/5 | Survey: "I feel heard" |
| Vote participation | >10% of DAU | People voting on proposals |

---

## Launch Plan

### Phase 1: Foundation (Week 1-2)
- [ ] Create Supabase tables
- [ ] Build commons.html page
- [ ] Add per-lesson Commons tab
- [ ] Basic proposal submission

### Phase 2: Voting (Week 3-4)
- [ ] Voting system
- [ ] Vote counts and sorting
- [ ] User contribution tracking

### Phase 3: Discussions (Week 5-6)
- [ ] Discussion threads
- [ ] Replies and reactions
- [ ] Notifications

### Phase 4: Recognition (Week 7-8)
- [ ] Contributor levels
- [ ] Badges
- [ ] Leaderboards

### Phase 5: Integration (Week 9-10)
- [ ] Content pipeline integration
- [ ] Analytics
- [ ] Moderation tools

---

## Open Questions

1. **Anonymous proposals**: Allow anonymous submission or require auth?
2. **Voting threshold**: How many votes before review?
3. **Expert verification**: Who verifies "expert" notes?
4. **Multilingual**: Commons content in EN only or per-language?

---

## Related Documents

- [SUPABASE_SCHEMA.md](../backend/SUPABASE_SCHEMA.md) - Core database schema
- [TRUST_AND_SAFETY_INDEX.md](../trust-safety/TRUST_AND_SAFETY_INDEX.md) - Moderation principles
- [PHASEDNA_V2_COMPLETE.md](../phasedna/PHASEDNA_V2_COMPLETE.md) - Lesson structure

---

*Last updated: December 2025*
*Contact: hello@curiouskelly.com*





