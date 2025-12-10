-- ============================================================
-- LEARNER COMMONS TABLES
-- ============================================================
-- Migration: 001_learner_commons.sql
-- Created: December 2025
-- Description: Community governance system for lesson improvements
-- ============================================================

-- ============================================================
-- TABLE: commons_proposals
-- ============================================================
-- Proposed changes to lessons from the community
CREATE TABLE IF NOT EXISTS commons_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lesson_id UUID NOT NULL REFERENCES core_lessons(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    
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
CREATE INDEX IF NOT EXISTS idx_proposals_lesson ON commons_proposals(lesson_id);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON commons_proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_type ON commons_proposals(type);
CREATE INDEX IF NOT EXISTS idx_proposals_score ON commons_proposals(vote_score DESC);
CREATE INDEX IF NOT EXISTS idx_proposals_user ON commons_proposals(user_id);
CREATE INDEX IF NOT EXISTS idx_proposals_created ON commons_proposals(created_at DESC);

-- ============================================================
-- TABLE: commons_votes
-- ============================================================
-- User votes on proposals
CREATE TABLE IF NOT EXISTS commons_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id UUID NOT NULL REFERENCES commons_proposals(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    
    vote_type TEXT NOT NULL CHECK (vote_type IN ('up', 'down')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- One vote per user per proposal
    UNIQUE(proposal_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_votes_proposal ON commons_votes(proposal_id);
CREATE INDEX IF NOT EXISTS idx_votes_user ON commons_votes(user_id);

-- ============================================================
-- TRIGGER: Update vote counts on proposals
-- ============================================================
CREATE OR REPLACE FUNCTION update_proposal_votes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.vote_type = 'up' THEN
            UPDATE commons_proposals SET upvotes = upvotes + 1, updated_at = NOW() WHERE id = NEW.proposal_id;
        ELSE
            UPDATE commons_proposals SET downvotes = downvotes + 1, updated_at = NOW() WHERE id = NEW.proposal_id;
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        IF OLD.vote_type = 'up' THEN
            UPDATE commons_proposals SET upvotes = GREATEST(upvotes - 1, 0), updated_at = NOW() WHERE id = OLD.proposal_id;
        ELSE
            UPDATE commons_proposals SET downvotes = GREATEST(downvotes - 1, 0), updated_at = NOW() WHERE id = OLD.proposal_id;
        END IF;
    ELSIF TG_OP = 'UPDATE' THEN
        -- Handle vote change
        IF OLD.vote_type = 'up' THEN
            UPDATE commons_proposals SET upvotes = GREATEST(upvotes - 1, 0) WHERE id = OLD.proposal_id;
        ELSE
            UPDATE commons_proposals SET downvotes = GREATEST(downvotes - 1, 0) WHERE id = OLD.proposal_id;
        END IF;
        IF NEW.vote_type = 'up' THEN
            UPDATE commons_proposals SET upvotes = upvotes + 1, updated_at = NOW() WHERE id = NEW.proposal_id;
        ELSE
            UPDATE commons_proposals SET downvotes = downvotes + 1, updated_at = NOW() WHERE id = NEW.proposal_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_proposal_votes ON commons_votes;
CREATE TRIGGER trigger_update_proposal_votes
AFTER INSERT OR UPDATE OR DELETE ON commons_votes
FOR EACH ROW EXECUTE FUNCTION update_proposal_votes();

-- ============================================================
-- TABLE: commons_discussions
-- ============================================================
-- Discussion threads on proposals or lessons
CREATE TABLE IF NOT EXISTS commons_discussions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Can be attached to a proposal OR directly to a lesson
    proposal_id UUID REFERENCES commons_proposals(id) ON DELETE CASCADE,
    lesson_id UUID REFERENCES core_lessons(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES commons_discussions(id) ON DELETE CASCADE,  -- For replies
    
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    
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

CREATE INDEX IF NOT EXISTS idx_discussions_proposal ON commons_discussions(proposal_id);
CREATE INDEX IF NOT EXISTS idx_discussions_lesson ON commons_discussions(lesson_id);
CREATE INDEX IF NOT EXISTS idx_discussions_parent ON commons_discussions(parent_id);
CREATE INDEX IF NOT EXISTS idx_discussions_user ON commons_discussions(user_id);
CREATE INDEX IF NOT EXISTS idx_discussions_created ON commons_discussions(created_at DESC);

-- ============================================================
-- TABLE: commons_lesson_notes
-- ============================================================
-- Community-contributed knowledge for each lesson
CREATE TABLE IF NOT EXISTS commons_lesson_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lesson_id UUID NOT NULL REFERENCES core_lessons(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    
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

CREATE INDEX IF NOT EXISTS idx_notes_lesson ON commons_lesson_notes(lesson_id);
CREATE INDEX IF NOT EXISTS idx_notes_type ON commons_lesson_notes(type);
CREATE INDEX IF NOT EXISTS idx_notes_featured ON commons_lesson_notes(is_featured) WHERE is_featured = TRUE;
CREATE INDEX IF NOT EXISTS idx_notes_user ON commons_lesson_notes(user_id);

-- ============================================================
-- TABLE: commons_user_contributions
-- ============================================================
-- Track user contribution stats for recognition
CREATE TABLE IF NOT EXISTS commons_user_contributions (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    
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

-- ============================================================
-- TABLE: commons_activity_log
-- ============================================================
-- Track all commons activity for transparency
CREATE TABLE IF NOT EXISTS commons_activity_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Context
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    lesson_id UUID REFERENCES core_lessons(id) ON DELETE SET NULL,
    proposal_id UUID REFERENCES commons_proposals(id) ON DELETE SET NULL,
    
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

CREATE INDEX IF NOT EXISTS idx_activity_lesson ON commons_activity_log(lesson_id);
CREATE INDEX IF NOT EXISTS idx_activity_user ON commons_activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_created ON commons_activity_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_public ON commons_activity_log(is_public) WHERE is_public = TRUE;

-- ============================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================

-- Enable RLS on all commons tables
ALTER TABLE commons_proposals ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_votes ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_discussions ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_lesson_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_user_contributions ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_activity_log ENABLE ROW LEVEL SECURITY;

-- PROPOSALS POLICIES
DROP POLICY IF EXISTS "Anyone can view proposals" ON commons_proposals;
CREATE POLICY "Anyone can view proposals" ON commons_proposals
    FOR SELECT USING (NOT is_flagged OR auth.uid() = user_id);

DROP POLICY IF EXISTS "Authenticated users can create proposals" ON commons_proposals;
CREATE POLICY "Authenticated users can create proposals" ON commons_proposals
    FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own proposals" ON commons_proposals;
CREATE POLICY "Users can update own proposals" ON commons_proposals
    FOR UPDATE USING (auth.uid() = user_id AND status = 'open');

-- VOTES POLICIES
DROP POLICY IF EXISTS "Anyone can view votes" ON commons_votes;
CREATE POLICY "Anyone can view votes" ON commons_votes
    FOR SELECT USING (TRUE);

DROP POLICY IF EXISTS "Authenticated users can vote" ON commons_votes;
CREATE POLICY "Authenticated users can vote" ON commons_votes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can change own vote" ON commons_votes;
CREATE POLICY "Users can change own vote" ON commons_votes
    FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can remove own vote" ON commons_votes;
CREATE POLICY "Users can remove own vote" ON commons_votes
    FOR DELETE USING (auth.uid() = user_id);

-- DISCUSSIONS POLICIES
DROP POLICY IF EXISTS "Anyone can view discussions" ON commons_discussions;
CREATE POLICY "Anyone can view discussions" ON commons_discussions
    FOR SELECT USING (NOT is_hidden OR auth.uid() = user_id);

DROP POLICY IF EXISTS "Authenticated users can post discussions" ON commons_discussions;
CREATE POLICY "Authenticated users can post discussions" ON commons_discussions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can edit own discussions" ON commons_discussions;
CREATE POLICY "Users can edit own discussions" ON commons_discussions
    FOR UPDATE USING (auth.uid() = user_id);

-- NOTES POLICIES
DROP POLICY IF EXISTS "Anyone can view notes" ON commons_lesson_notes;
CREATE POLICY "Anyone can view notes" ON commons_lesson_notes
    FOR SELECT USING (NOT is_hidden);

DROP POLICY IF EXISTS "Authenticated users can add notes" ON commons_lesson_notes;
CREATE POLICY "Authenticated users can add notes" ON commons_lesson_notes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can edit own notes" ON commons_lesson_notes;
CREATE POLICY "Users can edit own notes" ON commons_lesson_notes
    FOR UPDATE USING (auth.uid() = user_id);

-- USER CONTRIBUTIONS POLICIES
DROP POLICY IF EXISTS "Users can view their own contributions" ON commons_user_contributions;
CREATE POLICY "Users can view their own contributions" ON commons_user_contributions
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Anyone can view public contribution stats" ON commons_user_contributions;
CREATE POLICY "Anyone can view public contribution stats" ON commons_user_contributions
    FOR SELECT USING (TRUE);

-- ACTIVITY LOG POLICIES
DROP POLICY IF EXISTS "Anyone can view public activity" ON commons_activity_log;
CREATE POLICY "Anyone can view public activity" ON commons_activity_log
    FOR SELECT USING (is_public = TRUE OR auth.uid() = user_id);

-- ============================================================
-- FUNCTIONS: Auto-update user contribution stats
-- ============================================================
CREATE OR REPLACE FUNCTION update_user_contribution_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure user exists in contributions table
    INSERT INTO commons_user_contributions (user_id, created_at)
    VALUES (NEW.user_id, NOW())
    ON CONFLICT (user_id) DO NOTHING;
    
    -- Update stats based on table
    IF TG_TABLE_NAME = 'commons_proposals' THEN
        UPDATE commons_user_contributions 
        SET proposals_submitted = proposals_submitted + 1,
            last_contribution_at = NOW(),
            first_contribution_at = COALESCE(first_contribution_at, NOW()),
            updated_at = NOW()
        WHERE user_id = NEW.user_id;
    ELSIF TG_TABLE_NAME = 'commons_votes' THEN
        UPDATE commons_user_contributions 
        SET votes_cast = votes_cast + 1,
            last_contribution_at = NOW(),
            first_contribution_at = COALESCE(first_contribution_at, NOW()),
            updated_at = NOW()
        WHERE user_id = NEW.user_id;
    ELSIF TG_TABLE_NAME = 'commons_discussions' THEN
        UPDATE commons_user_contributions 
        SET discussions_posted = discussions_posted + 1,
            last_contribution_at = NOW(),
            first_contribution_at = COALESCE(first_contribution_at, NOW()),
            updated_at = NOW()
        WHERE user_id = NEW.user_id;
    ELSIF TG_TABLE_NAME = 'commons_lesson_notes' THEN
        UPDATE commons_user_contributions 
        SET notes_contributed = notes_contributed + 1,
            last_contribution_at = NOW(),
            first_contribution_at = COALESCE(first_contribution_at, NOW()),
            updated_at = NOW()
        WHERE user_id = NEW.user_id;
    END IF;
    
    -- Update contributor level
    UPDATE commons_user_contributions 
    SET contributor_level = CASE
        WHEN proposals_approved + notes_contributed >= 100 THEN 'expert'
        WHEN proposals_approved + notes_contributed >= 25 THEN 'trusted'
        WHEN proposals_approved + notes_contributed >= 5 THEN 'contributor'
        ELSE 'newcomer'
    END,
    updated_at = NOW()
    WHERE user_id = NEW.user_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for contribution tracking
DROP TRIGGER IF EXISTS trigger_proposal_contribution ON commons_proposals;
CREATE TRIGGER trigger_proposal_contribution
AFTER INSERT ON commons_proposals
FOR EACH ROW EXECUTE FUNCTION update_user_contribution_stats();

DROP TRIGGER IF EXISTS trigger_vote_contribution ON commons_votes;
CREATE TRIGGER trigger_vote_contribution
AFTER INSERT ON commons_votes
FOR EACH ROW EXECUTE FUNCTION update_user_contribution_stats();

DROP TRIGGER IF EXISTS trigger_discussion_contribution ON commons_discussions;
CREATE TRIGGER trigger_discussion_contribution
AFTER INSERT ON commons_discussions
FOR EACH ROW EXECUTE FUNCTION update_user_contribution_stats();

DROP TRIGGER IF EXISTS trigger_note_contribution ON commons_lesson_notes;
CREATE TRIGGER trigger_note_contribution
AFTER INSERT ON commons_lesson_notes
FOR EACH ROW EXECUTE FUNCTION update_user_contribution_stats();

-- ============================================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================================
-- Uncomment to insert sample proposals for testing

/*
INSERT INTO commons_proposals (lesson_id, type, title, description, rationale, status, upvotes, downvotes)
SELECT 
    id,
    'enhance',
    'Add historical context to ' || topic,
    'The current lesson is great but could benefit from more historical background to help learners understand how this topic evolved over time.',
    'Historical context helps learners remember information better and connects new knowledge to existing mental models.',
    'open',
    floor(random() * 50)::int,
    floor(random() * 10)::int
FROM core_lessons
WHERE day_number <= 5;
*/

-- ============================================================
-- VERIFICATION
-- ============================================================
-- Run this to verify tables were created:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'commons_%';

COMMENT ON TABLE commons_proposals IS 'Community proposals for lesson improvements';
COMMENT ON TABLE commons_votes IS 'User votes on proposals';
COMMENT ON TABLE commons_discussions IS 'Discussion threads on proposals or lessons';
COMMENT ON TABLE commons_lesson_notes IS 'Community-contributed knowledge for lessons';
COMMENT ON TABLE commons_user_contributions IS 'User contribution statistics and levels';
COMMENT ON TABLE commons_activity_log IS 'Activity log for transparency';





