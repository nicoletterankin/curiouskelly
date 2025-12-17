-- ============================================================
-- COMMONS NOTES & VOTES
-- ============================================================
-- Migration: 003_commons_notes_votes.sql
-- Created: December 17, 2025
-- Description: Community notes and voting system for Phase Commons
-- ============================================================

-- ============================================================
-- TABLE: commons_notes
-- ============================================================
-- Community-contributed context, tips, and citations for lesson content
CREATE TABLE IF NOT EXISTS commons_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to content
    content_address TEXT NOT NULL,  -- e.g., "017.hook.talk"
    
    -- Note details
    note_type TEXT NOT NULL CHECK (note_type IN (
        'expert_context',       -- Expert background info
        'historical_note',      -- Historical context
        'source_citation',      -- Academic/reference source
        'teaching_tip',         -- Pedagogical suggestion
        'common_misconception', -- What learners often misunderstand
        'real_world_example'    -- Practical application
    )),
    content TEXT NOT NULL,
    sources TEXT[] DEFAULT '{}',    -- URLs or citations
    
    -- Verification
    is_verified BOOLEAN DEFAULT false,  -- Staff-verified accuracy
    is_featured BOOLEAN DEFAULT false,  -- Highlighted by staff
    
    -- Engagement
    helpful_count INTEGER DEFAULT 0,
    insightful_count INTEGER DEFAULT 0,
    
    -- Author
    user_id UUID REFERENCES auth.users(id),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_commons_notes_address ON commons_notes(content_address);
CREATE INDEX IF NOT EXISTS idx_commons_notes_type ON commons_notes(note_type);
CREATE INDEX IF NOT EXISTS idx_commons_notes_featured ON commons_notes(is_featured) WHERE is_featured = true;
CREATE INDEX IF NOT EXISTS idx_commons_notes_verified ON commons_notes(is_verified) WHERE is_verified = true;

-- ============================================================
-- TABLE: commons_votes
-- ============================================================
-- Votes on proposals (one vote per user per proposal)
CREATE TABLE IF NOT EXISTS commons_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to proposal
    proposal_id UUID NOT NULL REFERENCES commons_proposals(id) ON DELETE CASCADE,
    
    -- Voter
    user_id UUID NOT NULL REFERENCES auth.users(id),
    
    -- Vote type
    vote_type TEXT NOT NULL CHECK (vote_type IN ('up', 'down')),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- One vote per user per proposal
    UNIQUE(proposal_id, user_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_commons_votes_proposal ON commons_votes(proposal_id);
CREATE INDEX IF NOT EXISTS idx_commons_votes_user ON commons_votes(user_id);

-- ============================================================
-- TABLE: commons_note_reactions
-- ============================================================
-- Reactions to community notes
CREATE TABLE IF NOT EXISTS commons_note_reactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to note
    note_id UUID NOT NULL REFERENCES commons_notes(id) ON DELETE CASCADE,
    
    -- Reactor
    user_id UUID NOT NULL REFERENCES auth.users(id),
    
    -- Reaction type
    reaction_type TEXT NOT NULL CHECK (reaction_type IN ('helpful', 'insightful')),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- One reaction type per user per note
    UNIQUE(note_id, user_id, reaction_type)
);

-- Index
CREATE INDEX IF NOT EXISTS idx_note_reactions_note ON commons_note_reactions(note_id);

-- ============================================================
-- FUNCTION: update_note_reaction_counts
-- ============================================================
-- Trigger to update reaction counts on notes
CREATE OR REPLACE FUNCTION update_note_reaction_counts()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.reaction_type = 'helpful' THEN
            UPDATE commons_notes SET helpful_count = helpful_count + 1 WHERE id = NEW.note_id;
        ELSIF NEW.reaction_type = 'insightful' THEN
            UPDATE commons_notes SET insightful_count = insightful_count + 1 WHERE id = NEW.note_id;
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        IF OLD.reaction_type = 'helpful' THEN
            UPDATE commons_notes SET helpful_count = GREATEST(0, helpful_count - 1) WHERE id = OLD.note_id;
        ELSIF OLD.reaction_type = 'insightful' THEN
            UPDATE commons_notes SET insightful_count = GREATEST(0, insightful_count - 1) WHERE id = OLD.note_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_note_reactions ON commons_note_reactions;
CREATE TRIGGER trigger_note_reactions
    AFTER INSERT OR DELETE ON commons_note_reactions
    FOR EACH ROW
    EXECUTE FUNCTION update_note_reaction_counts();

-- ============================================================
-- RLS POLICIES
-- ============================================================

-- commons_notes
ALTER TABLE commons_notes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view notes" ON commons_notes
    FOR SELECT USING (true);

CREATE POLICY "Authenticated users can create notes" ON commons_notes
    FOR INSERT WITH CHECK (auth.uid() IS NOT NULL);

CREATE POLICY "Users can update own notes" ON commons_notes
    FOR UPDATE USING (auth.uid() = user_id);

-- commons_votes
ALTER TABLE commons_votes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view votes" ON commons_votes
    FOR SELECT USING (true);

CREATE POLICY "Authenticated users can vote" ON commons_votes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can change own votes" ON commons_votes
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own votes" ON commons_votes
    FOR DELETE USING (auth.uid() = user_id);

-- commons_note_reactions
ALTER TABLE commons_note_reactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view reactions" ON commons_note_reactions
    FOR SELECT USING (true);

CREATE POLICY "Authenticated users can react" ON commons_note_reactions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can remove own reactions" ON commons_note_reactions
    FOR DELETE USING (auth.uid() = user_id);

-- ============================================================
-- COMMENTS
-- ============================================================
COMMENT ON TABLE commons_notes IS 'Community-contributed context, tips, and expert notes for lesson content';
COMMENT ON TABLE commons_votes IS 'User votes on content change proposals';
COMMENT ON TABLE commons_note_reactions IS 'User reactions (helpful/insightful) to community notes';
