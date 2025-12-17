-- ════════════════════════════════════════════════════════════════════
-- LEARNER COMMONS DATABASE SETUP
-- ════════════════════════════════════════════════════════════════════
-- 
-- Run this file in your Supabase SQL Editor to set up all tables
-- needed for the Commons-Governed Content system.
--
-- Order matters! This file includes all migrations in sequence.
--
-- After running this SQL:
--   npx tsx scripts/migrate-to-content-atoms.ts --all
--   npx tsx scripts/verify-content-atoms.ts
--
-- ════════════════════════════════════════════════════════════════════

-- Check if commons_proposals exists (dependency for votes)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'commons_proposals') THEN
        RAISE NOTICE 'Creating commons_proposals table...';
        CREATE TABLE commons_proposals (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES auth.users(id),
            title TEXT NOT NULL,
            description TEXT,
            type TEXT CHECK (type IN ('enhance', 'correct', 'simplify', 'expand', 'typo')),
            status TEXT DEFAULT 'open' CHECK (status IN ('open', 'reviewing', 'approved', 'rejected', 'implemented')),
            affected_phases TEXT[] DEFAULT '{}',
            upvotes INTEGER DEFAULT 0,
            downvotes INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            reviewed_at TIMESTAMPTZ,
            reviewed_by UUID REFERENCES auth.users(id),
            implemented_at TIMESTAMPTZ
        );
        
        ALTER TABLE commons_proposals ENABLE ROW LEVEL SECURITY;
        
        CREATE POLICY "Anyone can view proposals" ON commons_proposals
            FOR SELECT USING (true);
            
        CREATE POLICY "Authenticated users can create proposals" ON commons_proposals
            FOR INSERT WITH CHECK (auth.uid() IS NOT NULL);
    END IF;
END $$;

-- ════════════════════════════════════════════════════════════════════
-- MIGRATION 002: CONTENT ATOMS
-- ════════════════════════════════════════════════════════════════════

-- THE SINGLE SOURCE OF TRUTH FOR ALL LESSON CONTENT
CREATE TABLE IF NOT EXISTS content_atoms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Address (composite key for lookup)
    day_number INTEGER NOT NULL CHECK (day_number >= 1 AND day_number <= 365),
    phase TEXT NOT NULL CHECK (phase IN (
        'hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro'
    )),
    content_type TEXT NOT NULL CHECK (content_type IN (
        'talk', 'question', 'option', 'response', 'comment', 'fun_fact', 'prompt'
    )),
    variant TEXT,
    age_bucket TEXT,
    language TEXT DEFAULT 'en',
    
    -- Content address (generated)
    content_address TEXT GENERATED ALWAYS AS (
        day_number::text || '.' || phase || '.' || content_type || 
        COALESCE('.' || variant, '') ||
        COALESCE('.' || age_bucket, '') ||
        CASE WHEN language != 'en' THEN '.' || language ELSE '' END
    ) STORED,
    
    -- The actual content
    text_content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Versioning
    version INTEGER DEFAULT 1,
    is_live BOOLEAN DEFAULT true,
    
    -- Governance
    change_source TEXT NOT NULL DEFAULT 'initial_seed' CHECK (change_source IN (
        'initial_seed', 'commons_proposal', 'staff_direct', 'automated_translation', 'audio_regeneration', 'migration'
    )),
    change_reference UUID,
    change_reason TEXT,
    changed_by UUID,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_content_atoms_day ON content_atoms(day_number);
CREATE INDEX IF NOT EXISTS idx_content_atoms_phase ON content_atoms(day_number, phase);
CREATE INDEX IF NOT EXISTS idx_content_atoms_live ON content_atoms(is_live) WHERE is_live = true;
CREATE INDEX IF NOT EXISTS idx_content_atoms_address ON content_atoms(content_address);
CREATE INDEX IF NOT EXISTS idx_content_atoms_lookup ON content_atoms(
    day_number, phase, content_type, variant, age_bucket, language
) WHERE is_live = true;

-- CONTENT HISTORY (audit trail)
CREATE TABLE IF NOT EXISTS content_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_atom_id UUID NOT NULL REFERENCES content_atoms(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    metadata JSONB,
    change_source TEXT NOT NULL,
    change_reference UUID,
    change_reason TEXT,
    changed_by UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_rollback_target BOOLEAN DEFAULT true,
    UNIQUE(content_atom_id, version)
);

CREATE INDEX IF NOT EXISTS idx_content_history_atom ON content_history(content_atom_id);
CREATE INDEX IF NOT EXISTS idx_content_history_version ON content_history(content_atom_id, version DESC);

-- Auto-create history on update
CREATE OR REPLACE FUNCTION create_content_history_on_update()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.text_content IS DISTINCT FROM NEW.text_content OR
       OLD.metadata IS DISTINCT FROM NEW.metadata THEN
        INSERT INTO content_history (
            content_atom_id, version, text_content, metadata,
            change_source, change_reference, change_reason, changed_by, created_at
        ) VALUES (
            OLD.id, OLD.version, OLD.text_content, OLD.metadata,
            OLD.change_source, OLD.change_reference, OLD.change_reason, OLD.changed_by, OLD.updated_at
        );
        NEW.version := OLD.version + 1;
        NEW.updated_at := NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_content_history ON content_atoms;
CREATE TRIGGER trigger_content_history
    BEFORE UPDATE ON content_atoms
    FOR EACH ROW
    EXECUTE FUNCTION create_content_history_on_update();

-- Add Commons fields to proposals
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'commons_proposals' AND column_name = 'target_atoms') THEN
        ALTER TABLE commons_proposals ADD COLUMN target_atoms TEXT[] DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'commons_proposals' AND column_name = 'proposed_changes') THEN
        ALTER TABLE commons_proposals ADD COLUMN proposed_changes JSONB DEFAULT '{}'::jsonb;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'commons_proposals' AND column_name = 'requires_audio_regen') THEN
        ALTER TABLE commons_proposals ADD COLUMN requires_audio_regen BOOLEAN DEFAULT false;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'commons_proposals' AND column_name = 'requires_video_regen') THEN
        ALTER TABLE commons_proposals ADD COLUMN requires_video_regen BOOLEAN DEFAULT false;
    END IF;
END $$;

-- RLS for content_atoms
ALTER TABLE content_atoms ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_history ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view live content" ON content_atoms;
CREATE POLICY "Anyone can view live content" ON content_atoms
    FOR SELECT USING (is_live = true);

DROP POLICY IF EXISTS "Anyone can view history" ON content_history;
CREATE POLICY "Anyone can view history" ON content_history
    FOR SELECT USING (true);

-- ════════════════════════════════════════════════════════════════════
-- MIGRATION 003: COMMONS NOTES & VOTES
-- ════════════════════════════════════════════════════════════════════

-- COMMUNITY NOTES
CREATE TABLE IF NOT EXISTS commons_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_address TEXT NOT NULL,
    note_type TEXT NOT NULL CHECK (note_type IN (
        'expert_context', 'historical_note', 'source_citation',
        'teaching_tip', 'common_misconception', 'real_world_example'
    )),
    content TEXT NOT NULL,
    sources TEXT[] DEFAULT '{}',
    is_verified BOOLEAN DEFAULT false,
    is_featured BOOLEAN DEFAULT false,
    helpful_count INTEGER DEFAULT 0,
    insightful_count INTEGER DEFAULT 0,
    user_id UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_commons_notes_address ON commons_notes(content_address);
CREATE INDEX IF NOT EXISTS idx_commons_notes_featured ON commons_notes(is_featured) WHERE is_featured = true;

-- VOTES ON PROPOSALS
CREATE TABLE IF NOT EXISTS commons_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id UUID NOT NULL REFERENCES commons_proposals(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id),
    vote_type TEXT NOT NULL CHECK (vote_type IN ('up', 'down')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(proposal_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_commons_votes_proposal ON commons_votes(proposal_id);

-- NOTE REACTIONS
CREATE TABLE IF NOT EXISTS commons_note_reactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    note_id UUID NOT NULL REFERENCES commons_notes(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id),
    reaction_type TEXT NOT NULL CHECK (reaction_type IN ('helpful', 'insightful')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(note_id, user_id, reaction_type)
);

-- Auto-update reaction counts
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

DROP TRIGGER IF EXISTS trigger_note_reactions ON commons_note_reactions;
CREATE TRIGGER trigger_note_reactions
    AFTER INSERT OR DELETE ON commons_note_reactions
    FOR EACH ROW
    EXECUTE FUNCTION update_note_reaction_counts();

-- RLS for notes/votes
ALTER TABLE commons_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_votes ENABLE ROW LEVEL SECURITY;
ALTER TABLE commons_note_reactions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view notes" ON commons_notes;
CREATE POLICY "Anyone can view notes" ON commons_notes FOR SELECT USING (true);

DROP POLICY IF EXISTS "Authenticated can create notes" ON commons_notes;
CREATE POLICY "Authenticated can create notes" ON commons_notes FOR INSERT WITH CHECK (auth.uid() IS NOT NULL);

DROP POLICY IF EXISTS "Anyone can view votes" ON commons_votes;
CREATE POLICY "Anyone can view votes" ON commons_votes FOR SELECT USING (true);

DROP POLICY IF EXISTS "Authenticated can vote" ON commons_votes;
CREATE POLICY "Authenticated can vote" ON commons_votes FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can change votes" ON commons_votes;
CREATE POLICY "Users can change votes" ON commons_votes FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete votes" ON commons_votes;
CREATE POLICY "Users can delete votes" ON commons_votes FOR DELETE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Anyone can view reactions" ON commons_note_reactions;
CREATE POLICY "Anyone can view reactions" ON commons_note_reactions FOR SELECT USING (true);

DROP POLICY IF EXISTS "Authenticated can react" ON commons_note_reactions;
CREATE POLICY "Authenticated can react" ON commons_note_reactions FOR INSERT WITH CHECK (auth.uid() = user_id);

-- ════════════════════════════════════════════════════════════════════
-- HELPER FUNCTION: Get lesson content
-- ════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION get_lesson_content(
    p_day_number INTEGER,
    p_age_bucket TEXT DEFAULT NULL,
    p_language TEXT DEFAULT 'en'
)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_object_agg(phase, phase_content) INTO result
    FROM (
        SELECT
            phase,
            jsonb_object_agg(
                content_type || COALESCE('.' || variant, ''),
                jsonb_build_object(
                    'content', text_content,
                    'metadata', metadata,
                    'version', version,
                    'address', content_address
                )
            ) as phase_content
        FROM content_atoms
        WHERE day_number = p_day_number
          AND is_live = true
          AND language = p_language
          AND (age_bucket IS NULL OR age_bucket = p_age_bucket OR p_age_bucket IS NULL)
        GROUP BY phase
    ) phases;
    RETURN COALESCE(result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- ════════════════════════════════════════════════════════════════════
-- DONE!
-- ════════════════════════════════════════════════════════════════════

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '════════════════════════════════════════════════════';
    RAISE NOTICE '✅ ALL MIGRATIONS COMPLETE';
    RAISE NOTICE '════════════════════════════════════════════════════';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - content_atoms (lesson content)';
    RAISE NOTICE '  - content_history (version history)';
    RAISE NOTICE '  - commons_proposals (change proposals)';
    RAISE NOTICE '  - commons_notes (community notes)';
    RAISE NOTICE '  - commons_votes (proposal votes)';
    RAISE NOTICE '  - commons_note_reactions (note reactions)';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '  1. Run: npx tsx scripts/migrate-to-content-atoms.ts --all';
    RAISE NOTICE '  2. Run: npx tsx scripts/verify-content-atoms.ts';
    RAISE NOTICE '';
END $$;
