-- ============================================================
-- CONTENT ATOMS - COMMONS-GOVERNED CONTENT
-- ============================================================
-- Migration: 002_content_atoms.sql
-- Created: December 17, 2025
-- Description: Single source of truth for all lesson content
-- ============================================================

-- ============================================================
-- TABLE: content_atoms
-- ============================================================
-- THE SINGLE SOURCE OF TRUTH FOR ALL LESSON CONTENT
CREATE TABLE IF NOT EXISTS content_atoms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Address (composite key for lookup)
    day_number INTEGER NOT NULL CHECK (day_number >= 1 AND day_number <= 365),
    phase TEXT NOT NULL CHECK (phase IN (
        'hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro'
    )),
    content_type TEXT NOT NULL CHECK (content_type IN (
        'talk',        -- Main script for the phase
        'question',    -- Question Kelly asks
        'option',      -- Choice option (A, B, etc.)
        'response',    -- Kelly's response to choice
        'comment',     -- Simulated student comment
        'fun_fact',    -- Extra fun fact
        'prompt'       -- Cliff/question prompt
    )),
    variant TEXT,                    -- A, B, or null for main content
    age_bucket TEXT,                 -- 2-5, 6-12, 13-17, 18-35, 36-60, 61+, or null for universal
    language TEXT DEFAULT 'en',      -- en, es, fr
    
    -- Content address (generated, for easy reference)
    content_address TEXT GENERATED ALWAYS AS (
        day_number::text || '.' || phase || '.' || content_type || 
        COALESCE('.' || variant, '') ||
        COALESCE('.' || age_bucket, '') ||
        CASE WHEN language != 'en' THEN '.' || language ELSE '' END
    ) STORED,
    
    -- The actual content
    text_content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    -- metadata can include:
    -- - kellyPose: string
    -- - kellyEmotion: string
    -- - duration: number (seconds)
    -- - visual_cue: string
    -- - factNumber: number
    -- - factTitle: string
    -- - audio_url: string
    -- - video_url: string
    
    -- Versioning
    version INTEGER DEFAULT 1,
    is_live BOOLEAN DEFAULT true,
    
    -- Governance
    change_source TEXT NOT NULL DEFAULT 'initial_seed' CHECK (change_source IN (
        'initial_seed',              -- Launch content
        'commons_proposal',          -- Community proposal (approved)
        'staff_direct',              -- Staff edit (requires justification)
        'automated_translation',     -- Machine translation
        'audio_regeneration',        -- Content updated for TTS
        'migration'                  -- From legacy system
    )),
    change_reference UUID,           -- proposal_id if from commons
    change_reason TEXT,              -- Required for staff_direct
    changed_by UUID,                 -- User who made the change
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_content_atoms_day ON content_atoms(day_number);
CREATE INDEX IF NOT EXISTS idx_content_atoms_phase ON content_atoms(day_number, phase);
CREATE INDEX IF NOT EXISTS idx_content_atoms_live ON content_atoms(is_live) WHERE is_live = true;
CREATE INDEX IF NOT EXISTS idx_content_atoms_address ON content_atoms(content_address);
CREATE INDEX IF NOT EXISTS idx_content_atoms_lookup ON content_atoms(
    day_number, phase, content_type, variant, age_bucket, language
) WHERE is_live = true;

-- ============================================================
-- TABLE: content_history
-- ============================================================
-- AUDIT TRAIL - EVERY VERSION OF EVERY PIECE OF CONTENT
CREATE TABLE IF NOT EXISTS content_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_atom_id UUID NOT NULL REFERENCES content_atoms(id) ON DELETE CASCADE,
    
    -- Snapshot of content at this version
    version INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    metadata JSONB,
    
    -- Change tracking
    change_source TEXT NOT NULL,
    change_reference UUID,           -- proposal_id, etc.
    change_reason TEXT,
    changed_by UUID,
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- For rollback capability
    is_rollback_target BOOLEAN DEFAULT true,
    
    UNIQUE(content_atom_id, version)
);

CREATE INDEX IF NOT EXISTS idx_content_history_atom ON content_history(content_atom_id);
CREATE INDEX IF NOT EXISTS idx_content_history_version ON content_history(content_atom_id, version DESC);

-- ============================================================
-- FUNCTION: create_content_history_on_update
-- ============================================================
-- Automatically create history entry when content changes
CREATE OR REPLACE FUNCTION create_content_history_on_update()
RETURNS TRIGGER AS $$
BEGIN
    -- Only create history if content actually changed
    IF OLD.text_content IS DISTINCT FROM NEW.text_content OR 
       OLD.metadata IS DISTINCT FROM NEW.metadata THEN
        
        -- Archive the old version
        INSERT INTO content_history (
            content_atom_id,
            version,
            text_content,
            metadata,
            change_source,
            change_reference,
            change_reason,
            changed_by,
            created_at
        ) VALUES (
            OLD.id,
            OLD.version,
            OLD.text_content,
            OLD.metadata,
            OLD.change_source,
            OLD.change_reference,
            OLD.change_reason,
            OLD.changed_by,
            OLD.updated_at
        );
        
        -- Increment version
        NEW.version := OLD.version + 1;
        NEW.updated_at := NOW();
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_content_history ON content_atoms;
CREATE TRIGGER trigger_content_history
    BEFORE UPDATE ON content_atoms
    FOR EACH ROW
    EXECUTE FUNCTION create_content_history_on_update();

-- ============================================================
-- ENHANCE: commons_proposals
-- ============================================================
-- Add fields to link proposals to specific content atoms

DO $$
BEGIN
    -- Add target_atoms column if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'commons_proposals' 
                   AND column_name = 'target_atoms') THEN
        ALTER TABLE commons_proposals 
        ADD COLUMN target_atoms TEXT[] DEFAULT '{}';
    END IF;
    
    -- Add proposed_changes column if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'commons_proposals' 
                   AND column_name = 'proposed_changes') THEN
        ALTER TABLE commons_proposals 
        ADD COLUMN proposed_changes JSONB DEFAULT '{}'::jsonb;
    END IF;
    
    -- Add requires_audio_regen column if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'commons_proposals' 
                   AND column_name = 'requires_audio_regen') THEN
        ALTER TABLE commons_proposals 
        ADD COLUMN requires_audio_regen BOOLEAN DEFAULT false;
    END IF;
    
    -- Add requires_video_regen column if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'commons_proposals' 
                   AND column_name = 'requires_video_regen') THEN
        ALTER TABLE commons_proposals 
        ADD COLUMN requires_video_regen BOOLEAN DEFAULT false;
    END IF;
END $$;

-- ============================================================
-- VIEW: content_atoms_live
-- ============================================================
-- Convenient view for only live content
CREATE OR REPLACE VIEW content_atoms_live AS
SELECT * FROM content_atoms WHERE is_live = true;

-- ============================================================
-- FUNCTION: get_lesson_content
-- ============================================================
-- Get all content for a day in structured format
CREATE OR REPLACE FUNCTION get_lesson_content(
    p_day_number INTEGER,
    p_age_bucket TEXT DEFAULT NULL,
    p_language TEXT DEFAULT 'en'
)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_object_agg(
        phase,
        phase_content
    ) INTO result
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

-- ============================================================
-- FUNCTION: apply_proposal_changes
-- ============================================================
-- Apply approved proposal changes to content_atoms
CREATE OR REPLACE FUNCTION apply_proposal_changes(p_proposal_id UUID)
RETURNS INTEGER AS $$
DECLARE
    proposal RECORD;
    atom_address TEXT;
    change JSONB;
    atoms_updated INTEGER := 0;
BEGIN
    -- Get the proposal
    SELECT * INTO proposal 
    FROM commons_proposals 
    WHERE id = p_proposal_id AND status = 'approved';
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Proposal not found or not approved: %', p_proposal_id;
    END IF;
    
    -- Apply each change
    FOR atom_address, change IN 
        SELECT * FROM jsonb_each(proposal.proposed_changes)
    LOOP
        UPDATE content_atoms
        SET 
            text_content = change->>'proposed',
            change_source = 'commons_proposal',
            change_reference = p_proposal_id,
            change_reason = proposal.title || ': ' || proposal.description,
            changed_by = proposal.user_id
        WHERE content_address = atom_address
          AND is_live = true;
        
        IF FOUND THEN
            atoms_updated := atoms_updated + 1;
        END IF;
    END LOOP;
    
    -- Update proposal status
    UPDATE commons_proposals
    SET 
        status = 'implemented',
        implemented_at = NOW()
    WHERE id = p_proposal_id;
    
    RETURN atoms_updated;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- RLS POLICIES
-- ============================================================
ALTER TABLE content_atoms ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_history ENABLE ROW LEVEL SECURITY;

-- Anyone can read live content
CREATE POLICY "Anyone can view live content" ON content_atoms
    FOR SELECT USING (is_live = true);

-- Staff can read all versions
CREATE POLICY "Staff can view all content" ON content_atoms
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM auth.users u 
            WHERE u.id = auth.uid() 
            AND (u.raw_user_meta_data->>'role' = 'staff' 
                 OR u.raw_user_meta_data->>'role' = 'admin')
        )
    );

-- Only staff can insert/update content atoms
CREATE POLICY "Staff can modify content" ON content_atoms
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM auth.users u 
            WHERE u.id = auth.uid() 
            AND (u.raw_user_meta_data->>'role' = 'staff' 
                 OR u.raw_user_meta_data->>'role' = 'admin')
        )
    );

-- Anyone can view history
CREATE POLICY "Anyone can view history" ON content_history
    FOR SELECT USING (true);

-- ============================================================
-- COMMENTS
-- ============================================================
COMMENT ON TABLE content_atoms IS 'Single source of truth for all lesson content. Each row is an addressable piece of content (script, option, response, etc.)';
COMMENT ON TABLE content_history IS 'Audit trail of all content changes. Every version of every content atom is preserved.';
COMMENT ON COLUMN content_atoms.content_address IS 'Human-readable address like "017.hook.talk" for easy reference';
COMMENT ON COLUMN content_atoms.is_live IS 'Whether this is the current active version';
COMMENT ON COLUMN content_atoms.change_source IS 'How this content came to be (initial_seed, commons_proposal, staff_direct, etc.)';
