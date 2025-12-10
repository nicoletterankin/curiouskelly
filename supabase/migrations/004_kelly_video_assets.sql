-- ═══════════════════════════════════════════════════════════════════════════
-- MIGRATION 004: KELLY VIDEO ASSETS
-- ═══════════════════════════════════════════════════════════════════════════
-- 
-- Purpose: Store ElevenLabs Omnihuman 1.5 generated lip-sync videos
-- Each lesson phase can have a pre-generated video with Kelly speaking
-- Videos are keyed by lesson_day + phase + age_bucket + language
--
-- Created: December 3, 2025
-- ═══════════════════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 1: Create kelly_video_assets table
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kelly_video_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- ═══ CONTENT REFERENCE ═══
    lesson_day INTEGER NOT NULL,
    phase TEXT NOT NULL,  -- 'welcome', 'q1', 'q2', 'q3', 'wisdom'
    
    -- ═══ VARIANT KEYS (for caching unique combinations) ═══
    age_bucket TEXT NOT NULL,  -- 'toddler', 'child', 'teen', 'young_adult', 'adult', 'elder'
    language TEXT NOT NULL DEFAULT 'en',
    archetype TEXT,  -- Optional: 'Scientist', 'Explorer', etc.
    
    -- ═══ SOURCE ASSETS ═══
    source_image_path TEXT NOT NULL,  -- Path to static Kelly image used
    source_audio_url TEXT,  -- ElevenLabs TTS audio URL (temporary)
    script_text TEXT,  -- The text that was spoken (for reference)
    
    -- ═══ GENERATED VIDEO ═══
    video_storage_path TEXT,  -- Supabase Storage path
    video_public_url TEXT,  -- CDN URL for playback
    video_duration_ms INTEGER,
    video_file_size_bytes BIGINT,
    video_format TEXT DEFAULT 'mp4',
    video_resolution TEXT,  -- e.g., '1080x1920' or '1920x1080'
    
    -- ═══ GENERATION METADATA ═══
    elevenlabs_generation_id TEXT,
    model_used TEXT DEFAULT 'omnihuman-1.5',
    generation_credits_used INTEGER,
    generation_started_at TIMESTAMPTZ,
    generation_completed_at TIMESTAMPTZ,
    generation_duration_ms INTEGER,  -- How long it took to generate
    
    -- ═══ QUALITY METADATA ═══
    lip_sync_quality_score DECIMAL(4,3),  -- 0.000 to 1.000
    video_quality_score DECIMAL(4,3),
    is_approved BOOLEAN DEFAULT false,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    
    -- ═══ STATUS ═══
    status TEXT DEFAULT 'pending',  -- 'pending', 'generating', 'completed', 'failed', 'expired'
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- ═══ USAGE ANALYTICS ═══
    view_count INTEGER DEFAULT 0,
    last_viewed_at TIMESTAMPTZ,
    
    -- ═══ TIMESTAMPS ═══
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- ═══ CONSTRAINTS ═══
    CONSTRAINT valid_phase CHECK (phase IN ('welcome', 'q1', 'q2', 'q3', 'wisdom')),
    CONSTRAINT valid_age_bucket CHECK (age_bucket IN ('toddler', 'child', 'teen', 'young_adult', 'adult', 'elder')),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'generating', 'completed', 'failed', 'expired')),
    CONSTRAINT valid_quality_scores CHECK (
        (lip_sync_quality_score IS NULL OR (lip_sync_quality_score >= 0 AND lip_sync_quality_score <= 1)) AND
        (video_quality_score IS NULL OR (video_quality_score >= 0 AND video_quality_score <= 1))
    ),
    
    -- Unique constraint: one video per lesson/phase/variant combo
    UNIQUE(lesson_day, phase, age_bucket, language)
);

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 2: Create indexes for fast lookups
-- ═══════════════════════════════════════════════════════════════════════════

-- Primary lookup index
CREATE INDEX IF NOT EXISTS idx_kelly_video_lookup 
ON kelly_video_assets(lesson_day, phase, age_bucket, language, status);

-- Index for completed videos only (most common query)
CREATE INDEX IF NOT EXISTS idx_kelly_video_completed
ON kelly_video_assets(lesson_day, phase, age_bucket, language)
WHERE status = 'completed';

-- Index for generation queue (find pending/generating)
CREATE INDEX IF NOT EXISTS idx_kelly_video_queue
ON kelly_video_assets(status, created_at)
WHERE status IN ('pending', 'generating');

-- Index for analytics
CREATE INDEX IF NOT EXISTS idx_kelly_video_analytics
ON kelly_video_assets(view_count DESC, last_viewed_at DESC)
WHERE status = 'completed';

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 3: Create helper functions
-- ═══════════════════════════════════════════════════════════════════════════

-- Function to get video URL or NULL (for fallback logic)
CREATE OR REPLACE FUNCTION get_kelly_video_url(
    p_lesson_day INTEGER,
    p_phase TEXT,
    p_age_bucket TEXT,
    p_language TEXT DEFAULT 'en'
)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    v_url TEXT;
BEGIN
    SELECT video_public_url INTO v_url
    FROM kelly_video_assets
    WHERE lesson_day = p_lesson_day
      AND phase = p_phase
      AND age_bucket = p_age_bucket
      AND language = p_language
      AND status = 'completed';
    
    -- Update view count and last_viewed_at if found
    IF v_url IS NOT NULL THEN
        UPDATE kelly_video_assets
        SET view_count = view_count + 1,
            last_viewed_at = NOW()
        WHERE lesson_day = p_lesson_day
          AND phase = p_phase
          AND age_bucket = p_age_bucket
          AND language = p_language
          AND status = 'completed';
    END IF;
    
    RETURN v_url;  -- Returns NULL if not found (fallback to image+audio)
END;
$$;

-- Function to get full video asset info
CREATE OR REPLACE FUNCTION get_kelly_video_asset(
    p_lesson_day INTEGER,
    p_phase TEXT,
    p_age_bucket TEXT,
    p_language TEXT DEFAULT 'en'
)
RETURNS TABLE (
    video_url TEXT,
    duration_ms INTEGER,
    has_video BOOLEAN
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kva.video_public_url,
        kva.video_duration_ms,
        (kva.video_public_url IS NOT NULL)::BOOLEAN
    FROM kelly_video_assets kva
    WHERE kva.lesson_day = p_lesson_day
      AND kva.phase = p_phase
      AND kva.age_bucket = p_age_bucket
      AND kva.language = p_language
      AND kva.status = 'completed'
    LIMIT 1;
    
    -- If no rows returned, return a row with nulls
    IF NOT FOUND THEN
        RETURN QUERY SELECT NULL::TEXT, NULL::INTEGER, FALSE;
    END IF;
END;
$$;

-- Function to check generation status
CREATE OR REPLACE FUNCTION get_kelly_video_status(
    p_lesson_day INTEGER,
    p_phase TEXT,
    p_age_bucket TEXT,
    p_language TEXT DEFAULT 'en'
)
RETURNS TABLE (
    status TEXT,
    error_message TEXT,
    retry_count INTEGER,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kva.status,
        kva.error_message,
        kva.retry_count,
        kva.created_at
    FROM kelly_video_assets kva
    WHERE kva.lesson_day = p_lesson_day
      AND kva.phase = p_phase
      AND kva.age_bucket = p_age_bucket
      AND kva.language = p_language
    LIMIT 1;
END;
$$;

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 4: Create triggers for updated_at
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION update_kelly_video_assets_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_kelly_video_assets_updated_at ON kelly_video_assets;
CREATE TRIGGER trigger_kelly_video_assets_updated_at
    BEFORE UPDATE ON kelly_video_assets
    FOR EACH ROW
    EXECUTE FUNCTION update_kelly_video_assets_updated_at();

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 5: RLS Policies
-- ═══════════════════════════════════════════════════════════════════════════

ALTER TABLE kelly_video_assets ENABLE ROW LEVEL SECURITY;

-- Public can read completed videos
CREATE POLICY "Public can view completed videos"
ON kelly_video_assets FOR SELECT
TO anon, authenticated
USING (status = 'completed');

-- Service role full access
CREATE POLICY "Service role full access to kelly videos"
ON kelly_video_assets FOR ALL TO service_role
USING (true) WITH CHECK (true);

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 6: Create generation queue view
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW kelly_video_generation_queue AS
SELECT 
    kva.id,
    kva.lesson_day,
    kva.phase,
    kva.age_bucket,
    kva.language,
    kva.status,
    kva.retry_count,
    kva.created_at,
    cl.topic as lesson_topic
FROM kelly_video_assets kva
LEFT JOIN core_lessons cl ON cl.day_number = kva.lesson_day
WHERE kva.status IN ('pending', 'generating')
ORDER BY kva.created_at ASC;

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 7: Create statistics view
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW kelly_video_stats AS
SELECT 
    COUNT(*) FILTER (WHERE status = 'completed') as completed_count,
    COUNT(*) FILTER (WHERE status = 'pending') as pending_count,
    COUNT(*) FILTER (WHERE status = 'generating') as generating_count,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_count,
    COUNT(DISTINCT lesson_day) FILTER (WHERE status = 'completed') as lessons_with_videos,
    SUM(video_file_size_bytes) FILTER (WHERE status = 'completed') as total_storage_bytes,
    SUM(view_count) as total_views,
    AVG(video_duration_ms) FILTER (WHERE status = 'completed') as avg_duration_ms
FROM kelly_video_assets;

-- ═══════════════════════════════════════════════════════════════════════════
-- VERIFICATION
-- ═══════════════════════════════════════════════════════════════════════════

SELECT 'Migration 004: kelly_video_assets complete!' as status;

-- Verify table exists
SELECT 
    'kelly_video_assets table' as check_name,
    EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'kelly_video_assets'
    ) as exists;

-- Verify functions exist
SELECT 
    'get_kelly_video_url function' as check_name,
    EXISTS (
        SELECT 1 FROM information_schema.routines 
        WHERE routine_name = 'get_kelly_video_url'
    ) as exists;




