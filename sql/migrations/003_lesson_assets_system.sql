-- ═══════════════════════════════════════════════════════════════════════════
-- MIGRATION 003: LESSON ASSETS SYSTEM
-- ═══════════════════════════════════════════════════════════════════════════
-- 
-- Purpose: Create a permanent, scalable asset caching system so that:
-- 1. Each lesson has a canonical thumbnail_slug that matches actual files
-- 2. All generated assets (images, audio, video) are cached and shared
-- 3. Students never regenerate content that already exists
-- 4. Every phase of every lesson has defined asset slots
--
-- Created: December 3, 2025
-- ═══════════════════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 1: Add thumbnail_slug to core_lessons
-- ═══════════════════════════════════════════════════════════════════════════

-- Add the column if it doesn't exist
ALTER TABLE core_lessons 
ADD COLUMN IF NOT EXISTS thumbnail_slug TEXT;

-- Add index for fast lookups
CREATE INDEX IF NOT EXISTS idx_core_lessons_thumbnail_slug 
ON core_lessons(thumbnail_slug) WHERE thumbnail_slug IS NOT NULL;

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 2: Create lesson_assets table for ALL cached content
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS lesson_assets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- ═══ IDENTITY ═══
  lesson_day INT NOT NULL,                    -- 1-365
  asset_type TEXT NOT NULL,                   -- 'thumbnail', 'phase_image', 'audio', 'video'
  phase TEXT,                                 -- 'hero', 'q1', 'q2', 'q3', 'hook', 'wisdom', etc.
  
  -- ═══ VARIANT DIMENSIONS ═══
  -- NULL means "universal" (works for all values of that dimension)
  language TEXT,                              -- 'en', 'es', 'fr' or NULL for universal
  age_bucket TEXT,                            -- 'toddler', 'child', 'teen', 'adult', 'senior' or NULL
  archetype TEXT,                             -- 'The Survivor', 'The Explorer', etc. or NULL
  tone TEXT,                                  -- 'playful', 'curious', 'serious' or NULL
  
  -- ═══ STORAGE ═══
  storage_bucket TEXT DEFAULT 'lesson-assets',
  storage_path TEXT NOT NULL,                 -- 'thumbnails/raw/lesson-001-starting-fresh.png'
  public_url TEXT NOT NULL,                   -- Full CDN URL
  file_format TEXT NOT NULL,                  -- 'png', 'jpeg', 'mp3', 'mp4'
  file_size_bytes BIGINT,
  
  -- ═══ GENERATION METADATA ═══
  generator TEXT,                             -- 'flux-1.1-pro', 'elevenlabs', 'manual'
  prompt_used TEXT,
  seed BIGINT,
  generation_params JSONB DEFAULT '{}',
  
  -- ═══ QUALITY & STATUS ═══
  quality_score DECIMAL(4,3),
  is_approved BOOLEAN DEFAULT false,
  is_primary BOOLEAN DEFAULT false,           -- The "best" version for this slot
  approved_by TEXT,
  approved_at TIMESTAMPTZ,
  
  -- ═══ USAGE ANALYTICS ═══
  view_count INT DEFAULT 0,
  last_served_at TIMESTAMPTZ,
  
  -- ═══ TIMESTAMPS ═══
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- ═══ CONSTRAINTS ═══
  CONSTRAINT valid_asset_type CHECK (asset_type IN ('thumbnail', 'phase_image', 'audio', 'video', 'animation')),
  CONSTRAINT valid_quality CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1))
);

-- Composite index for fast lookups by lesson + phase + variant
CREATE INDEX IF NOT EXISTS idx_lesson_assets_lookup
ON lesson_assets(lesson_day, asset_type, phase, language, age_bucket, archetype)
WHERE is_approved = true;

-- Index for finding primary assets
CREATE INDEX IF NOT EXISTS idx_lesson_assets_primary
ON lesson_assets(lesson_day, asset_type, phase, is_primary)
WHERE is_primary = true;

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 3: Create variant_cache table for tracking what's been generated
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS lesson_variant_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- ═══ CACHE KEY (composite unique) ═══
  lesson_day INT NOT NULL,
  phase TEXT NOT NULL,
  language TEXT NOT NULL DEFAULT 'en',
  age_bucket TEXT NOT NULL DEFAULT 'adult',
  archetype TEXT,                             -- NULL for non-archetype content
  tone TEXT NOT NULL DEFAULT 'curious',
  
  -- ═══ CACHE STATUS ═══
  is_complete BOOLEAN DEFAULT false,          -- All assets for this variant exist
  assets_ready JSONB DEFAULT '{}',            -- { "audio": true, "image": true, "video": false }
  
  -- ═══ LINKED ASSETS ═══
  thumbnail_asset_id UUID REFERENCES lesson_assets(id),
  image_asset_id UUID REFERENCES lesson_assets(id),
  audio_asset_id UUID REFERENCES lesson_assets(id),
  video_asset_id UUID REFERENCES lesson_assets(id),
  
  -- ═══ GENERATION STATUS ═══
  generation_started_at TIMESTAMPTZ,
  generation_completed_at TIMESTAMPTZ,
  last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
  access_count INT DEFAULT 0,
  
  -- ═══ TIMESTAMPS ═══
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Unique constraint on the cache key dimensions
  CONSTRAINT unique_variant_cache 
  UNIQUE (lesson_day, phase, language, age_bucket, COALESCE(archetype, ''), tone)
);

-- Index for cache lookups
CREATE INDEX IF NOT EXISTS idx_variant_cache_lookup
ON lesson_variant_cache(lesson_day, phase, language, age_bucket, tone);

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 4: Function to get or create cached variant
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION get_or_create_variant_cache(
  p_lesson_day INT,
  p_phase TEXT,
  p_language TEXT DEFAULT 'en',
  p_age_bucket TEXT DEFAULT 'adult',
  p_archetype TEXT DEFAULT NULL,
  p_tone TEXT DEFAULT 'curious'
)
RETURNS TABLE (
  cache_id UUID,
  is_complete BOOLEAN,
  assets_ready JSONB,
  needs_generation BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
  v_cache_id UUID;
  v_is_complete BOOLEAN;
  v_assets_ready JSONB;
BEGIN
  -- Try to find existing cache entry
  SELECT id, lvc.is_complete, lvc.assets_ready
  INTO v_cache_id, v_is_complete, v_assets_ready
  FROM lesson_variant_cache lvc
  WHERE lvc.lesson_day = p_lesson_day
    AND lvc.phase = p_phase
    AND lvc.language = p_language
    AND lvc.age_bucket = p_age_bucket
    AND COALESCE(lvc.archetype, '') = COALESCE(p_archetype, '')
    AND lvc.tone = p_tone;
  
  IF v_cache_id IS NULL THEN
    -- Create new cache entry
    INSERT INTO lesson_variant_cache (lesson_day, phase, language, age_bucket, archetype, tone)
    VALUES (p_lesson_day, p_phase, p_language, p_age_bucket, p_archetype, p_tone)
    RETURNING id INTO v_cache_id;
    
    v_is_complete := false;
    v_assets_ready := '{}';
  ELSE
    -- Update access tracking
    UPDATE lesson_variant_cache
    SET last_accessed_at = NOW(), access_count = access_count + 1
    WHERE id = v_cache_id;
  END IF;
  
  RETURN QUERY SELECT v_cache_id, v_is_complete, v_assets_ready, NOT v_is_complete;
END;
$$;

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 5: Function to get lesson thumbnail URL
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION get_lesson_thumbnail(p_day_number INT)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
  v_slug TEXT;
  v_url TEXT;
BEGIN
  -- First try to get from core_lessons.thumbnail_slug
  SELECT thumbnail_slug INTO v_slug
  FROM core_lessons
  WHERE day_number = p_day_number;
  
  IF v_slug IS NOT NULL THEN
    RETURN '/kelly/thumbnails/raw/lesson-' || LPAD(p_day_number::TEXT, 3, '0') || '-' || v_slug || '.png';
  END IF;
  
  -- Fallback: try lesson_assets table
  SELECT public_url INTO v_url
  FROM lesson_assets
  WHERE lesson_day = p_day_number
    AND asset_type = 'thumbnail'
    AND is_approved = true
    AND is_primary = true
  LIMIT 1;
  
  RETURN v_url;
END;
$$;

-- ═══════════════════════════════════════════════════════════════════════════
-- PART 6: RLS Policies
-- ═══════════════════════════════════════════════════════════════════════════

ALTER TABLE lesson_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE lesson_variant_cache ENABLE ROW LEVEL SECURITY;

-- Public can read approved assets
CREATE POLICY "Public can view approved assets"
ON lesson_assets FOR SELECT
TO anon, authenticated
USING (is_approved = true);

-- Public can read cache status
CREATE POLICY "Public can view cache status"
ON lesson_variant_cache FOR SELECT
TO anon, authenticated
USING (true);

-- Service role full access
CREATE POLICY "Service role full access to assets"
ON lesson_assets FOR ALL TO service_role
USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access to cache"
ON lesson_variant_cache FOR ALL TO service_role
USING (true) WITH CHECK (true);

-- ═══════════════════════════════════════════════════════════════════════════
-- VERIFICATION
-- ═══════════════════════════════════════════════════════════════════════════

SELECT 'Migration 003 complete!' as status;

SELECT 
  'core_lessons.thumbnail_slug' as column_check,
  EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'core_lessons' AND column_name = 'thumbnail_slug'
  ) as exists;





