-- ============================================================================
-- VISUAL COMMONS MIGRATION
-- ============================================================================
-- Creates the database infrastructure for learner-powered visual generation.
-- Run this in Supabase SQL Editor.
--
-- Created: December 17, 2025
-- ============================================================================

-- ============================================================================
-- TABLE: visual_commons
-- The main cache for all generated educational visuals
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_commons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Content-addressable identity (the key to deduplication)
  content_hash TEXT UNIQUE NOT NULL,
  
  -- Context metadata (denormalized for fast queries)
  day_number INTEGER NOT NULL CHECK (day_number >= 1 AND day_number <= 365),
  phase TEXT NOT NULL CHECK (phase IN (
    'hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro', 'complete'
  )),
  topic TEXT NOT NULL,
  visual_type TEXT NOT NULL CHECK (visual_type IN (
    'infographic', 'diagram', 'scene', 'comparison', 'timeline', 'process'
  )),
  age_group TEXT DEFAULT 'all' CHECK (age_group IN (
    '2-5', '6-12', '13-17', '18+', 'all'
  )),
  style TEXT DEFAULT 'default',
  
  -- The actual asset
  storage_path TEXT NOT NULL,
  public_url TEXT NOT NULL,
  thumbnail_url TEXT,
  width INTEGER,
  height INTEGER,
  file_size_bytes INTEGER,
  format TEXT DEFAULT 'png' CHECK (format IN ('png', 'webp', 'svg', 'jpg')),
  
  -- Generation metadata
  prompt_used TEXT NOT NULL,
  model_used TEXT NOT NULL,
  generation_params JSONB DEFAULT '{}',
  generation_time_ms INTEGER,
  estimated_cost DECIMAL(10,6) DEFAULT 0,
  
  -- Attribution
  generated_by UUID REFERENCES auth.users(id),
  generated_by_display_name TEXT,
  generation_source TEXT NOT NULL CHECK (generation_source IN (
    'byok',      -- User's own API key
    'platform',  -- Our API key
    'staff',     -- Admin-generated
    'seed'       -- Pre-seeded content
  )),
  
  -- Usage tracking
  view_count INTEGER DEFAULT 0,
  unique_learners_helped INTEGER DEFAULT 0,
  last_viewed_at TIMESTAMPTZ,
  
  -- Moderation
  status TEXT DEFAULT 'active' CHECK (status IN (
    'pending', 'active', 'flagged', 'removed'
  )),
  flagged_reason TEXT,
  moderated_by UUID REFERENCES auth.users(id),
  moderated_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_vc_hash ON visual_commons(content_hash);
CREATE INDEX IF NOT EXISTS idx_vc_day_phase ON visual_commons(day_number, phase);
CREATE INDEX IF NOT EXISTS idx_vc_day_phase_age ON visual_commons(day_number, phase, age_group);
CREATE INDEX IF NOT EXISTS idx_vc_generator ON visual_commons(generated_by) WHERE generated_by IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_vc_status ON visual_commons(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_vc_popular ON visual_commons(unique_learners_helped DESC) WHERE status = 'active';

-- Enable RLS
ALTER TABLE visual_commons ENABLE ROW LEVEL SECURITY;

-- Anyone can read active visuals
CREATE POLICY "Public read active visuals" ON visual_commons
  FOR SELECT USING (status = 'active');

-- Authenticated users can insert
CREATE POLICY "Authenticated insert" ON visual_commons
  FOR INSERT WITH CHECK (true);

-- Service role can do anything (for API operations)
CREATE POLICY "Service role full access" ON visual_commons
  FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- TABLE: visual_generation_queue
-- Queue for background generation when immediate isn't possible
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_generation_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  content_hash TEXT NOT NULL,
  context JSONB NOT NULL,
  prompt TEXT NOT NULL,
  
  requested_by UUID REFERENCES auth.users(id),
  priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
  
  status TEXT DEFAULT 'pending' CHECK (status IN (
    'pending', 'processing', 'completed', 'failed'
  )),
  attempts INTEGER DEFAULT 0,
  last_error TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  
  visual_id UUID REFERENCES visual_commons(id)
);

CREATE INDEX IF NOT EXISTS idx_vgq_pending ON visual_generation_queue(priority, created_at) 
  WHERE status = 'pending';

-- ============================================================================
-- TABLE: user_visual_contributions
-- Aggregate stats for gamification
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_visual_contributions (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id),
  
  total_contributed INTEGER DEFAULT 0,
  total_learners_helped INTEGER DEFAULT 0,
  
  badges JSONB DEFAULT '[]',
  
  contributions_this_week INTEGER DEFAULT 0,
  contributions_this_month INTEGER DEFAULT 0,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- TABLE: visual_views
-- Track individual views for analytics (optional, can be disabled for privacy)
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_views (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  visual_id UUID NOT NULL REFERENCES visual_commons(id),
  viewer_id UUID REFERENCES auth.users(id),
  viewer_session TEXT,
  viewed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vv_visual ON visual_views(visual_id);
CREATE INDEX IF NOT EXISTS idx_vv_viewer ON visual_views(viewer_id) WHERE viewer_id IS NOT NULL;

-- ============================================================================
-- FUNCTION: increment_visual_views
-- Atomically increment view count and update contributor stats
-- ============================================================================

CREATE OR REPLACE FUNCTION increment_visual_views(
  p_visual_id UUID,
  p_viewer_id UUID DEFAULT NULL,
  p_session_id TEXT DEFAULT NULL
)
RETURNS void AS $$
DECLARE
  v_generator_id UUID;
  v_is_unique BOOLEAN := false;
BEGIN
  -- Get the generator
  SELECT generated_by INTO v_generator_id
  FROM visual_commons
  WHERE id = p_visual_id;
  
  -- Check if this is a unique view (different from generator, not seen before)
  IF p_viewer_id IS NOT NULL AND p_viewer_id IS DISTINCT FROM v_generator_id THEN
    -- Check if viewer has seen this before
    IF NOT EXISTS (
      SELECT 1 FROM visual_views 
      WHERE visual_id = p_visual_id AND viewer_id = p_viewer_id
    ) THEN
      v_is_unique := true;
    END IF;
  ELSIF p_session_id IS NOT NULL THEN
    -- For anonymous, check by session
    IF NOT EXISTS (
      SELECT 1 FROM visual_views 
      WHERE visual_id = p_visual_id AND viewer_session = p_session_id
    ) THEN
      v_is_unique := true;
    END IF;
  END IF;
  
  -- Always increment view count
  UPDATE visual_commons 
  SET view_count = view_count + 1,
      last_viewed_at = NOW(),
      unique_learners_helped = CASE WHEN v_is_unique THEN unique_learners_helped + 1 ELSE unique_learners_helped END
  WHERE id = p_visual_id;
  
  -- Record the view
  INSERT INTO visual_views (visual_id, viewer_id, viewer_session)
  VALUES (p_visual_id, p_viewer_id, p_session_id);
  
  -- Update contributor stats if unique and generator exists
  IF v_is_unique AND v_generator_id IS NOT NULL THEN
    INSERT INTO user_visual_contributions (user_id, total_learners_helped)
    VALUES (v_generator_id, 1)
    ON CONFLICT (user_id) DO UPDATE
    SET total_learners_helped = user_visual_contributions.total_learners_helped + 1,
        updated_at = NOW();
  END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FUNCTION: get_visual_by_context
-- Look up a visual by its context (computes hash server-side)
-- ============================================================================

CREATE OR REPLACE FUNCTION get_visual_by_hash(p_content_hash TEXT)
RETURNS TABLE (
  id UUID,
  public_url TEXT,
  thumbnail_url TEXT,
  generated_by_display_name TEXT,
  unique_learners_helped INTEGER,
  created_at TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    vc.id,
    vc.public_url,
    vc.thumbnail_url,
    vc.generated_by_display_name,
    vc.unique_learners_helped,
    vc.created_at
  FROM visual_commons vc
  WHERE vc.content_hash = p_content_hash
    AND vc.status = 'active'
  LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FUNCTION: get_user_visual_stats
-- Get contribution stats for a user
-- ============================================================================

CREATE OR REPLACE FUNCTION get_user_visual_stats(p_user_id UUID)
RETURNS TABLE (
  total_contributed INTEGER,
  total_learners_helped INTEGER,
  badges JSONB,
  recent_contributions JSONB
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    COALESCE(uvc.total_contributed, 0),
    COALESCE(uvc.total_learners_helped, 0),
    COALESCE(uvc.badges, '[]'::JSONB),
    (
      SELECT COALESCE(jsonb_agg(row_to_json(r)), '[]'::JSONB)
      FROM (
        SELECT 
          vc.id,
          vc.topic,
          vc.phase,
          vc.unique_learners_helped,
          vc.created_at
        FROM visual_commons vc
        WHERE vc.generated_by = p_user_id
        ORDER BY vc.created_at DESC
        LIMIT 5
      ) r
    ) as recent_contributions
  FROM user_visual_contributions uvc
  WHERE uvc.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGER: Update user contribution count on visual insert
-- ============================================================================

CREATE OR REPLACE FUNCTION on_visual_created()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.generated_by IS NOT NULL THEN
    INSERT INTO user_visual_contributions (user_id, total_contributed)
    VALUES (NEW.generated_by, 1)
    ON CONFLICT (user_id) DO UPDATE
    SET total_contributed = user_visual_contributions.total_contributed + 1,
        contributions_this_week = user_visual_contributions.contributions_this_week + 1,
        contributions_this_month = user_visual_contributions.contributions_this_month + 1,
        updated_at = NOW();
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_visual_created
  AFTER INSERT ON visual_commons
  FOR EACH ROW
  EXECUTE FUNCTION on_visual_created();

-- ============================================================================
-- STORAGE: Create bucket for visuals
-- ============================================================================

-- Note: Run this separately or via Supabase Dashboard
-- INSERT INTO storage.buckets (id, name, public)
-- VALUES ('visuals', 'visuals', true)
-- ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
BEGIN
  RAISE NOTICE '✅ visual_commons table created';
  RAISE NOTICE '✅ visual_generation_queue table created';
  RAISE NOTICE '✅ user_visual_contributions table created';
  RAISE NOTICE '✅ visual_views table created';
  RAISE NOTICE '✅ increment_visual_views function created';
  RAISE NOTICE '✅ get_visual_by_hash function created';
  RAISE NOTICE '✅ get_user_visual_stats function created';
  RAISE NOTICE '✅ Triggers created';
  RAISE NOTICE '';
  RAISE NOTICE '⚠️  Remember to create the storage bucket:';
  RAISE NOTICE '    Go to Storage > New Bucket > Name: "visuals" > Public: ON';
  RAISE NOTICE '';
  RAISE NOTICE '🎉 VISUAL COMMONS MIGRATION COMPLETE';
END $$;
