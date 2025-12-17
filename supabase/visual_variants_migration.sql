-- ============================================================================
-- VISUAL VARIANTS MIGRATION
-- ============================================================================
-- Enhances visual_commons with variant dimensions for mass personalization.
-- Every learner can contribute variants; every learner benefits from all.
--
-- Created: December 17, 2025
-- ============================================================================

-- ============================================================================
-- ENHANCE: visual_commons with variant dimensions
-- ============================================================================

-- Add style column (defaults to artistic for backwards compatibility)
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'visual_commons' AND column_name = 'style') THEN
    ALTER TABLE visual_commons ADD COLUMN style TEXT DEFAULT 'artistic';
  END IF;
END $$;

-- Update style column constraint
ALTER TABLE visual_commons DROP CONSTRAINT IF EXISTS visual_commons_style_check;
ALTER TABLE visual_commons ADD CONSTRAINT visual_commons_style_check 
  CHECK (style IN (
    'artistic',     -- Photorealistic, cinematic, emotional
    'textbook',     -- Educational illustration with labels
    'diagram',      -- Technical diagram, flowcharts
    'medical',      -- Anatomical accuracy, scientific
    'minimal',      -- Simple shapes, single concept
    'infographic',  -- Data visualization, statistics
    'illustrated',  -- Warm, hand-drawn feel
    '3d_render'     -- 3D visualization
  ));

-- Add complexity column
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'visual_commons' AND column_name = 'complexity') THEN
    ALTER TABLE visual_commons ADD COLUMN complexity TEXT DEFAULT 'standard';
  END IF;
END $$;

ALTER TABLE visual_commons DROP CONSTRAINT IF EXISTS visual_commons_complexity_check;
ALTER TABLE visual_commons ADD CONSTRAINT visual_commons_complexity_check
  CHECK (complexity IN ('simple', 'standard', 'detailed', 'expert'));

-- Add text inclusion mode
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'visual_commons' AND column_name = 'includes_text') THEN
    ALTER TABLE visual_commons ADD COLUMN includes_text TEXT DEFAULT 'none';
  END IF;
END $$;

ALTER TABLE visual_commons DROP CONSTRAINT IF EXISTS visual_commons_includes_text_check;
ALTER TABLE visual_commons ADD CONSTRAINT visual_commons_includes_text_check
  CHECK (includes_text IN ('none', 'labels', 'full', 'bilingual'));

-- Update existing records to have default values
UPDATE visual_commons 
SET 
  style = COALESCE(style, 'artistic'),
  complexity = COALESCE(complexity, 'standard'),
  includes_text = COALESCE(includes_text, 'none')
WHERE style IS NULL OR complexity IS NULL OR includes_text IS NULL;

-- Create composite index for variant queries
CREATE INDEX IF NOT EXISTS idx_vc_variants 
  ON visual_commons(day_number, phase, style, complexity, includes_text, age_group)
  WHERE status = 'active';

-- ============================================================================
-- TABLE: learner_visual_preferences
-- Track what styles each learner prefers
-- ============================================================================

CREATE TABLE IF NOT EXISTS learner_visual_preferences (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Default preferences
  preferred_style TEXT DEFAULT 'artistic' CHECK (preferred_style IN (
    'artistic', 'textbook', 'diagram', 'medical', 
    'minimal', 'infographic', 'illustrated', '3d_render'
  )),
  preferred_complexity TEXT DEFAULT 'standard' CHECK (preferred_complexity IN (
    'simple', 'standard', 'detailed', 'expert'
  )),
  preferred_text_mode TEXT DEFAULT 'none' CHECK (preferred_text_mode IN (
    'none', 'labels', 'full', 'bilingual'
  )),
  
  -- Usage history (counts per style)
  style_history JSONB DEFAULT '{
    "artistic": 0, "textbook": 0, "diagram": 0, "medical": 0,
    "minimal": 0, "infographic": 0, "illustrated": 0, "3d_render": 0
  }'::jsonb,
  
  -- Total selections made
  total_selections INTEGER DEFAULT 0,
  
  -- A/B testing
  experiment_cohort TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE learner_visual_preferences ENABLE ROW LEVEL SECURITY;

-- Users can read/write their own preferences
CREATE POLICY "Users manage own preferences" ON learner_visual_preferences
  FOR ALL USING (auth.uid() = user_id);

-- Service role full access
CREATE POLICY "Service role full access" ON learner_visual_preferences
  FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- TABLE: visual_selections
-- Track which variants learners choose (for learning what works)
-- ============================================================================

CREATE TABLE IF NOT EXISTS visual_selections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- What was selected
  visual_id UUID NOT NULL REFERENCES visual_commons(id) ON DELETE CASCADE,
  
  -- Who selected (nullable for anonymous)
  learner_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  session_id TEXT,
  
  -- What variants were shown as options
  variants_shown UUID[] NOT NULL DEFAULT '{}',
  variants_shown_count INTEGER GENERATED ALWAYS AS (array_length(variants_shown, 1)) STORED,
  
  -- Context
  day_number INTEGER NOT NULL,
  phase TEXT NOT NULL,
  
  -- Timing (how long did they take to choose?)
  time_to_select_ms INTEGER,
  
  -- Was this the recommended variant?
  was_recommended BOOLEAN DEFAULT false,
  
  selected_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_vs_visual ON visual_selections(visual_id);
CREATE INDEX IF NOT EXISTS idx_vs_learner ON visual_selections(learner_id) WHERE learner_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_vs_day_phase ON visual_selections(day_number, phase);
CREATE INDEX IF NOT EXISTS idx_vs_date ON visual_selections(selected_at);

-- Enable RLS
ALTER TABLE visual_selections ENABLE ROW LEVEL SECURITY;

-- Anyone can insert (for tracking)
CREATE POLICY "Insert selections" ON visual_selections
  FOR INSERT WITH CHECK (true);

-- Users can read their own
CREATE POLICY "Users read own" ON visual_selections
  FOR SELECT USING (auth.uid() = learner_id OR learner_id IS NULL);

-- Service role full access
CREATE POLICY "Service role full access" ON visual_selections
  FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- FUNCTION: get_variants_for_phase
-- Get all available variants for a lesson phase, ordered by relevance
-- ============================================================================

CREATE OR REPLACE FUNCTION get_variants_for_phase(
  p_day_number INTEGER,
  p_phase TEXT,
  p_preferred_style TEXT DEFAULT 'artistic',
  p_age_group TEXT DEFAULT 'all',
  p_limit INTEGER DEFAULT 8
)
RETURNS TABLE (
  id UUID,
  public_url TEXT,
  thumbnail_url TEXT,
  style TEXT,
  complexity TEXT,
  includes_text TEXT,
  age_group TEXT,
  unique_learners_helped INTEGER,
  generated_by_display_name TEXT,
  is_preferred_style BOOLEAN
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    vc.id,
    vc.public_url,
    vc.thumbnail_url,
    vc.style,
    vc.complexity,
    vc.includes_text,
    vc.age_group,
    vc.unique_learners_helped,
    vc.generated_by_display_name,
    (vc.style = p_preferred_style) as is_preferred_style
  FROM visual_commons vc
  WHERE vc.day_number = p_day_number
    AND vc.phase = p_phase
    AND vc.status = 'active'
    AND (vc.age_group = p_age_group OR vc.age_group = 'all' OR p_age_group = 'all')
  ORDER BY 
    -- Preferred style first
    CASE WHEN vc.style = p_preferred_style THEN 0 ELSE 1 END,
    -- Then by popularity
    vc.unique_learners_helped DESC,
    -- Then by recency
    vc.created_at DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FUNCTION: record_variant_selection
-- Record when a learner selects a variant and update preferences
-- ============================================================================

CREATE OR REPLACE FUNCTION record_variant_selection(
  p_visual_id UUID,
  p_learner_id UUID,
  p_session_id TEXT,
  p_variants_shown UUID[],
  p_day_number INTEGER,
  p_phase TEXT,
  p_time_to_select_ms INTEGER DEFAULT NULL
)
RETURNS void AS $$
DECLARE
  v_style TEXT;
BEGIN
  -- Get the style of selected visual
  SELECT style INTO v_style
  FROM visual_commons
  WHERE id = p_visual_id;

  -- Record the selection
  INSERT INTO visual_selections (
    visual_id, learner_id, session_id, variants_shown,
    day_number, phase, time_to_select_ms
  ) VALUES (
    p_visual_id, p_learner_id, p_session_id, p_variants_shown,
    p_day_number, p_phase, p_time_to_select_ms
  );

  -- Update visual view count
  UPDATE visual_commons
  SET view_count = view_count + 1,
      last_viewed_at = NOW()
  WHERE id = p_visual_id;

  -- Update learner preferences if logged in
  IF p_learner_id IS NOT NULL AND v_style IS NOT NULL THEN
    INSERT INTO learner_visual_preferences (user_id, preferred_style, total_selections, style_history)
    VALUES (
      p_learner_id,
      v_style,
      1,
      jsonb_build_object(v_style, 1)
    )
    ON CONFLICT (user_id) DO UPDATE
    SET 
      total_selections = learner_visual_preferences.total_selections + 1,
      style_history = jsonb_set(
        learner_visual_preferences.style_history,
        ARRAY[v_style],
        to_jsonb(COALESCE((learner_visual_preferences.style_history->>v_style)::int, 0) + 1)
      ),
      -- Update preferred style if this style is now most used
      preferred_style = CASE 
        WHEN COALESCE((learner_visual_preferences.style_history->>v_style)::int, 0) + 1 > 
             COALESCE((learner_visual_preferences.style_history->>learner_visual_preferences.preferred_style)::int, 0)
        THEN v_style
        ELSE learner_visual_preferences.preferred_style
      END,
      updated_at = NOW();
  END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FUNCTION: get_variant_stats
-- Get statistics about variant performance
-- ============================================================================

CREATE OR REPLACE FUNCTION get_variant_stats(
  p_day_number INTEGER DEFAULT NULL,
  p_phase TEXT DEFAULT NULL
)
RETURNS TABLE (
  style TEXT,
  total_visuals BIGINT,
  total_selections BIGINT,
  avg_learners_helped NUMERIC,
  selection_rate NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  WITH visual_stats AS (
    SELECT 
      vc.style,
      COUNT(DISTINCT vc.id) as visual_count,
      SUM(vc.unique_learners_helped) as total_helped
    FROM visual_commons vc
    WHERE vc.status = 'active'
      AND (p_day_number IS NULL OR vc.day_number = p_day_number)
      AND (p_phase IS NULL OR vc.phase = p_phase)
    GROUP BY vc.style
  ),
  selection_stats AS (
    SELECT 
      vc.style,
      COUNT(vs.id) as selection_count
    FROM visual_selections vs
    JOIN visual_commons vc ON vs.visual_id = vc.id
    WHERE (p_day_number IS NULL OR vs.day_number = p_day_number)
      AND (p_phase IS NULL OR vs.phase = p_phase)
    GROUP BY vc.style
  )
  SELECT 
    vs.style,
    vs.visual_count as total_visuals,
    COALESCE(ss.selection_count, 0) as total_selections,
    ROUND(vs.total_helped::numeric / NULLIF(vs.visual_count, 0), 2) as avg_learners_helped,
    ROUND(COALESCE(ss.selection_count, 0)::numeric / NULLIF(vs.visual_count, 0), 2) as selection_rate
  FROM visual_stats vs
  LEFT JOIN selection_stats ss ON vs.style = ss.style
  ORDER BY COALESCE(ss.selection_count, 0) DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
BEGIN
  RAISE NOTICE '✅ visual_commons enhanced with variant columns';
  RAISE NOTICE '✅ learner_visual_preferences table created';
  RAISE NOTICE '✅ visual_selections table created';
  RAISE NOTICE '✅ get_variants_for_phase function created';
  RAISE NOTICE '✅ record_variant_selection function created';
  RAISE NOTICE '✅ get_variant_stats function created';
  RAISE NOTICE '';
  RAISE NOTICE '🎉 VISUAL VARIANTS MIGRATION COMPLETE';
END $$;
