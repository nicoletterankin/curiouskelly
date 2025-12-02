-- ═══════════════════════════════════════════════════════════════════════════
-- KELLY IMAGE GENERATION SYSTEM - DATABASE SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════
-- 
-- This schema supports:
-- - Character consistency through reference images
-- - Per-lesson custom images with fallbacks
-- - AI generation job queue and tracking
-- - Quality control and approval workflow
-- - Cost tracking and analytics
--
-- Run this in your Supabase SQL Editor
-- ═══════════════════════════════════════════════════════════════════════════

-- Drop existing tables if recreating (CAREFUL in production!)
-- DROP TABLE IF EXISTS kelly_generation_usage CASCADE;
-- DROP TABLE IF EXISTS kelly_generation_jobs CASCADE;
-- DROP TABLE IF EXISTS kelly_images CASCADE;
-- DROP TABLE IF EXISTS kelly_prompt_templates CASCADE;
-- DROP TABLE IF EXISTS kelly_character_references CASCADE;

-- ═══════════════════════════════════════════════════════════════════════════
-- CHARACTER REFERENCES - The "soul" of Kelly's visual identity
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kelly_character_references (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Versioning
  version TEXT NOT NULL UNIQUE,              -- 'v2.0', 'v2.1-refined'
  description TEXT,
  is_active BOOLEAN DEFAULT false,           -- Only one active at a time
  
  -- Reference images (stored in Supabase Storage)
  reference_images JSONB NOT NULL DEFAULT '[]',  -- Array of URLs
  
  -- The master prompt that defines Kelly
  style_prompt TEXT NOT NULL,
  negative_prompt TEXT,
  
  -- Face embedding for consistency checking (optional, for advanced use)
  face_embedding_model TEXT,                 -- 'insightface', 'deepface', etc.
  face_embedding BYTEA,                      -- Serialized embedding vector
  
  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_by TEXT
);

-- Ensure only one active reference at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_character_ref 
ON kelly_character_references (is_active) WHERE is_active = true;

-- ═══════════════════════════════════════════════════════════════════════════
-- PROMPT TEMPLATES - Reusable prompts with variables
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kelly_prompt_templates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Identity
  name TEXT NOT NULL UNIQUE,                 -- 'lesson_hero', 'base_thinking'
  category TEXT NOT NULL,                    -- 'base_pose', 'lesson_specific', 'reaction'
  description TEXT,
  
  -- Template (uses {{variable}} syntax)
  prompt_template TEXT NOT NULL,
  required_variables TEXT[] NOT NULL DEFAULT '{}',
  
  -- Defaults
  default_negative_prompt TEXT,
  default_params JSONB DEFAULT '{}',         -- { "steps": 30, "guidance": 7 }
  
  -- Versioning
  version INT DEFAULT 1,
  is_active BOOLEAN DEFAULT true,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════════════════
-- KELLY IMAGES - Master catalog of all Kelly images
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kelly_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- ═══ IDENTITY ═══
  image_type TEXT NOT NULL,                  -- 'base_pose', 'lesson_specific', 'reaction'
  state TEXT NOT NULL,                       -- 'thinking', 'hero', 'q1', etc.
  
  -- Lesson context (NULL for base poses)
  lesson_day INT,                            -- 1-365, NULL for base poses
  lesson_topic TEXT,
  lesson_category TEXT,                      -- 'science', 'philosophy', etc.
  
  -- ═══ STORAGE ═══
  storage_bucket TEXT NOT NULL DEFAULT 'kelly-images',
  storage_path TEXT NOT NULL,                -- 'lessons/336/hero.png'
  public_url TEXT NOT NULL,                  -- Full CDN URL
  thumbnail_path TEXT,
  thumbnail_url TEXT,
  
  -- ═══ GENERATION ═══
  character_ref_id UUID REFERENCES kelly_character_references(id),
  template_id UUID REFERENCES kelly_prompt_templates(id),
  full_prompt TEXT NOT NULL,                 -- The actual prompt used
  negative_prompt TEXT,
  
  generator TEXT NOT NULL,                   -- 'flux-1.1-pro', 'dall-e-3'
  model_version TEXT,
  seed BIGINT,                               -- For reproducibility
  generation_params JSONB DEFAULT '{}',      -- { steps, guidance, etc. }
  
  -- ═══ QUALITY ═══
  quality_score DECIMAL(4,3),                -- 0.000 to 1.000
  consistency_score DECIMAL(4,3),            -- Face/style similarity to reference
  
  auto_approved BOOLEAN DEFAULT false,
  is_approved BOOLEAN DEFAULT false,
  approved_by TEXT,
  approved_at TIMESTAMPTZ,
  rejection_reason TEXT,
  
  -- ═══ TECHNICAL ═══
  width INT NOT NULL,
  height INT NOT NULL,
  file_size_bytes BIGINT,
  format TEXT DEFAULT 'png',
  
  -- ═══ ANALYTICS ═══
  view_count INT DEFAULT 0,
  engagement_score DECIMAL(4,3),             -- From A/B testing
  
  -- ═══ TIMESTAMPS ═══
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Constraints
  CONSTRAINT valid_quality CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)),
  CONSTRAINT valid_consistency CHECK (consistency_score IS NULL OR (consistency_score >= 0 AND consistency_score <= 1))
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_kelly_images_lesson 
ON kelly_images(lesson_day, state) WHERE lesson_day IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_kelly_images_type_state 
ON kelly_images(image_type, state, is_approved);

CREATE INDEX IF NOT EXISTS idx_kelly_images_pending_approval 
ON kelly_images(is_approved, auto_approved) WHERE is_approved = false;

-- ═══════════════════════════════════════════════════════════════════════════
-- GENERATION JOBS - Queue for AI image generation
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kelly_generation_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- ═══ JOB SPECIFICATION ═══
  job_type TEXT NOT NULL,                    -- 'lesson_batch', 'single', 'regenerate'
  lesson_day INT,                            -- NULL for base pose jobs
  image_types TEXT[] NOT NULL,               -- ['hero', 'q1', 'q2', 'q3', 'wisdom']
  priority INT DEFAULT 5,                    -- 1 = highest
  
  -- ═══ GENERATION PARAMS ═══
  character_ref_id UUID REFERENCES kelly_character_references(id),
  generator TEXT NOT NULL,                   -- Which AI to use
  prompt_variables JSONB DEFAULT '{}',       -- Variables for templates
  generation_options JSONB DEFAULT '{}',     -- { variations: 3, quality: 'premium' }
  
  -- ═══ STATUS ═══
  status TEXT DEFAULT 'pending',             -- pending, processing, completed, failed, cancelled
  progress DECIMAL(5,2) DEFAULT 0,           -- 0.00 to 100.00
  current_step TEXT,                         -- 'generating hero', 'quality check'
  error_message TEXT,
  error_details JSONB,
  
  -- ═══ RESULTS ═══
  generated_image_ids UUID[] DEFAULT '{}',
  approved_image_ids UUID[] DEFAULT '{}',
  rejected_image_ids UUID[] DEFAULT '{}',
  
  -- ═══ TIMING ═══
  created_at TIMESTAMPTZ DEFAULT NOW(),
  scheduled_for TIMESTAMPTZ,                 -- For delayed processing
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  
  -- ═══ RETRY ═══
  attempt_count INT DEFAULT 0,
  max_attempts INT DEFAULT 3,
  next_retry_at TIMESTAMPTZ,
  
  -- ═══ METADATA ═══
  created_by TEXT,
  notes TEXT
);

-- Index for job queue processing
CREATE INDEX IF NOT EXISTS idx_kelly_jobs_queue 
ON kelly_generation_jobs(status, priority DESC, created_at ASC) 
WHERE status IN ('pending', 'processing');

-- ═══════════════════════════════════════════════════════════════════════════
-- USAGE TRACKING - For cost analysis
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kelly_generation_usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  job_id UUID REFERENCES kelly_generation_jobs(id),
  image_id UUID REFERENCES kelly_images(id),
  
  -- Usage details
  generator TEXT NOT NULL,
  model TEXT,
  operation TEXT NOT NULL,                   -- 'generate', 'upscale', 'edit'
  
  -- Metrics
  prompt_tokens INT,
  completion_tokens INT,
  compute_seconds DECIMAL(10,3),
  estimated_cost_usd DECIMAL(10,6),
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for cost analysis
CREATE INDEX IF NOT EXISTS idx_kelly_usage_date 
ON kelly_generation_usage(created_at);

-- ═══════════════════════════════════════════════════════════════════════════
-- FUNCTIONS
-- ═══════════════════════════════════════════════════════════════════════════

-- Get Kelly image with intelligent fallback
CREATE OR REPLACE FUNCTION get_kelly_image_with_fallback(
  p_lesson_day INT,
  p_image_type TEXT,
  p_fallback_state TEXT DEFAULT 'welcome'
)
RETURNS TABLE (
  image_url TEXT,
  thumbnail_url TEXT,
  is_lesson_specific BOOLEAN,
  image_id UUID,
  source TEXT
) 
LANGUAGE plpgsql
AS $$
BEGIN
  -- 1. Try lesson-specific image
  RETURN QUERY
  SELECT 
    ki.public_url,
    ki.thumbnail_url,
    true,
    ki.id,
    'lesson_specific'::TEXT
  FROM kelly_images ki
  WHERE ki.lesson_day = p_lesson_day
    AND ki.state = p_image_type
    AND ki.image_type = 'lesson_specific'
    AND ki.is_approved = true
  ORDER BY ki.created_at DESC
  LIMIT 1;
  
  IF FOUND THEN RETURN; END IF;
  
  -- 2. Fall back to base pose
  RETURN QUERY
  SELECT 
    ki.public_url,
    ki.thumbnail_url,
    false,
    ki.id,
    'base_pose'::TEXT
  FROM kelly_images ki
  WHERE ki.image_type = 'base_pose'
    AND ki.state = p_fallback_state
    AND ki.is_approved = true
  ORDER BY ki.quality_score DESC NULLS LAST, ki.created_at DESC
  LIMIT 1;
  
  IF FOUND THEN RETURN; END IF;
  
  -- 3. Return NULL if nothing found (client handles ultimate fallback)
  RETURN;
END;
$$;

-- Get all images for a lesson (for preloading)
CREATE OR REPLACE FUNCTION get_kelly_lesson_images(p_lesson_day INT)
RETURNS TABLE (
  image_type TEXT,
  state TEXT,
  image_url TEXT,
  thumbnail_url TEXT,
  is_lesson_specific BOOLEAN
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  WITH lesson_images AS (
    SELECT 
      ki.state,
      ki.public_url,
      ki.thumbnail_url,
      true as is_specific,
      1 as priority
    FROM kelly_images ki
    WHERE ki.lesson_day = p_lesson_day
      AND ki.image_type = 'lesson_specific'
      AND ki.is_approved = true
  ),
  base_fallbacks AS (
    SELECT 
      ki.state,
      ki.public_url,
      ki.thumbnail_url,
      false as is_specific,
      2 as priority
    FROM kelly_images ki
    WHERE ki.image_type = 'base_pose'
      AND ki.is_approved = true
      AND ki.state NOT IN (SELECT state FROM lesson_images)
  )
  SELECT 
    'lesson_specific'::TEXT,
    li.state,
    li.public_url,
    li.thumbnail_url,
    li.is_specific
  FROM (
    SELECT * FROM lesson_images
    UNION ALL
    SELECT * FROM base_fallbacks
  ) li
  ORDER BY li.priority, li.state;
END;
$$;

-- ═══════════════════════════════════════════════════════════════════════════
-- ROW LEVEL SECURITY
-- ═══════════════════════════════════════════════════════════════════════════

-- Enable RLS
ALTER TABLE kelly_character_references ENABLE ROW LEVEL SECURITY;
ALTER TABLE kelly_prompt_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE kelly_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE kelly_generation_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE kelly_generation_usage ENABLE ROW LEVEL SECURITY;

-- Public can read approved images
CREATE POLICY "Public can view approved images"
ON kelly_images FOR SELECT
TO anon, authenticated
USING (is_approved = true);

-- Service role can do everything
CREATE POLICY "Service role full access to images"
ON kelly_images FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access to refs"
ON kelly_character_references FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access to templates"
ON kelly_prompt_templates FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access to jobs"
ON kelly_generation_jobs FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access to usage"
ON kelly_generation_usage FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Authenticated users can view templates
CREATE POLICY "Authenticated can view templates"
ON kelly_prompt_templates FOR SELECT
TO authenticated
USING (is_active = true);

-- ═══════════════════════════════════════════════════════════════════════════
-- INITIAL DATA - The Master Kelly Character Reference
-- ═══════════════════════════════════════════════════════════════════════════

INSERT INTO kelly_character_references (
  version,
  description,
  is_active,
  reference_images,
  style_prompt,
  negative_prompt
) VALUES (
  'v2.0',
  'Production Kelly character reference - December 2025',
  true,
  '["https://your-supabase-project.supabase.co/storage/v1/object/public/kelly-references/kelly-ref-1.png"]',
  E'A warm, intelligent woman in her late 20s named Kelly. Oval face with soft features, warm brown expressive eyes with slight smile lines, natural well-groomed eyebrows, straight proportional nose, natural pink lips often in a genuine warm smile. Medium to light brown hair with subtle caramel highlights, long soft waves past shoulders, healthy natural movement, slightly off-center parting. Warm olive Mediterranean complexion, healthy natural glow. Healthy average build, confident open posture. Wearing a comfortable light blue crewneck sweater, casual professional style. Seated in a vintage Hollywood director''s chair with wood frame and black canvas, in a bright clean studio with white/light gray background, soft natural light from camera-right casting gentle shadows. Professional photography, high quality, warm and inviting atmosphere.',
  E'cartoon, anime, illustration, painting, drawing, sketch, 3D render, CGI, plastic, doll-like, uncanny valley, harsh lighting, dark shadows, moody, cold colors, busy background, clutter, text, watermarks, logos, different clothing, different hair color, different eye color, different age, masculine features, uncomfortable expression, forced smile, stiff posture'
)
ON CONFLICT (version) DO NOTHING;

-- Insert base prompt templates
INSERT INTO kelly_prompt_templates (name, category, description, prompt_template, required_variables)
VALUES 
  ('base_thinking', 'base_pose', 'Kelly in thoughtful pose', 
   E'{{character}}\n\nKelly has a thoughtful expression, hand resting gently on her chin, eyes looking slightly upward as if pondering a deep question. Her posture is relaxed but engaged, curious about what the learner will discover.\n\n{{quality}}',
   ARRAY['character', 'quality']),
   
  ('base_excited', 'base_pose', 'Kelly showing excitement',
   E'{{character}}\n\nKelly''s face lights up with genuine excitement and discovery. Her eyes are bright and wide, a big authentic smile showing her enthusiasm. She leans slightly forward, hands animated as if she''s about to share something wonderful.\n\n{{quality}}',
   ARRAY['character', 'quality']),
   
  ('lesson_hero', 'lesson_specific', 'Hero image for lesson thumbnail',
   E'{{character}}\n\nKelly is introducing today''s lesson about "{{topic}}". She holds or gestures toward {{prop_description}}, which relates to the theme of {{category}}. Her expression is {{emotion}} - inviting the learner to explore this fascinating topic together. The prop is tastefully integrated, not dominating the frame.\n\n{{quality}}',
   ARRAY['character', 'topic', 'prop_description', 'category', 'emotion', 'quality'])
   
ON CONFLICT (name) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════
-- VERIFICATION
-- ═══════════════════════════════════════════════════════════════════════════

-- Verify tables created
SELECT 'kelly_character_references' as table_name, count(*) as rows FROM kelly_character_references
UNION ALL
SELECT 'kelly_prompt_templates', count(*) FROM kelly_prompt_templates
UNION ALL
SELECT 'kelly_images', count(*) FROM kelly_images
UNION ALL
SELECT 'kelly_generation_jobs', count(*) FROM kelly_generation_jobs
UNION ALL
SELECT 'kelly_generation_usage', count(*) FROM kelly_generation_usage;



