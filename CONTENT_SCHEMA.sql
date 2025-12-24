-- ============================================
-- KELLY OS CONTENT SCHEMA
-- Complete database schema for all lesson content
-- ============================================
-- Run this in Supabase SQL Editor
-- Project: tvjalxxsyryjphkforjv

-- ============================================
-- CORE LESSONS TABLE
-- ============================================
-- Stores the 365 daily lessons with metadata

CREATE TABLE IF NOT EXISTS public.core_lessons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  day_number INTEGER UNIQUE NOT NULL CHECK (day_number >= 1 AND day_number <= 365),
  topic TEXT NOT NULL,
  universal_truth TEXT,
  ideal_age_range TEXT,
  difficulty_level TEXT CHECK (difficulty_level IN ('beginner', 'intermediate', 'advanced')),
  estimated_duration INTEGER, -- minutes
  
  -- Content extensions (JSONB)
  quick_quiz_questions JSONB DEFAULT '[]', -- Array of quiz questions
  reflection_prompts JSONB DEFAULT '[]', -- Array of reflection prompts
  recommended_videos JSONB DEFAULT '[]', -- Array of {title, url, source, duration}
  recommended_books JSONB DEFAULT '[]', -- Array of {title, author, url}
  interactive_simulations JSONB DEFAULT '[]', -- Array of {title, url, type}
  downloadable_resources JSONB DEFAULT '[]', -- Array of {title, url, type}
  discussion_questions JSONB DEFAULT '[]', -- Array of discussion questions
  hands_on_activities JSONB DEFAULT '[]', -- Array of {title, description, materials}
  creative_prompts JSONB DEFAULT '[]', -- Array of creative prompts
  challenge_questions JSONB DEFAULT '[]', -- Array of challenge questions
  historical_context TEXT, -- Historical context text
  
  -- Media
  hero_image_url TEXT,
  thumbnail_url TEXT,
  demo_video_url TEXT,
  
  -- Metadata
  tags TEXT[] DEFAULT '{}',
  is_published BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_core_lessons_day_number ON public.core_lessons(day_number);
CREATE INDEX idx_core_lessons_published ON public.core_lessons(is_published) WHERE is_published = true;

-- ============================================
-- LESSON ATOMS TABLE
-- ============================================
-- Stores individual lesson phases per archetype
-- 365 days × 5 phases × 12 archetypes = 21,915 atoms

CREATE TABLE IF NOT EXISTS public.lesson_atoms (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  core_lesson_id UUID REFERENCES public.core_lessons(id) ON DELETE CASCADE,
  archetype TEXT NOT NULL, -- 'The Explorer', 'The Rebel', etc.
  phase TEXT NOT NULL CHECK (phase IN ('Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom')),
  
  -- Content (JSONB)
  content JSONB NOT NULL, -- {script, script_video_url, options: [{letter, text, quality, response, response_video_url}]}
  
  -- Media
  visual_url TEXT, -- Infographic URL
  hd_video_url TEXT, -- HD video URL
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(core_lesson_id, archetype, phase)
);

CREATE INDEX idx_lesson_atoms_core_lesson ON public.lesson_atoms(core_lesson_id);
CREATE INDEX idx_lesson_atoms_lookup ON public.lesson_atoms(core_lesson_id, archetype, phase);

-- ============================================
-- LESSON VISUALS TABLE
-- ============================================
-- Tracks visual asset generation and status

CREATE TABLE IF NOT EXISTS public.lesson_visuals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  core_lesson_id UUID REFERENCES public.core_lessons(id) ON DELETE CASCADE,
  day_number INTEGER UNIQUE NOT NULL,
  
  -- Visual URLs
  thumbnail_url TEXT,
  infographic_url TEXT,
  infographic_urls JSONB DEFAULT '[]', -- Array of infographic URLs per phase
  illustration_url TEXT,
  
  -- Status
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'generating', 'completed', 'failed')),
  error_message TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_lesson_visuals_day_number ON public.lesson_visuals(day_number);
CREATE INDEX idx_lesson_visuals_status ON public.lesson_visuals(status);

-- ============================================
-- KELLY VIDEO ASSETS TABLE
-- ============================================
-- Comprehensive video asset tracking with quality metadata

CREATE TABLE IF NOT EXISTS public.kelly_video_assets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Content reference
  lesson_day INTEGER NOT NULL,
  phase TEXT NOT NULL CHECK (phase IN ('welcome', 'q1', 'q2', 'q3', 'wisdom')),
  
  -- Variant keys
  age_bucket TEXT NOT NULL CHECK (age_bucket IN ('toddler', 'child', 'teen', 'young_adult', 'adult', 'elder')),
  language TEXT NOT NULL DEFAULT 'en',
  archetype TEXT, -- Optional: 'Scientist', 'Explorer', etc.
  
  -- Source assets
  source_image_path TEXT NOT NULL,
  source_audio_url TEXT,
  script_text TEXT,
  
  -- Generated video
  video_storage_path TEXT,
  video_public_url TEXT,
  video_duration_ms INTEGER,
  video_file_size_bytes BIGINT,
  video_format TEXT DEFAULT 'mp4',
  video_resolution TEXT, -- e.g., '1080x1920'
  
  -- Generation metadata
  elevenlabs_generation_id TEXT,
  model_used TEXT DEFAULT 'omnihuman-1.5',
  generation_credits_used INTEGER,
  generation_started_at TIMESTAMPTZ,
  generation_completed_at TIMESTAMPTZ,
  generation_duration_ms INTEGER,
  
  -- Quality metadata
  lip_sync_quality_score DECIMAL(4,3) CHECK (lip_sync_quality_score >= 0 AND lip_sync_quality_score <= 1),
  video_quality_score DECIMAL(4,3) CHECK (video_quality_score >= 0 AND video_quality_score <= 1),
  is_approved BOOLEAN DEFAULT false,
  approved_by TEXT,
  approved_at TIMESTAMPTZ,
  
  -- Status
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'generating', 'completed', 'failed', 'expired')),
  error_message TEXT,
  retry_count INTEGER DEFAULT 0,
  
  -- Usage analytics
  view_count INTEGER DEFAULT 0,
  last_viewed_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(lesson_day, phase, age_bucket, language)
);

CREATE INDEX idx_kelly_video_lookup ON public.kelly_video_assets(lesson_day, phase, age_bucket, language, status);
CREATE INDEX idx_kelly_video_completed ON public.kelly_video_assets(lesson_day, phase, age_bucket, language) WHERE status = 'completed';
CREATE INDEX idx_kelly_video_queue ON public.kelly_video_assets(status, created_at) WHERE status IN ('pending', 'generating');

-- ============================================
-- LESSON VIDEO GENERATION STATUS TABLE
-- ============================================
-- Tracks video generation progress for 5-phase journey

CREATE TABLE IF NOT EXISTS public.lesson_video_generation_status (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  core_lesson_id UUID NOT NULL REFERENCES public.core_lessons(id) ON DELETE CASCADE,
  archetype TEXT NOT NULL,
  phase TEXT NOT NULL CHECK (phase IN ('Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom')),
  video_type TEXT NOT NULL CHECK (video_type IN ('main', 'response_A', 'response_B', 'response_C')),
  
  -- Generation status
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'generating', 'completed', 'failed')),
  video_url TEXT,
  error_message TEXT,
  
  -- Metadata
  duration_seconds NUMERIC(6,2),
  file_size_bytes BIGINT,
  resolution TEXT,
  
  -- Timestamps
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(core_lesson_id, archetype, phase, video_type)
);

CREATE INDEX idx_video_generation_status ON public.lesson_video_generation_status(core_lesson_id, archetype, phase);
CREATE INDEX idx_video_generation_pending ON public.lesson_video_generation_status(status) WHERE status = 'pending';

-- ============================================
-- LESSON SHARDS TABLE
-- ============================================
-- Age/language/tone variants of lessons

CREATE TABLE IF NOT EXISTS public.lesson_shards (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  core_lesson_id UUID REFERENCES public.core_lessons(id) ON DELETE CASCADE,
  age INTEGER NOT NULL,
  region TEXT NOT NULL,
  tone TEXT NOT NULL,
  birth_year INTEGER,
  script_content JSONB NOT NULL,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_lesson_shards_core_lesson ON public.lesson_shards(core_lesson_id);
CREATE INDEX idx_lesson_shards_lookup ON public.lesson_shards(age, region, tone, birth_year);

-- ============================================
-- COMMONS LESSON NOTES TABLE
-- ============================================
-- Community-generated notes and research

CREATE TABLE IF NOT EXISTS public.commons_lesson_notes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  lesson_id UUID REFERENCES public.core_lessons(id) ON DELETE CASCADE,
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Note content
  type TEXT NOT NULL CHECK (type IN ('expert_context', 'historical_note', 'real_world_example', 'discussion_prompt')),
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  sources TEXT[], -- Array of source URLs
  
  -- Related content
  related_lessons UUID[], -- Array of related lesson IDs
  
  -- Moderation
  is_approved BOOLEAN DEFAULT false,
  votes_up INTEGER DEFAULT 0,
  votes_down INTEGER DEFAULT 0,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_commons_lesson_notes_lesson ON public.commons_lesson_notes(lesson_id);
CREATE INDEX idx_commons_lesson_notes_user ON public.commons_lesson_notes(user_id);
CREATE INDEX idx_commons_lesson_notes_type ON public.commons_lesson_notes(type);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Get video URL helper
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
  FROM public.kelly_video_assets
  WHERE lesson_day = p_lesson_day
    AND phase = p_phase
    AND age_bucket = p_age_bucket
    AND language = p_language
    AND status = 'completed';
  
  RETURN v_url;
END;
$$;

-- Get lesson video count
CREATE OR REPLACE FUNCTION get_lesson_video_count(lesson_id UUID)
RETURNS TABLE (
  total_videos INTEGER,
  completed_videos INTEGER,
  pending_videos INTEGER,
  failed_videos INTEGER
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    COUNT(*)::INTEGER as total_videos,
    COUNT(*) FILTER (WHERE status = 'completed')::INTEGER as completed_videos,
    COUNT(*) FILTER (WHERE status = 'pending')::INTEGER as pending_videos,
    COUNT(*) FILTER (WHERE status = 'failed')::INTEGER as failed_videos
  FROM public.lesson_video_generation_status
  WHERE core_lesson_id = lesson_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- TRIGGERS
-- ============================================

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_core_lessons_updated_at BEFORE UPDATE ON public.core_lessons
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_lesson_atoms_updated_at BEFORE UPDATE ON public.lesson_atoms
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_lesson_visuals_updated_at BEFORE UPDATE ON public.lesson_visuals
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_kelly_video_assets_updated_at BEFORE UPDATE ON public.kelly_video_assets
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_commons_lesson_notes_updated_at BEFORE UPDATE ON public.commons_lesson_notes
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

-- Core lessons: public read for published lessons
ALTER TABLE public.core_lessons ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view published lessons" ON public.core_lessons
  FOR SELECT USING (is_published = true);

-- Lesson atoms: public read
ALTER TABLE public.lesson_atoms ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view lesson atoms" ON public.lesson_atoms
  FOR SELECT USING (true);

-- Lesson visuals: public read
ALTER TABLE public.lesson_visuals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view lesson visuals" ON public.lesson_visuals
  FOR SELECT USING (true);

-- Kelly video assets: public read for completed videos
ALTER TABLE public.kelly_video_assets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public can view completed videos" ON public.kelly_video_assets
  FOR SELECT USING (status = 'completed');

-- Commons lesson notes: public read for approved notes
ALTER TABLE public.commons_lesson_notes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view approved notes" ON public.commons_lesson_notes
  FOR SELECT USING (is_approved = true);

CREATE POLICY "Users can create own notes" ON public.commons_lesson_notes
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own notes" ON public.commons_lesson_notes
  FOR UPDATE USING (auth.uid() = user_id);

-- ============================================
-- COMMENTS
-- ============================================

COMMENT ON TABLE public.core_lessons IS '365 daily lessons with metadata and content extensions';
COMMENT ON TABLE public.lesson_atoms IS 'Individual lesson phases per archetype (21,915 total)';
COMMENT ON TABLE public.lesson_visuals IS 'Visual asset tracking and generation status';
COMMENT ON TABLE public.kelly_video_assets IS 'Comprehensive video asset registry with quality metadata';
COMMENT ON TABLE public.lesson_video_generation_status IS 'Video generation progress tracking';
COMMENT ON TABLE public.lesson_shards IS 'Age/language/tone variants of lessons';
COMMENT ON TABLE public.commons_lesson_notes IS 'Community-generated notes and research';


