-- ============================================================================
-- LESSON SCHEMA CONSOLIDATION
-- ============================================================================
-- Migrates to the canonical core_lessons + lesson_atoms model.
-- This is the production schema for the 365-day curriculum.
--
-- Tables:
--   core_lessons: Master list of 365 lessons (one per day)
--   lesson_atoms: Content fragments by (lesson, archetype, phase, language)
--   lesson_audio_cache: Cached ElevenLabs audio URLs
--
-- The old tables (lessons, lesson_shards) are preserved but deprecated.
-- ============================================================================

-- 1. CREATE CORE_LESSONS TABLE (if not exists)
CREATE TABLE IF NOT EXISTS public.core_lessons (
  id SERIAL PRIMARY KEY,
  day_number INTEGER UNIQUE NOT NULL CHECK (day_number >= 1 AND day_number <= 365),
  topic VARCHAR(255) NOT NULL,
  universal_truth TEXT NOT NULL,
  description TEXT,
  category VARCHAR(100),
  subcategory VARCHAR(100),
  difficulty VARCHAR(20) DEFAULT 'beginner' CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
  estimated_minutes INTEGER DEFAULT 8,
  tags TEXT[] DEFAULT '{}',
  calendar_date VARCHAR(50), -- "January 1, 2025"
  calendar_month VARCHAR(20), -- "January"
  is_published BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. CREATE LESSON_ATOMS TABLE (if not exists)
CREATE TABLE IF NOT EXISTS public.lesson_atoms (
  id SERIAL PRIMARY KEY,
  core_lesson_id INTEGER NOT NULL REFERENCES public.core_lessons(id) ON DELETE CASCADE,
  age_bucket VARCHAR(10) NOT NULL CHECK (age_bucket IN ('2-5', '6-12', '13-17', '18-35', '36-60', '61-102')),
  phase VARCHAR(20) NOT NULL CHECK (phase IN ('welcome', 'teaching', 'practice', 'reflection', 'wisdom')),
  language VARCHAR(5) NOT NULL DEFAULT 'en' CHECK (language IN ('en', 'es', 'fr')),

  -- Content fields
  title VARCHAR(255),
  content TEXT NOT NULL, -- Kelly's script for this phase
  choices JSONB, -- For teaching/practice: [{text, response}, {text, response}]
  expression_cues JSONB, -- [{timestamp, type, intensity}]

  -- Metadata
  duration_seconds INTEGER,
  is_complete BOOLEAN DEFAULT false,
  validation_status VARCHAR(20) DEFAULT 'pending' CHECK (validation_status IN ('pending', 'valid', 'invalid', 'needs_review')),

  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),

  -- Prevent duplicates
  UNIQUE(core_lesson_id, age_bucket, phase, language)
);

-- 3. CREATE LESSON_AUDIO_CACHE TABLE
CREATE TABLE IF NOT EXISTS public.lesson_audio_cache (
  id SERIAL PRIMARY KEY,
  lesson_atom_id INTEGER NOT NULL REFERENCES public.lesson_atoms(id) ON DELETE CASCADE,

  -- Audio metadata
  audio_url TEXT NOT NULL,
  audio_hash VARCHAR(64) NOT NULL, -- SHA-256 of text content
  duration_ms INTEGER,
  file_size_bytes INTEGER,

  -- ElevenLabs metadata
  voice_id VARCHAR(100),
  model_id VARCHAR(100),
  voice_settings JSONB,

  created_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ, -- For cache invalidation

  UNIQUE(lesson_atom_id, audio_hash)
);

-- 4. INDEXES
CREATE INDEX IF NOT EXISTS idx_core_lessons_day ON public.core_lessons(day_number);
CREATE INDEX IF NOT EXISTS idx_core_lessons_category ON public.core_lessons(category);
CREATE INDEX IF NOT EXISTS idx_core_lessons_published ON public.core_lessons(is_published) WHERE is_published = true;

CREATE INDEX IF NOT EXISTS idx_atoms_lesson ON public.lesson_atoms(core_lesson_id);
CREATE INDEX IF NOT EXISTS idx_atoms_lookup ON public.lesson_atoms(core_lesson_id, age_bucket, phase, language);
CREATE INDEX IF NOT EXISTS idx_atoms_validation ON public.lesson_atoms(validation_status);

CREATE INDEX IF NOT EXISTS idx_audio_cache_atom ON public.lesson_audio_cache(lesson_atom_id);
CREATE INDEX IF NOT EXISTS idx_audio_cache_hash ON public.lesson_audio_cache(audio_hash);

-- 5. ROW LEVEL SECURITY
ALTER TABLE public.core_lessons ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lesson_atoms ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lesson_audio_cache ENABLE ROW LEVEL SECURITY;

-- Public read access for lessons
CREATE POLICY IF NOT EXISTS "Anyone can read published lessons"
  ON public.core_lessons FOR SELECT
  USING (is_published = true);

CREATE POLICY IF NOT EXISTS "Anyone can read lesson atoms"
  ON public.lesson_atoms FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.core_lessons cl
      WHERE cl.id = lesson_atoms.core_lesson_id AND cl.is_published = true
    )
  );

CREATE POLICY IF NOT EXISTS "Anyone can read audio cache"
  ON public.lesson_audio_cache FOR SELECT USING (true);

-- 6. TRIGGERS
DROP TRIGGER IF EXISTS update_core_lessons_updated_at ON public.core_lessons;
CREATE TRIGGER update_core_lessons_updated_at
  BEFORE UPDATE ON public.core_lessons
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_lesson_atoms_updated_at ON public.lesson_atoms;
CREATE TRIGGER update_lesson_atoms_updated_at
  BEFORE UPDATE ON public.lesson_atoms
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 7. HELPER FUNCTIONS

-- Get complete lesson content for a day
CREATE OR REPLACE FUNCTION public.get_lesson_for_day(
  p_day_number INTEGER,
  p_age_bucket VARCHAR DEFAULT '18-35',
  p_language VARCHAR DEFAULT 'en'
)
RETURNS JSONB AS $$
DECLARE
  v_result JSONB;
BEGIN
  SELECT jsonb_build_object(
    'lesson', row_to_json(cl),
    'atoms', (
      SELECT jsonb_agg(row_to_json(la) ORDER BY
        CASE la.phase
          WHEN 'welcome' THEN 1
          WHEN 'teaching' THEN 2
          WHEN 'practice' THEN 3
          WHEN 'reflection' THEN 4
          WHEN 'wisdom' THEN 5
        END
      )
      FROM public.lesson_atoms la
      WHERE la.core_lesson_id = cl.id
        AND la.age_bucket = p_age_bucket
        AND la.language = p_language
    )
  )
  INTO v_result
  FROM public.core_lessons cl
  WHERE cl.day_number = p_day_number
    AND cl.is_published = true;

  RETURN v_result;
END;
$$ LANGUAGE plpgsql STABLE;

-- Get lesson completion status
CREATE OR REPLACE FUNCTION public.get_lesson_completion_status(p_lesson_id INTEGER)
RETURNS JSONB AS $$
DECLARE
  v_total INTEGER;
  v_complete INTEGER;
  v_by_bucket JSONB;
BEGIN
  SELECT
    COUNT(*),
    COUNT(*) FILTER (WHERE is_complete = true)
  INTO v_total, v_complete
  FROM public.lesson_atoms
  WHERE core_lesson_id = p_lesson_id;

  SELECT jsonb_object_agg(
    age_bucket,
    jsonb_build_object(
      'total', COUNT(*),
      'complete', COUNT(*) FILTER (WHERE is_complete),
      'phases', jsonb_agg(DISTINCT phase)
    )
  )
  INTO v_by_bucket
  FROM public.lesson_atoms
  WHERE core_lesson_id = p_lesson_id
  GROUP BY age_bucket;

  RETURN jsonb_build_object(
    'total_atoms', v_total,
    'complete_atoms', v_complete,
    'completion_percent', CASE WHEN v_total > 0 THEN ROUND(v_complete::numeric / v_total * 100) ELSE 0 END,
    'by_age_bucket', v_by_bucket
  );
END;
$$ LANGUAGE plpgsql STABLE;

-- 8. COMMENTS
COMMENT ON TABLE public.core_lessons IS 'Master list of 365 daily lessons';
COMMENT ON TABLE public.lesson_atoms IS 'Content fragments by lesson/age/phase/language';
COMMENT ON TABLE public.lesson_audio_cache IS 'Cached ElevenLabs TTS audio URLs';
COMMENT ON COLUMN public.lesson_atoms.choices IS 'JSON array: [{text, response}, {text, response}] for interactive phases';
COMMENT ON COLUMN public.lesson_atoms.expression_cues IS 'JSON array: [{timestamp, type, intensity}] for Kelly avatar animation';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- To populate from JSON files, use the import script:
--   node scripts/import-lessons-to-supabase.js
-- ============================================================================
