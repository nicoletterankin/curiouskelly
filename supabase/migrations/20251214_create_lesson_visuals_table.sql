-- Lesson visuals tracking (thumbnails, infographics, illustrations)
-- Used by scripts/generate-lesson-visuals.ts

CREATE TABLE IF NOT EXISTS public.lesson_visuals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  core_lesson_id UUID REFERENCES public.core_lessons(id) ON DELETE SET NULL,
  day_number INTEGER UNIQUE NOT NULL,
  topic TEXT,

  thumbnail_url TEXT,
  thumbnail_path TEXT,

  -- Keep a single "primary" infographic_url for convenience, plus a full list.
  infographic_url TEXT,
  infographic_urls JSONB DEFAULT '[]'::jsonb,
  illustration_url TEXT,
  illustration_path TEXT,

  status TEXT NOT NULL DEFAULT 'pending',
  error TEXT,

  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (public read, service-role write)
ALTER TABLE public.lesson_visuals ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can read lesson visuals" ON public.lesson_visuals;
CREATE POLICY "Anyone can read lesson visuals"
  ON public.lesson_visuals
  FOR SELECT
  USING (true);

DROP POLICY IF EXISTS "Service role insert lesson visuals" ON public.lesson_visuals;
CREATE POLICY "Service role insert lesson visuals"
  ON public.lesson_visuals
  FOR INSERT
  TO service_role
  WITH CHECK (true);

DROP POLICY IF EXISTS "Service role update lesson visuals" ON public.lesson_visuals;
CREATE POLICY "Service role update lesson visuals"
  ON public.lesson_visuals
  FOR UPDATE
  TO service_role
  USING (true);

DROP POLICY IF EXISTS "Service role delete lesson visuals" ON public.lesson_visuals;
CREATE POLICY "Service role delete lesson visuals"
  ON public.lesson_visuals
  FOR DELETE
  TO service_role
  USING (true);

CREATE INDEX IF NOT EXISTS idx_lesson_visuals_day_number ON public.lesson_visuals(day_number);
CREATE INDEX IF NOT EXISTS idx_lesson_visuals_status ON public.lesson_visuals(status);
CREATE INDEX IF NOT EXISTS idx_lesson_visuals_core_lesson_id ON public.lesson_visuals(core_lesson_id);

-- updated_at trigger (function is created in 001_create_users_table.sql)
DROP TRIGGER IF EXISTS update_lesson_visuals_updated_at ON public.lesson_visuals;
CREATE TRIGGER update_lesson_visuals_updated_at
  BEFORE UPDATE ON public.lesson_visuals
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();











