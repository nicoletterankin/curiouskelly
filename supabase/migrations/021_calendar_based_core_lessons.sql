-- Calendar-based core_lessons (Kelly Temporal System)
--
-- Goals:
-- - Make core lesson lookup stable by calendar month/day across years
-- - Preserve existing day_number (1–365) as the canonical topic index
-- - Add Feb 29 as a special bonus lesson (day_number = 366)
-- - Add tables for time-anchored variants + emergency lessons + commons voting

-- 1) Extend core_lessons with calendar columns
ALTER TABLE public.core_lessons
  ADD COLUMN IF NOT EXISTS calendar_month INTEGER,
  ADD COLUMN IF NOT EXISTS calendar_day INTEGER,
  ADD COLUMN IF NOT EXISTS is_leap_day BOOLEAN DEFAULT FALSE;

-- 2) Populate calendar_month/calendar_day for existing 1–365 day_number rows
-- Use a non-leap anchor year so month/day mapping is stable.
UPDATE public.core_lessons
SET
  calendar_month = EXTRACT(MONTH FROM (DATE '2026-01-01' + (day_number - 1) * INTERVAL '1 day'))::INTEGER,
  calendar_day = EXTRACT(DAY FROM (DATE '2026-01-01' + (day_number - 1) * INTERVAL '1 day'))::INTEGER,
  is_leap_day = FALSE
WHERE day_number BETWEEN 1 AND 365;

-- 3) Indexes for fast lookup by calendar date
CREATE INDEX IF NOT EXISTS idx_core_lessons_calendar
  ON public.core_lessons(calendar_month, calendar_day);

CREATE INDEX IF NOT EXISTS idx_core_lessons_is_leap_day
  ON public.core_lessons(is_leap_day);

-- 4) Ensure Leap Day lesson exists (day_number = 366, Feb 29)
INSERT INTO public.core_lessons (
  day_number,
  calendar_month,
  calendar_day,
  is_leap_day,
  topic,
  universal_truth
)
SELECT
  366,
  2,
  29,
  TRUE,
  'The Gift of Extra Time',
  'Some days are rare. Use them wisely.'
WHERE NOT EXISTS (
  SELECT 1 FROM public.core_lessons WHERE is_leap_day = TRUE OR day_number = 366
);

-- 5) Temporal content (time-anchored variants)
CREATE TABLE IF NOT EXISTS public.temporal_content (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  core_lesson_id UUID REFERENCES public.core_lessons(id),
  anchor_year INTEGER NOT NULL,
  archetype TEXT NOT NULL,
  phase TEXT NOT NULL,
  content JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(core_lesson_id, anchor_year, archetype, phase)
);

CREATE INDEX IF NOT EXISTS idx_temporal_content_lookup
  ON public.temporal_content(core_lesson_id, anchor_year, archetype);

-- 6) Emergency lessons
CREATE TABLE IF NOT EXISTS public.emergency_lessons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  trigger_date DATE NOT NULL,
  event_name TEXT NOT NULL,
  event_type TEXT NOT NULL, -- 'tragedy', 'celebration', 'global_event'
  topic TEXT NOT NULL,
  universal_truth TEXT NOT NULL,
  replaces_regular_lesson BOOLEAN DEFAULT FALSE,
  active_start TIMESTAMPTZ NOT NULL,
  active_end TIMESTAMPTZ, -- NULL means indefinitely available
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by TEXT -- 'system', 'admin', 'learner_commons'
);

CREATE TABLE IF NOT EXISTS public.emergency_lesson_atoms (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  emergency_lesson_id UUID REFERENCES public.emergency_lessons(id),
  archetype TEXT NOT NULL,
  phase TEXT NOT NULL,
  content JSONB NOT NULL,
  age_bucket TEXT NOT NULL
);

-- 7) Learner commons voting for topic proposals
CREATE TABLE IF NOT EXISTS public.topic_proposals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  calendar_month INTEGER NOT NULL,
  calendar_day INTEGER NOT NULL,
  current_topic TEXT NOT NULL,
  proposed_topic TEXT NOT NULL,
  proposed_truth TEXT NOT NULL,
  reason TEXT NOT NULL,
  proposed_by UUID REFERENCES public.users(id),
  votes_for INTEGER DEFAULT 0,
  votes_against INTEGER DEFAULT 0,
  status TEXT DEFAULT 'voting', -- 'voting', 'approved', 'rejected'
  voting_ends_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.topic_votes (
  proposal_id UUID REFERENCES public.topic_proposals(id),
  user_id UUID REFERENCES public.users(id),
  vote BOOLEAN NOT NULL, -- true = for, false = against
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (proposal_id, user_id)
);














