-- ============================================
-- USERS TABLE: Add Lifetime Tracking Fields
-- ============================================
-- Extend users table with lifetime engagement metrics

-- Add new columns (IF NOT EXISTS not supported for columns, use DO block)
DO $$
BEGIN
  -- Lifetime engagement
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'first_lesson_at') THEN
    ALTER TABLE public.users ADD COLUMN first_lesson_at TIMESTAMPTZ;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lifetime_lessons_completed') THEN
    ALTER TABLE public.users ADD COLUMN lifetime_lessons_completed INTEGER DEFAULT 0;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lifetime_contributions') THEN
    ALTER TABLE public.users ADD COLUMN lifetime_contributions INTEGER DEFAULT 0;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lifetime_value_usd') THEN
    ALTER TABLE public.users ADD COLUMN lifetime_value_usd DECIMAL(10,2) DEFAULT 0;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'longest_streak') THEN
    ALTER TABLE public.users ADD COLUMN longest_streak INTEGER DEFAULT 0;
  END IF;
  
  -- Preferences
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'preferred_language') THEN
    ALTER TABLE public.users ADD COLUMN preferred_language VARCHAR(10) DEFAULT 'en';
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'timezone') THEN
    ALTER TABLE public.users ADD COLUMN timezone VARCHAR(50);
  END IF;
  
  -- Acquisition tracking
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'acquisition_source') THEN
    ALTER TABLE public.users ADD COLUMN acquisition_source VARCHAR(100);
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'acquisition_campaign') THEN
    ALTER TABLE public.users ADD COLUMN acquisition_campaign VARCHAR(100);
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'referred_by_user_id') THEN
    ALTER TABLE public.users ADD COLUMN referred_by_user_id UUID REFERENCES public.users(id);
  END IF;
END $$;

-- Update longest_streak trigger
CREATE OR REPLACE FUNCTION update_longest_streak()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.streak_days > COALESCE(NEW.longest_streak, 0) THEN
    NEW.longest_streak = NEW.streak_days;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS track_longest_streak ON public.users;

CREATE TRIGGER track_longest_streak
  BEFORE UPDATE ON public.users
  FOR EACH ROW
  WHEN (NEW.streak_days IS DISTINCT FROM OLD.streak_days)
  EXECUTE FUNCTION update_longest_streak();

-- Function to update first_lesson_at on first completion
CREATE OR REPLACE FUNCTION set_first_lesson_at()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.completed AND NOT COALESCE(OLD.completed, false) THEN
    UPDATE public.users
    SET 
      first_lesson_at = COALESCE(first_lesson_at, NOW()),
      lifetime_lessons_completed = lifetime_lessons_completed + 1
    WHERE id = NEW.user_id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS track_first_lesson ON public.user_progress;

CREATE TRIGGER track_first_lesson
  AFTER INSERT OR UPDATE ON public.user_progress
  FOR EACH ROW EXECUTE FUNCTION set_first_lesson_at();

-- Index for acquisition analysis
CREATE INDEX IF NOT EXISTS idx_users_acquisition ON public.users(acquisition_source, acquisition_campaign);
CREATE INDEX IF NOT EXISTS idx_users_referred_by ON public.users(referred_by_user_id) WHERE referred_by_user_id IS NOT NULL;

-- Comment
COMMENT ON COLUMN public.users.first_lesson_at IS 'Timestamp of first lesson completion';
COMMENT ON COLUMN public.users.lifetime_lessons_completed IS 'Total lessons completed (ever)';
COMMENT ON COLUMN public.users.lifetime_contributions IS 'Total comments + artwork submissions';
COMMENT ON COLUMN public.users.lifetime_value_usd IS 'Total revenue from this user';
COMMENT ON COLUMN public.users.longest_streak IS 'Longest streak achieved (ever)';
