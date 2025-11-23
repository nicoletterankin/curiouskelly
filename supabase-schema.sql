-- ============================================
-- CURIOUS KELLY - PRODUCTION DATABASE SCHEMA
-- ============================================
-- Run this in Supabase SQL Editor
-- Project: tvjalxxsyryjphkforjv

-- ============================================
-- 1. USERS TABLE (extends auth.users)
-- ============================================

CREATE TABLE IF NOT EXISTS public.users (
  id UUID REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
  email TEXT NOT NULL,
  name TEXT,
  age INTEGER,
  subscription_tier TEXT DEFAULT 'free' CHECK (subscription_tier IN ('free', 'annual', 'gift', 'enterprise')),
  subscription_status TEXT DEFAULT 'inactive' CHECK (subscription_status IN ('active', 'inactive', 'cancelled', 'expired')),
  subscription_started_at TIMESTAMPTZ,
  subscription_expires_at TIMESTAMPTZ,
  stripe_customer_id TEXT UNIQUE,
  current_day INTEGER DEFAULT 1,
  streak_days INTEGER DEFAULT 0,
  last_lesson_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own data" ON public.users
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own data" ON public.users
  FOR UPDATE USING (auth.uid() = id);

-- Trigger to create user record on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.users (id, email, name)
  VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'name');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================
-- 2. LESSONS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.lessons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  day_number INTEGER UNIQUE NOT NULL,
  title TEXT NOT NULL,
  subtitle TEXT,
  content JSONB NOT NULL, -- PhaseDNA structure
  audio_url TEXT,
  duration_seconds INTEGER,
  difficulty TEXT CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
  tags TEXT[],
  is_published BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.lessons ENABLE ROW LEVEL SECURITY;

-- Anyone can view published lessons
CREATE POLICY "Anyone can view published lessons" ON public.lessons
  FOR SELECT USING (is_published = true);

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_lessons_day_number ON public.lessons(day_number);
CREATE INDEX IF NOT EXISTS idx_lessons_published ON public.lessons(is_published) WHERE is_published = true;

-- ============================================
-- 3. USER PROGRESS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.user_progress (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
  lesson_id UUID REFERENCES public.lessons(id) ON DELETE CASCADE NOT NULL,
  completed BOOLEAN DEFAULT false,
  progress_percent INTEGER DEFAULT 0 CHECK (progress_percent >= 0 AND progress_percent <= 100),
  last_position_seconds INTEGER DEFAULT 0,
  time_spent_seconds INTEGER DEFAULT 0,
  completed_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, lesson_id)
);

-- Enable RLS
ALTER TABLE public.user_progress ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own progress" ON public.user_progress
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own progress" ON public.user_progress
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own progress" ON public.user_progress
  FOR UPDATE USING (auth.uid() = user_id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON public.user_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_user_progress_lesson_id ON public.user_progress(lesson_id);
CREATE INDEX IF NOT EXISTS idx_user_progress_completed ON public.user_progress(completed) WHERE completed = true;

-- ============================================
-- 4. AFFILIATES TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.affiliates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
  referral_code TEXT UNIQUE NOT NULL,
  tier TEXT DEFAULT 'scholar' CHECK (tier IN ('scholar', 'fellow', 'ambassador', 'founding')),
  commission_rate DECIMAL(5,2) DEFAULT 20.00 CHECK (commission_rate >= 0 AND commission_rate <= 100),
  is_founding_100 BOOLEAN DEFAULT false,
  total_referrals INTEGER DEFAULT 0,
  active_referrals INTEGER DEFAULT 0,
  lifetime_earnings DECIMAL(10,2) DEFAULT 0.00,
  last_payout_at TIMESTAMPTZ,
  status TEXT DEFAULT 'active' CHECK (status IN ('pending', 'active', 'suspended', 'terminated')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.affiliates ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own affiliate data" ON public.affiliates
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own affiliate data" ON public.affiliates
  FOR UPDATE USING (auth.uid() = user_id);

-- Index
CREATE INDEX IF NOT EXISTS idx_affiliates_referral_code ON public.affiliates(referral_code);
CREATE INDEX IF NOT EXISTS idx_affiliates_user_id ON public.affiliates(user_id);

-- ============================================
-- 5. REFERRALS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.referrals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  affiliate_id UUID REFERENCES public.affiliates(id) ON DELETE CASCADE NOT NULL,
  referred_user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
  referral_code TEXT NOT NULL,
  subscription_value DECIMAL(10,2),
  commission_earned DECIMAL(10,2),
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'cancelled', 'paid')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.referrals ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Affiliates can view own referrals" ON public.referrals
  FOR SELECT USING (
    affiliate_id IN (SELECT id FROM public.affiliates WHERE user_id = auth.uid())
  );

-- Indexes
CREATE INDEX IF NOT EXISTS idx_referrals_affiliate_id ON public.referrals(affiliate_id);
CREATE INDEX IF NOT EXISTS idx_referrals_referred_user_id ON public.referrals(referred_user_id);

-- ============================================
-- 6. AFFILIATE APPLICATIONS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.affiliate_applications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  platform TEXT NOT NULL,
  url TEXT,
  audience TEXT,
  focus TEXT,
  why TEXT,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  reviewed_at TIMESTAMPTZ,
  reviewed_by UUID REFERENCES auth.users(id),
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.affiliate_applications ENABLE ROW LEVEL SECURITY;

-- Public can submit applications
CREATE POLICY "Anyone can submit applications" ON public.affiliate_applications
  FOR INSERT WITH CHECK (true);

-- ============================================
-- 7. ENTERPRISE INQUIRIES TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.enterprise_inquiries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  organization TEXT NOT NULL,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  phone TEXT,
  org_type TEXT,
  size TEXT,
  use_case TEXT,
  timeline TEXT,
  status TEXT DEFAULT 'new' CHECK (status IN ('new', 'contacted', 'qualified', 'proposal', 'closed', 'lost')),
  assigned_to UUID REFERENCES auth.users(id),
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.enterprise_inquiries ENABLE ROW LEVEL SECURITY;

-- Public can submit inquiries
CREATE POLICY "Anyone can submit inquiries" ON public.enterprise_inquiries
  FOR INSERT WITH CHECK (true);

-- ============================================
-- 8. NEWSLETTER SUBSCRIBERS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.newsletter_subscribers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  source TEXT DEFAULT 'website',
  status TEXT DEFAULT 'subscribed' CHECK (status IN ('subscribed', 'unsubscribed')),
  subscribed_at TIMESTAMPTZ DEFAULT NOW(),
  unsubscribed_at TIMESTAMPTZ
);

-- Enable RLS
ALTER TABLE public.newsletter_subscribers ENABLE ROW LEVEL SECURITY;

-- Public can subscribe
CREATE POLICY "Anyone can subscribe" ON public.newsletter_subscribers
  FOR INSERT WITH CHECK (true);

-- Index
CREATE INDEX IF NOT EXISTS idx_newsletter_email ON public.newsletter_subscribers(email);

-- ============================================
-- 9. ANALYTICS EVENTS TABLE (Optional)
-- ============================================

CREATE TABLE IF NOT EXISTS public.analytics_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
  event_type TEXT NOT NULL,
  event_data JSONB,
  session_id TEXT,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.analytics_events ENABLE ROW LEVEL SECURITY;

-- Users can insert their own events
CREATE POLICY "Users can log own events" ON public.analytics_events
  FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON public.analytics_events(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON public.analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON public.analytics_events(created_at DESC);

-- ============================================
-- 10. FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_lessons_updated_at BEFORE UPDATE ON public.lessons
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_progress_updated_at BEFORE UPDATE ON public.user_progress
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_affiliates_updated_at BEFORE UPDATE ON public.affiliates
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_referrals_updated_at BEFORE UPDATE ON public.referrals
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_enterprise_inquiries_updated_at BEFORE UPDATE ON public.enterprise_inquiries
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate affiliate tier based on referrals
CREATE OR REPLACE FUNCTION update_affiliate_tier()
RETURNS TRIGGER AS $$
BEGIN
  -- Don't change tier for Founding 100
  IF NEW.is_founding_100 THEN
    NEW.tier = 'founding';
    NEW.commission_rate = 30.00;
    RETURN NEW;
  END IF;

  -- Update tier based on active referrals
  IF NEW.active_referrals >= 500 THEN
    NEW.tier = 'ambassador';
    NEW.commission_rate = 30.00;
  ELSIF NEW.active_referrals >= 100 THEN
    NEW.tier = 'fellow';
    NEW.commission_rate = 25.00;
  ELSE
    NEW.tier = 'scholar';
    NEW.commission_rate = 20.00;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calculate_affiliate_tier BEFORE INSERT OR UPDATE ON public.affiliates
  FOR EACH ROW EXECUTE FUNCTION update_affiliate_tier();

-- Function to update user streak
CREATE OR REPLACE FUNCTION update_user_streak()
RETURNS TRIGGER AS $$
DECLARE
  last_lesson_date DATE;
  today_date DATE;
BEGIN
  -- Only update streak when lesson is completed
  IF NEW.completed AND (OLD.completed IS NULL OR NOT OLD.completed) THEN
    SELECT last_lesson_at::DATE INTO last_lesson_date
    FROM public.users
    WHERE id = NEW.user_id;

    today_date := NOW()::DATE;

    -- If last lesson was yesterday, increment streak
    IF last_lesson_date = today_date - INTERVAL '1 day' THEN
      UPDATE public.users
      SET 
        streak_days = streak_days + 1,
        last_lesson_at = NOW(),
        current_day = current_day + 1
      WHERE id = NEW.user_id;
    
    -- If last lesson was today, don't change streak
    ELSIF last_lesson_date = today_date THEN
      -- Do nothing
      NULL;
    
    -- Otherwise, reset streak to 1
    ELSE
      UPDATE public.users
      SET 
        streak_days = 1,
        last_lesson_at = NOW(),
        current_day = (SELECT COUNT(*) FROM public.user_progress WHERE user_id = NEW.user_id AND completed = true)
      WHERE id = NEW.user_id;
    END IF;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER maintain_user_streak AFTER INSERT OR UPDATE ON public.user_progress
  FOR EACH ROW EXECUTE FUNCTION update_user_streak();

-- ============================================
-- 11. SEED DATA (Sample Lessons)
-- ============================================

-- Insert first 5 lessons as examples
INSERT INTO public.lessons (day_number, title, subtitle, content, duration_seconds, difficulty, tags, is_published) VALUES
(1, 'The Sun', 'Our Star and Source of Life', '{"phases": [{"type": "welcome", "content": "Welcome to Day 1!"}, {"type": "question", "content": "What is the Sun?"}]}', 600, 'beginner', ARRAY['science', 'astronomy'], true),
(2, 'Habit Stacking', 'Building Better Daily Routines', '{"phases": [{"type": "welcome", "content": "Welcome to Day 2!"}]}', 480, 'beginner', ARRAY['life-skills', 'productivity'], true),
(3, 'Planet Earth', 'Our Home in the Solar System', '{"phases": [{"type": "welcome", "content": "Welcome to Day 3!"}]}', 540, 'beginner', ARRAY['science', 'earth'], true),
(4, 'Simple Machines', 'Levers, Pulleys, and Wheels', '{"phases": [{"type": "welcome", "content": "Welcome to Day 4!"}]}', 510, 'beginner', ARRAY['science', 'physics'], true),
(5, 'Emotional Intelligence', 'Understanding Your Feelings', '{"phases": [{"type": "welcome", "content": "Welcome to Day 5!"}]}', 600, 'intermediate', ARRAY['life-skills', 'psychology'], true)
ON CONFLICT (day_number) DO NOTHING;

-- ============================================
-- SCHEMA COMPLETE
-- ============================================
-- Next steps:
-- 1. Run this SQL in Supabase SQL Editor
-- 2. Configure OAuth providers in Authentication settings
-- 3. Set up Storage buckets for images/audio
-- 4. Deploy backend API
-- 5. Wire up frontend authentication



