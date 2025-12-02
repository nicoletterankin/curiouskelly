-- ============================================
-- CURIOUS KELLY - PRODUCTION DATABASE SCHEMA
-- ============================================
-- CLEAN VERSION - Drops existing objects first

-- Drop existing policies first (to avoid conflicts)
DROP POLICY IF EXISTS "Users can view own data" ON public.users;
DROP POLICY IF EXISTS "Users can update own data" ON public.users;
DROP POLICY IF EXISTS "Anyone can view published lessons" ON public.lessons;
DROP POLICY IF EXISTS "Users can view own progress" ON public.user_progress;
DROP POLICY IF EXISTS "Users can insert own progress" ON public.user_progress;
DROP POLICY IF EXISTS "Users can update own progress" ON public.user_progress;
DROP POLICY IF EXISTS "Users can view own affiliate data" ON public.affiliates;
DROP POLICY IF EXISTS "Users can update own affiliate data" ON public.affiliates;
DROP POLICY IF EXISTS "Affiliates can view own referrals" ON public.referrals;
DROP POLICY IF EXISTS "Anyone can submit applications" ON public.affiliate_applications;
DROP POLICY IF EXISTS "Anyone can submit inquiries" ON public.enterprise_inquiries;
DROP POLICY IF EXISTS "Anyone can subscribe" ON public.newsletter_subscribers;
DROP POLICY IF EXISTS "Users can log own events" ON public.analytics_events;

-- Drop existing triggers
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
DROP TRIGGER IF EXISTS update_users_updated_at ON public.users;
DROP TRIGGER IF EXISTS update_lessons_updated_at ON public.lessons;
DROP TRIGGER IF EXISTS update_user_progress_updated_at ON public.user_progress;
DROP TRIGGER IF EXISTS update_affiliates_updated_at ON public.affiliates;
DROP TRIGGER IF EXISTS update_referrals_updated_at ON public.referrals;
DROP TRIGGER IF EXISTS update_enterprise_inquiries_updated_at ON public.enterprise_inquiries;
DROP TRIGGER IF EXISTS calculate_affiliate_tier ON public.affiliates;
DROP TRIGGER IF EXISTS maintain_user_streak ON public.user_progress;

-- Drop existing functions
DROP FUNCTION IF EXISTS public.handle_new_user();
DROP FUNCTION IF EXISTS update_updated_at_column();
DROP FUNCTION IF EXISTS update_affiliate_tier();
DROP FUNCTION IF EXISTS update_user_streak();

-- Drop existing tables (cascade to remove dependencies)
DROP TABLE IF EXISTS public.analytics_events CASCADE;
DROP TABLE IF EXISTS public.newsletter_subscribers CASCADE;
DROP TABLE IF EXISTS public.enterprise_inquiries CASCADE;
DROP TABLE IF EXISTS public.affiliate_applications CASCADE;
DROP TABLE IF EXISTS public.referrals CASCADE;
DROP TABLE IF EXISTS public.affiliates CASCADE;
DROP TABLE IF EXISTS public.user_progress CASCADE;
DROP TABLE IF EXISTS public.lessons CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;

-- ============================================
-- NOW CREATE EVERYTHING FRESH
-- ============================================

-- 1. USERS TABLE
CREATE TABLE public.users (
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

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own data" ON public.users
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own data" ON public.users
  FOR UPDATE USING (auth.uid() = id);

-- 2. LESSONS TABLE
CREATE TABLE public.lessons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  day_number INTEGER UNIQUE NOT NULL,
  title TEXT NOT NULL,
  subtitle TEXT,
  content JSONB NOT NULL,
  audio_url TEXT,
  duration_seconds INTEGER,
  difficulty TEXT CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
  tags TEXT[],
  is_published BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.lessons ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view published lessons" ON public.lessons
  FOR SELECT USING (is_published = true);

CREATE INDEX idx_lessons_day_number ON public.lessons(day_number);
CREATE INDEX idx_lessons_published ON public.lessons(is_published) WHERE is_published = true;

-- 3. USER PROGRESS TABLE
CREATE TABLE public.user_progress (
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

ALTER TABLE public.user_progress ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own progress" ON public.user_progress
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own progress" ON public.user_progress
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own progress" ON public.user_progress
  FOR UPDATE USING (auth.uid() = user_id);

CREATE INDEX idx_user_progress_user_id ON public.user_progress(user_id);
CREATE INDEX idx_user_progress_lesson_id ON public.user_progress(lesson_id);

-- 4. AFFILIATES TABLE
CREATE TABLE public.affiliates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
  referral_code TEXT UNIQUE NOT NULL,
  tier TEXT DEFAULT 'scholar' CHECK (tier IN ('scholar', 'fellow', 'ambassador', 'founding')),
  commission_rate DECIMAL(5,2) DEFAULT 20.00,
  is_founding_100 BOOLEAN DEFAULT false,
  total_referrals INTEGER DEFAULT 0,
  active_referrals INTEGER DEFAULT 0,
  lifetime_earnings DECIMAL(10,2) DEFAULT 0.00,
  status TEXT DEFAULT 'active' CHECK (status IN ('pending', 'active', 'suspended', 'terminated')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.affiliates ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own affiliate data" ON public.affiliates
  FOR SELECT USING (auth.uid() = user_id);

CREATE INDEX idx_affiliates_referral_code ON public.affiliates(referral_code);

-- 5. REFERRALS TABLE
CREATE TABLE public.referrals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  affiliate_id UUID REFERENCES public.affiliates(id) ON DELETE CASCADE NOT NULL,
  referred_user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
  referral_code TEXT NOT NULL,
  subscription_value DECIMAL(10,2),
  commission_earned DECIMAL(10,2),
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'cancelled', 'paid')),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.referrals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Affiliates can view own referrals" ON public.referrals
  FOR SELECT USING (
    affiliate_id IN (SELECT id FROM public.affiliates WHERE user_id = auth.uid())
  );

-- 6. AFFILIATE APPLICATIONS TABLE
CREATE TABLE public.affiliate_applications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  platform TEXT NOT NULL,
  url TEXT,
  audience TEXT,
  focus TEXT,
  why TEXT,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.affiliate_applications ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can submit applications" ON public.affiliate_applications
  FOR INSERT WITH CHECK (true);

-- 7. ENTERPRISE INQUIRIES TABLE
CREATE TABLE public.enterprise_inquiries (
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
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.enterprise_inquiries ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can submit inquiries" ON public.enterprise_inquiries
  FOR INSERT WITH CHECK (true);

-- 8. NEWSLETTER SUBSCRIBERS TABLE
CREATE TABLE public.newsletter_subscribers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  source TEXT DEFAULT 'website',
  status TEXT DEFAULT 'subscribed' CHECK (status IN ('subscribed', 'unsubscribed')),
  subscribed_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.newsletter_subscribers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can subscribe" ON public.newsletter_subscribers
  FOR INSERT WITH CHECK (true);

-- 9. ANALYTICS EVENTS TABLE
CREATE TABLE public.analytics_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
  event_type TEXT NOT NULL,
  event_data JSONB,
  session_id TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.analytics_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can log own events" ON public.analytics_events
  FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE INDEX idx_analytics_user_id ON public.analytics_events(user_id);
CREATE INDEX idx_analytics_event_type ON public.analytics_events(event_type);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to create user record on signup
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

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_lessons_updated_at BEFORE UPDATE ON public.lessons
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_progress_updated_at BEFORE UPDATE ON public.user_progress
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_affiliates_updated_at BEFORE UPDATE ON public.affiliates
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- SEED DATA - Sample Lessons
-- ============================================

INSERT INTO public.lessons (day_number, title, subtitle, content, duration_seconds, difficulty, tags, is_published) VALUES
(1, 'The Sun', 'Our Star and Source of Life', '{"phases": [{"type": "welcome", "content": "Welcome to Day 1! Today we explore our amazing Sun."}]}', 600, 'beginner', ARRAY['science', 'astronomy'], true),
(2, 'Habit Stacking', 'Building Better Daily Routines', '{"phases": [{"type": "welcome", "content": "Welcome to Day 2! Learn to build powerful habits."}]}', 480, 'beginner', ARRAY['life-skills', 'productivity'], true),
(3, 'Planet Earth', 'Our Home in the Solar System', '{"phases": [{"type": "welcome", "content": "Welcome to Day 3! Discover our beautiful planet."}]}', 540, 'beginner', ARRAY['science', 'earth'], true),
(4, 'Simple Machines', 'Levers, Pulleys, and Wheels', '{"phases": [{"type": "welcome", "content": "Welcome to Day 4! Physics in everyday life."}]}', 510, 'beginner', ARRAY['science', 'physics'], true),
(5, 'Emotional Intelligence', 'Understanding Your Feelings', '{"phases": [{"type": "welcome", "content": "Welcome to Day 5! Master your emotions."}]}', 600, 'intermediate', ARRAY['life-skills', 'psychology'], true);

-- ============================================
-- COMPLETE!
-- ============================================
















