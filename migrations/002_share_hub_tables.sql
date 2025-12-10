-- ============================================
-- MIGRATION: Share Hub Tables
-- ============================================
-- Run this in Supabase SQL Editor to add:
-- - learning_groups (for group learning feature)
-- - group_members (group membership)
-- - daily_lesson_stats (for Global Perspectives)
-- 
-- Date: December 2024
-- ============================================

-- ============================================
-- 1. LEARNING GROUPS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.learning_groups (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  emoji TEXT DEFAULT '👨‍👩‍👧‍👦',
  invite_code TEXT UNIQUE NOT NULL,
  created_by UUID REFERENCES public.users(id) ON DELETE SET NULL,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.learning_groups ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Members can view their groups" ON public.learning_groups
  FOR SELECT USING (
    id IN (SELECT group_id FROM public.group_members WHERE user_id = auth.uid())
    OR created_by = auth.uid()
  );

CREATE POLICY "Users can create groups" ON public.learning_groups
  FOR INSERT WITH CHECK (auth.uid() = created_by);

CREATE POLICY "Owners can update groups" ON public.learning_groups
  FOR UPDATE USING (auth.uid() = created_by);

-- Index
CREATE INDEX IF NOT EXISTS idx_learning_groups_invite_code ON public.learning_groups(invite_code);
CREATE INDEX IF NOT EXISTS idx_learning_groups_created_by ON public.learning_groups(created_by);

-- ============================================
-- 2. GROUP MEMBERS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.group_members (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  group_id UUID REFERENCES public.learning_groups(id) ON DELETE CASCADE NOT NULL,
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
  role TEXT DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member')),
  joined_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(group_id, user_id)
);

-- Enable RLS
ALTER TABLE public.group_members ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Members can view group memberships" ON public.group_members
  FOR SELECT USING (
    auth.uid() = user_id 
    OR group_id IN (SELECT group_id FROM public.group_members WHERE user_id = auth.uid())
  );

CREATE POLICY "Users can join groups" ON public.group_members
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can leave groups" ON public.group_members
  FOR DELETE USING (auth.uid() = user_id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_group_members_group_id ON public.group_members(group_id);
CREATE INDEX IF NOT EXISTS idx_group_members_user_id ON public.group_members(user_id);

-- ============================================
-- 3. DAILY LESSON STATS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.daily_lesson_stats (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  stat_date DATE UNIQUE NOT NULL DEFAULT CURRENT_DATE,
  learners_count INTEGER DEFAULT 0,
  lessons_completed INTEGER DEFAULT 0,
  countries_count INTEGER DEFAULT 0,
  countries_list TEXT[],
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS - public read for stats
ALTER TABLE public.daily_lesson_stats ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view daily stats" ON public.daily_lesson_stats
  FOR SELECT USING (true);

-- Index
CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON public.daily_lesson_stats(stat_date DESC);

-- ============================================
-- 4. TRIGGERS
-- ============================================

-- Trigger for updated_at on learning_groups
DROP TRIGGER IF EXISTS update_learning_groups_updated_at ON public.learning_groups;
CREATE TRIGGER update_learning_groups_updated_at 
  BEFORE UPDATE ON public.learning_groups
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for updated_at on daily_lesson_stats
DROP TRIGGER IF EXISTS update_daily_stats_updated_at ON public.daily_lesson_stats;
CREATE TRIGGER update_daily_stats_updated_at 
  BEFORE UPDATE ON public.daily_lesson_stats
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 5. FUNCTION: Update daily stats on lesson completion
-- ============================================

CREATE OR REPLACE FUNCTION update_daily_lesson_stats()
RETURNS TRIGGER AS $$
BEGIN
  -- When a lesson is completed, update today's stats
  IF NEW.completed AND (OLD IS NULL OR NOT OLD.completed) THEN
    INSERT INTO public.daily_lesson_stats (stat_date, learners_count, lessons_completed)
    VALUES (CURRENT_DATE, 1, 1)
    ON CONFLICT (stat_date) 
    DO UPDATE SET 
      learners_count = daily_lesson_stats.learners_count + 1,
      lessons_completed = daily_lesson_stats.lessons_completed + 1,
      updated_at = NOW();
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger for lesson completion stats
DROP TRIGGER IF EXISTS on_lesson_completed ON public.user_progress;
CREATE TRIGGER on_lesson_completed
  AFTER INSERT OR UPDATE ON public.user_progress
  FOR EACH ROW EXECUTE FUNCTION update_daily_lesson_stats();

-- ============================================
-- MIGRATION COMPLETE
-- ============================================
-- Verify by running:
-- SELECT * FROM public.learning_groups LIMIT 1;
-- SELECT * FROM public.group_members LIMIT 1;
-- SELECT * FROM public.daily_lesson_stats LIMIT 1;




