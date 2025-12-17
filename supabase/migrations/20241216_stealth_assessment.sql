-- ============================================
-- STEALTH ASSESSMENT SYSTEM
-- Migration: December 16, 2025
-- Purpose: Enable invisible learning observation and insights
-- ============================================

-- ============================================
-- 1. LEARNER OBSERVATIONS TABLE
-- Captures behavioral signals during lessons
-- ============================================

CREATE TABLE IF NOT EXISTS public.learner_observations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
  lesson_id UUID REFERENCES public.lessons(id) ON DELETE SET NULL,
  day_number INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  
  -- Response quality metrics
  first_try_correct BOOLEAN,
  option_quality_sequence TEXT[], -- ['best', 'good', 'redirect', 'best', 'good']
  hints_used INTEGER DEFAULT 0,
  redirects_count INTEGER DEFAULT 0,
  redirect_recoveries INTEGER DEFAULT 0, -- good choice after redirect
  
  -- Timing metrics (all in milliseconds)
  phase_durations JSONB, -- {"welcome": 15000, "q1": 32000, "q2": 28000, ...}
  choice_timings INTEGER[], -- [3200, 4100, 2800, 5500] per question phase
  avg_choice_time INTEGER,
  rushed_choices_count INTEGER DEFAULT 0, -- choices < 2000ms
  deliberate_choices_count INTEGER DEFAULT 0, -- choices 5000-25000ms
  
  -- Engagement metrics
  audio_replays INTEGER DEFAULT 0,
  video_replays INTEGER DEFAULT 0,
  pauses_count INTEGER DEFAULT 0,
  total_session_duration INTEGER, -- total ms from start to finish
  completed BOOLEAN DEFAULT false,
  abandoned_at_phase TEXT, -- NULL if completed, otherwise phase name
  
  -- Context for analysis
  archetype TEXT,
  age_setting TEXT,
  language TEXT DEFAULT 'en',
  device_type TEXT CHECK (device_type IN ('mobile', 'tablet', 'desktop', 'unknown')),
  
  -- Timestamps
  started_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  
  UNIQUE(user_id, session_id)
);

-- Enable RLS
ALTER TABLE public.learner_observations ENABLE ROW LEVEL SECURITY;

-- Policies: Users can only access their own observations
CREATE POLICY "Users can view own observations" ON public.learner_observations
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own observations" ON public.learner_observations
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own observations" ON public.learner_observations
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own observations" ON public.learner_observations
  FOR DELETE USING (auth.uid() = user_id);

-- Indexes for efficient queries
CREATE INDEX idx_observations_user_id ON public.learner_observations(user_id);
CREATE INDEX idx_observations_day ON public.learner_observations(day_number);
CREATE INDEX idx_observations_completed ON public.learner_observations(completed) WHERE completed = true;
CREATE INDEX idx_observations_started_at ON public.learner_observations(started_at DESC);

-- ============================================
-- 2. LEARNER INSIGHTS TABLE
-- Aggregated, computed insights for user display
-- ============================================

CREATE TABLE IF NOT EXISTS public.learner_insights (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
  
  -- Engagement Profile
  engagement_style TEXT CHECK (engagement_style IN (
    'explorer',     -- Takes time, explores options, curious
    'deliberator',  -- Careful consideration before choosing
    'speedrunner',  -- Quick, efficient completion
    'reflector',    -- Pauses, replays, thinks deeply
    'undetermined'  -- Not enough data yet
  )) DEFAULT 'undetermined',
  
  -- Learning Trajectory
  learning_velocity TEXT CHECK (learning_velocity IN (
    'accelerating', -- Getting better faster
    'steady',       -- Consistent performance
    'warming_up',   -- Still finding rhythm
    'undetermined'
  )) DEFAULT 'undetermined',
  
  -- Proficiency Metrics (0-100 scale)
  overall_mastery INTEGER DEFAULT 0 CHECK (overall_mastery >= 0 AND overall_mastery <= 100),
  subject_proficiencies JSONB DEFAULT '{}', -- {"science": 72, "life-skills": 85, "history": 45}
  difficulty_comfort TEXT CHECK (difficulty_comfort IN (
    'beginner', 'intermediate', 'advanced', 'undetermined'
  )) DEFAULT 'undetermined',
  
  -- Session Patterns
  avg_session_duration INTEGER, -- average ms per completed lesson
  optimal_session_length INTEGER, -- recommended based on engagement patterns
  best_time_of_day TEXT CHECK (best_time_of_day IN (
    'early_morning', 'morning', 'afternoon', 'evening', 'night', 'undetermined'
  )) DEFAULT 'undetermined',
  preferred_session_count INTEGER DEFAULT 1, -- lessons per sitting
  
  -- Streak & Habit Metrics
  streak_reliability DECIMAL(3,2) DEFAULT 0.00 CHECK (streak_reliability >= 0 AND streak_reliability <= 1),
  avg_days_between_lessons DECIMAL(5,2),
  longest_streak INTEGER DEFAULT 0,
  
  -- Strengths & Growth (stored as text arrays for flexibility)
  strengths TEXT[] DEFAULT '{}', -- ['quick thinker', 'persistent', 'curious']
  growth_areas TEXT[] DEFAULT '{}', -- ['taking time to reflect', 'exploring options']
  
  -- Archetype Affinity
  preferred_archetype TEXT,
  archetype_scores JSONB DEFAULT '{}', -- {"explorer": 0.8, "scientist": 0.6, "rebel": 0.4}
  
  -- Confidence & Data Quality
  lessons_analyzed INTEGER DEFAULT 0,
  confidence_level DECIMAL(3,2) DEFAULT 0.00 CHECK (confidence_level >= 0 AND confidence_level <= 1),
  -- Confidence builds: 0.0-0.3 (< 5 lessons), 0.3-0.6 (5-15), 0.6-0.9 (15-30), 0.9+ (30+)
  
  -- Timestamps
  last_analyzed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.learner_insights ENABLE ROW LEVEL SECURITY;

-- Policies: Users can only view their own insights
CREATE POLICY "Users can view own insights" ON public.learner_insights
  FOR SELECT USING (auth.uid() = user_id);

-- System can upsert insights (via service role)
-- Note: Direct user insert not allowed - computed by system

-- Index
CREATE INDEX idx_insights_user_id ON public.learner_insights(user_id);

-- Trigger for updated_at
CREATE TRIGGER update_learner_insights_updated_at 
  BEFORE UPDATE ON public.learner_insights
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 3. EXTEND USER_PROGRESS TABLE
-- Add observation summary to existing progress
-- ============================================

ALTER TABLE public.user_progress 
  ADD COLUMN IF NOT EXISTS observation_summary JSONB DEFAULT '{}';
-- Contains: {first_try_correct, hints_used, session_duration, avg_choice_time}

-- ============================================
-- 4. COMPUTE INSIGHTS FUNCTION
-- Analyzes observations and updates insights
-- ============================================

CREATE OR REPLACE FUNCTION compute_learner_insights(target_user_id UUID)
RETURNS JSONB AS $$
DECLARE
  obs_count INTEGER;
  completed_count INTEGER;
  avg_accuracy DECIMAL;
  avg_session INTEGER;
  avg_choice INTEGER;
  rush_ratio DECIMAL;
  deliberate_ratio DECIMAL;
  engagement_type TEXT;
  velocity_type TEXT;
  confidence DECIMAL;
  strengths_arr TEXT[];
  growth_arr TEXT[];
  result JSONB;
BEGIN
  -- Count observations
  SELECT COUNT(*), COUNT(*) FILTER (WHERE completed = true)
  INTO obs_count, completed_count
  FROM learner_observations
  WHERE user_id = target_user_id;
  
  -- Need at least 3 completed lessons for meaningful insights
  IF completed_count < 3 THEN
    RETURN jsonb_build_object(
      'status', 'insufficient_data',
      'lessons_needed', 3 - completed_count,
      'message', 'Complete a few more lessons for Kelly to understand your style'
    );
  END IF;
  
  -- Calculate aggregate metrics from completed lessons
  SELECT 
    AVG(CASE WHEN first_try_correct THEN 1.0 ELSE 0.0 END),
    AVG(total_session_duration),
    AVG(avg_choice_time),
    AVG(COALESCE(rushed_choices_count, 0)::DECIMAL / 
        GREATEST(COALESCE(array_length(choice_timings, 1), 1), 1)),
    AVG(COALESCE(deliberate_choices_count, 0)::DECIMAL / 
        GREATEST(COALESCE(array_length(choice_timings, 1), 1), 1))
  INTO avg_accuracy, avg_session, avg_choice, rush_ratio, deliberate_ratio
  FROM learner_observations
  WHERE user_id = target_user_id AND completed = true;
  
  -- Determine engagement style
  IF rush_ratio > 0.5 THEN
    engagement_type := 'speedrunner';
  ELSIF deliberate_ratio > 0.5 THEN
    engagement_type := 'deliberator';
  ELSIF avg_accuracy > 0.7 AND deliberate_ratio > 0.3 THEN
    engagement_type := 'explorer';
  ELSE
    engagement_type := 'reflector';
  END IF;
  
  -- Calculate learning velocity (compare recent half vs. first half)
  -- Simplified: more sophisticated would use regression
  velocity_type := 'steady';
  
  -- Calculate confidence level based on data quantity
  confidence := LEAST(completed_count::DECIMAL / 20.0, 1.0);
  
  -- Determine strengths based on metrics
  strengths_arr := ARRAY[]::TEXT[];
  IF avg_accuracy > 0.75 THEN
    strengths_arr := array_append(strengths_arr, 'First-try thinker');
  END IF;
  IF deliberate_ratio > 0.4 THEN
    strengths_arr := array_append(strengths_arr, 'Thoughtful explorer');
  END IF;
  -- Add redirect recovery strength
  IF (SELECT AVG(redirect_recoveries::DECIMAL / GREATEST(redirects_count, 1)) 
      FROM learner_observations 
      WHERE user_id = target_user_id AND completed = true AND redirects_count > 0) > 0.6 THEN
    strengths_arr := array_append(strengths_arr, 'Great at bouncing back');
  END IF;
  IF completed_count >= 7 THEN
    strengths_arr := array_append(strengths_arr, 'Committed learner');
  END IF;
  
  -- Determine growth areas (positive framing)
  growth_arr := ARRAY[]::TEXT[];
  IF rush_ratio > 0.4 THEN
    growth_arr := array_append(growth_arr, 'Take your time with tricky questions');
  END IF;
  IF (SELECT AVG(hints_used) FROM learner_observations 
      WHERE user_id = target_user_id AND completed = true) > 2 THEN
    growth_arr := array_append(growth_arr, 'Trust your first instinct more');
  END IF;
  
  -- Upsert the insights
  INSERT INTO learner_insights (
    user_id,
    engagement_style,
    learning_velocity,
    overall_mastery,
    avg_session_duration,
    strengths,
    growth_areas,
    lessons_analyzed,
    confidence_level,
    last_analyzed_at
  ) VALUES (
    target_user_id,
    engagement_type,
    velocity_type,
    LEAST(ROUND(COALESCE(avg_accuracy, 0) * 100)::INTEGER, 100),
    ROUND(COALESCE(avg_session, 0))::INTEGER,
    strengths_arr,
    growth_arr,
    completed_count,
    confidence,
    NOW()
  )
  ON CONFLICT (user_id) DO UPDATE SET
    engagement_style = EXCLUDED.engagement_style,
    learning_velocity = EXCLUDED.learning_velocity,
    overall_mastery = EXCLUDED.overall_mastery,
    avg_session_duration = EXCLUDED.avg_session_duration,
    strengths = EXCLUDED.strengths,
    growth_areas = EXCLUDED.growth_areas,
    lessons_analyzed = EXCLUDED.lessons_analyzed,
    confidence_level = EXCLUDED.confidence_level,
    last_analyzed_at = EXCLUDED.last_analyzed_at;
  
  -- Return summary
  result := jsonb_build_object(
    'status', 'success',
    'engagement_style', engagement_type,
    'mastery', LEAST(ROUND(COALESCE(avg_accuracy, 0) * 100), 100),
    'lessons_analyzed', completed_count,
    'confidence', confidence,
    'strengths_count', array_length(strengths_arr, 1),
    'growth_areas_count', array_length(growth_arr, 1)
  );
  
  RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- 5. TRIGGER TO AUTO-COMPUTE ON COMPLETION
-- Updates insights when observation is marked complete
-- ============================================

CREATE OR REPLACE FUNCTION trigger_compute_insights_on_completion()
RETURNS TRIGGER AS $$
BEGIN
  -- Only compute when a lesson is marked as completed
  IF NEW.completed = true AND (OLD.completed IS NULL OR OLD.completed = false) THEN
    -- Run insight computation asynchronously (non-blocking)
    PERFORM compute_learner_insights(NEW.user_id);
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_observation_complete
  AFTER INSERT OR UPDATE ON public.learner_observations
  FOR EACH ROW EXECUTE FUNCTION trigger_compute_insights_on_completion();

-- ============================================
-- 6. HELPER VIEW FOR USER JOURNEY DISPLAY
-- Simplified view for frontend consumption
-- ============================================

CREATE OR REPLACE VIEW public.user_learning_journey AS
SELECT 
  i.user_id,
  i.engagement_style,
  CASE i.engagement_style
    WHEN 'explorer' THEN '🧭 Explorer'
    WHEN 'deliberator' THEN '🎯 Deliberator'
    WHEN 'speedrunner' THEN '⚡ Quick Thinker'
    WHEN 'reflector' THEN '🌙 Deep Reflector'
    ELSE '✨ Discovering...'
  END AS style_display,
  CASE i.engagement_style
    WHEN 'explorer' THEN 'You love understanding deeply before moving on'
    WHEN 'deliberator' THEN 'You think carefully before choosing'
    WHEN 'speedrunner' THEN 'You''re efficient and trust your instincts'
    WHEN 'reflector' THEN 'You take time to really absorb each lesson'
    ELSE 'Kelly is still learning how you learn best'
  END AS style_description,
  i.overall_mastery,
  i.learning_velocity,
  i.strengths,
  i.growth_areas,
  i.lessons_analyzed,
  i.confidence_level,
  u.streak_days,
  u.current_day,
  (SELECT COUNT(*) FROM user_progress up WHERE up.user_id = i.user_id AND up.completed = true) as completed_lessons
FROM learner_insights i
JOIN users u ON i.user_id = u.id;

-- ============================================
-- 7. DATA EXPORT FUNCTION
-- For user privacy: export all their learning data
-- ============================================

CREATE OR REPLACE FUNCTION export_my_learning_data(target_user_id UUID)
RETURNS JSONB AS $$
DECLARE
  result JSONB;
BEGIN
  -- Verify the user is requesting their own data
  IF auth.uid() != target_user_id THEN
    RETURN jsonb_build_object('error', 'Unauthorized');
  END IF;
  
  SELECT jsonb_build_object(
    'exported_at', NOW(),
    'user_id', target_user_id,
    'insights', (SELECT row_to_json(i.*) FROM learner_insights i WHERE i.user_id = target_user_id),
    'observations', (SELECT json_agg(o.*) FROM learner_observations o WHERE o.user_id = target_user_id),
    'progress', (SELECT json_agg(p.*) FROM user_progress p WHERE p.user_id = target_user_id)
  ) INTO result;
  
  RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- 8. DATA DELETION FUNCTION
-- For user privacy: delete all their learning observation data
-- ============================================

CREATE OR REPLACE FUNCTION delete_my_learning_history(target_user_id UUID)
RETURNS JSONB AS $$
DECLARE
  obs_deleted INTEGER;
  insights_deleted INTEGER;
BEGIN
  -- Verify the user is requesting deletion of their own data
  IF auth.uid() != target_user_id THEN
    RETURN jsonb_build_object('error', 'Unauthorized');
  END IF;
  
  -- Delete observations
  DELETE FROM learner_observations WHERE user_id = target_user_id;
  GET DIAGNOSTICS obs_deleted = ROW_COUNT;
  
  -- Delete insights
  DELETE FROM learner_insights WHERE user_id = target_user_id;
  GET DIAGNOSTICS insights_deleted = ROW_COUNT;
  
  RETURN jsonb_build_object(
    'status', 'success',
    'deleted_at', NOW(),
    'observations_deleted', obs_deleted,
    'insights_deleted', insights_deleted
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- SCHEMA COMPLETE
-- ============================================
-- Run this migration in Supabase SQL Editor
-- Then integrate the JavaScript observer into learn.html
