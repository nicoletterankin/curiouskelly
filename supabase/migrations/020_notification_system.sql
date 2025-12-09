-- =============================================================================
-- Curious Kelly Notification System
-- Migration: 020_notification_system
-- Date: December 2025
-- =============================================================================
-- This migration creates the infrastructure for Kelly's daily notification system
-- across iOS, Android, Web, Desktop, and other platforms.
--
-- IMPORTANT: Date Display Convention
-- - Internal: day_number (1-365) - used in database, APIs, URLs
-- - User-Facing: Real calendar dates - "December 17" not "Day 1"
-- - Year 1 Content: December 17, 2025 → December 16, 2026
-- - Day 1 = December 17 (launch day)
-- =============================================================================

-- 1. Notification Preferences
-- Where users control their notification experience
CREATE TABLE IF NOT EXISTS public.notification_preferences (
  user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  
  -- Timing Preferences
  preferred_time TIME DEFAULT '09:00',
  timezone TEXT DEFAULT 'America/New_York',
  auto_timing BOOLEAN DEFAULT true,           -- Let Kelly learn optimal time
  learned_optimal_time TIME,                   -- Kelly's learned best time for this user
  last_timing_analysis_at TIMESTAMPTZ,
  
  -- Channel Preferences
  push_enabled BOOLEAN DEFAULT true,
  email_enabled BOOLEAN DEFAULT true,
  web_push_enabled BOOLEAN DEFAULT true,
  
  -- Notification Type Preferences
  daily_reminder BOOLEAN DEFAULT true,
  streak_alerts BOOLEAN DEFAULT true,
  milestone_celebrations BOOLEAN DEFAULT true,
  gentle_returns BOOLEAN DEFAULT true,
  family_updates BOOLEAN DEFAULT false,
  collective_milestones BOOLEAN DEFAULT false,
  
  -- Quiet Hours (local time)
  quiet_start TIME DEFAULT '22:00',
  quiet_end TIME DEFAULT '07:00',
  weekend_quiet BOOLEAN DEFAULT false,         -- Quiet on weekends?
  
  -- Streak Protection
  streak_shields_available INTEGER DEFAULT 0,
  streak_shields_used INTEGER DEFAULT 0,
  last_shield_earned_at TIMESTAMPTZ,
  
  -- Metadata
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Push Tokens
-- Store device tokens for all platforms
CREATE TABLE IF NOT EXISTS public.push_tokens (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  
  -- Token Data
  device_token TEXT NOT NULL,
  platform TEXT NOT NULL CHECK (platform IN ('ios', 'android', 'web', 'macos', 'windows', 'linux')),
  
  -- Device Info
  device_name TEXT,                            -- "John's iPhone"
  device_model TEXT,                           -- "iPhone 15 Pro"
  app_version TEXT,                            -- "1.0.0"
  os_version TEXT,                             -- "iOS 17.2"
  
  -- Status
  is_active BOOLEAN DEFAULT true,
  last_active_at TIMESTAMPTZ DEFAULT now(),
  last_notification_at TIMESTAMPTZ,
  failed_count INTEGER DEFAULT 0,              -- Track delivery failures
  
  -- Metadata
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  
  UNIQUE(user_id, device_token)
);

-- 3. Notification Log
-- Track all sent notifications for analytics
CREATE TABLE IF NOT EXISTS public.notification_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  
  -- Notification Content
  notification_type TEXT NOT NULL,             -- 'daily_reminder', 'streak_alert', etc.
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  
  -- A/B Testing
  copy_variant TEXT,                           -- 'A', 'B', etc.
  ab_test_id UUID,
  
  -- Timing
  scheduled_for TIMESTAMPTZ,
  sent_at TIMESTAMPTZ DEFAULT now(),
  
  -- Delivery Tracking
  platform TEXT,                               -- Which platform received it
  device_token_id UUID REFERENCES push_tokens(id) ON DELETE SET NULL,
  
  -- Engagement Tracking
  delivered_at TIMESTAMPTZ,                    -- APNs/FCM confirmed delivery
  opened_at TIMESTAMPTZ,                       -- User tapped notification
  converted_at TIMESTAMPTZ,                    -- User started lesson
  dismissed_at TIMESTAMPTZ,                    -- User dismissed without opening
  
  -- Context
  lesson_day INTEGER,
  streak_count INTEGER,
  metadata JSONB DEFAULT '{}'::jsonb,
  
  -- Error Tracking
  error_message TEXT,
  retry_count INTEGER DEFAULT 0
);

-- 4. Notification Copy Library
-- Store all Kelly's notification copy variants
CREATE TABLE IF NOT EXISTS public.notification_copy (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Classification
  notification_type TEXT NOT NULL,             -- 'daily_reminder', 'streak_save', etc.
  variant_code TEXT NOT NULL,                  -- 'A', 'B', 'C', etc.
  
  -- Content (with {placeholders})
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  
  -- Conditions for use
  min_streak INTEGER DEFAULT 0,
  max_streak INTEGER,
  day_of_week INTEGER[],                       -- 0=Sunday, 6=Saturday
  special_occasion TEXT,                       -- 'birthday', 'first_lesson', etc.
  
  -- Performance
  send_count INTEGER DEFAULT 0,
  open_count INTEGER DEFAULT 0,
  conversion_count INTEGER DEFAULT 0,
  
  -- Status
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  
  UNIQUE(notification_type, variant_code)
);

-- 5. A/B Test Configuration
CREATE TABLE IF NOT EXISTS public.notification_ab_tests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Test Setup
  test_name TEXT NOT NULL UNIQUE,
  notification_type TEXT NOT NULL,
  
  -- Variants
  variant_a_id UUID REFERENCES notification_copy(id),
  variant_b_id UUID REFERENCES notification_copy(id),
  
  -- Split
  traffic_percentage NUMERIC DEFAULT 50,       -- % to variant A (rest to B)
  
  -- Duration
  start_date TIMESTAMPTZ DEFAULT now(),
  end_date TIMESTAMPTZ,
  
  -- Results
  winner TEXT,                                 -- 'A' or 'B' or NULL
  statistical_significance NUMERIC,
  results_summary JSONB DEFAULT '{}'::jsonb,
  
  -- Status
  status TEXT DEFAULT 'active' CHECK (status IN ('draft', 'active', 'paused', 'completed')),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- 6. Notification Queue (for scheduled sends)
CREATE TABLE IF NOT EXISTS public.notification_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  notification_type TEXT NOT NULL,
  
  -- Content (copied from copy library with personalization applied)
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  
  -- Scheduling
  scheduled_for TIMESTAMPTZ NOT NULL,
  priority INTEGER DEFAULT 5,                  -- 1=highest, 10=lowest
  
  -- Status
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'sent', 'failed', 'cancelled')),
  processed_at TIMESTAMPTZ,
  error_message TEXT,
  retry_count INTEGER DEFAULT 0,
  
  -- Metadata
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Push tokens
CREATE INDEX IF NOT EXISTS idx_push_tokens_user ON public.push_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_push_tokens_platform ON public.push_tokens(platform);
CREATE INDEX IF NOT EXISTS idx_push_tokens_active ON public.push_tokens(is_active) WHERE is_active = true;

-- Notification log
CREATE INDEX IF NOT EXISTS idx_notification_log_user ON public.notification_log(user_id);
CREATE INDEX IF NOT EXISTS idx_notification_log_type ON public.notification_log(notification_type);
CREATE INDEX IF NOT EXISTS idx_notification_log_sent ON public.notification_log(sent_at);
CREATE INDEX IF NOT EXISTS idx_notification_log_opened ON public.notification_log(opened_at) WHERE opened_at IS NOT NULL;

-- Notification queue
CREATE INDEX IF NOT EXISTS idx_notification_queue_scheduled ON public.notification_queue(scheduled_for);
CREATE INDEX IF NOT EXISTS idx_notification_queue_status ON public.notification_queue(status) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_notification_queue_user ON public.notification_queue(user_id);

-- Notification copy
CREATE INDEX IF NOT EXISTS idx_notification_copy_type ON public.notification_copy(notification_type);
CREATE INDEX IF NOT EXISTS idx_notification_copy_active ON public.notification_copy(is_active) WHERE is_active = true;

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

ALTER TABLE public.notification_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.push_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notification_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notification_copy ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notification_ab_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notification_queue ENABLE ROW LEVEL SECURITY;

-- Users can read/update their own preferences
CREATE POLICY "Users can view own notification preferences" 
  ON public.notification_preferences FOR SELECT 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can update own notification preferences" 
  ON public.notification_preferences FOR UPDATE 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own notification preferences" 
  ON public.notification_preferences FOR INSERT 
  WITH CHECK (auth.uid() = user_id);

-- Users can manage their own push tokens
CREATE POLICY "Users can view own push tokens" 
  ON public.push_tokens FOR SELECT 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own push tokens" 
  ON public.push_tokens FOR INSERT 
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own push tokens" 
  ON public.push_tokens FOR UPDATE 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own push tokens" 
  ON public.push_tokens FOR DELETE 
  USING (auth.uid() = user_id);

-- Users can view their own notification history
CREATE POLICY "Users can view own notification log" 
  ON public.notification_log FOR SELECT 
  USING (auth.uid() = user_id);

-- Notification copy is readable by all authenticated users (for preview)
CREATE POLICY "Authenticated users can view notification copy" 
  ON public.notification_copy FOR SELECT 
  USING (auth.role() = 'authenticated');

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to get user's optimal notification time
CREATE OR REPLACE FUNCTION get_optimal_notification_time(p_user_id UUID)
RETURNS TIME AS $$
DECLARE
  v_learned_time TIME;
  v_explicit_time TIME;
  v_auto_timing BOOLEAN;
BEGIN
  SELECT learned_optimal_time, preferred_time, auto_timing
  INTO v_learned_time, v_explicit_time, v_auto_timing
  FROM notification_preferences
  WHERE user_id = p_user_id;
  
  -- If user has auto-timing on and we've learned their time, use that
  IF v_auto_timing AND v_learned_time IS NOT NULL THEN
    RETURN v_learned_time;
  END IF;
  
  -- Otherwise use their explicit preference or default
  RETURN COALESCE(v_explicit_time, '09:00'::TIME);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to analyze and update optimal notification time
CREATE OR REPLACE FUNCTION analyze_user_notification_time(p_user_id UUID)
RETURNS TIME AS $$
DECLARE
  v_optimal_hour INTEGER;
  v_optimal_time TIME;
BEGIN
  -- Find the hour when user most commonly completes lessons
  SELECT 
    EXTRACT(HOUR FROM completed_at)::INTEGER
  INTO v_optimal_hour
  FROM lesson_history
  WHERE user_id = p_user_id
    AND completed_at > now() - INTERVAL '30 days'
  GROUP BY EXTRACT(HOUR FROM completed_at)
  ORDER BY COUNT(*) DESC
  LIMIT 1;
  
  -- Default to 9am if no data
  IF v_optimal_hour IS NULL THEN
    v_optimal_hour := 9;
  END IF;
  
  -- Convert to time (send notification 30 min before typical completion)
  v_optimal_time := make_time(GREATEST(v_optimal_hour - 1, 0), 30, 0);
  
  -- Update the preference
  UPDATE notification_preferences
  SET learned_optimal_time = v_optimal_time,
      last_timing_analysis_at = now()
  WHERE user_id = p_user_id;
  
  RETURN v_optimal_time;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to award streak shield (called when user hits 30-day streak)
CREATE OR REPLACE FUNCTION award_streak_shield(p_user_id UUID)
RETURNS VOID AS $$
BEGIN
  UPDATE notification_preferences
  SET 
    streak_shields_available = streak_shields_available + 1,
    last_shield_earned_at = now()
  WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to use streak shield
CREATE OR REPLACE FUNCTION use_streak_shield(p_user_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
  v_shields_available INTEGER;
BEGIN
  SELECT streak_shields_available INTO v_shields_available
  FROM notification_preferences
  WHERE user_id = p_user_id;
  
  IF v_shields_available > 0 THEN
    UPDATE notification_preferences
    SET 
      streak_shields_available = streak_shields_available - 1,
      streak_shields_used = streak_shields_used + 1
    WHERE user_id = p_user_id;
    RETURN true;
  END IF;
  
  RETURN false;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- SEED DATA: Kelly's Notification Copy Library
-- =============================================================================

-- Placeholder Reference:
-- {name} - User's display name
-- {lesson_title} - Today's lesson title
-- {lesson_emoji} - Today's lesson emoji
-- {streak_days} - Current streak count
-- {date_formatted} - "December 17" (user-facing date)
-- {date_short} - "Dec 17"
-- {day_of_week} - "Wednesday"
-- {month_name} - "December"
-- {special_occasion} - "Christmas" or empty

INSERT INTO public.notification_copy (notification_type, variant_code, title, body, is_active) VALUES

-- Daily Reminders (use {date_formatted} for dates, NEVER day numbers)
('daily_reminder', 'A', '✨ Your 5 minutes of wonder', 'Today: {lesson_title}. Ready when you are.', true),
('daily_reminder', 'B', '{lesson_emoji} {lesson_title}', '5 minutes. I think you''ll love this one.', true),
('daily_reminder', 'C', 'Good morning, {name}', '{date_formatted}: {lesson_title}. Shall we?', true),
('daily_reminder', 'D', 'Something wonderful today', '{lesson_emoji} {lesson_title} — 5 minutes with Kelly.', true),
('daily_reminder', 'E', 'The world learned something new', 'You haven''t yet. {lesson_title} is waiting.', true),
('daily_reminder', 'F', '{day_of_week} wonder', '{lesson_emoji} {lesson_title} is ready for you.', true),

-- Streak Saves (evening, if no lesson yet)
('streak_save', 'A', 'Keep it going?', 'Day {streak_days} is waiting. Just 5 minutes.', true),
('streak_save', 'B', 'Don''t let this streak slip 🔥', '{streak_days} days strong. Today''s lesson won''t take long.', true),
('streak_save', 'C', 'Your streak needs you', '{streak_days} days of curiosity. Worth protecting, don''t you think?', true),
('streak_save', 'D', 'A legendary streak at risk', '{streak_days} days. Most people never get here. You did. Keep going?', true),

-- Streak Celebrations
('streak_celebration', '7', '🌟 One week of wonder!', '7 days of curiosity. That''s no small thing.', true),
('streak_celebration', '14', '✨ Two weeks strong!', '14 days in. You''re building something beautiful.', true),
('streak_celebration', '30', '🔥 A month together!', '30 days of daily curiosity. Habits are forming. Keep going.', true),
('streak_celebration', '60', '💪 60 days!', 'Most people quit at 3. You''re extraordinary.', true),
('streak_celebration', '100', '💯 One hundred days!', 'I don''t have words. Just gratitude. You''re a true learner.', true),
('streak_celebration', '200', '⭐ 200 days!', 'Half a year of daily curiosity. You inspire me.', true),
('streak_celebration', '365', '🏆 A FULL YEAR!', 'Every. Single. Day. You''re a legendary learner.', true),

-- Gentle Returns (after absence)
('gentle_return', 'day_3', 'Miss you a little', 'No lesson reminder. Just wanted to say hi. Hope you''re okay. 💙', true),
('gentle_return', 'day_7', 'Your spot is still here', 'A week without learning together. Whenever you''re ready.', true),
('gentle_return', 'day_14', 'Still curious?', 'Two weeks is a long time. I hope you''re okay. I''m here.', true),
('gentle_return', 'day_30', 'I''m still here', 'A month apart. Your streak is gone, but you''re not. Come back?', true),

-- Birthday
('birthday', 'A', '🎂 Happy Birthday, {name}!', 'Your birthday lesson is waiting. Same lesson, different you. That''s the magic.', true),

-- Year Complete
('year_complete', 'A', '🏆 You did it. 365 lessons.', '{name}, you learned something new every single day. That''s extraordinary.', true),

-- Surprise Delights (rare, random)
('surprise', 'A', 'Just checking in', 'No lesson reminder. Just wanted to say: you''re doing great. 💙', true),
('surprise', 'B', 'Look around today', 'Notice something you''ve never noticed before. That''s curiosity.', true),
('surprise', 'C', 'Full moon tonight 🌕', 'Remember what we learned about tides?', true),
('surprise', 'D', 'You''ve been learning for {total_days} days', 'I''m proud to be your teacher.', true)

ON CONFLICT (notification_type, variant_code) DO UPDATE SET
  title = EXCLUDED.title,
  body = EXCLUDED.body,
  is_active = EXCLUDED.is_active,
  updated_at = now();

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE public.notification_preferences IS 'User notification preferences and timing settings for the Kelly notification system';
COMMENT ON TABLE public.push_tokens IS 'Device push tokens for iOS, Android, Web, and Desktop platforms';
COMMENT ON TABLE public.notification_log IS 'Complete log of all sent notifications for analytics and debugging';
COMMENT ON TABLE public.notification_copy IS 'Kelly''s notification copy library with A/B variants';
COMMENT ON TABLE public.notification_ab_tests IS 'A/B test configuration and results for notification optimization';
COMMENT ON TABLE public.notification_queue IS 'Scheduled notifications waiting to be sent';

