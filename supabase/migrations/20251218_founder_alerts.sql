-- ═══════════════════════════════════════════════════════════════════════════
-- FOUNDER ALERTS & AUTONOMOUS MODERATION SYSTEM
-- 
-- Philosophy: Remove founder from day-to-day operations.
-- Only escalate what the community can't solve itself.
-- 
-- Created: 2025-12-18
-- ═══════════════════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════════════════
-- NOTIFICATION TABLES
-- ═══════════════════════════════════════════════════════════════════════════

-- Log of all notifications sent to founder
CREATE TABLE IF NOT EXISTS founder_notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  type TEXT NOT NULL CHECK (type IN ('milestone', 'escalation', 'happy_digest', 'escalation_digest', 'weekly_digest')),
  data JSONB,
  sent_at TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_founder_notifications_type ON founder_notifications(type);
CREATE INDEX IF NOT EXISTS idx_founder_notifications_sent ON founder_notifications(sent_at DESC);

-- Happy learner events (celebrations)
CREATE TABLE IF NOT EXISTS happy_learner_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  type TEXT NOT NULL CHECK (type IN ('first_lesson', 'streak_7', 'streak_30', 'streak_100', 'streak_365', 'completed_track', 'helpful_comment', 'first_comment')),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  detail TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_happy_events_type ON happy_learner_events(type);
CREATE INDEX IF NOT EXISTS idx_happy_events_created ON happy_learner_events(created_at DESC);

-- Lesson completions tracking
CREATE TABLE IF NOT EXISTS lesson_completions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  lesson_year INTEGER NOT NULL DEFAULT 1,
  lesson_day INTEGER NOT NULL,
  completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, lesson_year, lesson_day)
);

CREATE INDEX IF NOT EXISTS idx_lesson_completions_user ON lesson_completions(user_id);
CREATE INDEX IF NOT EXISTS idx_lesson_completions_date ON lesson_completions(completed_at DESC);

-- Payment events for billing alerts
CREATE TABLE IF NOT EXISTS payment_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  stripe_customer_id TEXT,
  event_type TEXT NOT NULL,
  amount_cents INTEGER,
  currency TEXT DEFAULT 'usd',
  resolved BOOLEAN DEFAULT false,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_payment_events_type ON payment_events(event_type);
CREATE INDEX IF NOT EXISTS idx_payment_events_resolved ON payment_events(resolved) WHERE resolved = false;

-- HeyGen performance logs
CREATE TABLE IF NOT EXISTS heygen_performance_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  checked_at TIMESTAMP WITH TIME ZONE NOT NULL,
  completed_count INTEGER NOT NULL DEFAULT 0,
  pending_count INTEGER NOT NULL DEFAULT 0,
  failed_count INTEGER NOT NULL DEFAULT 0,
  sample_data JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_heygen_logs_checked ON heygen_performance_logs(checked_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════
-- CURRICULUM SUGGESTIONS ENHANCEMENTS
-- ═══════════════════════════════════════════════════════════════════════════

ALTER TABLE curriculum_suggestions ADD COLUMN IF NOT EXISTS upvotes INTEGER DEFAULT 0;
ALTER TABLE curriculum_suggestions ADD COLUMN IF NOT EXISTS downvotes INTEGER DEFAULT 0;
ALTER TABLE curriculum_suggestions ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE curriculum_suggestions ADD COLUMN IF NOT EXISTS resolution_notes TEXT;
ALTER TABLE curriculum_suggestions ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'open';

-- ═══════════════════════════════════════════════════════════════════════════
-- AUTO-MODERATION TRIGGERS
-- ═══════════════════════════════════════════════════════════════════════════

-- Auto-log happy events on lesson completion
CREATE OR REPLACE FUNCTION log_happy_event() RETURNS TRIGGER AS $$
DECLARE
  completion_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO completion_count
  FROM lesson_completions
  WHERE user_id = NEW.user_id;
  
  -- First lesson
  IF completion_count = 1 THEN
    INSERT INTO happy_learner_events (type, user_id, detail)
    VALUES ('first_lesson', NEW.user_id, 'Completed their first lesson!');
  END IF;
  
  -- Track completion (365 lessons)
  IF completion_count = 365 THEN
    INSERT INTO happy_learner_events (type, user_id, detail)
    VALUES ('completed_track', NEW.user_id, 'Completed 365 lessons!');
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS on_lesson_completion ON lesson_completions;
CREATE TRIGGER on_lesson_completion
  AFTER INSERT ON lesson_completions
  FOR EACH ROW
  EXECUTE FUNCTION log_happy_event();

-- Auto-approve trusted users (5+ approved comments)
CREATE OR REPLACE FUNCTION auto_moderate_comment() RETURNS TRIGGER AS $$
DECLARE
  approved_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO approved_count
  FROM phase_comments
  WHERE user_id = NEW.user_id
    AND moderation_status IN ('approved', 'featured')
    AND deleted_at IS NULL;
  
  IF approved_count >= 5 THEN
    NEW.moderation_status := 'approved';
    NEW.moderated_at := NOW();
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS auto_moderate_on_insert ON phase_comments;
CREATE TRIGGER auto_moderate_on_insert
  BEFORE INSERT ON phase_comments
  FOR EACH ROW
  EXECUTE FUNCTION auto_moderate_comment();

-- Auto-feature popular comments (10+ upvotes)
CREATE OR REPLACE FUNCTION auto_feature_popular_comment() RETURNS TRIGGER AS $$
BEGIN
  IF NEW.upvotes >= 10 AND NEW.moderation_status = 'approved' THEN
    NEW.moderation_status := 'featured';
    NEW.moderated_at := NOW();
    
    INSERT INTO happy_learner_events (type, user_id, detail)
    VALUES ('helpful_comment', NEW.user_id, 'Your comment was featured!');
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS auto_feature_on_upvote ON phase_comments;
CREATE TRIGGER auto_feature_on_upvote
  BEFORE UPDATE OF upvotes ON phase_comments
  FOR EACH ROW
  WHEN (NEW.upvotes > OLD.upvotes)
  EXECUTE FUNCTION auto_feature_popular_comment();

-- Auto-resolve suggestions by community consensus
CREATE OR REPLACE FUNCTION auto_resolve_suggestion() RETURNS TRIGGER AS $$
BEGIN
  -- 20+ upvotes = community accepts
  IF NEW.upvotes >= 20 AND NEW.status = 'open' THEN
    NEW.status := 'accepted';
    NEW.resolved_at := NOW();
    NEW.resolution_notes := 'Auto-accepted: Community consensus (20+ upvotes)';
  END IF;
  
  -- 10+ downvotes = community rejects
  IF NEW.downvotes >= 10 AND NEW.status = 'open' THEN
    NEW.status := 'declined';
    NEW.resolved_at := NOW();
    NEW.resolution_notes := 'Auto-declined: Community consensus (10+ downvotes)';
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS auto_resolve_on_vote ON curriculum_suggestions;
CREATE TRIGGER auto_resolve_on_vote
  BEFORE UPDATE OF upvotes, downvotes ON curriculum_suggestions
  FOR EACH ROW
  EXECUTE FUNCTION auto_resolve_suggestion();

-- ═══════════════════════════════════════════════════════════════════════════
-- STREAK CHECKER FUNCTION (called by cron)
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION check_and_log_streaks() RETURNS void AS $$
DECLARE
  user_record RECORD;
  streak_days INTEGER;
BEGIN
  FOR user_record IN 
    SELECT DISTINCT user_id FROM lesson_completions 
    WHERE completed_at > NOW() - INTERVAL '1 day'
  LOOP
    -- Simple streak calculation
    WITH consecutive_days AS (
      SELECT DATE(completed_at) as d,
             DATE(completed_at) - ROW_NUMBER() OVER (ORDER BY DATE(completed_at))::INTEGER AS grp
      FROM lesson_completions
      WHERE user_id = user_record.user_id
      GROUP BY DATE(completed_at)
    ),
    streak_groups AS (
      SELECT grp, COUNT(*) as streak_length, MAX(d) as end_date
      FROM consecutive_days
      GROUP BY grp
    )
    SELECT COALESCE(streak_length, 0) INTO streak_days
    FROM streak_groups
    WHERE end_date = CURRENT_DATE
    LIMIT 1;
    
    -- Log milestone streaks
    IF streak_days = 7 AND NOT EXISTS (
      SELECT 1 FROM happy_learner_events 
      WHERE user_id = user_record.user_id AND type = 'streak_7'
      AND created_at > NOW() - INTERVAL '30 days'
    ) THEN
      INSERT INTO happy_learner_events (type, user_id, detail)
      VALUES ('streak_7', user_record.user_id, '7-day streak!');
    END IF;
    
    IF streak_days = 30 AND NOT EXISTS (
      SELECT 1 FROM happy_learner_events 
      WHERE user_id = user_record.user_id AND type = 'streak_30'
      AND created_at > NOW() - INTERVAL '60 days'
    ) THEN
      INSERT INTO happy_learner_events (type, user_id, detail)
      VALUES ('streak_30', user_record.user_id, '30-day streak!');
    END IF;
    
    IF streak_days = 100 AND NOT EXISTS (
      SELECT 1 FROM happy_learner_events 
      WHERE user_id = user_record.user_id AND type = 'streak_100'
      AND created_at > NOW() - INTERVAL '120 days'
    ) THEN
      INSERT INTO happy_learner_events (type, user_id, detail)
      VALUES ('streak_100', user_record.user_id, '100-day streak!');
    END IF;
    
    IF streak_days = 365 AND NOT EXISTS (
      SELECT 1 FROM happy_learner_events 
      WHERE user_id = user_record.user_id AND type = 'streak_365'
    ) THEN
      INSERT INTO happy_learner_events (type, user_id, detail)
      VALUES ('streak_365', user_record.user_id, '365-day streak! A full year!');
    END IF;
  END LOOP;
END;
$$ LANGUAGE plpgsql;
