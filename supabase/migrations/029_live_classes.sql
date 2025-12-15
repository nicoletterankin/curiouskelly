-- ============================================
-- LIVE CLASSES: Hourly Kelly Sessions
-- ============================================

-- ============================================
-- LIVE CLASS SESSIONS
-- ============================================

CREATE TABLE IF NOT EXISTS public.live_class_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Schedule
  scheduled_at TIMESTAMPTZ NOT NULL,  -- Top of hour (e.g., 2025-12-15 14:00:00 UTC)
  day_number INTEGER NOT NULL,  -- Which lesson is being taught
  
  -- Session info
  title TEXT,
  description TEXT,
  
  -- Status
  status VARCHAR(20) DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'live', 'completed', 'cancelled')),
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  
  -- Stream info (YouTube Live, Daily.co, etc.)
  stream_platform VARCHAR(20) DEFAULT 'youtube' CHECK (stream_platform IN ('youtube', 'daily', 'zoom', 'custom')),
  stream_url TEXT,
  stream_id TEXT,  -- YouTube video ID or Daily room name
  
  -- Capacity
  max_attendees INTEGER DEFAULT 10000,
  actual_attendees INTEGER DEFAULT 0,
  peak_concurrent INTEGER DEFAULT 0,
  
  -- Recording
  recording_url TEXT,
  recording_duration_seconds INTEGER,
  transcript_url TEXT,
  
  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Unique constraint: one session per hour per day
  UNIQUE(scheduled_at)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_live_sessions_scheduled ON public.live_class_sessions(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_live_sessions_day ON public.live_class_sessions(day_number);
CREATE INDEX IF NOT EXISTS idx_live_sessions_status ON public.live_class_sessions(status);
CREATE INDEX IF NOT EXISTS idx_live_sessions_upcoming ON public.live_class_sessions(scheduled_at) WHERE status = 'scheduled' AND scheduled_at > NOW();

-- Enable RLS
ALTER TABLE public.live_class_sessions ENABLE ROW LEVEL SECURITY;

-- Anyone can view sessions
CREATE POLICY "Anyone can view sessions" ON public.live_class_sessions
  FOR SELECT USING (true);

-- Trigger for updated_at
CREATE TRIGGER update_live_sessions_updated_at
  BEFORE UPDATE ON public.live_class_sessions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- LIVE CLASS ATTENDANCE
-- ============================================

CREATE TABLE IF NOT EXISTS public.live_class_attendance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- What session
  session_id UUID NOT NULL REFERENCES public.live_class_sessions(id) ON DELETE CASCADE,
  
  -- Who attended (can be anonymous for free users)
  user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
  anonymous_id VARCHAR(100),  -- For non-logged-in users
  
  -- Timing
  joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  left_at TIMESTAMPTZ,
  duration_seconds INTEGER,
  
  -- Engagement
  questions_asked INTEGER DEFAULT 0,
  reactions_sent INTEGER DEFAULT 0,
  chat_messages INTEGER DEFAULT 0,
  
  -- Access type
  access_type VARCHAR(20) CHECK (access_type IN ('free_today', 'subscriber', 'purchased', 'anonymous')),
  
  -- Device
  device_type VARCHAR(20),
  platform VARCHAR(20),
  
  UNIQUE(session_id, user_id),
  UNIQUE(session_id, anonymous_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_attendance_session ON public.live_class_attendance(session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_user ON public.live_class_attendance(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_attendance_joined ON public.live_class_attendance(joined_at);

-- Enable RLS
ALTER TABLE public.live_class_attendance ENABLE ROW LEVEL SECURITY;

-- Users can view their own attendance
CREATE POLICY "Users can view own attendance" ON public.live_class_attendance
  FOR SELECT USING (auth.uid() = user_id);

-- Users can record their own attendance
CREATE POLICY "Users can record attendance" ON public.live_class_attendance
  FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Users can update their own attendance (for leave time)
CREATE POLICY "Users can update own attendance" ON public.live_class_attendance
  FOR UPDATE USING (auth.uid() = user_id OR user_id IS NULL);

-- Trigger to update session attendee count
CREATE OR REPLACE FUNCTION update_session_attendees()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE public.live_class_sessions 
    SET 
      actual_attendees = actual_attendees + 1,
      peak_concurrent = GREATEST(peak_concurrent, actual_attendees + 1)
    WHERE id = NEW.session_id;
  ELSIF TG_OP = 'UPDATE' AND NEW.left_at IS NOT NULL AND OLD.left_at IS NULL THEN
    -- User left, calculate duration
    NEW.duration_seconds = EXTRACT(EPOCH FROM (NEW.left_at - NEW.joined_at))::INTEGER;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS track_session_attendees ON public.live_class_attendance;

CREATE TRIGGER track_session_attendees
  AFTER INSERT OR UPDATE ON public.live_class_attendance
  FOR EACH ROW EXECUTE FUNCTION update_session_attendees();

-- ============================================
-- FUNCTION: Get next live class
-- ============================================

CREATE OR REPLACE FUNCTION get_next_live_class()
RETURNS TABLE (
  id UUID,
  scheduled_at TIMESTAMPTZ,
  day_number INTEGER,
  stream_url TEXT,
  status VARCHAR(20),
  minutes_until INTEGER
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    s.id,
    s.scheduled_at,
    s.day_number,
    s.stream_url,
    s.status,
    EXTRACT(EPOCH FROM (s.scheduled_at - NOW()))::INTEGER / 60 as minutes_until
  FROM public.live_class_sessions s
  WHERE s.scheduled_at > NOW() - INTERVAL '1 hour'
    AND s.status IN ('scheduled', 'live')
  ORDER BY s.scheduled_at
  LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Comments
COMMENT ON TABLE public.live_class_sessions IS 'Hourly live class sessions with Kelly';
COMMENT ON TABLE public.live_class_attendance IS 'Who attended which live class';
