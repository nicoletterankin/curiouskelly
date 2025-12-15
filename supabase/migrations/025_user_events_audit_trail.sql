-- ============================================
-- USER EVENTS: Zero-Trust Audit Trail
-- ============================================
-- Every interaction tracked, immutable, queryable
-- Enables "pull any user_id, see complete history"

-- Create the main events table
CREATE TABLE IF NOT EXISTS public.user_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Who
  user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
  session_id UUID,
  
  -- What
  event_type VARCHAR(50) NOT NULL,
  event_category VARCHAR(30) NOT NULL CHECK (event_category IN ('learner_action', 'kelly_action', 'system')),
  
  -- Details (flexible JSON payload)
  payload JSONB NOT NULL DEFAULT '{}',
  
  -- When (immutable)
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  
  -- Context
  ip_address INET,
  user_agent TEXT,
  device_type VARCHAR(20) CHECK (device_type IN ('mobile', 'tablet', 'desktop', 'tv', 'unknown')),
  platform VARCHAR(20) CHECK (platform IN ('web', 'ios', 'android', 'roku', 'api', 'unknown')),
  
  -- Lesson context (if applicable)
  day_number INTEGER,
  
  -- Immutability verification
  checksum VARCHAR(64)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON public.user_events(user_id);
CREATE INDEX IF NOT EXISTS idx_user_events_type ON public.user_events(event_type);
CREATE INDEX IF NOT EXISTS idx_user_events_category ON public.user_events(event_category);
CREATE INDEX IF NOT EXISTS idx_user_events_created ON public.user_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_events_day ON public.user_events(day_number) WHERE day_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_events_user_created ON public.user_events(user_id, created_at DESC);

-- Composite index for audit queries
CREATE INDEX IF NOT EXISTS idx_user_events_audit ON public.user_events(user_id, event_type, created_at DESC);

-- Enable RLS
ALTER TABLE public.user_events ENABLE ROW LEVEL SECURITY;

-- Users can view their own events
CREATE POLICY "Users can view own events" ON public.user_events
  FOR SELECT USING (auth.uid() = user_id);

-- Users can insert their own events
CREATE POLICY "Users can log own events" ON public.user_events
  FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Service role can insert any events (for Kelly → User tracking)
CREATE POLICY "Service can log any events" ON public.user_events
  FOR INSERT WITH CHECK (auth.role() = 'service_role');

-- IMMUTABILITY: Prevent updates and deletes
CREATE OR REPLACE FUNCTION prevent_event_modification()
RETURNS TRIGGER AS $$
BEGIN
  RAISE EXCEPTION 'user_events table is immutable - modifications not allowed';
END;
$$ LANGUAGE plpgsql;

-- Drop trigger if exists (for re-running migration)
DROP TRIGGER IF EXISTS no_update_events ON public.user_events;
DROP TRIGGER IF EXISTS no_delete_events ON public.user_events;

CREATE TRIGGER no_update_events
  BEFORE UPDATE ON public.user_events
  FOR EACH ROW EXECUTE FUNCTION prevent_event_modification();

CREATE TRIGGER no_delete_events
  BEFORE DELETE ON public.user_events
  FOR EACH ROW EXECUTE FUNCTION prevent_event_modification();

-- Function to generate checksum on insert
CREATE OR REPLACE FUNCTION generate_event_checksum()
RETURNS TRIGGER AS $$
BEGIN
  NEW.checksum = encode(
    sha256(
      (NEW.user_id::text || NEW.event_type || NEW.event_category || NEW.created_at::text || NEW.payload::text)::bytea
    ),
    'hex'
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_event_checksum ON public.user_events;

CREATE TRIGGER set_event_checksum
  BEFORE INSERT ON public.user_events
  FOR EACH ROW EXECUTE FUNCTION generate_event_checksum();

-- Comment on table
COMMENT ON TABLE public.user_events IS 'Zero-trust audit trail of all user<->Kelly interactions. IMMUTABLE.';

-- ============================================
-- EVENT TYPE REFERENCE (for documentation)
-- ============================================
-- LEARNER_ACTION:
--   lesson.started, lesson.completed, lesson.paused, lesson.skipped
--   comment.posted, comment.edited, comment.deleted
--   artwork.submitted, artwork.withdrawn
--   reaction.added, reaction.removed
--   purchase.initiated, purchase.completed, purchase.failed, purchase.refunded
--   subscription.started, subscription.renewed, subscription.cancelled
--   download.requested, download.completed, download.bundle
--   liveclass.joined, liveclass.left, liveclass.question
--   support.ticket_opened, support.message_sent
--   settings.updated, profile.updated
--
-- KELLY_ACTION:
--   kelly.email_sent, kelly.push_sent, kelly.sms_sent
--   kelly.reminder_sent, kelly.streak_celebrated
--   kelly.welcome_sent, kelly.comeback_sent
--   kelly.gift_delivered, kelly.birthday_message
--   moderation.comment_approved, moderation.comment_rejected
--   moderation.artwork_approved, moderation.artwork_rejected
--
-- SYSTEM:
--   system.session_started, system.session_ended
--   system.error, system.migration
