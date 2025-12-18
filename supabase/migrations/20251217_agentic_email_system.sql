-- Agentic Email System Tables
-- Kelly's autonomous email handling infrastructure
-- Created: December 17, 2025

-- ============================================
-- EMAIL THREADS - Main thread tracking
-- ============================================
CREATE TABLE IF NOT EXISTS email_threads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  thread_id TEXT UNIQUE,                    -- For grouping replies (e.g., In-Reply-To header)
  from_email TEXT NOT NULL,
  from_name TEXT,
  subject TEXT,
  
  -- Classification
  category TEXT CHECK (category IN (
    'support', 'billing', 'enterprise', 'press', 
    'family', 'partner', 'feedback', 'spam', 'other'
  )),
  urgency TEXT DEFAULT 'normal' CHECK (urgency IN ('low', 'normal', 'high', 'critical')),
  sentiment TEXT CHECK (sentiment IN ('positive', 'neutral', 'negative', 'angry')),
  
  -- Status tracking
  status TEXT DEFAULT 'open' CHECK (status IN (
    'open', 'pending_approval', 'responded', 'escalated', 'closed', 'spam'
  )),
  requires_human BOOLEAN DEFAULT false,
  
  -- Escalation
  escalated_to TEXT,                        -- nicoletterankin@gmail.com if escalated
  escalation_reason TEXT,
  escalated_at TIMESTAMPTZ,
  
  -- AI processing metadata
  classification_confidence NUMERIC(3,2),   -- 0.00 to 1.00
  ai_summary TEXT,                          -- Brief AI-generated summary
  detected_intent TEXT,                     -- password_reset, refund_request, etc.
  detected_entities JSONB,                  -- { "product": "subscription", "amount": "$199" }
  
  -- Timestamps
  first_message_at TIMESTAMPTZ DEFAULT NOW(),
  last_message_at TIMESTAMPTZ DEFAULT NOW(),
  responded_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_email_threads_status ON email_threads(status);
CREATE INDEX IF NOT EXISTS idx_email_threads_category ON email_threads(category);
CREATE INDEX IF NOT EXISTS idx_email_threads_from_email ON email_threads(from_email);
CREATE INDEX IF NOT EXISTS idx_email_threads_created_at ON email_threads(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_email_threads_requires_human ON email_threads(requires_human) WHERE requires_human = true;
CREATE INDEX IF NOT EXISTS idx_email_threads_escalated ON email_threads(escalated_to) WHERE escalated_to IS NOT NULL;

-- ============================================
-- EMAIL MESSAGES - Individual messages in threads
-- ============================================
CREATE TABLE IF NOT EXISTS email_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  thread_id UUID REFERENCES email_threads(id) ON DELETE CASCADE,
  
  -- Direction and addresses
  direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound', 'draft')),
  from_email TEXT,
  from_name TEXT,
  to_email TEXT,
  reply_to TEXT,
  cc TEXT[],
  
  -- Content
  subject TEXT,
  body_text TEXT,
  body_html TEXT,
  attachments JSONB,                        -- [{ "filename": "...", "size": 1234, "url": "..." }]
  
  -- Resend integration
  resend_message_id TEXT,                   -- Resend's ID for tracking
  resend_status TEXT,                       -- sent, delivered, bounced, etc.
  
  -- For drafts awaiting approval
  is_approved BOOLEAN DEFAULT false,
  approved_by TEXT,
  approved_at TIMESTAMPTZ,
  
  -- Timestamps
  sent_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_email_messages_thread_id ON email_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_email_messages_direction ON email_messages(direction);
CREATE INDEX IF NOT EXISTS idx_email_messages_created_at ON email_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_email_messages_drafts ON email_messages(direction, is_approved) 
  WHERE direction = 'draft' AND is_approved = false;

-- ============================================
-- EMAIL ACTIONS - Audit trail
-- ============================================
CREATE TABLE IF NOT EXISTS email_actions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  thread_id UUID REFERENCES email_threads(id) ON DELETE CASCADE,
  message_id UUID REFERENCES email_messages(id) ON DELETE SET NULL,
  
  -- Action details
  action TEXT NOT NULL CHECK (action IN (
    'received', 'classified', 'drafted', 'approved', 'sent', 
    'escalated', 'closed', 'reopened', 'merged', 'marked_spam'
  )),
  actor TEXT NOT NULL,                      -- kelly_ai, nicolette, system, resend_webhook
  
  -- Additional context
  details JSONB,                            -- { "reason": "...", "old_status": "...", "new_status": "..." }
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for audit trail queries
CREATE INDEX IF NOT EXISTS idx_email_actions_thread_id ON email_actions(thread_id);
CREATE INDEX IF NOT EXISTS idx_email_actions_created_at ON email_actions(created_at DESC);

-- ============================================
-- EMAIL TEMPLATES - Reusable response patterns
-- ============================================
CREATE TABLE IF NOT EXISTS email_templates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Template identification
  name TEXT NOT NULL UNIQUE,
  category TEXT NOT NULL,                   -- support, billing, enterprise, etc.
  intent TEXT,                              -- password_reset, refund_request, etc.
  
  -- Content
  subject_template TEXT,
  body_template TEXT NOT NULL,              -- Supports {{variables}}
  
  -- Metadata
  is_active BOOLEAN DEFAULT true,
  use_count INTEGER DEFAULT 0,
  last_used_at TIMESTAMPTZ,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- EMAIL ESCALATION RULES - Configurable escalation
-- ============================================
CREATE TABLE IF NOT EXISTS email_escalation_rules (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Rule definition
  name TEXT NOT NULL,
  description TEXT,
  
  -- Matching criteria (all must match if specified)
  match_category TEXT[],                    -- ['billing', 'enterprise']
  match_sentiment TEXT[],                   -- ['angry', 'negative']
  match_keywords TEXT[],                    -- ['refund', 'lawyer', 'cancel']
  match_from_domain TEXT[],                 -- ['nytimes.com', 'wsj.com']
  
  -- Action
  escalate_to TEXT NOT NULL,                -- email address
  priority TEXT DEFAULT 'normal',
  notification_subject TEXT,
  
  -- Status
  is_active BOOLEAN DEFAULT true,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default escalation rules
INSERT INTO email_escalation_rules (name, description, match_category, escalate_to, priority) VALUES
  ('Enterprise Deals', 'All enterprise inquiries need human review', 
   ARRAY['enterprise'], 'nicoletterankin@gmail.com', 'high'),
  ('Press Inquiries', 'Media and journalist requests', 
   ARRAY['press'], 'nicoletterankin@gmail.com', 'high'),
  ('Billing Issues', 'Refunds, disputes, and payment problems', 
   ARRAY['billing'], 'nicoletterankin@gmail.com', 'high'),
  ('Partner Requests', 'Affiliate and partnership inquiries', 
   ARRAY['partner'], 'nicoletterankin@gmail.com', 'normal')
ON CONFLICT DO NOTHING;

INSERT INTO email_escalation_rules (name, description, match_keywords, escalate_to, priority) VALUES
  ('Legal Mentions', 'Any mention of legal action', 
   ARRAY['lawyer', 'legal', 'lawsuit', 'attorney', 'sue'], 'nicoletterankin@gmail.com', 'critical'),
  ('Angry Customers', 'Upset or frustrated customers', 
   ARRAY['furious', 'outraged', 'terrible', 'worst', 'horrible', 'disgusted'], 'nicoletterankin@gmail.com', 'high'),
  ('Refund Requests', 'Direct refund mentions', 
   ARRAY['refund', 'money back', 'chargeback'], 'nicoletterankin@gmail.com', 'high')
ON CONFLICT DO NOTHING;

INSERT INTO email_escalation_rules (name, description, match_sentiment, escalate_to, priority) VALUES
  ('Very Negative Sentiment', 'AI detected angry tone', 
   ARRAY['angry'], 'nicoletterankin@gmail.com', 'high')
ON CONFLICT DO NOTHING;

-- ============================================
-- FUNCTIONS
-- ============================================

-- Update thread timestamps when messages are added
CREATE OR REPLACE FUNCTION update_thread_timestamps()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE email_threads 
  SET 
    last_message_at = NEW.created_at,
    updated_at = NOW()
  WHERE id = NEW.thread_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for message inserts
DROP TRIGGER IF EXISTS trigger_update_thread_timestamps ON email_messages;
CREATE TRIGGER trigger_update_thread_timestamps
  AFTER INSERT ON email_messages
  FOR EACH ROW
  EXECUTE FUNCTION update_thread_timestamps();

-- Auto-update updated_at on threads
CREATE OR REPLACE FUNCTION update_email_thread_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_email_thread_updated_at ON email_threads;
CREATE TRIGGER trigger_email_thread_updated_at
  BEFORE UPDATE ON email_threads
  FOR EACH ROW
  EXECUTE FUNCTION update_email_thread_updated_at();

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

-- Enable RLS
ALTER TABLE email_threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE email_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE email_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE email_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE email_escalation_rules ENABLE ROW LEVEL SECURITY;

-- Service role can do everything (for API)
CREATE POLICY "Service role full access on email_threads" ON email_threads
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access on email_messages" ON email_messages
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access on email_actions" ON email_actions
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access on email_templates" ON email_templates
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access on email_escalation_rules" ON email_escalation_rules
  FOR ALL USING (auth.role() = 'service_role');

-- ============================================
-- VIEWS
-- ============================================

-- Pending drafts view (for admin dashboard)
CREATE OR REPLACE VIEW email_pending_drafts AS
SELECT 
  t.id as thread_id,
  t.from_email,
  t.from_name,
  t.subject,
  t.category,
  t.urgency,
  t.sentiment,
  t.ai_summary,
  m.id as draft_id,
  m.body_text as draft_text,
  m.body_html as draft_html,
  m.created_at as draft_created_at,
  t.first_message_at
FROM email_threads t
JOIN email_messages m ON m.thread_id = t.id
WHERE t.status = 'pending_approval'
  AND m.direction = 'draft'
  AND m.is_approved = false
ORDER BY 
  CASE t.urgency 
    WHEN 'critical' THEN 1 
    WHEN 'high' THEN 2 
    WHEN 'normal' THEN 3 
    ELSE 4 
  END,
  t.first_message_at ASC;

-- Daily email stats view
CREATE OR REPLACE VIEW email_daily_stats AS
SELECT 
  DATE(created_at) as date,
  COUNT(*) as total_threads,
  COUNT(*) FILTER (WHERE status = 'responded') as auto_responded,
  COUNT(*) FILTER (WHERE status = 'pending_approval') as pending_approval,
  COUNT(*) FILTER (WHERE escalated_to IS NOT NULL) as escalated,
  COUNT(*) FILTER (WHERE category = 'spam') as spam,
  AVG(EXTRACT(EPOCH FROM (responded_at - first_message_at))/60) as avg_response_time_minutes
FROM email_threads
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

COMMENT ON TABLE email_threads IS 'Kelly Agentic Email System - Main thread tracking';
COMMENT ON TABLE email_messages IS 'Individual messages within email threads';
COMMENT ON TABLE email_actions IS 'Audit trail of all actions taken on emails';
COMMENT ON TABLE email_templates IS 'Reusable response templates for Kelly';
COMMENT ON TABLE email_escalation_rules IS 'Rules for when to escalate emails to humans';
