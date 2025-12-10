-- Migration: 20251207_earn_to_learn
-- Description: Integrate affiliate/earnings system directly into learner experience
-- Author: System Architect
-- Date: December 7, 2025
-- 
-- PHILOSOPHY: Every learner is an affiliate from Day 1
-- LIFETIME COOKIES: Attribution never expires
-- EARN TO LEARN: Commission rates increase with learning progress

BEGIN;

-- ============================================================
-- PART 1: Extend users table with earning capabilities
-- ============================================================

-- Add referral/earning columns to users (if not exists)
DO $$
BEGIN
  -- Referral identity
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'referral_code') THEN
    ALTER TABLE users ADD COLUMN referral_code TEXT UNIQUE;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'referred_by_user_id') THEN
    ALTER TABLE users ADD COLUMN referred_by_user_id UUID REFERENCES users(id);
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'referred_at') THEN
    ALTER TABLE users ADD COLUMN referred_at TIMESTAMPTZ;
  END IF;
  
  -- Commission tracking
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'commission_rate') THEN
    ALTER TABLE users ADD COLUMN commission_rate DECIMAL(5,4) DEFAULT 0.10; -- 10% starting
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'commission_tier') THEN
    ALTER TABLE users ADD COLUMN commission_tier TEXT DEFAULT 'new_learner';
  END IF;
  
  -- Referral stats
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'total_referrals') THEN
    ALTER TABLE users ADD COLUMN total_referrals INTEGER DEFAULT 0;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'active_referrals') THEN
    ALTER TABLE users ADD COLUMN active_referrals INTEGER DEFAULT 0;
  END IF;
  
  -- Earnings
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lifetime_earnings') THEN
    ALTER TABLE users ADD COLUMN lifetime_earnings DECIMAL(12,2) DEFAULT 0.00;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'pending_earnings') THEN
    ALTER TABLE users ADD COLUMN pending_earnings DECIMAL(12,2) DEFAULT 0.00;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'available_earnings') THEN
    ALTER TABLE users ADD COLUMN available_earnings DECIMAL(12,2) DEFAULT 0.00;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'last_payout_at') THEN
    ALTER TABLE users ADD COLUMN last_payout_at TIMESTAMPTZ;
  END IF;
  
  -- Payout preferences
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'payout_method') THEN
    ALTER TABLE users ADD COLUMN payout_method TEXT CHECK (payout_method IN ('paypal', 'stripe', 'bank', 'gift_credit'));
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'payout_details') THEN
    ALTER TABLE users ADD COLUMN payout_details JSONB DEFAULT '{}';
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'tax_form_status') THEN
    ALTER TABLE users ADD COLUMN tax_form_status TEXT DEFAULT 'not_required' CHECK (tax_form_status IN ('not_required', 'pending', 'submitted', 'verified'));
  END IF;
END $$;

-- ============================================================
-- PART 2: Referral tracking (LIFETIME attribution)
-- ============================================================

CREATE TABLE IF NOT EXISTS referral_clicks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- The referrer
  referrer_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  referral_code TEXT NOT NULL,
  
  -- The visitor (may not be a user yet)
  visitor_fingerprint TEXT, -- Browser fingerprint for pre-signup tracking
  visitor_ip_hash TEXT, -- Hashed IP for privacy-respecting tracking
  visitor_email TEXT, -- If they enter email before signup
  
  -- Tracking
  clicked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  source_url TEXT,
  landing_page TEXT,
  
  -- UTM tracking
  utm_source TEXT,
  utm_medium TEXT,
  utm_campaign TEXT,
  utm_content TEXT,
  utm_term TEXT,
  
  -- Conversion tracking
  converted_to_user_id UUID REFERENCES users(id),
  converted_at TIMESTAMPTZ,
  conversion_type TEXT, -- 'signup', 'subscription', 'gift'
  
  -- LIFETIME ATTRIBUTION - NO EXPIRATION
  -- This is the key difference from standard affiliate programs
  attribution_expires_at TIMESTAMPTZ DEFAULT NULL, -- NULL = NEVER expires
  
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_referral_clicks_referrer ON referral_clicks(referrer_id);
CREATE INDEX IF NOT EXISTS idx_referral_clicks_code ON referral_clicks(referral_code);
CREATE INDEX IF NOT EXISTS idx_referral_clicks_fingerprint ON referral_clicks(visitor_fingerprint);
CREATE INDEX IF NOT EXISTS idx_referral_clicks_email ON referral_clicks(visitor_email);

-- ============================================================
-- PART 3: Commission transactions
-- ============================================================

CREATE TABLE IF NOT EXISTS commission_transactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Parties
  referrer_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  referred_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  referral_click_id UUID REFERENCES referral_clicks(id),
  
  -- Transaction type
  transaction_type TEXT NOT NULL CHECK (transaction_type IN (
    'initial_subscription',
    'subscription_renewal', 
    'subscription_upgrade',
    'gift_purchase',
    'lifetime_purchase',
    'adjustment',
    'refund_clawback'
  )),
  
  -- Financial details
  gross_amount DECIMAL(12,2) NOT NULL, -- Amount the customer paid
  commission_rate DECIMAL(5,4) NOT NULL, -- Rate at time of transaction
  commission_amount DECIMAL(12,2) NOT NULL, -- What the referrer earns
  currency TEXT DEFAULT 'USD',
  
  -- Status
  status TEXT DEFAULT 'pending' CHECK (status IN (
    'pending',      -- Waiting for payment to clear
    'approved',     -- Cleared, ready for payout
    'paid',         -- Already paid out
    'clawed_back',  -- Reversed due to refund
    'cancelled'     -- Cancelled
  )),
  
  -- Payout tracking
  payout_id UUID, -- Links to payouts table when paid
  paid_at TIMESTAMPTZ,
  
  -- Stripe integration
  stripe_payment_intent_id TEXT,
  stripe_invoice_id TEXT,
  stripe_subscription_id TEXT,
  
  -- Notes
  notes TEXT,
  
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_commission_referrer ON commission_transactions(referrer_id);
CREATE INDEX IF NOT EXISTS idx_commission_referred ON commission_transactions(referred_user_id);
CREATE INDEX IF NOT EXISTS idx_commission_status ON commission_transactions(status);
CREATE INDEX IF NOT EXISTS idx_commission_payout ON commission_transactions(payout_id);

-- ============================================================
-- PART 4: Payouts
-- ============================================================

CREATE TABLE IF NOT EXISTS payouts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  
  -- Amount
  amount DECIMAL(12,2) NOT NULL,
  currency TEXT DEFAULT 'USD',
  
  -- Method
  method TEXT NOT NULL CHECK (method IN ('paypal', 'stripe', 'bank', 'gift_credit')),
  method_details JSONB, -- PayPal email, bank details, etc.
  
  -- Status
  status TEXT DEFAULT 'pending' CHECK (status IN (
    'pending',
    'processing',
    'completed',
    'failed',
    'cancelled'
  )),
  
  -- Processing
  requested_at TIMESTAMPTZ DEFAULT now(),
  processed_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  
  -- External references
  paypal_payout_id TEXT,
  stripe_transfer_id TEXT,
  bank_reference TEXT,
  
  -- Failure handling
  failure_reason TEXT,
  retry_count INTEGER DEFAULT 0,
  
  -- Notes
  notes TEXT,
  
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_payouts_user ON payouts(user_id);
CREATE INDEX IF NOT EXISTS idx_payouts_status ON payouts(status);

-- ============================================================
-- PART 5: Commission tier definitions
-- ============================================================

CREATE TABLE IF NOT EXISTS commission_tiers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  tier_name TEXT UNIQUE NOT NULL,
  display_name TEXT NOT NULL,
  
  -- Requirements
  min_lessons_completed INTEGER NOT NULL,
  min_unique_lessons INTEGER DEFAULT 0,
  min_referrals INTEGER DEFAULT 0,
  
  -- Rewards
  base_commission_rate DECIMAL(5,4) NOT NULL,
  bonus_commission_rate DECIMAL(5,4) DEFAULT 0, -- Additional bonuses
  
  -- Perks
  perks JSONB DEFAULT '[]',
  
  -- Ordering
  sort_order INTEGER NOT NULL,
  
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Insert default tiers
INSERT INTO commission_tiers (tier_name, display_name, min_lessons_completed, base_commission_rate, sort_order, perks)
VALUES 
  ('new_learner', 'New Learner', 0, 0.10, 1, '["Share & Earn access"]'),
  ('active_learner', 'Active Learner', 7, 0.15, 2, '["Weekly earnings email"]'),
  ('committed_learner', 'Committed Learner', 30, 0.20, 3, '["Monthly earnings report", "Priority support"]'),
  ('dedicated_learner', 'Dedicated Learner', 100, 0.25, 4, '["Dedicated dashboard", "Custom share links"]'),
  ('complete_learner', 'Complete Learner', 365, 0.30, 5, '["Kelly Companion badge", "Direct payout"]'),
  ('legendary_learner', 'Legendary Learner', 1000, 0.35, 6, '["VIP status", "Founding member perks", "API access"]')
ON CONFLICT (tier_name) DO UPDATE SET
  base_commission_rate = EXCLUDED.base_commission_rate,
  min_lessons_completed = EXCLUDED.min_lessons_completed;

-- ============================================================
-- PART 6: Bonus programs
-- ============================================================

CREATE TABLE IF NOT EXISTS bonus_programs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  program_name TEXT UNIQUE NOT NULL,
  display_name TEXT NOT NULL,
  description TEXT,
  
  -- Type
  bonus_type TEXT NOT NULL CHECK (bonus_type IN (
    'rate_increase',     -- Add to commission rate
    'flat_bonus',        -- One-time payment
    'multiplier'         -- Multiply commission
  )),
  
  bonus_value DECIMAL(8,4) NOT NULL, -- Rate increase, flat amount, or multiplier
  
  -- Eligibility
  eligibility_rules JSONB NOT NULL DEFAULT '{}',
  -- Examples:
  -- {"referred_teacher": true}
  -- {"referred_family_count": {"gte": 3}}
  -- {"community_referrals": {"gte": 10}}
  
  -- Limits
  max_applications_per_user INTEGER, -- NULL = unlimited
  total_budget DECIMAL(12,2), -- NULL = unlimited
  budget_used DECIMAL(12,2) DEFAULT 0,
  
  -- Timing
  starts_at TIMESTAMPTZ DEFAULT now(),
  ends_at TIMESTAMPTZ, -- NULL = ongoing
  
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Insert default bonus programs
INSERT INTO bonus_programs (program_name, display_name, description, bonus_type, bonus_value, eligibility_rules)
VALUES
  ('teacher_referral', 'Teacher Bonus', 'Extra commission for referring verified teachers', 'rate_increase', 0.05, '{"referred_is_teacher": true}'),
  ('family_bundle', 'Family Bundle', 'Bonus for referring 3+ family members', 'rate_increase', 0.05, '{"family_referrals_gte": 3}'),
  ('community_builder', 'Community Builder', 'Bonus for referring 10+ learners', 'rate_increase', 0.05, '{"total_referrals_gte": 10}'),
  ('first_share', 'First Share Bonus', 'One-time bonus for first successful referral', 'flat_bonus', 5.00, '{"first_referral": true}')
ON CONFLICT (program_name) DO NOTHING;

-- ============================================================
-- PART 7: Functions and Triggers
-- ============================================================

-- Function: Generate unique, memorable referral code
CREATE OR REPLACE FUNCTION generate_referral_code(user_id UUID, display_name TEXT, email TEXT)
RETURNS TEXT AS $$
DECLARE
  base_code TEXT;
  final_code TEXT;
  counter INTEGER := 0;
BEGIN
  -- Use display name if available, otherwise email prefix
  IF display_name IS NOT NULL AND display_name != '' THEN
    base_code := lower(regexp_replace(display_name, '[^a-zA-Z0-9]', '', 'g'));
  ELSE
    base_code := lower(split_part(email, '@', 1));
    base_code := regexp_replace(base_code, '[^a-zA-Z0-9]', '', 'g');
  END IF;
  
  -- Truncate to reasonable length
  base_code := left(base_code, 15);
  
  -- If base is too short, add some characters from user_id
  IF length(base_code) < 3 THEN
    base_code := base_code || left(user_id::text, 4);
  END IF;
  
  -- Try the base code first
  final_code := base_code;
  
  -- If taken, add incrementing numbers
  WHILE EXISTS (SELECT 1 FROM users WHERE referral_code = final_code) LOOP
    counter := counter + 1;
    final_code := base_code || counter::text;
  END LOOP;
  
  RETURN final_code;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-generate referral code on user insert
CREATE OR REPLACE FUNCTION set_referral_code_on_insert()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.referral_code IS NULL OR NEW.referral_code = '' THEN
    NEW.referral_code := generate_referral_code(NEW.id, NEW.display_name, NEW.email);
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_set_referral_code ON users;
CREATE TRIGGER tr_set_referral_code
BEFORE INSERT ON users
FOR EACH ROW
EXECUTE FUNCTION set_referral_code_on_insert();

-- Function: Update commission tier based on learning progress
CREATE OR REPLACE FUNCTION update_commission_tier()
RETURNS TRIGGER AS $$
DECLARE
  new_tier RECORD;
BEGIN
  -- Find the highest tier the user qualifies for
  SELECT * INTO new_tier
  FROM commission_tiers
  WHERE min_lessons_completed <= NEW.total_lessons_completed
    AND is_active = true
  ORDER BY sort_order DESC
  LIMIT 1;
  
  IF new_tier IS NOT NULL THEN
    NEW.commission_tier := new_tier.tier_name;
    NEW.commission_rate := new_tier.base_commission_rate;
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_update_commission_tier ON users;
CREATE TRIGGER tr_update_commission_tier
BEFORE UPDATE OF total_lessons_completed, unique_lessons_completed ON users
FOR EACH ROW
EXECUTE FUNCTION update_commission_tier();

-- Function: Record commission when payment occurs
CREATE OR REPLACE FUNCTION record_commission(
  p_referrer_id UUID,
  p_referred_user_id UUID,
  p_transaction_type TEXT,
  p_gross_amount DECIMAL,
  p_stripe_payment_intent_id TEXT DEFAULT NULL,
  p_stripe_invoice_id TEXT DEFAULT NULL,
  p_stripe_subscription_id TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
  v_commission_rate DECIMAL;
  v_commission_amount DECIMAL;
  v_transaction_id UUID;
BEGIN
  -- Get the referrer's current commission rate
  SELECT commission_rate INTO v_commission_rate
  FROM users
  WHERE id = p_referrer_id;
  
  -- Calculate commission
  v_commission_amount := p_gross_amount * v_commission_rate;
  
  -- Insert the transaction
  INSERT INTO commission_transactions (
    referrer_id,
    referred_user_id,
    transaction_type,
    gross_amount,
    commission_rate,
    commission_amount,
    stripe_payment_intent_id,
    stripe_invoice_id,
    stripe_subscription_id,
    status
  ) VALUES (
    p_referrer_id,
    p_referred_user_id,
    p_transaction_type,
    p_gross_amount,
    v_commission_rate,
    v_commission_amount,
    p_stripe_payment_intent_id,
    p_stripe_invoice_id,
    p_stripe_subscription_id,
    'pending'
  ) RETURNING id INTO v_transaction_id;
  
  -- Update referrer's pending earnings
  UPDATE users
  SET pending_earnings = pending_earnings + v_commission_amount,
      lifetime_earnings = lifetime_earnings + v_commission_amount,
      updated_at = now()
  WHERE id = p_referrer_id;
  
  RETURN v_transaction_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Approve pending commissions (called after payment clears)
CREATE OR REPLACE FUNCTION approve_pending_commissions(p_days_threshold INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
  v_count INTEGER;
BEGIN
  -- Move pending to approved after threshold days
  WITH approved AS (
    UPDATE commission_transactions
    SET status = 'approved',
        updated_at = now()
    WHERE status = 'pending'
      AND created_at < now() - (p_days_threshold || ' days')::interval
    RETURNING referrer_id, commission_amount
  )
  UPDATE users u
  SET pending_earnings = pending_earnings - approved.commission_amount,
      available_earnings = available_earnings + approved.commission_amount,
      updated_at = now()
  FROM approved
  WHERE u.id = approved.referrer_id;
  
  GET DIAGNOSTICS v_count = ROW_COUNT;
  RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- PART 8: Views for dashboards
-- ============================================================

-- Learner earnings dashboard view
CREATE OR REPLACE VIEW learner_earnings_dashboard AS
SELECT 
  u.id as user_id,
  u.referral_code,
  'kelly.me/' || u.referral_code as referral_link,
  u.commission_tier,
  u.commission_rate,
  u.total_referrals,
  u.active_referrals,
  u.pending_earnings,
  u.available_earnings,
  u.lifetime_earnings,
  u.total_lessons_completed,
  u.unique_lessons_completed,
  ct.display_name as tier_display_name,
  ct.perks as tier_perks,
  -- Next tier info
  next_tier.display_name as next_tier_name,
  next_tier.min_lessons_completed as lessons_to_next_tier,
  next_tier.base_commission_rate as next_tier_rate
FROM users u
LEFT JOIN commission_tiers ct ON u.commission_tier = ct.tier_name
LEFT JOIN LATERAL (
  SELECT * FROM commission_tiers
  WHERE min_lessons_completed > u.total_lessons_completed
    AND is_active = true
  ORDER BY sort_order ASC
  LIMIT 1
) next_tier ON true;

-- Recent transactions view
CREATE OR REPLACE VIEW recent_commission_transactions AS
SELECT 
  ct.*,
  referred.display_name as referred_name,
  referred.email as referred_email
FROM commission_transactions ct
JOIN users referred ON ct.referred_user_id = referred.id
ORDER BY ct.created_at DESC;

-- ============================================================
-- PART 9: Row Level Security
-- ============================================================

-- Enable RLS
ALTER TABLE referral_clicks ENABLE ROW LEVEL SECURITY;
ALTER TABLE commission_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE payouts ENABLE ROW LEVEL SECURITY;

-- Policies for referral_clicks
DROP POLICY IF EXISTS "Users can view their own referral clicks" ON referral_clicks;
CREATE POLICY "Users can view their own referral clicks" ON referral_clicks
  FOR SELECT USING (referrer_id = auth.uid());

-- Policies for commission_transactions  
DROP POLICY IF EXISTS "Users can view their own commissions" ON commission_transactions;
CREATE POLICY "Users can view their own commissions" ON commission_transactions
  FOR SELECT USING (referrer_id = auth.uid());

-- Policies for payouts
DROP POLICY IF EXISTS "Users can view their own payouts" ON payouts;
CREATE POLICY "Users can view their own payouts" ON payouts
  FOR SELECT USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can request payouts" ON payouts;
CREATE POLICY "Users can request payouts" ON payouts
  FOR INSERT WITH CHECK (user_id = auth.uid());

-- ============================================================
-- PART 10: Generate codes for existing users
-- ============================================================

-- Update existing users who don't have referral codes
UPDATE users
SET referral_code = generate_referral_code(id, display_name, email),
    commission_tier = 'new_learner',
    commission_rate = 0.10
WHERE referral_code IS NULL;

-- Update commission tiers based on existing progress
UPDATE users
SET commission_tier = (
  SELECT tier_name FROM commission_tiers
  WHERE min_lessons_completed <= users.total_lessons_completed
    AND is_active = true
  ORDER BY sort_order DESC
  LIMIT 1
),
commission_rate = (
  SELECT base_commission_rate FROM commission_tiers
  WHERE min_lessons_completed <= users.total_lessons_completed
    AND is_active = true
  ORDER BY sort_order DESC
  LIMIT 1
)
WHERE total_lessons_completed > 0;

COMMIT;

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check all users have referral codes
-- SELECT COUNT(*) as users_without_codes FROM users WHERE referral_code IS NULL;

-- Check tier distribution
-- SELECT commission_tier, COUNT(*) as count FROM users GROUP BY commission_tier ORDER BY count DESC;

-- Sample referral codes
-- SELECT id, email, referral_code, commission_tier, commission_rate FROM users LIMIT 10;



