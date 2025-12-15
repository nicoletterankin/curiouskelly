-- ============================================
-- GIFT CODES: Gift Subscription Management
-- ============================================

CREATE TABLE IF NOT EXISTS public.gift_codes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- The code itself (12 alphanumeric characters)
  code VARCHAR(12) NOT NULL UNIQUE,
  
  -- Purchase information
  stripe_checkout_session_id VARCHAR(255),
  stripe_payment_intent_id VARCHAR(255),
  purchase_price DECIMAL(10,2),
  currency VARCHAR(3) DEFAULT 'USD',
  
  -- Gift details
  duration_months INTEGER NOT NULL DEFAULT 12, -- 3, 6, 12, or 0 for lifetime
  plan_type VARCHAR(20) NOT NULL, -- 'gift_3mo', 'gift_6mo', 'gift_12mo', 'gift_lifetime'
  
  -- Gifter info
  gifter_email VARCHAR(255),
  gifter_name VARCHAR(255),
  message TEXT,
  
  -- Recipient info
  recipient_email VARCHAR(255),
  delivery_date DATE,
  
  -- Redemption tracking
  redeemed_at TIMESTAMPTZ,
  redeemed_by_email VARCHAR(255),
  redeemed_by_user_id UUID REFERENCES public.users(id),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ, -- Optional expiry for the code itself
  
  -- Status
  status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'redeemed', 'expired', 'cancelled'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_gift_codes_code ON public.gift_codes(code);
CREATE INDEX IF NOT EXISTS idx_gift_codes_recipient ON public.gift_codes(recipient_email);
CREATE INDEX IF NOT EXISTS idx_gift_codes_status ON public.gift_codes(status);
CREATE INDEX IF NOT EXISTS idx_gift_codes_stripe ON public.gift_codes(stripe_checkout_session_id);

-- Enable RLS
ALTER TABLE public.gift_codes ENABLE ROW LEVEL SECURITY;

-- Service role can manage gift codes
CREATE POLICY "Service can manage gift codes" ON public.gift_codes
  FOR ALL USING (auth.role() = 'service_role');

-- Users can view their own redeemed gifts
CREATE POLICY "Users can view own redeemed gifts" ON public.gift_codes
  FOR SELECT USING (auth.uid() = redeemed_by_user_id);

-- Function to generate random gift code
CREATE OR REPLACE FUNCTION generate_gift_code()
RETURNS VARCHAR(12) AS $$
DECLARE
  chars VARCHAR(36) := 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; -- Removed confusing chars: I, O, 0, 1
  code VARCHAR(12) := '';
  i INTEGER;
BEGIN
  FOR i IN 1..12 LOOP
    code := code || substr(chars, floor(random() * length(chars) + 1)::int, 1);
  END LOOP;
  RETURN code;
END;
$$ LANGUAGE plpgsql;

-- Comment
COMMENT ON TABLE public.gift_codes IS 'Gift subscription codes for purchase and redemption';
