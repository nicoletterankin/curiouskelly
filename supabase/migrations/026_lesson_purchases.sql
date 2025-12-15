-- ============================================
-- LESSON PURCHASES: Pay-Per-Lesson
-- ============================================
-- Track individual lesson purchases ($1.99 each)

CREATE TABLE IF NOT EXISTS public.lesson_purchases (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Who bought what
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  day_number INTEGER NOT NULL,
  
  -- Payment details
  purchase_price DECIMAL(10,2) NOT NULL,
  currency VARCHAR(3) DEFAULT 'USD',
  stripe_payment_intent_id VARCHAR(255),
  stripe_checkout_session_id VARCHAR(255),
  
  -- Status
  status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'refunded', 'failed')),
  
  -- Timestamps
  purchased_at TIMESTAMPTZ DEFAULT NOW(),
  refunded_at TIMESTAMPTZ,
  
  -- One purchase per lesson per user
  UNIQUE(user_id, day_number)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_lesson_purchases_user ON public.lesson_purchases(user_id);
CREATE INDEX IF NOT EXISTS idx_lesson_purchases_day ON public.lesson_purchases(day_number);
CREATE INDEX IF NOT EXISTS idx_lesson_purchases_stripe ON public.lesson_purchases(stripe_payment_intent_id);

-- Enable RLS
ALTER TABLE public.lesson_purchases ENABLE ROW LEVEL SECURITY;

-- Users can view their own purchases
CREATE POLICY "Users can view own purchases" ON public.lesson_purchases
  FOR SELECT USING (auth.uid() = user_id);

-- Service role can insert (from webhook)
CREATE POLICY "Service can insert purchases" ON public.lesson_purchases
  FOR INSERT WITH CHECK (auth.role() = 'service_role' OR auth.uid() = user_id);

-- Comment
COMMENT ON TABLE public.lesson_purchases IS 'Individual lesson purchases ($1.99 each) for pay-per-lesson model';

-- ============================================
-- REGIONAL PRICES: Market-Tailored Pricing
-- ============================================

CREATE TABLE IF NOT EXISTS public.regional_prices (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Region identification
  region VARCHAR(50) NOT NULL,  -- 'us', 'in', 'br', 'eu', etc.
  
  -- Product type
  product_type VARCHAR(50) NOT NULL CHECK (product_type IN (
    'single_lesson', 'monthly', 'annual', 'lifetime', 'family',
    'gift_3mo', 'gift_6mo', 'gift_12mo', 'gift_lifetime'
  )),
  
  -- Pricing
  price DECIMAL(10,2) NOT NULL,
  currency VARCHAR(3) NOT NULL,
  
  -- Stripe integration
  stripe_price_id VARCHAR(255),
  
  -- Active/inactive
  is_active BOOLEAN DEFAULT true,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(region, product_type)
);

-- Index
CREATE INDEX IF NOT EXISTS idx_regional_prices_region ON public.regional_prices(region);
CREATE INDEX IF NOT EXISTS idx_regional_prices_active ON public.regional_prices(is_active) WHERE is_active = true;

-- Enable RLS (public read for pricing)
ALTER TABLE public.regional_prices ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view active prices" ON public.regional_prices
  FOR SELECT USING (is_active = true);

-- Seed with initial prices
INSERT INTO public.regional_prices (region, product_type, price, currency) VALUES
  ('us', 'single_lesson', 1.99, 'USD'),
  ('us', 'monthly', 9.99, 'USD'),
  ('us', 'annual', 79.00, 'USD'),
  ('us', 'lifetime', 199.00, 'USD'),
  ('in', 'single_lesson', 0.49, 'USD'),
  ('in', 'monthly', 2.99, 'USD'),
  ('in', 'annual', 24.99, 'USD'),
  ('br', 'single_lesson', 0.99, 'USD'),
  ('br', 'monthly', 4.99, 'USD'),
  ('eu', 'single_lesson', 1.99, 'EUR'),
  ('eu', 'monthly', 8.99, 'EUR')
ON CONFLICT (region, product_type) DO NOTHING;

-- Comment
COMMENT ON TABLE public.regional_prices IS 'Market-tailored pricing per region';

-- ============================================
-- USER PRICING TIER: Personalized Pricing
-- ============================================

CREATE TABLE IF NOT EXISTS public.user_pricing_tiers (
  user_id UUID PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Detected region
  region VARCHAR(50) DEFAULT 'us',
  detected_country VARCHAR(2),  -- ISO country code
  
  -- Custom tier
  tier VARCHAR(50) DEFAULT 'standard' CHECK (tier IN ('standard', 'student', 'educator', 'nonprofit', 'custom')),
  
  -- Discounts
  custom_discount_pct INTEGER DEFAULT 0 CHECK (custom_discount_pct >= 0 AND custom_discount_pct <= 100),
  discount_reason TEXT,
  discount_expires_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.user_pricing_tiers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own pricing tier" ON public.user_pricing_tiers
  FOR SELECT USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_user_pricing_tiers_updated_at
  BEFORE UPDATE ON public.user_pricing_tiers
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comment
COMMENT ON TABLE public.user_pricing_tiers IS 'Per-user pricing tier and discounts';
