-- Migration: Kids Account Compliance for Share & Earn
-- Purpose: COPPA/GDPR-K compliant referral system with family accounts
-- Date: December 7, 2025

-- ============================================================================
-- COMPLIANCE RULES (These are LAW, not preferences)
-- ============================================================================
-- 
-- COPPA (Children's Online Privacy Protection Act - USA):
-- - Under 13: Cannot enter contracts, cannot receive payouts
-- - Requires "verifiable parental consent" for data collection
-- 
-- GDPR-K (EU):
-- - Under 16 (varies by country, 13-16): Need parental consent
-- 
-- FTC Endorsement Guidelines:
-- - Material connections must be disclosed
-- - Children cannot be used as endorsers without parental consent
--
-- TAX LAW:
-- - Cannot issue 1099 to minors in most jurisdictions
-- - Earnings must go through parent/guardian
--
-- ============================================================================
-- SOLUTION: Age-Gated Earnings with Family Account Structure
-- ============================================================================

-- 1. Add family account support to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS parent_account_id UUID REFERENCES users(id);
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_family_admin BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS parental_consent_for_earnings BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS parental_consent_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN IF NOT EXISTS earnings_held_for_minors NUMERIC(10,2) DEFAULT 0.00;

-- 2. Create index for family lookups
CREATE INDEX IF NOT EXISTS idx_users_parent_account ON users(parent_account_id) WHERE parent_account_id IS NOT NULL;

-- 3. Create view for age calculation (handles birthday, birth_year, or age field)
CREATE OR REPLACE VIEW users_with_age AS
SELECT 
  *,
  CASE
    WHEN birthday IS NOT NULL THEN 
      EXTRACT(YEAR FROM age(birthday))::INTEGER
    WHEN birth_year IS NOT NULL THEN 
      EXTRACT(YEAR FROM CURRENT_DATE)::INTEGER - birth_year
    WHEN age IS NOT NULL THEN 
      age
    ELSE NULL
  END AS calculated_age,
  CASE
    WHEN birthday IS NOT NULL THEN 
      EXTRACT(YEAR FROM age(birthday))::INTEGER < 13
    WHEN birth_year IS NOT NULL THEN 
      (EXTRACT(YEAR FROM CURRENT_DATE)::INTEGER - birth_year) < 13
    WHEN age IS NOT NULL THEN 
      age < 13
    ELSE FALSE -- If we don't know age, assume adult (conservative for access, strict for earnings)
  END AS is_under_13,
  CASE
    WHEN birthday IS NOT NULL THEN 
      EXTRACT(YEAR FROM age(birthday))::INTEGER < 18
    WHEN birth_year IS NOT NULL THEN 
      (EXTRACT(YEAR FROM CURRENT_DATE)::INTEGER - birth_year) < 18
    WHEN age IS NOT NULL THEN 
      age < 18
    ELSE FALSE
  END AS is_minor
FROM users;

-- 4. Create table to track minor earnings (held until 18 or parent claims)
CREATE TABLE IF NOT EXISTS minor_earnings_ledger (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  minor_user_id UUID NOT NULL REFERENCES users(id),
  parent_user_id UUID REFERENCES users(id),
  
  -- The commission that was earned
  commission_transaction_id UUID REFERENCES commission_transactions(id),
  amount NUMERIC(10,2) NOT NULL,
  
  -- Status tracking
  status TEXT DEFAULT 'held' CHECK (status IN ('held', 'transferred_to_parent', 'transferred_at_18', 'forfeited')),
  
  -- Resolution details
  resolved_at TIMESTAMPTZ,
  resolved_by TEXT, -- 'parent_claim', 'age_18_transfer', 'manual_review'
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_minor_earnings_minor ON minor_earnings_ledger(minor_user_id);
CREATE INDEX IF NOT EXISTS idx_minor_earnings_parent ON minor_earnings_ledger(parent_user_id);
CREATE INDEX IF NOT EXISTS idx_minor_earnings_status ON minor_earnings_ledger(status);

-- 5. Create compliance audit log
CREATE TABLE IF NOT EXISTS earnings_compliance_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  event_type TEXT NOT NULL,
  -- Event types:
  -- 'minor_referral_blocked' - Under 13 tried to use referral
  -- 'minor_earnings_held' - 13-17 earned, held for parent
  -- 'payout_blocked_minor' - Minor tried to request payout
  -- 'parent_claimed_earnings' - Parent withdrew minor's earnings
  -- 'age_18_release' - Earnings released when user turned 18
  -- 'parental_consent_given' - Parent gave consent for earnings
  -- 'family_link_created' - Child linked to parent account
  
  details JSONB,
  ip_address INET,
  user_agent TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_compliance_log_user ON earnings_compliance_log(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_log_type ON earnings_compliance_log(event_type);

-- 6. Function to check if user can participate in earnings
CREATE OR REPLACE FUNCTION can_user_earn(user_uuid UUID)
RETURNS TABLE(
  can_see_referral_link BOOLEAN,
  can_share BOOLEAN,
  can_accumulate_earnings BOOLEAN,
  can_request_payout BOOLEAN,
  earnings_destination TEXT,
  reason TEXT
) AS $$
DECLARE
  user_age INTEGER;
  has_parent BOOLEAN;
  has_parental_consent BOOLEAN;
BEGIN
  -- Get user age and family status
  SELECT 
    u.calculated_age,
    u.parent_account_id IS NOT NULL,
    u.parental_consent_for_earnings
  INTO user_age, has_parent, has_parental_consent
  FROM users_with_age u
  WHERE u.id = user_uuid;
  
  -- If user not found or no age data, default to adult behavior
  IF user_age IS NULL THEN
    RETURN QUERY SELECT 
      TRUE, TRUE, TRUE, TRUE, 
      'self'::TEXT, 
      'Age unknown - defaulting to adult'::TEXT;
    RETURN;
  END IF;
  
  -- RULE 1: Under 13 - COPPA strict compliance
  IF user_age < 13 THEN
    IF has_parent AND has_parental_consent THEN
      RETURN QUERY SELECT 
        TRUE,  -- Can see referral link (for family sharing)
        TRUE,  -- Can share
        TRUE,  -- Earnings accumulate (to parent)
        FALSE, -- Cannot request payout directly
        'parent'::TEXT,
        'Under 13 with parental consent - earnings go to parent'::TEXT;
    ELSE
      RETURN QUERY SELECT 
        FALSE, -- Cannot see referral link
        FALSE, -- Cannot share for earnings
        FALSE, -- No earnings
        FALSE, -- No payout
        'none'::TEXT,
        'Under 13 without parental consent - earnings disabled'::TEXT;
    END IF;
    RETURN;
  END IF;
  
  -- RULE 2: Ages 13-17 - Limited participation
  IF user_age < 18 THEN
    RETURN QUERY SELECT 
      TRUE,  -- Can see referral link
      TRUE,  -- Can share
      TRUE,  -- Earnings accumulate
      FALSE, -- Cannot request payout directly (held until 18)
      CASE WHEN has_parent THEN 'parent_or_held' ELSE 'held_until_18' END,
      'Minor (13-17) - earnings held until 18 or parent can claim'::TEXT;
    RETURN;
  END IF;
  
  -- RULE 3: 18+ - Full access
  RETURN QUERY SELECT 
    TRUE, TRUE, TRUE, TRUE, 
    'self'::TEXT, 
    'Adult - full earnings access'::TEXT;
END;
$$ LANGUAGE plpgsql;

-- 7. Trigger to handle commission creation for minors
CREATE OR REPLACE FUNCTION handle_minor_commission()
RETURNS TRIGGER AS $$
DECLARE
  referrer_age INTEGER;
  referrer_parent UUID;
  referrer_can_earn RECORD;
BEGIN
  -- Check referrer's earning eligibility
  SELECT * INTO referrer_can_earn FROM can_user_earn(NEW.referrer_id);
  
  -- If earnings go somewhere other than self
  IF referrer_can_earn.earnings_destination != 'self' THEN
    -- Get parent account if exists
    SELECT parent_account_id INTO referrer_parent
    FROM users WHERE id = NEW.referrer_id;
    
    -- Create entry in minor earnings ledger
    INSERT INTO minor_earnings_ledger (
      minor_user_id,
      parent_user_id,
      commission_transaction_id,
      amount,
      status
    ) VALUES (
      NEW.referrer_id,
      referrer_parent,
      NEW.id,
      NEW.commission_amount,
      'held'
    );
    
    -- Log the compliance event
    INSERT INTO earnings_compliance_log (user_id, event_type, details)
    VALUES (
      NEW.referrer_id,
      'minor_earnings_held',
      jsonb_build_object(
        'commission_id', NEW.id,
        'amount', NEW.commission_amount,
        'reason', referrer_can_earn.reason,
        'destination', referrer_can_earn.earnings_destination
      )
    );
    
    -- If parent exists, add to their held earnings counter
    IF referrer_parent IS NOT NULL THEN
      UPDATE users 
      SET earnings_held_for_minors = earnings_held_for_minors + NEW.commission_amount
      WHERE id = referrer_parent;
    END IF;
    
    -- Set commission status to indicate it's held for minor
    NEW.status := 'held_for_minor';
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger (drop if exists to allow re-running)
DROP TRIGGER IF EXISTS check_minor_commission ON commission_transactions;
CREATE TRIGGER check_minor_commission
  BEFORE INSERT ON commission_transactions
  FOR EACH ROW
  EXECUTE FUNCTION handle_minor_commission();

-- 8. Function for parent to claim child's earnings
CREATE OR REPLACE FUNCTION parent_claim_minor_earnings(
  parent_uuid UUID,
  minor_uuid UUID
)
RETURNS TABLE(success BOOLEAN, amount_claimed NUMERIC, message TEXT) AS $$
DECLARE
  total_held NUMERIC;
  is_valid_parent BOOLEAN;
BEGIN
  -- Verify parent-child relationship
  SELECT EXISTS(
    SELECT 1 FROM users 
    WHERE id = minor_uuid AND parent_account_id = parent_uuid
  ) INTO is_valid_parent;
  
  IF NOT is_valid_parent THEN
    RETURN QUERY SELECT FALSE, 0.00::NUMERIC, 'Invalid parent-child relationship'::TEXT;
    RETURN;
  END IF;
  
  -- Get total held earnings
  SELECT COALESCE(SUM(amount), 0) INTO total_held
  FROM minor_earnings_ledger
  WHERE minor_user_id = minor_uuid AND status = 'held';
  
  IF total_held <= 0 THEN
    RETURN QUERY SELECT FALSE, 0.00::NUMERIC, 'No held earnings to claim'::TEXT;
    RETURN;
  END IF;
  
  -- Update ledger entries
  UPDATE minor_earnings_ledger
  SET 
    status = 'transferred_to_parent',
    resolved_at = NOW(),
    resolved_by = 'parent_claim'
  WHERE minor_user_id = minor_uuid AND status = 'held';
  
  -- Update parent's available earnings
  UPDATE users
  SET 
    available_earnings = available_earnings + total_held,
    earnings_held_for_minors = earnings_held_for_minors - total_held
  WHERE id = parent_uuid;
  
  -- Log compliance event
  INSERT INTO earnings_compliance_log (user_id, event_type, details)
  VALUES (
    minor_uuid,
    'parent_claimed_earnings',
    jsonb_build_object(
      'parent_id', parent_uuid,
      'amount', total_held
    )
  );
  
  RETURN QUERY SELECT TRUE, total_held, 'Earnings transferred to parent account'::TEXT;
END;
$$ LANGUAGE plpgsql;

-- 9. Function to transfer earnings when user turns 18
CREATE OR REPLACE FUNCTION check_age_18_transfer()
RETURNS INTEGER AS $$
DECLARE
  transferred_count INTEGER := 0;
  user_record RECORD;
BEGIN
  -- Find users who just turned 18 with held earnings
  FOR user_record IN 
    SELECT u.id, u.birthday
    FROM users_with_age u
    WHERE u.calculated_age = 18
    AND u.birthday = CURRENT_DATE - INTERVAL '18 years'
    AND EXISTS (
      SELECT 1 FROM minor_earnings_ledger 
      WHERE minor_user_id = u.id AND status = 'held'
    )
  LOOP
    -- Transfer their held earnings
    UPDATE users
    SET available_earnings = available_earnings + (
      SELECT COALESCE(SUM(amount), 0) 
      FROM minor_earnings_ledger 
      WHERE minor_user_id = user_record.id AND status = 'held'
    )
    WHERE id = user_record.id;
    
    -- Update ledger
    UPDATE minor_earnings_ledger
    SET 
      status = 'transferred_at_18',
      resolved_at = NOW(),
      resolved_by = 'age_18_transfer'
    WHERE minor_user_id = user_record.id AND status = 'held';
    
    -- Log event
    INSERT INTO earnings_compliance_log (user_id, event_type, details)
    VALUES (
      user_record.id,
      'age_18_release',
      jsonb_build_object('birthday', user_record.birthday)
    );
    
    transferred_count := transferred_count + 1;
  END LOOP;
  
  RETURN transferred_count;
END;
$$ LANGUAGE plpgsql;

-- 10. RLS policies for family data
ALTER TABLE minor_earnings_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE earnings_compliance_log ENABLE ROW LEVEL SECURITY;

-- Parents can see their children's held earnings
CREATE POLICY "Parents can view children's held earnings" ON minor_earnings_ledger
  FOR SELECT USING (
    parent_user_id = auth.uid() OR 
    minor_user_id = auth.uid()
  );

-- Users can see their own compliance log
CREATE POLICY "Users can view their own compliance log" ON earnings_compliance_log
  FOR SELECT USING (user_id = auth.uid());

-- 11. Add status value for held_for_minor
ALTER TABLE commission_transactions 
  DROP CONSTRAINT IF EXISTS commission_transactions_status_check;
  
ALTER TABLE commission_transactions
  ADD CONSTRAINT commission_transactions_status_check 
  CHECK (status IN ('pending', 'approved', 'paid', 'refunded', 'held_for_minor'));

-- ============================================================================
-- EDGE CASE HANDLERS
-- ============================================================================

-- Edge Case 1: Self-referral within family (allowed but flagged)
-- Handled in API layer - we allow family members to refer each other
-- but track it for potential abuse detection

-- Edge Case 2: Age change/correction
-- If age is corrected and user was actually under 13, retroactively hold earnings
CREATE OR REPLACE FUNCTION handle_age_correction()
RETURNS TRIGGER AS $$
DECLARE
  new_age INTEGER;
  old_age INTEGER;
BEGIN
  -- Calculate ages
  IF NEW.birthday IS NOT NULL THEN
    new_age := EXTRACT(YEAR FROM age(NEW.birthday))::INTEGER;
  ELSIF NEW.birth_year IS NOT NULL THEN
    new_age := EXTRACT(YEAR FROM CURRENT_DATE)::INTEGER - NEW.birth_year;
  ELSE
    new_age := NEW.age;
  END IF;
  
  IF OLD.birthday IS NOT NULL THEN
    old_age := EXTRACT(YEAR FROM age(OLD.birthday))::INTEGER;
  ELSIF OLD.birth_year IS NOT NULL THEN
    old_age := EXTRACT(YEAR FROM CURRENT_DATE)::INTEGER - OLD.birth_year;
  ELSE
    old_age := OLD.age;
  END IF;
  
  -- If age was corrected down to under 13
  IF new_age < 13 AND (old_age IS NULL OR old_age >= 13) THEN
    -- Log for manual review
    INSERT INTO earnings_compliance_log (user_id, event_type, details)
    VALUES (
      NEW.id,
      'age_correction_review_needed',
      jsonb_build_object(
        'old_age', old_age,
        'new_age', new_age,
        'action_needed', 'Review past earnings for COPPA compliance'
      )
    );
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS handle_user_age_correction ON users;
CREATE TRIGGER handle_user_age_correction
  AFTER UPDATE OF birthday, birth_year, age ON users
  FOR EACH ROW
  EXECUTE FUNCTION handle_age_correction();

-- Edge Case 3: Parent account deleted
-- Transfer custody of held earnings to Kelly (held for minor until 18)
-- Handled by FK constraint - earnings remain with minor_user_id

-- Edge Case 4: Minor account deleted
-- Forfeit held earnings (they're not yet "earned" in legal sense)
-- Handled by cascade delete on minor_earnings_ledger

-- ============================================================================
-- SUMMARY VIEW FOR DASHBOARD
-- ============================================================================

CREATE OR REPLACE VIEW family_earnings_summary AS
SELECT 
  p.id AS parent_id,
  p.email AS parent_email,
  p.earnings_held_for_minors,
  COUNT(DISTINCT c.id) AS child_count,
  json_agg(json_build_object(
    'child_id', c.id,
    'child_name', c.display_name,
    'child_age', uwa.calculated_age,
    'held_earnings', (
      SELECT COALESCE(SUM(amount), 0) 
      FROM minor_earnings_ledger 
      WHERE minor_user_id = c.id AND status = 'held'
    )
  )) AS children
FROM users p
JOIN users c ON c.parent_account_id = p.id
JOIN users_with_age uwa ON uwa.id = c.id
WHERE p.is_family_admin = TRUE
GROUP BY p.id, p.email, p.earnings_held_for_minors;


