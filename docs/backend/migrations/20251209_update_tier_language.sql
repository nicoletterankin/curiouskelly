-- ============================================================
-- UPDATE COMMISSION TIER LANGUAGE
-- Date: 2024-12-09
-- Purpose: Replace "Learner" with more natural, Kelly-aligned language
-- ============================================================

-- PHILOSOPHY:
-- Kelly never calls people "students," "users," or "learners" in conversation.
-- She uses "you," "we," "friend," and refers to people as "explorers" or companions.
-- Commission tier names are USER-FACING in the earnings dashboard,
-- so they should match Kelly's warm, natural voice.

-- ============================================================
-- OPTION 1: EXPLORER-BASED (Recommended)
-- Aligns with "The Explorer" archetype and Kelly's adventurous spirit
-- ============================================================

UPDATE commission_tiers 
SET 
  display_name = 'New Explorer',
  perks = '["Share & Earn access", "Welcome to the journey!"]'
WHERE tier_name = 'new_learner';

UPDATE commission_tiers 
SET 
  display_name = 'Active Explorer',
  perks = '["Weekly earnings email", "You''re making progress!"]'
WHERE tier_name = 'active_learner';

UPDATE commission_tiers 
SET 
  display_name = 'Committed Explorer',
  perks = '["Monthly earnings report", "Priority support", "You''re dedicated!"]'
WHERE tier_name = 'committed_learner';

UPDATE commission_tiers 
SET 
  display_name = 'Dedicated Explorer',
  perks = '["Dedicated dashboard", "Custom share links", "You''re unstoppable!"]'
WHERE tier_name = 'dedicated_learner';

UPDATE commission_tiers 
SET 
  display_name = 'Complete Explorer',
  perks = '["Kelly Companion badge", "Direct payout", "You did it!"]'
WHERE tier_name = 'complete_learner';

UPDATE commission_tiers 
SET 
  display_name = 'Legendary Explorer',
  perks = '["VIP status", "Founding member perks", "API access", "You''re a legend!"]'
WHERE tier_name = 'legendary_learner';

-- ============================================================
-- Update bonus program descriptions
-- Replace "learners" with "friends" or "people"
-- ============================================================

UPDATE bonus_programs 
SET description = 'Bonus for referring 10+ friends'
WHERE program_name = 'community_builder' 
  AND description LIKE '%learners%';

-- Update any other bonus descriptions that mention "learners"
UPDATE bonus_programs 
SET description = REPLACE(description, 'learners', 'friends')
WHERE description LIKE '%learners%';

UPDATE bonus_programs 
SET description = REPLACE(description, 'Learners', 'Friends')
WHERE description LIKE '%Learners%';

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check updated tier names
-- SELECT tier_name, display_name, perks FROM commission_tiers ORDER BY sort_order;

-- Check for any remaining "learner" references in user-facing text
-- SELECT * FROM bonus_programs WHERE description LIKE '%learner%';

-- ============================================================
-- ROLLBACK (if needed)
-- ============================================================

-- To rollback to original names:
/*
UPDATE commission_tiers SET display_name = 'New Learner' WHERE tier_name = 'new_learner';
UPDATE commission_tiers SET display_name = 'Active Learner' WHERE tier_name = 'active_learner';
UPDATE commission_tiers SET display_name = 'Committed Learner' WHERE tier_name = 'committed_learner';
UPDATE commission_tiers SET display_name = 'Dedicated Learner' WHERE tier_name = 'dedicated_learner';
UPDATE commission_tiers SET display_name = 'Complete Learner' WHERE tier_name = 'complete_learner';
UPDATE commission_tiers SET display_name = 'Legendary Learner' WHERE tier_name = 'legendary_learner';
*/

-- ============================================================
-- NOTES
-- ============================================================

-- Technical names (tier_name column) remain unchanged:
-- - 'new_learner', 'active_learner', etc.
-- - These are internal identifiers used in code
-- - Changing them would break existing references

-- Table names remain unchanged:
-- - commission_tiers, bonus_programs, etc.
-- - These are developer-facing, not user-facing

-- Only display_name and description fields are updated:
-- - These appear in the user interface
-- - They should match Kelly's conversational voice





