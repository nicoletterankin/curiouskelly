-- VISUAL VARIANTS QUICKSTART
-- Run this in Supabase SQL Editor to enable variant generation

-- Add variant columns to visual_commons
ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS 
  style TEXT DEFAULT 'artistic';

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  complexity TEXT DEFAULT 'standard';

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  includes_text TEXT DEFAULT 'none';

-- Create index for variant queries
CREATE INDEX IF NOT EXISTS idx_vc_variants 
  ON visual_commons(day_number, phase, style, complexity, includes_text);

-- Done!
SELECT 'Variant columns added!' as status;

