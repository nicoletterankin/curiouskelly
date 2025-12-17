-- VISUAL COMMONS QUICK START
-- Run this in Supabase SQL Editor to start generating immediately

-- 1. Create the main table
CREATE TABLE IF NOT EXISTS visual_commons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content_hash TEXT UNIQUE NOT NULL,
  day_number INTEGER NOT NULL,
  phase TEXT NOT NULL,
  topic TEXT NOT NULL,
  visual_type TEXT DEFAULT 'scene',
  age_group TEXT DEFAULT 'all',
  style TEXT DEFAULT 'default',
  storage_path TEXT NOT NULL,
  public_url TEXT NOT NULL,
  format TEXT DEFAULT 'png',
  prompt_used TEXT,
  model_used TEXT,
  generation_params JSONB DEFAULT '{}',
  estimated_cost DECIMAL(10,6) DEFAULT 0,
  generated_by UUID,
  generated_by_display_name TEXT,
  generation_source TEXT DEFAULT 'seed',
  view_count INTEGER DEFAULT 0,
  unique_learners_helped INTEGER DEFAULT 0,
  status TEXT DEFAULT 'active',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create indexes
CREATE INDEX IF NOT EXISTS idx_vc_hash ON visual_commons(content_hash);
CREATE INDEX IF NOT EXISTS idx_vc_day_phase ON visual_commons(day_number, phase);

-- 3. Enable public read
ALTER TABLE visual_commons ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read" ON visual_commons FOR SELECT USING (true);
CREATE POLICY "Service insert" ON visual_commons FOR INSERT WITH CHECK (true);
CREATE POLICY "Service update" ON visual_commons FOR UPDATE USING (true);

-- Done!
SELECT 'visual_commons table created!' as status;
