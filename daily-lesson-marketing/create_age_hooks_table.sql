-- Create table for age-specific hooks
-- Run this in Supabase Dashboard > SQL Editor

CREATE TABLE IF NOT EXISTS lesson_age_hooks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    day_number INTEGER NOT NULL,
    topic TEXT NOT NULL,
    age_bucket TEXT NOT NULL CHECK (age_bucket IN ('5-7', '8-12', '13-17', '18-35', '36-60', '61+')),
    hook TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(day_number, age_bucket)
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_lesson_age_hooks_day ON lesson_age_hooks(day_number);
CREATE INDEX IF NOT EXISTS idx_lesson_age_hooks_bucket ON lesson_age_hooks(age_bucket);

-- Enable Row Level Security (but allow public read)
ALTER TABLE lesson_age_hooks ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read hooks
CREATE POLICY "Allow public read access" ON lesson_age_hooks
    FOR SELECT USING (true);

-- Allow service role to insert/update
CREATE POLICY "Allow service role full access" ON lesson_age_hooks
    FOR ALL USING (auth.role() = 'service_role');

-- Grant permissions
GRANT SELECT ON lesson_age_hooks TO anon;
GRANT SELECT ON lesson_age_hooks TO authenticated;
GRANT ALL ON lesson_age_hooks TO service_role;

-- Verify creation
SELECT 'Table lesson_age_hooks created successfully!' AS status;










