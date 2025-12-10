-- Create lesson_shards table to store all 365 lessons
CREATE TABLE IF NOT EXISTS public.lesson_shards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    day INTEGER UNIQUE NOT NULL,
    date TEXT NOT NULL,
    title TEXT NOT NULL,
    lesson_id TEXT NOT NULL,
    learning_objective TEXT,
    category TEXT,
    tags JSONB DEFAULT '[]',
    age_variants JSONB DEFAULT '[]',
    languages JSONB DEFAULT '["en"]',
    difficulty TEXT DEFAULT 'beginner',
    duration JSONB DEFAULT '{"min": 5, "max": 10}',
    has_dna BOOLEAN DEFAULT FALSE,
    dna_file TEXT,
    dna_content JSONB,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.lesson_shards ENABLE ROW LEVEL SECURITY;

-- Policy: Anyone can read lessons (public access)
CREATE POLICY "Anyone can read lessons"
    ON public.lesson_shards
    FOR SELECT
    USING (true);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_lesson_shards_day ON public.lesson_shards(day);
CREATE INDEX IF NOT EXISTS idx_lesson_shards_lesson_id ON public.lesson_shards(lesson_id);
CREATE INDEX IF NOT EXISTS idx_lesson_shards_category ON public.lesson_shards(category);

-- Create updated_at trigger
DROP TRIGGER IF EXISTS update_lesson_shards_updated_at ON public.lesson_shards;
CREATE TRIGGER update_lesson_shards_updated_at
    BEFORE UPDATE ON public.lesson_shards
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();



















