-- Create users table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    current_day INTEGER DEFAULT 1,
    streak_days INTEGER DEFAULT 0,
    last_lesson_date DATE,
    openai_connected BOOLEAN DEFAULT FALSE,
    openai_user_id TEXT,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read their own data
CREATE POLICY "Users can read own data"
    ON public.users
    FOR SELECT
    USING (auth.uid() = id);

-- Policy: Users can update their own data
CREATE POLICY "Users can update own data"
    ON public.users
    FOR UPDATE
    USING (auth.uid() = id);

-- Policy: Users can insert their own data
CREATE POLICY "Users can insert own data"
    ON public.users
    FOR INSERT
    WITH CHECK (auth.uid() = id);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_current_day ON public.users(current_day);

-- Insert sample data for testing (remove in production)
-- This will automatically create a profile for existing auth users
INSERT INTO public.users (id, email, full_name, current_day, streak_days)
SELECT 
    id,
    email,
    raw_user_meta_data->>'full_name',
    1,
    0
FROM auth.users
WHERE NOT EXISTS (SELECT 1 FROM public.users WHERE users.id = auth.users.id)
ON CONFLICT (id) DO NOTHING;



































