-- ═══════════════════════════════════════════════════════════════════
-- LESSON COMMENTS TABLE
-- Pre-generated AI comments for each lesson phase
-- Per CURIOUS-KELLY-COMPLETE-SYSTEM-SPEC.md
-- ═══════════════════════════════════════════════════════════════════

-- Create the lesson_comments table
CREATE TABLE IF NOT EXISTS lesson_comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Lesson/Phase targeting
  lesson_day INT NOT NULL,                    -- 1-365
  phase TEXT NOT NULL,                        -- 'welcome', 'q1', 'q2', 'q3', 'hook', 'complete'
  option_context TEXT,                        -- NULL or 'A', 'B', 'C' (for option-specific comments)
  
  -- Persona info
  persona_name TEXT NOT NULL,                 -- "Hans", "Yuki", "Maria"
  persona_country TEXT NOT NULL,              -- "DE", "JP", "BR"
  persona_flag TEXT NOT NULL,                 -- "🇩🇪", "🇯🇵", "🇧🇷"
  
  -- Comment content
  comment_text TEXT NOT NULL,
  comment_type TEXT NOT NULL,                 -- 'insight', 'reaction', 'question', 'funny'
  
  -- Metadata
  age_appropriate_min INT DEFAULT 2,
  age_appropriate_max INT DEFAULT 102,
  language TEXT DEFAULT 'en',
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast lookup by lesson and phase
CREATE INDEX IF NOT EXISTS idx_lesson_comments_lookup 
ON lesson_comments(lesson_day, phase, option_context);

-- Index for phase-only queries
CREATE INDEX IF NOT EXISTS idx_lesson_comments_phase
ON lesson_comments(phase);

-- Index for lesson-day queries
CREATE INDEX IF NOT EXISTS idx_lesson_comments_day
ON lesson_comments(lesson_day);

-- Comment for documentation
COMMENT ON TABLE lesson_comments IS 'Pre-generated AI comments displayed during lessons, organized by day, phase, and optionally by selected option';

COMMENT ON COLUMN lesson_comments.phase IS 'Phase ID: welcome, q1, q2, q3, hook, complete';
COMMENT ON COLUMN lesson_comments.option_context IS 'NULL for general phase comments, or A/B/C for option-specific reactions';
COMMENT ON COLUMN lesson_comments.comment_type IS 'Type of comment: insight (educational), reaction (emotional), question (prompts thinking), funny (humor)';

-- ═══════════════════════════════════════════════════════════════════
-- ROW LEVEL SECURITY (RLS)
-- ═══════════════════════════════════════════════════════════════════

ALTER TABLE lesson_comments ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read comments (public content)
CREATE POLICY "Anyone can read lesson comments"
ON lesson_comments FOR SELECT
TO authenticated, anon
USING (true);

-- Only service role can insert/update (batch generation)
CREATE POLICY "Service role can manage comments"
ON lesson_comments FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- ═══════════════════════════════════════════════════════════════════
-- SAMPLE DATA (Day 1 example)
-- ═══════════════════════════════════════════════════════════════════

-- Welcome phase comments
INSERT INTO lesson_comments (lesson_day, phase, option_context, persona_name, persona_country, persona_flag, comment_text, comment_type) VALUES
(1, 'welcome', NULL, 'Maria', 'BR', '🇧🇷', 'Good morning everyone! ☀️', 'reaction'),
(1, 'welcome', NULL, 'James', 'GB', '🇬🇧', 'Ready to learn something new!', 'reaction'),
(1, 'welcome', NULL, 'Yuki', 'JP', '🇯🇵', 'おはよう! Let''s do this', 'reaction'),
(1, 'welcome', NULL, 'Hans', 'DE', '🇩🇪', 'Kelly is the best teacher 💙', 'reaction'),
(1, 'welcome', NULL, 'Sofia', 'MX', '🇲🇽', 'Day 1 of 365! Here we go', 'reaction');

-- Q1 phase comments
INSERT INTO lesson_comments (lesson_day, phase, option_context, persona_name, persona_country, persona_flag, comment_text, comment_type) VALUES
(1, 'q1', NULL, 'Emma', 'US', '🇺🇸', 'Hmm this is a good question 🤔', 'engagement'),
(1, 'q1', NULL, 'Lucas', 'FR', '🇫🇷', 'I think I know this one!', 'reaction'),
(1, 'q1', NULL, 'Priya', 'IN', '🇮🇳', 'Wait let me think...', 'engagement');

-- Option-specific comments (when user hovers/selects A or B)
INSERT INTO lesson_comments (lesson_day, phase, option_context, persona_name, persona_country, persona_flag, comment_text, comment_type) VALUES
(1, 'q1', 'A', 'Ahmed', 'EG', '🇪🇬', 'That''s what I picked! 🙋', 'reaction'),
(1, 'q1', 'B', 'Chen', 'CN', '🇨🇳', 'B makes sense to me', 'reaction');

-- Hook phase (wisdom reveal)
INSERT INTO lesson_comments (lesson_day, phase, option_context, persona_name, persona_country, persona_flag, comment_text, comment_type) VALUES
(1, 'hook', NULL, 'Isabella', 'IT', '🇮🇹', 'Mind = BLOWN 🤯', 'reaction'),
(1, 'hook', NULL, 'Kofi', 'GH', '🇬🇭', 'I never thought of it that way!', 'insight'),
(1, 'hook', NULL, 'Mei', 'TW', '🇹🇼', 'This is so deep 💎', 'reaction'),
(1, 'hook', NULL, 'Carlos', 'AR', '🇦🇷', 'Screenshotting this 📸', 'reaction');

-- Complete phase
INSERT INTO lesson_comments (lesson_day, phase, option_context, persona_name, persona_country, persona_flag, comment_text, comment_type) VALUES
(1, 'complete', NULL, 'Sarah', 'CA', '🇨🇦', 'Great first lesson! 🎉', 'reaction'),
(1, 'complete', NULL, 'Jin', 'KR', '🇰🇷', 'See you tomorrow! 👋', 'reaction'),
(1, 'complete', NULL, 'Olga', 'UA', '🇺🇦', 'Day 1 complete! 364 to go 💪', 'reaction');

-- ═══════════════════════════════════════════════════════════════════
-- HELPER FUNCTIONS
-- ═══════════════════════════════════════════════════════════════════

-- Function to get comments for a specific phase
CREATE OR REPLACE FUNCTION get_lesson_comments(
  p_lesson_day INT,
  p_phase TEXT,
  p_option_context TEXT DEFAULT NULL,
  p_limit INT DEFAULT 10
)
RETURNS SETOF lesson_comments
LANGUAGE sql
STABLE
AS $$
  SELECT * FROM lesson_comments
  WHERE lesson_day = p_lesson_day
    AND phase = p_phase
    AND (p_option_context IS NULL OR option_context IS NULL OR option_context = p_option_context)
  ORDER BY random()
  LIMIT p_limit;
$$;

-- Function to get random comments for any phase (fallback)
CREATE OR REPLACE FUNCTION get_random_comments(
  p_phase TEXT,
  p_limit INT DEFAULT 5
)
RETURNS SETOF lesson_comments
LANGUAGE sql
STABLE
AS $$
  SELECT * FROM lesson_comments
  WHERE phase = p_phase
    AND option_context IS NULL
  ORDER BY random()
  LIMIT p_limit;
$$;

COMMENT ON FUNCTION get_lesson_comments IS 'Get comments for a specific lesson day and phase, optionally filtered by option context';
COMMENT ON FUNCTION get_random_comments IS 'Get random comments for a phase (fallback when lesson-specific not available)';





