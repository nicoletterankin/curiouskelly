-- ============================================================
-- COMPLETE 5-PHASE VIDEO JOURNEY SCHEMA
-- Date: 2024-12-09
-- Purpose: Support full video matrix with responses for all options
-- ============================================================

-- PHILOSOPHY:
-- Every phase needs:
-- 1. Main script video (Kelly presents the content)
-- 2. Response videos for each option (Kelly responds to A/B/C)
-- This gives learners ALL paths, not just one path.

-- ============================================================
-- PART 1: Update lesson_atoms content structure
-- ============================================================

-- The content JSONB now needs to include video URLs
-- New structure:
-- {
--   "script": "Main content text",
--   "script_video_url": "path/to/main_video.mp4",
--   "options": [
--     {
--       "text": "Option A text",
--       "letter": "A",
--       "quality": "good",
--       "response": "Kelly's response text",
--       "response_video_url": "path/to/response_A.mp4"
--     },
--     ...
--   ]
-- }

-- No schema change needed - JSONB is flexible
-- But we'll add helper functions to validate structure

-- ============================================================
-- PART 2: User lesson progress tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS user_lesson_paths (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  core_lesson_id UUID NOT NULL REFERENCES core_lessons(id) ON DELETE CASCADE,
  archetype TEXT NOT NULL,
  phase TEXT NOT NULL CHECK (phase IN ('Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom')),
  option_selected TEXT CHECK (option_selected IN ('A', 'B', 'C', 'auto', NULL)),
  completed_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Track which videos were watched
  watched_main_video BOOLEAN DEFAULT FALSE,
  watched_response_video BOOLEAN DEFAULT FALSE,
  
  -- Performance tracking
  time_spent_seconds INTEGER,
  
  UNIQUE(user_id, core_lesson_id, archetype, phase)
);

CREATE INDEX idx_user_lesson_paths_user ON user_lesson_paths(user_id);
CREATE INDEX idx_user_lesson_paths_lesson ON user_lesson_paths(core_lesson_id);

COMMENT ON TABLE user_lesson_paths IS 'Tracks which paths users take through lessons and which videos they watch';

-- ============================================================
-- PART 3: User lesson preferences
-- ============================================================

-- Add lesson experience settings to users table
DO $$
BEGIN
  -- Autoplay mode: automatically select "best" option and continue
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lesson_autoplay') THEN
    ALTER TABLE users ADD COLUMN lesson_autoplay BOOLEAN DEFAULT FALSE;
  END IF;
  
  -- Show progress indicator (Hook → Fact1 → Fact2 → Fact3 → Wisdom)
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lesson_show_progress') THEN
    ALTER TABLE users ADD COLUMN lesson_show_progress BOOLEAN DEFAULT TRUE;
  END IF;
  
  -- Transition delay between phases (milliseconds)
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lesson_transition_delay') THEN
    ALTER TABLE users ADD COLUMN lesson_transition_delay INTEGER DEFAULT 2000 CHECK (lesson_transition_delay BETWEEN 500 AND 5000);
  END IF;
  
  -- Show option quality badges ("good", "best", "redirect")
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lesson_show_quality_badges') THEN
    ALTER TABLE users ADD COLUMN lesson_show_quality_badges BOOLEAN DEFAULT TRUE;
  END IF;
  
  -- Enable replay mode (explore all paths)
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'lesson_replay_mode') THEN
    ALTER TABLE users ADD COLUMN lesson_replay_mode BOOLEAN DEFAULT FALSE;
  END IF;
END $$;

COMMENT ON COLUMN users.lesson_autoplay IS 'Auto-select best option and continue without user interaction';
COMMENT ON COLUMN users.lesson_show_progress IS 'Show visual progress indicator through 5 phases';
COMMENT ON COLUMN users.lesson_transition_delay IS 'Delay in milliseconds between phases (500-5000)';
COMMENT ON COLUMN users.lesson_show_quality_badges IS 'Show hints about option quality (good/best/redirect)';
COMMENT ON COLUMN users.lesson_replay_mode IS 'Allow exploring all paths and responses';

-- ============================================================
-- PART 4: Video generation tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS lesson_video_generation_status (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  core_lesson_id UUID NOT NULL REFERENCES core_lessons(id) ON DELETE CASCADE,
  archetype TEXT NOT NULL,
  phase TEXT NOT NULL CHECK (phase IN ('Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom')),
  video_type TEXT NOT NULL CHECK (video_type IN ('main', 'response_A', 'response_B', 'response_C')),
  
  -- Generation status
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'generating', 'completed', 'failed')),
  video_url TEXT,
  error_message TEXT,
  
  -- Metadata
  duration_seconds NUMERIC(6,2),
  file_size_bytes BIGINT,
  resolution TEXT,
  
  -- Timestamps
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(core_lesson_id, archetype, phase, video_type)
);

CREATE INDEX idx_video_generation_status ON lesson_video_generation_status(core_lesson_id, archetype, phase);
CREATE INDEX idx_video_generation_pending ON lesson_video_generation_status(status) WHERE status = 'pending';

COMMENT ON TABLE lesson_video_generation_status IS 'Tracks video generation progress for the complete 5-phase journey';

-- ============================================================
-- PART 5: Helper functions
-- ============================================================

-- Function to get video count for a lesson
CREATE OR REPLACE FUNCTION get_lesson_video_count(lesson_id UUID)
RETURNS TABLE (
  total_videos INTEGER,
  completed_videos INTEGER,
  pending_videos INTEGER,
  failed_videos INTEGER
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    COUNT(*)::INTEGER as total_videos,
    COUNT(*) FILTER (WHERE status = 'completed')::INTEGER as completed_videos,
    COUNT(*) FILTER (WHERE status = 'pending')::INTEGER as pending_videos,
    COUNT(*) FILTER (WHERE status = 'failed')::INTEGER as failed_videos
  FROM lesson_video_generation_status
  WHERE core_lesson_id = lesson_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get user's lesson completion stats
CREATE OR REPLACE FUNCTION get_user_lesson_stats(p_user_id UUID, p_lesson_id UUID)
RETURNS TABLE (
  phases_completed INTEGER,
  total_phases INTEGER,
  paths_explored INTEGER,
  total_paths INTEGER,
  completion_percentage NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    COUNT(DISTINCT phase)::INTEGER as phases_completed,
    5 as total_phases,  -- Always 5 phases
    COUNT(*)::INTEGER as paths_explored,
    15 as total_paths,  -- 3 archetypes × 5 phases
    (COUNT(DISTINCT phase)::NUMERIC / 5 * 100) as completion_percentage
  FROM user_lesson_paths
  WHERE user_id = p_user_id AND core_lesson_id = p_lesson_id;
END;
$$ LANGUAGE plpgsql;

-- Function to initialize video generation tracking for a lesson
CREATE OR REPLACE FUNCTION initialize_lesson_video_tracking(lesson_id UUID)
RETURNS INTEGER AS $$
DECLARE
  inserted_count INTEGER := 0;
  archetype TEXT;
  phase TEXT;
  video_type TEXT;
BEGIN
  -- For each archetype
  FOR archetype IN SELECT unnest(ARRAY['The Explorer', 'The Rebel', 'The Scientist']) LOOP
    -- For each phase
    FOR phase IN SELECT unnest(ARRAY['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom']) LOOP
      -- Main video
      INSERT INTO lesson_video_generation_status (core_lesson_id, archetype, phase, video_type, status)
      VALUES (lesson_id, archetype, phase, 'main', 'pending')
      ON CONFLICT (core_lesson_id, archetype, phase, video_type) DO NOTHING;
      
      IF FOUND THEN
        inserted_count := inserted_count + 1;
      END IF;
      
      -- Response videos (except for Wisdom phase)
      IF phase != 'Wisdom' THEN
        FOR video_type IN SELECT unnest(ARRAY['response_A', 'response_B', 'response_C']) LOOP
          INSERT INTO lesson_video_generation_status (core_lesson_id, archetype, phase, video_type, status)
          VALUES (lesson_id, archetype, phase, video_type, 'pending')
          ON CONFLICT (core_lesson_id, archetype, phase, video_type) DO NOTHING;
          
          IF FOUND THEN
            inserted_count := inserted_count + 1;
          END IF;
        END LOOP;
      END IF;
    END LOOP;
  END LOOP;
  
  RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- PART 6: Row Level Security (RLS)
-- ============================================================

-- Users can view their own lesson paths
DROP POLICY IF EXISTS "Users can view their own lesson paths" ON user_lesson_paths;
CREATE POLICY "Users can view their own lesson paths" ON user_lesson_paths
  FOR SELECT USING (auth.uid() = user_id);

-- Users can insert their own lesson paths
DROP POLICY IF EXISTS "Users can insert their own lesson paths" ON user_lesson_paths;
CREATE POLICY "Users can insert their own lesson paths" ON user_lesson_paths
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can update their own lesson paths
DROP POLICY IF EXISTS "Users can update their own lesson paths" ON user_lesson_paths;
CREATE POLICY "Users can update their own lesson paths" ON user_lesson_paths
  FOR UPDATE USING (auth.uid() = user_id);

-- Video generation status is publicly readable (for progress indicators)
DROP POLICY IF EXISTS "Anyone can view video generation status" ON lesson_video_generation_status;
CREATE POLICY "Anyone can view video generation status" ON lesson_video_generation_status
  FOR SELECT USING (true);

-- ============================================================
-- PART 7: Initialize tracking for existing lessons
-- ============================================================

-- Initialize video tracking for Day 1 (for testing)
DO $$
DECLARE
  day1_lesson_id UUID;
BEGIN
  SELECT id INTO day1_lesson_id FROM core_lessons WHERE day_number = 1 LIMIT 1;
  
  IF day1_lesson_id IS NOT NULL THEN
    PERFORM initialize_lesson_video_tracking(day1_lesson_id);
    RAISE NOTICE 'Initialized video tracking for Day 1 lesson: %', day1_lesson_id;
  END IF;
END $$;

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check video generation status for Day 1
-- SELECT archetype, phase, video_type, status FROM lesson_video_generation_status WHERE core_lesson_id = (SELECT id FROM core_lessons WHERE day_number = 1 LIMIT 1) ORDER BY archetype, phase, video_type;

-- Check total videos needed per lesson
-- SELECT get_lesson_video_count((SELECT id FROM core_lessons WHERE day_number = 1 LIMIT 1));

-- Check user lesson preferences
-- SELECT lesson_autoplay, lesson_show_progress, lesson_transition_delay FROM users LIMIT 5;







