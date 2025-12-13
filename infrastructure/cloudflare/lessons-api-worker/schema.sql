-- Kelly Lessons Mirror Database
-- This mirrors Supabase for redundancy
-- Cloudflare D1 SQLite-based edge database

DROP TABLE IF EXISTS shards;
DROP TABLE IF EXISTS atoms;
DROP TABLE IF EXISTS lessons;

-- Core lessons table (mirrors core_lessons from Supabase)
CREATE TABLE lessons (
  day_number INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  topic TEXT,
  subtitle TEXT,
  marketing_hook TEXT,
  marketing_headline TEXT,
  marketing_tagline TEXT,
  marketing_pitch TEXT,
  hook_question TEXT,
  universal_truth TEXT,
  content TEXT,
  category TEXT,
  difficulty TEXT DEFAULT 'beginner',
  duration_estimate INTEGER DEFAULT 5,
  hero_image_url TEXT,
  thumbnail_url TEXT,
  audio_url TEXT,
  video_url TEXT,
  quick_quiz_questions TEXT, -- JSON array stored as text
  reflection_prompts TEXT,   -- JSON array stored as text
  mastery_criteria TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

-- Atoms table (mirrors lesson_atoms - archetype-specific dialog)
CREATE TABLE atoms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lesson_day INTEGER NOT NULL,
  core_lesson_id TEXT,
  archetype TEXT NOT NULL,
  phase TEXT,
  dialog_type TEXT,
  content TEXT,           -- JSON content stored as text
  kelly_script TEXT,
  kelly_pose TEXT,
  kelly_emotion TEXT,
  trigger_context TEXT,
  is_active INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (lesson_day) REFERENCES lessons(day_number)
);

-- Shards table (mirrors lesson_shards - age/region personalized content)
CREATE TABLE shards (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lesson_day INTEGER NOT NULL,
  core_lesson_id TEXT,
  archetype TEXT,
  region TEXT NOT NULL,
  age INTEGER,
  script_content TEXT,    -- JSON content stored as text
  diff_type TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (lesson_day) REFERENCES lessons(day_number)
);

-- Indexes for fast queries
CREATE INDEX idx_atoms_lookup ON atoms(lesson_day, archetype);
CREATE INDEX idx_atoms_phase ON atoms(lesson_day, phase);
CREATE INDEX idx_atoms_active ON atoms(is_active);
CREATE INDEX idx_shards_lookup ON shards(lesson_day, region);
CREATE INDEX idx_shards_archetype ON shards(lesson_day, archetype);
CREATE INDEX idx_shards_age ON shards(age);
CREATE INDEX idx_lessons_category ON lessons(category);

-- Metadata table for sync tracking
CREATE TABLE sync_metadata (
  id INTEGER PRIMARY KEY,
  last_sync_at TEXT,
  lessons_count INTEGER DEFAULT 0,
  atoms_count INTEGER DEFAULT 0,
  shards_count INTEGER DEFAULT 0,
  sync_source TEXT DEFAULT 'supabase',
  sync_duration_ms INTEGER
);

-- Insert initial metadata row
INSERT INTO sync_metadata (id, last_sync_at) VALUES (1, datetime('now'));
