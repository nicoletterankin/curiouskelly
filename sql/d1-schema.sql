-- Cloudflare D1 Schema for Lesson Mirror
-- This mirrors the Supabase schema for redundancy
--
-- Run: wrangler d1 execute lessons-db --file=./sql/d1-schema.sql

-- Core Lessons Table (365 rows)
CREATE TABLE IF NOT EXISTS core_lessons (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  day_number INTEGER UNIQUE NOT NULL,
  topic TEXT NOT NULL,
  universal_truth TEXT,
  marketing_headline TEXT,
  marketing_tagline TEXT,
  marketing_pitch TEXT,
  hook_question TEXT,
  quick_quiz_questions TEXT, -- JSON array
  reflection_prompts TEXT,   -- JSON array
  mastery_criteria TEXT,
  hero_image_url TEXT,
  thumbnail_url TEXT,
  audio_url TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

-- Lesson Atoms Table (archetype-specific dialog)
CREATE TABLE IF NOT EXISTS lesson_atoms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  core_lesson_id INTEGER NOT NULL,
  archetype TEXT NOT NULL,
  phase TEXT NOT NULL, -- Hook, Fact1, Fact2, Fact3, Wisdom
  content TEXT NOT NULL, -- JSON object with script, text, etc.
  hd_video_url TEXT,
  visual_url TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (core_lesson_id) REFERENCES core_lessons(id)
);

-- Lesson Shards Table (age/region personalized content)
CREATE TABLE IF NOT EXISTS lesson_shards (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  core_lesson_id INTEGER NOT NULL,
  archetype TEXT NOT NULL,
  region TEXT, -- kid, teen, adult, mature, elder
  age INTEGER,
  tone TEXT,
  birth_year INTEGER,
  script_content TEXT NOT NULL, -- JSON object with phase-specific scripts
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (core_lesson_id) REFERENCES core_lessons(id)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_lessons_day ON core_lessons(day_number);
CREATE INDEX IF NOT EXISTS idx_atoms_lesson_archetype ON lesson_atoms(core_lesson_id, archetype);
CREATE INDEX IF NOT EXISTS idx_shards_lesson_archetype ON lesson_shards(core_lesson_id, archetype);

-- Sync status table (for tracking mirror freshness)
CREATE TABLE IF NOT EXISTS sync_status (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  table_name TEXT UNIQUE NOT NULL,
  last_sync TEXT NOT NULL,
  row_count INTEGER,
  checksum TEXT
);






