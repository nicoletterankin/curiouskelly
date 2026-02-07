-- KellyOS Core Schema Migration
-- Sprint A: Database Foundation
-- Run against Neon: soft-block-64917198

-- 1. core_lessons: Master curriculum (365 days)
CREATE TABLE IF NOT EXISTS core_lessons_v2 (
  id SERIAL PRIMARY KEY,
  day_number INTEGER NOT NULL UNIQUE CHECK (day_number >= 1 AND day_number <= 365),
  title TEXT NOT NULL,
  subject TEXT,
  learning_objective TEXT,
  category TEXT DEFAULT 'general',
  difficulty TEXT DEFAULT 'beginner',
  seed_data JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. lesson_atoms: Per-lesson variants (day × phase × age × language)
CREATE TABLE IF NOT EXISTS lesson_atoms (
  id SERIAL PRIMARY KEY,
  lesson_id INTEGER NOT NULL REFERENCES core_lessons_v2(id) ON DELETE CASCADE,
  phase INTEGER NOT NULL CHECK (phase >= 1 AND phase <= 7),
  variant TEXT DEFAULT 'default',
  age_group TEXT DEFAULT 'adult' CHECK (age_group IN ('kid', 'teen', 'adult', 'elder')),
  language TEXT DEFAULT 'en',
  script TEXT,
  audio_url TEXT,
  video_url TEXT,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'script_complete', 'audio_complete', 'video_complete', 'published')),
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(lesson_id, phase, variant, age_group, language)
);

-- 3. lesson_scripts: Phase scripts with options (7 phases × 2 options per atom)
CREATE TABLE IF NOT EXISTS lesson_scripts (
  id SERIAL PRIMARY KEY,
  atom_id INTEGER NOT NULL REFERENCES lesson_atoms(id) ON DELETE CASCADE,
  phase INTEGER NOT NULL CHECK (phase >= 1 AND phase <= 7),
  option_number INTEGER NOT NULL DEFAULT 1 CHECK (option_number IN (1, 2)),
  content TEXT NOT NULL,
  duration_seconds INTEGER,
  word_count INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(atom_id, phase, option_number)
);

-- 4. generation_jobs: Track all async generation work
CREATE TABLE IF NOT EXISTS generation_jobs (
  id SERIAL PRIMARY KEY,
  atom_id INTEGER REFERENCES lesson_atoms(id) ON DELETE SET NULL,
  job_type TEXT NOT NULL CHECK (job_type IN ('script', 'audio', 'video', 'lipsync')),
  provider TEXT NOT NULL,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'complete', 'failed', 'cancelled')),
  external_id TEXT,
  input_params JSONB DEFAULT '{}',
  output_url TEXT,
  error TEXT,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_lesson_atoms_lesson_id ON lesson_atoms(lesson_id);
CREATE INDEX IF NOT EXISTS idx_lesson_atoms_status ON lesson_atoms(status);
CREATE INDEX IF NOT EXISTS idx_lesson_atoms_day_phase ON lesson_atoms(lesson_id, phase);
CREATE INDEX IF NOT EXISTS idx_lesson_scripts_atom ON lesson_scripts(atom_id);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_atom ON generation_jobs(atom_id);
CREATE INDEX IF NOT EXISTS idx_core_lessons_v2_day ON core_lessons_v2(day_number);
