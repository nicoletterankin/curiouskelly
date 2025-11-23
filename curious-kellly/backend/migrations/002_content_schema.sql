-- Content Engine Schema
-- Adds tables for the 365-day curriculum and generated atoms

-- 1. Core Lessons (The 365 Map)
CREATE TABLE IF NOT EXISTS core_lessons (
  id SERIAL PRIMARY KEY,
  day_number INTEGER UNIQUE NOT NULL, -- 1 to 365
  topic VARCHAR(255) NOT NULL,
  universal_truth TEXT NOT NULL, -- The core fact/concept
  description TEXT, -- Optional expanded description
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Lesson Atoms (The Generated Content)
CREATE TABLE IF NOT EXISTS lesson_atoms (
  id SERIAL PRIMARY KEY,
  core_lesson_id INTEGER NOT NULL REFERENCES core_lessons(id) ON DELETE CASCADE,
  archetype VARCHAR(50) NOT NULL, -- 'The Survivor', 'The Scientist', etc.
  phase VARCHAR(50) NOT NULL, -- 'Hook', 'Fact1', 'Wisdom', etc.
  content JSONB NOT NULL, -- The full JSON content (script, options, etc.)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(core_lesson_id, archetype, phase) -- Prevent duplicates
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_core_lessons_day ON core_lessons(day_number);
CREATE INDEX IF NOT EXISTS idx_atoms_lesson_id ON lesson_atoms(core_lesson_id);
CREATE INDEX IF NOT EXISTS idx_atoms_lookup ON lesson_atoms(core_lesson_id, archetype, phase);

-- Triggers
CREATE TRIGGER update_core_lessons_updated_at BEFORE UPDATE ON core_lessons
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_lesson_atoms_updated_at BEFORE UPDATE ON lesson_atoms
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();






