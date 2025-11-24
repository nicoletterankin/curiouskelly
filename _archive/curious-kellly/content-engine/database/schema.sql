-- Core Lessons (The Immutable Truth)
CREATE TABLE IF NOT EXISTS core_lessons (
    id TEXT PRIMARY KEY, -- Changed from UUID to TEXT to support legacy string IDs
    topic TEXT NOT NULL,
    day_number INTEGER UNIQUE NOT NULL CHECK (day_number BETWEEN 0 AND 365), -- Expanded to allow Day 0 (which exists in data)
    universal_facts JSONB NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Lesson Atoms (The Personalised Shards)
CREATE TABLE IF NOT EXISTS lesson_atoms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lesson_id TEXT REFERENCES core_lessons(id) ON DELETE CASCADE, -- Changed FK type
    language TEXT NOT NULL DEFAULT 'en-US',
    
    -- The Persona Matrix
    archetype TEXT NOT NULL,
    phase_type TEXT NOT NULL,
    verbosity_level TEXT NOT NULL CHECK (verbosity_level IN ('Brief', 'Balanced', 'Detailed')),
    
    -- The Content (Gold Standard)
    content JSONB NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique atoms per configuration
    CONSTRAINT unique_atom UNIQUE (lesson_id, language, archetype, phase_type, verbosity_level)
);

-- Indexes
CREATE INDEX idx_atoms_lookup ON lesson_atoms(lesson_id, language, archetype, phase_type);
