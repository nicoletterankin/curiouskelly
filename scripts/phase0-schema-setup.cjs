/**
 * Phase 0: Create all new database tables for the content empire
 * Tables: kellyos_lesson_graph, kellyos_tags, kellyos_quotes, kellyos_facts_v2,
 *         kellyos_teacher_guides, kellyos_clusters, kellyos_cluster_lessons,
 *         kellyos_learning_paths, lesson_atoms (if not exists)
 * Also: add columns to kellyos_audio and core_lessons_v2 if missing
 */
require('dotenv').config();
const { Client } = require('pg');

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  console.log('[SCHEMA] Connected to database');

  // 1. Lesson Graph (prerequisites, followups, related)
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_lesson_graph (
      id SERIAL PRIMARY KEY,
      from_day INTEGER NOT NULL,
      to_day INTEGER NOT NULL,
      relationship TEXT NOT NULL CHECK (relationship IN ('prerequisite', 'followup', 'related')),
      strength FLOAT DEFAULT 0.5,
      created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_lesson_graph_from ON kellyos_lesson_graph(from_day);
    CREATE INDEX IF NOT EXISTS idx_lesson_graph_to ON kellyos_lesson_graph(to_day);
  `);
  console.log('[SCHEMA] kellyos_lesson_graph created');

  // 2. Tags and categories
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_tags (
      id SERIAL PRIMARY KEY,
      day_number INTEGER NOT NULL,
      tag TEXT NOT NULL,
      category TEXT,
      is_primary BOOLEAN DEFAULT FALSE,
      created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_tags_day ON kellyos_tags(day_number);
    CREATE INDEX IF NOT EXISTS idx_tags_tag ON kellyos_tags(tag);
    CREATE INDEX IF NOT EXISTS idx_tags_category ON kellyos_tags(category);
  `);
  console.log('[SCHEMA] kellyos_tags created');

  // 3. Kelly quotes
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_quotes (
      id SERIAL PRIMARY KEY,
      day_number INTEGER NOT NULL,
      quote_type TEXT NOT NULL CHECK (quote_type IN ('hook', 'wonder', 'wisdom')),
      quote_text TEXT NOT NULL,
      attribution TEXT DEFAULT 'Kelly',
      created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_quotes_day ON kellyos_quotes(day_number);
    CREATE INDEX IF NOT EXISTS idx_quotes_type ON kellyos_quotes(quote_type);
  `);
  console.log('[SCHEMA] kellyos_quotes created');

  // 4. Facts (expanded version with tricky flag)
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_facts_v2 (
      id SERIAL PRIMARY KEY,
      day_number INTEGER NOT NULL,
      statement TEXT NOT NULL,
      is_true BOOLEAN NOT NULL,
      is_tricky BOOLEAN DEFAULT FALSE,
      explanation TEXT NOT NULL,
      difficulty INTEGER DEFAULT 5 CHECK (difficulty >= 1 AND difficulty <= 10),
      created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_facts_v2_day ON kellyos_facts_v2(day_number);
  `);
  console.log('[SCHEMA] kellyos_facts_v2 created');

  // 5. Teacher guides
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_teacher_guides (
      id SERIAL PRIMARY KEY,
      day_number INTEGER NOT NULL UNIQUE,
      grade_range TEXT,
      standards_alignment JSONB,
      prep_notes TEXT,
      discussion_questions JSONB,
      extension_activities JSONB,
      assessment_rubric JSONB,
      materials TEXT,
      time_15min TEXT,
      time_30min TEXT,
      time_45min TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_teacher_guides_day ON kellyos_teacher_guides(day_number);
  `);
  console.log('[SCHEMA] kellyos_teacher_guides created');

  // 6. Clusters
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_clusters (
      id SERIAL PRIMARY KEY,
      cluster_name TEXT NOT NULL UNIQUE,
      cluster_description TEXT,
      icon TEXT,
      color TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    );
  `);
  console.log('[SCHEMA] kellyos_clusters created');

  // 7. Cluster-lesson mapping
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_cluster_lessons (
      cluster_id INTEGER REFERENCES kellyos_clusters(id),
      day_number INTEGER NOT NULL,
      relevance_score FLOAT DEFAULT 1.0,
      PRIMARY KEY (cluster_id, day_number)
    );
    CREATE INDEX IF NOT EXISTS idx_cluster_lessons_day ON kellyos_cluster_lessons(day_number);
  `);
  console.log('[SCHEMA] kellyos_cluster_lessons created');

  // 8. Learning paths
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_learning_paths (
      id SERIAL PRIMARY KEY,
      path_name TEXT NOT NULL UNIQUE,
      path_description TEXT,
      difficulty TEXT CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
      age_range TEXT,
      estimated_days INTEGER,
      icon TEXT,
      lessons JSONB NOT NULL,
      created_at TIMESTAMP DEFAULT NOW()
    );
  `);
  console.log('[SCHEMA] kellyos_learning_paths created');

  // 9. Lesson atoms for archetype content
  await client.query(`
    CREATE TABLE IF NOT EXISTS lesson_atoms (
      id SERIAL PRIMARY KEY,
      day_number INTEGER NOT NULL,
      phase TEXT NOT NULL,
      archetype TEXT,
      kelly_script TEXT,
      kelly_emotion TEXT,
      age_group TEXT DEFAULT 'adult',
      language TEXT DEFAULT 'en',
      created_at TIMESTAMP DEFAULT NOW(),
      UNIQUE(day_number, phase, archetype, age_group, language)
    );
    CREATE INDEX IF NOT EXISTS idx_lesson_atoms_day ON lesson_atoms(day_number);
    CREATE INDEX IF NOT EXISTS idx_lesson_atoms_archetype ON lesson_atoms(archetype);
    CREATE INDEX IF NOT EXISTS idx_lesson_atoms_age ON lesson_atoms(age_group);
  `);
  console.log('[SCHEMA] lesson_atoms created');

  // 10. Add columns to kellyos_audio if missing
  const addColumnSafe = async (table, col, type) => {
    try {
      await client.query(`ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${col} ${type}`);
      console.log(`[SCHEMA] Added ${col} to ${table}`);
    } catch (e) {
      console.log(`[SCHEMA] ${col} already exists on ${table} or error: ${e.message}`);
    }
  };

  await addColumnSafe('kellyos_audio', 'srt_text', 'TEXT');
  await addColumnSafe('kellyos_audio', 'age_group', 'TEXT');
  await addColumnSafe('core_lessons_v2', 'learning_objectives', 'JSONB');
  await addColumnSafe('core_lessons_v2', 'difficulty_data', 'JSONB');
  await addColumnSafe('core_lessons_v2', 'summary_short', 'TEXT');
  await addColumnSafe('core_lessons_v2', 'summary_teaser', 'TEXT');
  await addColumnSafe('core_lessons_v2', 'meta_description', 'TEXT');
  await addColumnSafe('core_lessons_v2', 'search_vector', 'tsvector');

  // 11. Full-text search index
  try {
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_core_lessons_search ON core_lessons_v2 USING gin(search_vector);
    `);
    console.log('[SCHEMA] Full-text search index created');
  } catch (e) {
    console.log(`[SCHEMA] Search index: ${e.message}`);
  }

  // 12. Verify all tables exist
  const tables = await client.query(`
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name LIKE 'kellyos_%' OR table_name = 'lesson_atoms'
    ORDER BY table_name
  `);
  console.log('\n[SCHEMA] All kellyos_* tables:');
  tables.rows.forEach(r => console.log(`  - ${r.table_name}`));

  console.log('\n[SCHEMA] ALL TABLES CREATED SUCCESSFULLY');
  await client.end();
}

main().catch(e => { console.error('[SCHEMA ERROR]', e); process.exit(1); });
