/**
 * Phase 0 Fix: Add missing columns to existing tables
 * lesson_atoms already exists — skip recreation
 * Just add columns to kellyos_audio and core_lessons_v2
 */
require('dotenv').config();
const { Client } = require('pg');

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  console.log('[SCHEMA-FIX] Connected');

  const addColumnSafe = async (table, col, type) => {
    try {
      await client.query(`ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${col} ${type}`);
      console.log(`[SCHEMA-FIX] Added ${col} to ${table}`);
    } catch (e) {
      console.log(`[SCHEMA-FIX] ${col} on ${table}: ${e.message.substring(0, 80)}`);
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

  // Create index on search_vector
  try {
    await client.query('CREATE INDEX IF NOT EXISTS idx_core_lessons_search ON core_lessons_v2 USING gin(search_vector)');
    console.log('[SCHEMA-FIX] GIN search index created');
  } catch (e) {
    console.log(`[SCHEMA-FIX] Search index: ${e.message.substring(0, 80)}`);
  }

  // Verify all tables
  const tables = await client.query(`
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND (table_name LIKE 'kellyos_%' OR table_name = 'lesson_atoms' OR table_name = 'core_lessons_v2')
    ORDER BY table_name
  `);
  console.log('\n[SCHEMA-FIX] All relevant tables:');
  tables.rows.forEach(r => console.log(`  - ${r.table_name}`));

  console.log('\n[SCHEMA-FIX] COMPLETE');
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
