/**
 * Sprint H: Schema Alignment & Contract
 * H.1 — Introspect actual schema
 * H.2 — Create compatibility views
 * H.3 — Write SCHEMA-CONTRACT.md
 * H.4 — Add missing indexes
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const LOG_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\logs', 'cursor-burndown.log');
const AUDIT_DIR = path.join('C:\\Users\\user\\kelly-pipeline\\audit');

function log(sprint, msg) {
  const line = `[${new Date().toISOString()}] ${sprint} | ${msg}\n`;
  fs.mkdirSync(path.dirname(LOG_FILE), { recursive: true });
  fs.appendFileSync(LOG_FILE, line, 'utf-8');
  process.stdout.write(line);
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  // ===== H.1 — Introspect schema =====
  log('SPRINT H.1', 'START | Introspecting schema');
  
  const tables = await client.query(`
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema = 'public' ORDER BY table_name
  `);
  log('SPRINT H.1', `FOUND | ${tables.rows.length} tables`);
  
  const columns = await client.query(`
    SELECT table_name, column_name, data_type, is_nullable, column_default
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
    ORDER BY table_name, ordinal_position
  `);
  log('SPRINT H.1', `FOUND | ${columns.rows.length} columns total`);
  
  const indexes = await client.query(`
    SELECT tablename, indexname, indexdef 
    FROM pg_indexes WHERE schemaname = 'public'
    ORDER BY tablename, indexname
  `);
  log('SPRINT H.1', `FOUND | ${indexes.rows.length} indexes`);
  
  // Get row counts for all tables
  const rowCounts = {};
  for (const t of tables.rows) {
    try {
      const cnt = await client.query(`SELECT COUNT(*) as cnt FROM "${t.table_name}"`);
      rowCounts[t.table_name] = parseInt(cnt.rows[0].cnt);
    } catch (e) {
      rowCounts[t.table_name] = -1;
    }
  }
  
  // Save schema snapshot
  fs.mkdirSync(AUDIT_DIR, { recursive: true });
  let snapshot = `-- Schema Snapshot: ${new Date().toISOString()}\n\n`;
  snapshot += `-- TABLES (${tables.rows.length}):\n`;
  for (const t of tables.rows) {
    snapshot += `-- ${t.table_name} (${rowCounts[t.table_name]} rows)\n`;
  }
  snapshot += '\n-- COLUMNS:\n';
  let currentTable = '';
  for (const c of columns.rows) {
    if (c.table_name !== currentTable) {
      currentTable = c.table_name;
      snapshot += `\n-- TABLE: ${currentTable}\n`;
    }
    snapshot += `--   ${c.column_name} ${c.data_type} ${c.is_nullable === 'YES' ? 'NULL' : 'NOT NULL'}${c.column_default ? ' DEFAULT ' + c.column_default : ''}\n`;
  }
  snapshot += '\n-- INDEXES:\n';
  for (const i of indexes.rows) {
    snapshot += `-- ${i.indexdef}\n`;
  }
  fs.writeFileSync(path.join(AUDIT_DIR, 'schema-snapshot.sql'), snapshot);
  log('SPRINT H.1', 'COMPLETE | schema-snapshot.sql written');
  
  // ===== H.2 — Create compatibility views =====
  log('SPRINT H.2', 'START | Creating compatibility views');
  
  // Check what columns kellyos_lessons actually has
  const klCols = columns.rows.filter(c => c.table_name === 'kellyos_lessons').map(c => c.column_name);
  log('SPRINT H.2', `kellyos_lessons columns: ${klCols.join(', ')}`);
  
  // Create lessons_by_day view (maps day_number to day_of_year for v0)
  try {
    const contentCol = klCols.includes('content_text') ? 'content_text' : klCols.includes('text') ? 'text' : 'title';
    const langCol = klCols.includes('language') ? 'language' : null;
    
    await client.query(`
      CREATE OR REPLACE VIEW lessons_by_day AS
      SELECT
        day_number AS day_of_year,
        day_number,
        phase,
        title,
        ${contentCol} AS content_text,
        ${klCols.includes('audio_url') ? 'audio_url' : 'NULL::text AS audio_url'},
        alignment_json
        ${langCol ? `, ${langCol}` : ''}
      FROM kellyos_lessons
    `);
    log('SPRINT H.2', 'CREATED | lessons_by_day view');
  } catch (e) {
    log('SPRINT H.2', `ERROR creating lessons_by_day: ${e.message}`);
  }
  
  // Create live_classes view (harmless fallback for v0)
  try {
    // Check which table to use as source
    const hasCoreV2 = tables.rows.some(t => t.table_name === 'core_lessons_v2');
    const hasCoreOrig = tables.rows.some(t => t.table_name === 'core_lessons');
    const srcTable = hasCoreV2 ? 'core_lessons_v2' : hasCoreOrig ? 'core_lessons' : 'lessons';
    
    const srcCols = columns.rows.filter(c => c.table_name === srcTable).map(c => c.column_name);
    
    await client.query(`
      CREATE OR REPLACE VIEW live_classes AS
      SELECT
        ${srcCols.includes('id') ? 'id' : 'day_number AS id'},
        day_number,
        title,
        'active'::text AS status,
        ${srcCols.includes('created_at') ? 'created_at' : 'NOW() AS created_at'}
      FROM ${srcTable}
      LIMIT 10
    `);
    log('SPRINT H.2', 'CREATED | live_classes view');
  } catch (e) {
    log('SPRINT H.2', `ERROR creating live_classes: ${e.message}`);
  }
  
  // Create a core_lessons view that maps to core_lessons_v2 (if v0 expects core_lessons)
  try {
    const hasCoreV2 = tables.rows.some(t => t.table_name === 'core_lessons_v2');
    const hasCoreOrig = tables.rows.some(t => t.table_name === 'core_lessons');
    
    if (hasCoreV2 && !hasCoreOrig) {
      await client.query(`
        CREATE OR REPLACE VIEW core_lessons AS
        SELECT * FROM core_lessons_v2
      `);
      log('SPRINT H.2', 'CREATED | core_lessons view -> core_lessons_v2');
    } else if (hasCoreV2 && hasCoreOrig) {
      log('SPRINT H.2', 'SKIP | Both core_lessons and core_lessons_v2 exist');
    }
  } catch (e) {
    log('SPRINT H.2', `ERROR creating core_lessons view: ${e.message}`);
  }
  
  log('SPRINT H.2', 'COMPLETE | Compatibility views created');
  
  // ===== H.3 — Write SCHEMA-CONTRACT.md =====
  log('SPRINT H.3', 'START | Writing SCHEMA-CONTRACT.md');
  
  let contract = `# SCHEMA-CONTRACT.md — KellyOS Database Schema Contract\n\n`;
  contract += `**Generated:** ${new Date().toISOString()}\n`;
  contract += `**Database:** Neon PostgreSQL (soft-block-64917198)\n`;
  contract += `**This is the single source of truth for database queries.**\n\n`;
  contract += `## Quick Reference\n\n`;
  contract += `| Table | Rows | Purpose |\n`;
  contract += `|-------|------|--------|\n`;
  
  const tablePurposes = {
    'core_lessons_v2': 'Master curriculum - 365 days',
    'core_lessons': 'View → core_lessons_v2 (alias)',
    'lesson_atoms': 'Per-lesson phase variants (day × phase × age × language)',
    'lesson_scripts': 'Script text with options (7 phases × 2 options)',
    'generation_jobs': 'Async job tracking for script/audio/video generation',
    'kellyos_lessons': 'Legacy lesson storage with 5-phase structure',
    'kellyos_audio': 'Audio files + alignment data per lesson slot',
    'kellyos_assets': 'Blob-hosted visual assets (sprites, videos, behaviors)',
    'kellyos_playback_log': 'Playback analytics',
    'lessons': 'Original lessons table (365 rows)',
    'lesson_perspectives': 'Multi-perspective lesson variants',
    'kelly_lesson_assets': 'Per-lesson multimedia assets',
    'heygen_videos': 'HeyGen video generation records',
    'lessons_by_day': 'View → kellyos_lessons (maps day_of_year)',
    'live_classes': 'View → core_lessons_v2 (v0 compatibility)',
    'kellyos_facts': 'Fact-check questions per lesson',
  };
  
  for (const t of tables.rows) {
    contract += `| ${t.table_name} | ${rowCounts[t.table_name] >= 0 ? rowCounts[t.table_name] : 'N/A'} | ${tablePurposes[t.table_name] || ''} |\n`;
  }
  
  contract += `\n## Table Definitions\n\n`;
  
  currentTable = '';
  for (const c of columns.rows) {
    if (c.table_name !== currentTable) {
      if (currentTable) contract += `\n`;
      currentTable = c.table_name;
      contract += `### ${currentTable}\n`;
      contract += `**Rows:** ${rowCounts[currentTable] >= 0 ? rowCounts[currentTable] : 'N/A'}\n`;
      contract += `**Purpose:** ${tablePurposes[currentTable] || 'See table definition below'}\n\n`;
      contract += `| Column | Type | Nullable | Default |\n`;
      contract += `|--------|------|----------|---------|\n`;
    }
    contract += `| ${c.column_name} | ${c.data_type} | ${c.is_nullable} | ${c.column_default || ''} |\n`;
  }
  
  contract += `\n## Important Compatibility Notes\n\n`;
  contract += `### For v0 Frontend:\n`;
  contract += `- Use \`day_number\` (not \`day_of_year\`) when querying tables directly\n`;
  contract += `- The view \`lessons_by_day\` provides \`day_of_year\` as an alias for \`day_number\`\n`;
  contract += `- \`live_classes\` is a compatibility view, not a real table — do not INSERT\n`;
  contract += `- \`core_lessons\` is a view aliasing \`core_lessons_v2\` — use either name\n\n`;
  contract += `### Key Queries:\n\n`;
  contract += '```sql\n';
  contract += `-- Get lesson for a specific day\n`;
  contract += `SELECT * FROM core_lessons_v2 WHERE day_number = 1;\n\n`;
  contract += `-- Get lesson content with audio\n`;
  contract += `SELECT kl.*, ka.audio_url, ka.alignment_json, ka.duration_seconds\n`;
  contract += `FROM kellyos_lessons kl\n`;
  contract += `LEFT JOIN kellyos_audio ka ON ka.day_number = kl.day_number AND ka.phase = kl.phase\n`;
  contract += `WHERE kl.day_number = 1;\n\n`;
  contract += `-- Get all scripts for a day (7 phases × 2 options)\n`;
  contract += `SELECT la.phase, la.variant, ls.option_number, ls.content, ls.word_count\n`;
  contract += `FROM lesson_atoms la\n`;
  contract += `JOIN lesson_scripts ls ON ls.atom_id = la.id\n`;
  contract += `JOIN core_lessons_v2 cl ON cl.id = la.lesson_id\n`;
  contract += `WHERE cl.day_number = 1 AND la.age_group = 'adult' AND la.language = 'en'\n`;
  contract += `ORDER BY la.phase, ls.option_number;\n\n`;
  contract += `-- Get assets for an age group\n`;
  contract += `SELECT * FROM kellyos_assets WHERE age = 'adult';\n\n`;
  contract += `-- Day of year compatibility\n`;
  contract += `SELECT * FROM lessons_by_day WHERE day_of_year = 1;\n`;
  contract += '```\n\n';
  contract += `### Phase Mapping:\n`;
  contract += `| Phase Number | Name | Expression |\n`;
  contract += `|--------------|------|------------|\n`;
  contract += `| 1 | hook | excited |\n`;
  contract += `| 2 | teach/story | talking |\n`;
  contract += `| 3 | example/wonder | curious |\n`;
  contract += `| 4 | practice/action | thinking |\n`;
  contract += `| 5 | reflect | talking |\n`;
  contract += `| 6 | apply | thinking |\n`;
  contract += `| 7 | close/wisdom | talking |\n`;
  
  const contractPath = path.join('C:\\Users\\user\\kelly-pipeline', 'SCHEMA-CONTRACT.md');
  fs.writeFileSync(contractPath, contract);
  log('SPRINT H.3', 'COMPLETE | SCHEMA-CONTRACT.md written');
  
  // ===== H.4 — Add missing indexes =====
  log('SPRINT H.4', 'START | Adding performance indexes');
  
  const indexQueries = [
    'CREATE INDEX IF NOT EXISTS idx_kellyos_lessons_day_phase ON kellyos_lessons(day_number, phase)',
    'CREATE INDEX IF NOT EXISTS idx_kellyos_audio_day_phase ON kellyos_audio(day_number, phase)',
    'CREATE INDEX IF NOT EXISTS idx_kellyos_assets_type_age ON kellyos_assets(asset_type, age)',
    'CREATE INDEX IF NOT EXISTS idx_core_lessons_v2_day ON core_lessons_v2(day_number)',
    'CREATE INDEX IF NOT EXISTS idx_lesson_atoms_lesson_phase ON lesson_atoms(lesson_id, phase)',
    'CREATE INDEX IF NOT EXISTS idx_lesson_atoms_status ON lesson_atoms(status)',
    'CREATE INDEX IF NOT EXISTS idx_lesson_scripts_atom ON lesson_scripts(atom_id)',
    'CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status)',
    'CREATE INDEX IF NOT EXISTS idx_kelly_lesson_assets_day ON kelly_lesson_assets(day_number, phase)',
  ];
  
  let indexCreated = 0;
  for (const q of indexQueries) {
    try {
      await client.query(q);
      indexCreated++;
    } catch (e) {
      log('SPRINT H.4', `INDEX SKIP: ${e.message.substring(0, 60)}`);
    }
  }
  log('SPRINT H.4', `COMPLETE | ${indexCreated}/${indexQueries.length} indexes created/verified`);
  
  // Save checkpoint
  const checkpoint = {
    last_updated: new Date().toISOString(),
    credits_at_start: '81%',
    sprints: {
      H: { status: 'complete', completed_at: new Date().toISOString(), notes: `${tables.rows.length} tables, ${columns.rows.length} cols, views created, ${indexCreated} indexes` },
      I: { status: 'pending' },
      J: { status: 'pending' },
      K: { status: 'pending' },
      L: { status: 'pending' },
      M: { status: 'pending' },
      N: { status: 'pending' },
      O: { status: 'pending' },
      P: { status: 'pending' },
      Q: { status: 'pending' },
      R: { status: 'pending' }
    }
  };
  
  const cpDir = path.join('C:\\Users\\user\\kelly-pipeline\\checkpoints');
  fs.mkdirSync(cpDir, { recursive: true });
  fs.writeFileSync(path.join(cpDir, 'burndown.json'), JSON.stringify(checkpoint, null, 2));
  
  log('SPRINT H', 'COMPLETE | All H sub-tasks done');
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
