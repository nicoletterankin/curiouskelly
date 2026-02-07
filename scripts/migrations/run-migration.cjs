/**
 * Run SQL migrations against Neon database
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

async function runMigration(sqlFile) {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log(`Running migration: ${sqlFile}`);
  const sql = fs.readFileSync(sqlFile, 'utf-8');
  
  try {
    await client.query(sql);
    console.log('Migration complete.');
    
    // Verify tables created
    const tables = ['core_lessons_v2', 'lesson_atoms', 'lesson_scripts', 'generation_jobs'];
    for (const table of tables) {
      const res = await client.query(`SELECT COUNT(*) as cnt FROM ${table}`);
      console.log(`  ${table}: ${res.rows[0].cnt} rows`);
    }
  } catch (e) {
    console.error('Migration error:', e.message);
    throw e;
  } finally {
    await client.end();
  }
}

const migrationFile = process.argv[2] || path.join(__dirname, '001_core_schema.sql');
runMigration(migrationFile).catch(e => process.exit(1));
