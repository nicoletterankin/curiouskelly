require('dotenv').config();
const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.DATABASE_URL });
  await c.connect();

  // Check constraint values
  const checks = await c.query(`
    SELECT conname, pg_get_constraintdef(oid) as def
    FROM pg_constraint
    WHERE conrelid = 'lesson_atoms'::regclass AND contype = 'c'
  `);
  console.log('Check constraints:');
  checks.rows.forEach(r => console.log(`  ${r.conname}: ${r.def}`));

  // Check lesson_id → day_number mapping
  const mapping = await c.query(`
    SELECT id, day_number FROM core_lessons_v2 ORDER BY day_number LIMIT 10
  `);
  console.log('\ncore_lessons_v2 id→day mapping (first 10):');
  mapping.rows.forEach(r => console.log(`  id=${r.id} → day=${r.day_number}`));

  // Check existing lesson_atoms data
  const existing = await c.query('SELECT COUNT(*) as cnt FROM lesson_atoms');
  console.log(`\nExisting lesson_atoms: ${existing.rows[0].cnt}`);
  
  const sample = await c.query('SELECT * FROM lesson_atoms LIMIT 3');
  console.log('Sample:', JSON.stringify(sample.rows, null, 2));

  await c.end();
})();
