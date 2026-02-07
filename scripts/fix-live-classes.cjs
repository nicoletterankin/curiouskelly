require('dotenv').config();
const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.DATABASE_URL });
  await c.connect();
  try {
    await c.query('DROP TABLE IF EXISTS live_classes CASCADE');
    console.log('Dropped live_classes table');
  } catch (e) {
    console.log('Drop failed:', e.message);
  }
  try {
    await c.query(`CREATE OR REPLACE VIEW live_classes AS SELECT id, day_number, title, 'active' AS status, created_at FROM core_lessons_v2 LIMIT 10`);
    console.log('Created live_classes view');
  } catch (e) {
    console.log('View creation error:', e.message);
  }
  // Verify
  const r = await c.query('SELECT * FROM live_classes LIMIT 3');
  console.log('Verified:', r.rows.length, 'rows');
  r.rows.forEach(row => console.log(`  Day ${row.day_number}: ${row.title}`));
  await c.end();
})();
