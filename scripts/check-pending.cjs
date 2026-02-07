require('dotenv').config();
const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.DATABASE_URL });
  await c.connect();
  const r = await c.query("SELECT la.phase, la.variant, COUNT(*) as cnt FROM lesson_atoms la WHERE la.status='pending' GROUP BY la.phase, la.variant ORDER BY la.phase");
  r.rows.forEach(r => console.log('Phase', r.phase, '(' + r.variant + '):', r.cnt, 'pending'));
  const t = await c.query("SELECT COUNT(*) as cnt FROM lesson_atoms WHERE status='pending'");
  console.log('Total pending:', t.rows[0].cnt);
  // Which days have all pending
  const days = await c.query("SELECT cl.day_number FROM core_lessons_v2 cl WHERE NOT EXISTS (SELECT 1 FROM lesson_atoms la JOIN lesson_scripts ls ON ls.atom_id=la.id WHERE la.lesson_id=cl.id) ORDER BY cl.day_number");
  console.log('Days with NO scripts:', days.rows.length, '(first 10:', days.rows.slice(0, 10).map(r => r.day_number).join(', ') + ')');
  await c.end();
})();
