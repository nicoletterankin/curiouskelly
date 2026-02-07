require('dotenv').config();
const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.DATABASE_URL });
  await c.connect();
  
  // Check current kellyos_audio schema
  const cols = await c.query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='kellyos_audio' ORDER BY ordinal_position");
  console.log('kellyos_audio columns:');
  cols.rows.forEach(r => console.log(`  ${r.column_name} - ${r.data_type}`));
  
  // Check row count
  const cnt = await c.query('SELECT COUNT(*) as cnt FROM kellyos_audio');
  console.log(`\nRows: ${cnt.rows[0].cnt}`);
  
  // Check existing constraints
  const constraints = await c.query("SELECT conname, contype FROM pg_constraint WHERE conrelid='kellyos_audio'::regclass");
  console.log('\nExisting constraints:');
  constraints.rows.forEach(r => console.log(`  ${r.conname} (${r.contype})`));
  
  // Add unique constraint if missing
  try {
    await c.query('ALTER TABLE kellyos_audio ADD CONSTRAINT kellyos_audio_day_phase_uniq UNIQUE (day_number, phase)');
    console.log('\nAdded unique constraint on (day_number, phase)');
  } catch (e) {
    if (e.message.includes('already exists')) {
      console.log('\nUnique constraint already exists');
    } else if (e.message.includes('duplicate key')) {
      // Deduplicate first
      console.log('\nDuplicates exist, deduplicating...');
      await c.query(`
        DELETE FROM kellyos_audio a USING kellyos_audio b
        WHERE a.id > b.id AND a.day_number = b.day_number AND a.phase = b.phase
      `);
      const cnt2 = await c.query('SELECT COUNT(*) as cnt FROM kellyos_audio');
      console.log(`After dedup: ${cnt2.rows[0].cnt} rows`);
      await c.query('ALTER TABLE kellyos_audio ADD CONSTRAINT kellyos_audio_day_phase_uniq UNIQUE (day_number, phase)');
      console.log('Added unique constraint');
    } else {
      console.log('Constraint error:', e.message);
    }
  }
  
  // Also check which slots already have audio
  const existing = await c.query(`
    SELECT COUNT(*) as cnt FROM kellyos_audio WHERE audio_url IS NOT NULL
  `);
  console.log(`\nSlots with audio_url: ${existing.rows[0].cnt}`);
  
  // Check which slots need audio
  const missing = await c.query(`
    SELECT kl.day_number, kl.phase FROM kellyos_lessons kl
    LEFT JOIN kellyos_audio ka ON ka.day_number = kl.day_number AND ka.phase = kl.phase AND ka.audio_url IS NOT NULL
    WHERE (kl.language = 'en' OR kl.language IS NULL) AND ka.id IS NULL
    ORDER BY kl.day_number, kl.phase
  `);
  console.log(`Slots needing audio: ${missing.rows.length}`);
  
  await c.end();
})();
