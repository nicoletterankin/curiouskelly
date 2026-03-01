const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');
const day = 39;
(async () => {
  console.log('=== core_lessons columns ===');
  try {
    const cols = await sql`SELECT column_name FROM information_schema.columns WHERE table_name = 'core_lessons' ORDER BY ordinal_position`;
    console.log(cols.map(c => c.column_name).join(', '));
  } catch(e) { console.log('ERROR:', e.message); }

  console.log('\n=== core_lessons row for day 39 ===');
  try {
    const r = await sql`SELECT * FROM core_lessons WHERE day_number = ${day}`;
    console.log(JSON.stringify(r, null, 2));
  } catch(e) { console.log('ERROR:', e.message); }

  console.log('\n=== kellyos_lessons columns ===');
  try {
    const cols2 = await sql`SELECT column_name FROM information_schema.columns WHERE table_name = 'kellyos_lessons' ORDER BY ordinal_position`;
    console.log(cols2.map(c => c.column_name).join(', '));
  } catch(e) { console.log('ERROR:', e.message); }

  console.log('\n=== kellyos_audio columns ===');
  try {
    const cols3 = await sql`SELECT column_name FROM information_schema.columns WHERE table_name = 'kellyos_audio' ORDER BY ordinal_position`;
    console.log(cols3.map(c => c.column_name).join(', '));
  } catch(e) { console.log('ERROR:', e.message); }
})();
