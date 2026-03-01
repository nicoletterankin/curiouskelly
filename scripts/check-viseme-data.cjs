const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');
const day = 39;
(async () => {
  console.log('=== viseme_timeline sample for day 39 ===');
  try {
    const r = await sql`SELECT phase, viseme_timeline, srt_text FROM kellyos_audio WHERE day_number = ${day} AND phase = 'hook' LIMIT 1`;
    console.log(JSON.stringify(r[0], null, 2));
  } catch(e) { console.log('ERROR:', e.message); }
})();
