const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

const day = 39;

(async () => {
  console.log('=== QUERY 1: lessons table (day_of_year=39) ===');
  try {
    const r1 = await sql`SELECT day_of_year, title, topic, LEFT(hook_script, 80) as hook_preview FROM lessons WHERE day_of_year = ${day}`;
    console.log(JSON.stringify(r1, null, 2));
  } catch(e) { console.log('ERROR:', e.message); }

  console.log('\n=== QUERY 2: kellyos_lessons (day_number=39, en) ===');
  try {
    const lang = 'en';
    const r2 = await sql`SELECT day_number, phase, LEFT(content_text, 80) as content_preview, language, tone FROM kellyos_lessons WHERE day_number = ${day} AND language = ${lang} LIMIT 10`;
    console.log(JSON.stringify(r2, null, 2));
  } catch(e) { console.log('ERROR:', e.message); }

  console.log('\n=== QUERY 3: kellyos_audio (day_number=39) ===');
  try {
    const r3 = await sql`SELECT day_number, phase, audio_url FROM kellyos_audio WHERE day_number = ${day}`;
    console.log(JSON.stringify(r3, null, 2));
  } catch(e) { console.log('ERROR:', e.message); }

  console.log('\n=== QUERY 4: heygen_videos (day_of_year=39) ===');
  try {
    const r4 = await sql`SELECT day_of_year, phase, video_url, age_category FROM heygen_videos WHERE day_of_year = ${day}`;
    console.log(JSON.stringify(r4, null, 2));
  } catch(e) { console.log('ERROR:', e.message); }

  console.log('\n=== QUERY 5: kellyos_facts (day_number=39) ===');
  try {
    const r5 = await sql`SELECT day_number, statement, is_true FROM kellyos_facts WHERE day_number = ${day}`;
    console.log(JSON.stringify(r5, null, 2));
  } catch(e) { console.log('ERROR:', e.message); }

  console.log('\n=== QUERY 6: core_lessons (day_number=39) ===');
  try {
    const r6 = await sql`SELECT day_number, title, subject, theme, universal_truth, icon_emoji FROM core_lessons WHERE day_number = ${day}`;
    console.log(JSON.stringify(r6, null, 2));
  } catch(e) { console.log('ERROR:', e.message); }
})();
