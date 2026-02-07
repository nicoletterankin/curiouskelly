require('dotenv').config();
const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.DATABASE_URL });
  await c.connect();
  
  // Check alignment coverage
  const r = await c.query(`
    SELECT 
      COUNT(*) as total,
      COUNT(CASE WHEN alignment_json IS NOT NULL THEN 1 END) as with_alignment,
      COUNT(CASE WHEN alignment_json IS NOT NULL AND alignment_json::text != 'null' AND alignment_json::text != '[]' AND alignment_json::text != '{}' THEN 1 END) as valid_alignment,
      COUNT(CASE WHEN audio_url IS NOT NULL THEN 1 END) as with_audio,
      COUNT(CASE WHEN duration_seconds > 0 THEN 1 END) as with_duration
    FROM kellyos_audio
  `);
  const v = r.rows[0];
  console.log('kellyos_audio coverage:');
  console.log(`  Total: ${v.total}`);
  console.log(`  With audio_url: ${v.with_audio}`);
  console.log(`  With alignment_json: ${v.with_alignment}`);
  console.log(`  Valid alignment (non-empty): ${v.valid_alignment}`);
  console.log(`  With duration: ${v.with_duration}`);
  
  // Sample an alignment entry
  const sample = await c.query(`
    SELECT day_number, phase, alignment_json, duration_seconds 
    FROM kellyos_audio 
    WHERE alignment_json IS NOT NULL AND alignment_json::text != 'null' 
    LIMIT 1
  `);
  if (sample.rows.length > 0) {
    const s = sample.rows[0];
    const align = typeof s.alignment_json === 'string' ? JSON.parse(s.alignment_json) : s.alignment_json;
    console.log(`\nSample alignment (Day ${s.day_number}, ${s.phase}):`);
    console.log(`  Type: ${typeof align}`);
    console.log(`  Keys: ${Object.keys(align || {}).join(', ')}`);
    if (Array.isArray(align)) {
      console.log(`  Array length: ${align.length}`);
      console.log(`  First entry: ${JSON.stringify(align[0]).substring(0, 100)}`);
    } else if (align && align.characters) {
      console.log(`  Characters: ${align.characters?.length || 0}`);
    }
  }
  
  // Also check kellyos_lessons alignment  
  const kl = await c.query(`
    SELECT 
      COUNT(*) as total,
      COUNT(CASE WHEN alignment_json IS NOT NULL AND alignment_json::text != 'null' THEN 1 END) as with_alignment
    FROM kellyos_lessons
    WHERE language = 'en' OR language IS NULL
  `);
  console.log(`\nkellyos_lessons coverage:`);
  console.log(`  Total: ${kl.rows[0].total}`);
  console.log(`  With alignment: ${kl.rows[0].with_alignment}`);
  
  await c.end();
})();
