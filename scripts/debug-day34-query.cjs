require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Exact query the API uses
  const ageGroup = 'adult'; // age 30 -> adult
  const archetype = 'storyteller';
  const phase = 'hook';
  const dayNumber = 34;
  
  console.log('Running EXACT API query for Day 34...');
  console.log(`ageGroup: ${ageGroup}, archetype: ${archetype}, phase: ${phase}`);
  
  const result = await pool.query(`
    SELECT video_url, audio_url, script, thumbnail_url, age_category, archetype, day_of_year, status
    FROM heygen_videos
    WHERE day_of_year = $1
      AND phase = $2
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY 
      CASE WHEN age_category = $3 THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END,
      CASE WHEN archetype = $4 THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END,
      updated_at DESC NULLS LAST,
      created_at DESC
    LIMIT 1
  `, [dayNumber, phase, ageGroup, archetype]);
  
  console.log('Query returned:', result.rows.length, 'rows');
  
  if (result.rows.length > 0) {
    const row = result.rows[0];
    console.log('First row:');
    console.log('  age_category:', row.age_category);
    console.log('  archetype:', row.archetype);
    console.log('  status:', row.status);
    console.log('  video_url:', row.video_url?.substring(0, 70));
    console.log('  Is HeyGen:', row.video_url?.includes('files2.heygen'));
  } else {
    console.log('NO ROWS RETURNED!');
    
    // Debug: show what's actually there
    const all = await pool.query(`
      SELECT age_category, archetype, status, video_url 
      FROM heygen_videos 
      WHERE day_of_year = 34 AND phase = 'hook' AND video_url IS NOT NULL
      LIMIT 10
    `);
    console.log('\nAll Day 34 hook entries:');
    all.rows.forEach(r => console.log(`  ${r.age_category}/${r.archetype}/${r.status}: ${r.video_url?.substring(0, 40)}...`));
  }
  
  await pool.end();
}

main().catch(console.error);
