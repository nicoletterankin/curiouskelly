require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Check what the API query would return
  const result = await pool.query(`
    SELECT video_url, status, age_category, archetype, updated_at
    FROM heygen_videos 
    WHERE day_of_year = 34 AND phase = 'hook' 
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY updated_at DESC NULLS LAST
    LIMIT 5
  `);
  
  console.log('Day 34 hook videos matching API query:');
  if (result.rows.length === 0) {
    console.log('NO MATCHING ROWS - this is why API falls back!');
  } else {
    result.rows.forEach((row, i) => {
      console.log(`${i+1}. ${row.age_category}/${row.archetype}`);
      console.log(`   URL: ${row.video_url?.substring(0, 60)}...`);
      console.log(`   Status: ${row.status}`);
    });
  }
  
  // Also check without status filter
  const all = await pool.query(`
    SELECT status, COUNT(*) as count FROM heygen_videos 
    WHERE day_of_year = 34 AND phase = 'hook'
    GROUP BY status
  `);
  console.log('\nDay 34 hook by status:');
  all.rows.forEach(r => console.log(`  ${r.status}: ${r.count}`));
  
  // Check if storyteller specifically has video
  const storyteller = await pool.query(`
    SELECT video_url FROM heygen_videos 
    WHERE day_of_year = 34 AND phase = 'hook' AND archetype = 'storyteller'
      AND status IN ('completed', 'placeholder', 'ready') AND video_url IS NOT NULL
    LIMIT 1
  `);
  console.log('\nStoryteller (API default) has video:', storyteller.rows.length > 0 ? 'YES' : 'NO');
  if (storyteller.rows.length > 0) {
    console.log('URL:', storyteller.rows[0].video_url?.substring(0, 60));
  }
  
  await pool.end();
}

main().catch(console.error);
