require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Compare Day 1 vs Day 34 entries
  console.log('=== DAY 1 (WORKS) ===');
  const day1 = await pool.query(`
    SELECT age_category, archetype, status, video_url 
    FROM heygen_videos 
    WHERE day_of_year = 1 AND phase = 'hook' AND video_url IS NOT NULL
    ORDER BY updated_at DESC
    LIMIT 3
  `);
  day1.rows.forEach(r => console.log(`${r.age_category}/${r.archetype}/${r.status}: ${r.video_url?.substring(0, 50)}...`));
  
  console.log('\n=== DAY 34 (FAILS) ===');
  const day34 = await pool.query(`
    SELECT age_category, archetype, status, video_url 
    FROM heygen_videos 
    WHERE day_of_year = 34 AND phase = 'hook' AND video_url IS NOT NULL
    ORDER BY updated_at DESC
    LIMIT 3
  `);
  day34.rows.forEach(r => console.log(`${r.age_category}/${r.archetype}/${r.status}: ${r.video_url?.substring(0, 50)}...`));
  
  // Check if the status values are exactly the same
  console.log('\n=== STATUS VALUES ===');
  const statuses = await pool.query(`
    SELECT DISTINCT status FROM heygen_videos WHERE day_of_year IN (1, 34) AND video_url IS NOT NULL
  `);
  statuses.rows.forEach(r => console.log(`"${r.status}"`));
  
  await pool.end();
}

main().catch(console.error);
