require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Check Day 34 ordered by created_at DESC (what old production code uses)
  const result = await pool.query(`
    SELECT video_url, age_category, archetype, status, created_at
    FROM heygen_videos
    WHERE day_of_year = 34
      AND phase = 'hook'
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY 
      CASE WHEN age_category = 'adult' THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END,
      CASE WHEN archetype = 'storyteller' THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END,
      created_at DESC
    LIMIT 3
  `);
  
  console.log('Day 34 hook with OLD query (created_at DESC only):');
  result.rows.forEach((r, i) => {
    const urlType = r.video_url?.includes('files2.heygen') ? 'HEYGEN' : 'BLOB';
    console.log(`${i+1}. ${urlType} | ${r.age_category}/${r.archetype}`);
    console.log(`   created: ${r.created_at}`);
  });
  
  // Also check if there are entries with NULL video_url that might mess things up
  const allStatuses = await pool.query(`
    SELECT status, COUNT(*) as count
    FROM heygen_videos
    WHERE day_of_year = 34 AND phase = 'hook'
    GROUP BY status
  `);
  console.log('\nAll Day 34 hook by status:');
  allStatuses.rows.forEach(r => console.log(`  ${r.status}: ${r.count}`));
  
  await pool.end();
}

main().catch(console.error);
