require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  for (const day of [1, 18, 31, 32, 33, 34]) {
    // Run exact same query as API
    const result = await pool.query(`
      SELECT video_url, age_category, archetype, status
      FROM heygen_videos
      WHERE day_of_year = $1
        AND phase = 'hook'
        AND status IN ('completed', 'placeholder', 'ready')
        AND video_url IS NOT NULL
      ORDER BY 
        CASE WHEN age_category = 'adult' THEN 0 WHEN age_category = 'adult' THEN 1 ELSE 2 END,
        CASE WHEN archetype = 'storyteller' THEN 0 WHEN archetype = 'storyteller' THEN 1 ELSE 2 END,
        updated_at DESC NULLS LAST,
        created_at DESC
      LIMIT 1
    `, [day]);
    
    if (result.rows.length > 0) {
      const r = result.rows[0];
      const urlType = r.video_url?.includes('files2.heygen') || r.video_url?.includes('files.heygen') ? 'HEYGEN' : 
                     r.video_url?.includes('blob.vercel') ? 'BLOB' : 'OTHER';
      console.log(`Day ${String(day).padStart(2)}: ${urlType.padEnd(6)} | ${r.age_category}/${r.archetype} | ${r.status}`);
    } else {
      console.log(`Day ${String(day).padStart(2)}: NO ROWS matching query`);
    }
  }
  
  await pool.end();
}

main().catch(console.error);
