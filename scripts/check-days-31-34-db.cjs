require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  for (const day of [31, 32, 33, 34]) {
    const result = await pool.query(`
      SELECT video_url, status, age_category, archetype
      FROM heygen_videos 
      WHERE day_of_year = $1 AND phase = 'hook' 
        AND status IN ('completed', 'placeholder', 'ready')
        AND video_url IS NOT NULL
        AND (age_category = 'adult' OR age_category = 'youngAdult' OR age_category = 'middleAge')
      ORDER BY 
        CASE WHEN age_category = 'adult' THEN 0 ELSE 1 END,
        CASE WHEN archetype = 'storyteller' THEN 0 ELSE 1 END
      LIMIT 1
    `, [day]);
    
    if (result.rows.length > 0) {
      const r = result.rows[0];
      const urlPrefix = r.video_url?.includes('files2.heygen') ? 'HEYGEN' : 
                       r.video_url?.includes('blob.vercel') ? 'BLOB' : 'OTHER';
      console.log(`Day ${day}: ${urlPrefix} | ${r.age_category}/${r.archetype} | ${r.status}`);
    } else {
      console.log(`Day ${day}: NO MATCHING ROWS`);
    }
  }
  
  await pool.end();
}

main().catch(console.error);
