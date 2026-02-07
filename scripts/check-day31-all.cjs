require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  const result = await pool.query(`
    SELECT video_url, status, age_category, archetype, updated_at, created_at
    FROM heygen_videos 
    WHERE day_of_year = 31 AND phase = 'hook' 
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY updated_at DESC NULLS LAST, created_at DESC
    LIMIT 5
  `);
  
  console.log('Day 31 hook videos (ordered by updated_at DESC):');
  result.rows.forEach((r, i) => {
    const urlType = r.video_url?.includes('files2.heygen') ? 'HEYGEN' : 
                   r.video_url?.includes('blob.vercel') ? 'BLOB' : 'OTHER';
    console.log(`${i+1}. ${urlType} | ${r.age_category}/${r.archetype} | updated: ${r.updated_at}`);
    console.log(`   ${r.video_url?.substring(0, 60)}...`);
  });
  
  await pool.end();
}

main().catch(console.error);
