require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Check Day 1 - ordered by created_at ASC (oldest first) - what production might return
  const oldestFirst = await pool.query(`
    SELECT video_url, age_category, archetype, status, created_at, updated_at
    FROM heygen_videos
    WHERE day_of_year = 1 AND phase = 'hook'
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY created_at ASC
    LIMIT 3
  `);
  
  console.log('Day 1 - OLDEST first (created_at ASC):');
  oldestFirst.rows.forEach((r, i) => {
    const urlType = r.video_url?.includes('files2.heygen') || r.video_url?.includes('files.heygen') ? 'HEYGEN' : 
                   r.video_url?.includes('blob.vercel') ? 'BLOB' : 'OTHER';
    console.log(`${i+1}. ${urlType} | ${r.age_category}/${r.archetype}`);
    console.log(`   created: ${r.created_at}`);
    console.log(`   URL: ${r.video_url?.substring(0, 50)}...`);
  });
  
  console.log('\nDay 1 - NEWEST first (updated_at DESC):');
  const newestFirst = await pool.query(`
    SELECT video_url, age_category, archetype, status, created_at, updated_at
    FROM heygen_videos
    WHERE day_of_year = 1 AND phase = 'hook'
      AND status IN ('completed', 'placeholder', 'ready')
      AND video_url IS NOT NULL
    ORDER BY updated_at DESC NULLS LAST, created_at DESC
    LIMIT 3
  `);
  newestFirst.rows.forEach((r, i) => {
    const urlType = r.video_url?.includes('files2.heygen') || r.video_url?.includes('files.heygen') ? 'HEYGEN' : 
                   r.video_url?.includes('blob.vercel') ? 'BLOB' : 'OTHER';
    console.log(`${i+1}. ${urlType} | ${r.age_category}/${r.archetype}`);
    console.log(`   updated: ${r.updated_at}`);
  });
  
  await pool.end();
}

main().catch(console.error);
