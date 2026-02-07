require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Search for Blob URLs in heygen_videos
  const blobUrls = await pool.query(`
    SELECT id, day_of_year, phase, age_category, archetype, video_url, created_at, updated_at
    FROM heygen_videos
    WHERE video_url LIKE '%blob.vercel%'
    LIMIT 10
  `);
  
  console.log('Rows with Blob URLs in heygen_videos:', blobUrls.rows.length);
  blobUrls.rows.forEach(r => {
    console.log(`Day ${r.day_of_year}/${r.phase}/${r.age_category}/${r.archetype}:`);
    console.log(`  URL: ${r.video_url?.substring(0, 60)}...`);
    console.log(`  created: ${r.created_at}`);
    console.log(`  updated: ${r.updated_at}`);
  });
  
  // Also check kelly_lesson_assets which might be queried before heygen_videos
  console.log('\n--- kelly_lesson_assets ---');
  const kla = await pool.query(`
    SELECT day_number, phase, age_group, video_url
    FROM kelly_lesson_assets
    WHERE video_url LIKE '%blob.vercel%'
      AND day_number = 1 AND phase = 'hook'
    LIMIT 5
  `);
  
  console.log('Day 1 hook Blob URLs in kelly_lesson_assets:', kla.rows.length);
  kla.rows.forEach(r => {
    console.log(`  ${r.age_group}: ${r.video_url?.substring(0, 60)}...`);
  });
  
  await pool.end();
}

main().catch(console.error);
