require('dotenv').config();
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || process.env.NEON_DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

async function main() {
  // Check if ANY videos have URLs
  const countResult = await pool.query(`
    SELECT COUNT(*) as total,
           COUNT(video_url) as with_url
    FROM heygen_videos
  `);
  console.log('Total rows:', countResult.rows[0].total);
  console.log('With video_url:', countResult.rows[0].with_url);
  
  // Get any row with a video URL
  const result = await pool.query(`
    SELECT day_of_year, phase, video_url, heygen_video_id, updated_at 
    FROM heygen_videos 
    WHERE video_url IS NOT NULL
    LIMIT 5
  `);
  
  console.log('\nRows with video_url:');
  for (const row of result.rows) {
    console.log(`  Day ${row.day_of_year} ${row.phase}: ${row.video_url?.substring(0, 50)}...`);
  }
  if (result.rows.length === 0) {
    console.log('  NONE FOUND');
  }
  
  await pool.end();
}

main().catch(console.error);
