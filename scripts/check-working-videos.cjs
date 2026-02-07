require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Get existing working videos to see what avatar was used
  const result = await pool.query(`
    SELECT heygen_video_id, avatar_key, video_url, day_of_year, phase
    FROM heygen_videos 
    WHERE video_url IS NOT NULL AND video_url LIKE '%heygen%'
    LIMIT 5
  `);
  
  console.log('WORKING VIDEOS (to copy their settings):');
  result.rows.forEach(row => {
    console.log(`Day ${row.day_of_year} ${row.phase}:`);
    console.log(`  HeyGen ID: ${row.heygen_video_id}`);
    console.log(`  Avatar: ${row.avatar_key}`);
    console.log(`  URL: ${row.video_url?.substring(0, 70)}...`);
  });
  
  // Check one of these videos to see what was sent to HeyGen
  if (result.rows[0]?.heygen_video_id) {
    const videoId = result.rows[0].heygen_video_id;
    console.log('\nChecking HeyGen status for:', videoId);
    
    const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': process.env.HEYGEN_API_KEY }
    });
    const data = await res.json();
    console.log('Status:', data.data?.status);
  }
  
  await pool.end();
}

main().catch(console.error);
