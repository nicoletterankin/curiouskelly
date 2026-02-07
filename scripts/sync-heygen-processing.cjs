require('dotenv').config();
const {Pool} = require('pg');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function checkAndSync() {
  // Get all processing videos
  const processing = await pool.query(`
    SELECT id, day_of_year, phase, archetype, heygen_video_id
    FROM heygen_videos 
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    ORDER BY day_of_year, phase
  `);
  
  console.log(`Found ${processing.rows.length} videos in "processing" status\n`);
  
  let completed = 0;
  let stillProcessing = 0;
  let failed = 0;
  
  for (const row of processing.rows) {
    const response = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${row.heygen_video_id}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    });
    
    const data = await response.json();
    const status = data.data?.status;
    const videoUrl = data.data?.video_url;
    
    if (status === 'completed' && videoUrl) {
      await pool.query(`
        UPDATE heygen_videos 
        SET status = 'completed', video_url = $1, updated_at = NOW(), completed_at = NOW()
        WHERE id = $2
      `, [videoUrl, row.id]);
      console.log(`✅ Day ${row.day_of_year} ${row.phase}: COMPLETED`);
      completed++;
    } else if (status === 'failed') {
      await pool.query(`
        UPDATE heygen_videos SET status = 'failed', error_message = $1, updated_at = NOW()
        WHERE id = $2
      `, [data.data?.error || 'Unknown error', row.id]);
      console.log(`❌ Day ${row.day_of_year} ${row.phase}: FAILED`);
      failed++;
    } else {
      console.log(`⏳ Day ${row.day_of_year} ${row.phase}: ${status || 'pending'}`);
      stillProcessing++;
    }
    
    // Rate limit
    await new Promise(r => setTimeout(r, 500));
  }
  
  console.log(`\n=== SUMMARY ===`);
  console.log(`Completed: ${completed}`);
  console.log(`Still processing: ${stillProcessing}`);
  console.log(`Failed: ${failed}`);
  
  await pool.end();
}

checkAndSync().catch(console.error);
