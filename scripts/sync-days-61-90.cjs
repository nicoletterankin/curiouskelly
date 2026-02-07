require('dotenv').config();
const {Pool} = require('pg');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function sync() {
  // Get all processing videos for days 61-90
  const processing = await pool.query(`
    SELECT id, day_of_year, phase, heygen_video_id
    FROM heygen_videos 
    WHERE day_of_year BETWEEN 61 AND 90
      AND status = 'processing' 
      AND heygen_video_id IS NOT NULL
    ORDER BY day_of_year, phase
  `);
  
  console.log(`Found ${processing.rows.length} videos to sync\n`);
  
  let completed = 0, stillProcessing = 0, failed = 0;
  
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
      completed++;
      if (completed % 10 === 0) console.log(`✅ ${completed} completed...`);
    } else if (status === 'failed') {
      await pool.query(`UPDATE heygen_videos SET status = 'failed' WHERE id = $1`, [row.id]);
      failed++;
    } else {
      stillProcessing++;
    }
    
    await new Promise(r => setTimeout(r, 300));
  }
  
  console.log(`\n=== SUMMARY ===`);
  console.log(`Completed: ${completed}`);
  console.log(`Still processing: ${stillProcessing}`);
  console.log(`Failed: ${failed}`);
  
  await pool.end();
}

sync().catch(console.error);
