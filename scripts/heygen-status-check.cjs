require('dotenv').config();
const {Pool} = require('pg');
const pool = new Pool({connectionString: process.env.DATABASE_URL, ssl:{rejectUnauthorized:false}});

async function main() {
  // Count videos by day
  const byDay = await pool.query(`
    SELECT day_of_year, COUNT(*) as count
    FROM heygen_videos 
    WHERE status = 'completed' AND video_url IS NOT NULL AND video_url LIKE '%heygen%'
    GROUP BY day_of_year
    ORDER BY day_of_year
  `);
  
  console.log('=== HEYGEN VIDEOS BY DAY ===');
  let total = 0;
  byDay.rows.forEach(r => {
    console.log(`Day ${String(r.day_of_year).padStart(2)}: ${r.count} videos`);
    total += parseInt(r.count);
  });
  console.log(`\nTOTAL: ${total} HeyGen videos in database`);
  
  // Check what days are missing between 1-40
  const existingDays = byDay.rows.map(r => r.day_of_year);
  const missingDays = [];
  for (let d = 1; d <= 40; d++) {
    if (!existingDays.includes(d)) missingDays.push(d);
  }
  console.log('\nMissing days (1-40):', missingDays.length > 0 ? missingDays.join(', ') : 'NONE');
  
  // Check recent video generation
  const recent = await pool.query(`
    SELECT day_of_year, phase, archetype, updated_at
    FROM heygen_videos 
    WHERE status = 'completed' AND video_url IS NOT NULL
    ORDER BY updated_at DESC
    LIMIT 5
  `);
  console.log('\n=== MOST RECENT VIDEOS ===');
  recent.rows.forEach(r => {
    console.log(`Day ${r.day_of_year} ${r.phase} (${r.archetype}) - ${r.updated_at}`);
  });
  
  await pool.end();
}

main().catch(console.error);
