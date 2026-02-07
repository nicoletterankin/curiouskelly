#!/usr/bin/env node
require('dotenv').config();
const fs = require('fs');
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// Load video IDs from file
const data = JSON.parse(fs.readFileSync('days-31-33-videos.json', 'utf8'));
const VIDEOS = data.videos;

async function checkAndSync() {
  console.log(`Checking ${VIDEOS.length} videos...`);
  console.log('='.repeat(50));
  
  let completed = 0;
  let processing = 0;
  
  for (const video of VIDEOS) {
    const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${video.videoId}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    });
    const statusData = await res.json();
    const status = statusData.data?.status || 'unknown';
    const url = statusData.data?.video_url;
    
    if (status === 'completed' && url) {
      // Update database
      await pool.query(`
        INSERT INTO heygen_videos (id, day_of_year, phase, age_category, archetype, heygen_video_id, video_url, status, created_at, updated_at)
        VALUES (gen_random_uuid(), $1, $2, 'adult', 'storyteller', $3, $4, 'completed', NOW(), NOW())
        ON CONFLICT DO NOTHING
      `, [video.day, video.phase, video.videoId, url]).catch(() => {});
      
      await pool.query(`
        UPDATE heygen_videos SET video_url = $1, status = 'completed', updated_at = NOW()
        WHERE day_of_year = $2 AND phase = $3 AND age_category = 'adult'
      `, [url, video.day, video.phase]);
      
      completed++;
    } else if (status === 'processing' || status === 'pending' || status === 'waiting') {
      processing++;
    }
  }
  
  console.log(`✅ Completed: ${completed}/${VIDEOS.length}`);
  console.log(`⏳ Processing: ${processing}`);
  
  if (completed === VIDEOS.length) {
    console.log('\n✅ ALL DAYS 31-33 VIDEOS SYNCED!');
  } else {
    console.log('\nRun again in 1-2 minutes.');
  }
  
  await pool.end();
}

checkAndSync().catch(console.error);
