#!/usr/bin/env node
require('dotenv').config();
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// REAL Curious Kelly videos
const VIDEOS = [
  { phase: 'hook', videoId: '1cb5222e20a74348b41786b17dec410b' },
  { phase: 'story', videoId: '600e05f3b67c4ae99fa82cbcae9d6aa5' },
  { phase: 'wonder', videoId: '27a9eaf679da48d3928d1257610496ce' },
  { phase: 'action', videoId: '4575921af41f41bb9d278f4f0be3725f' },
  { phase: 'wisdom', videoId: '14641def9bd94ad0a3589a1104339bf2' }
];

async function checkAndSync() {
  console.log('Checking Day 34 REAL Curious Kelly videos...');
  console.log('='.repeat(50));
  
  let completed = 0;
  
  for (const video of VIDEOS) {
    const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${video.videoId}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    });
    const data = await res.json();
    const status = data.data?.status || 'unknown';
    const url = data.data?.video_url;
    
    if (status === 'completed' && url) {
      console.log(`✅ ${video.phase}: READY`);
      
      // Update database
      await pool.query(`
        UPDATE heygen_videos 
        SET video_url = $1, status = 'completed', updated_at = NOW()
        WHERE day_of_year = 34 AND phase = $2 AND age_category = 'adult'
      `, [url, video.phase]);
      
      // Insert if not exists
      await pool.query(`
        INSERT INTO heygen_videos (id, day_of_year, phase, age_category, archetype, heygen_video_id, video_url, status, created_at, updated_at)
        VALUES (gen_random_uuid(), 34, $1, 'adult', 'storyteller', $2, $3, 'completed', NOW(), NOW())
        ON CONFLICT DO NOTHING
      `, [video.phase, video.videoId, url]).catch(() => {});
      
      completed++;
    } else if (status === 'processing' || status === 'pending' || status === 'waiting') {
      console.log(`⏳ ${video.phase}: ${status}`);
    } else {
      console.log(`❌ ${video.phase}: ${status}`);
    }
  }
  
  console.log('');
  console.log(`Completed: ${completed}/5`);
  
  if (completed === 5) {
    console.log('');
    console.log('✅ ALL DAY 34 REAL CURIOUS KELLY VIDEOS READY!');
  } else {
    console.log('');
    console.log('Run again in 1-2 minutes.');
  }
  
  await pool.end();
}

checkAndSync().catch(console.error);
