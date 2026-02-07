#!/usr/bin/env node
require('dotenv').config();
const fs = require('fs');
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

// Video IDs - REAL KELLY (Blue Shirt)
const VIDEOS = [
  { phase: 'hook', videoId: '1e57b239d2a94ffa88e60cf6c6dddfa0' },
  { phase: 'story', videoId: '39e3d5c9f22741ed9ce3fa069a6dd40b' },
  { phase: 'wonder', videoId: '51dd4f66666f459f987e638d26f5af07' },
  { phase: 'action', videoId: '277b8f7c873d4013affa1b243d7b0718' },
  { phase: 'wisdom', videoId: 'b0c89b4ebddd46d8b0d5167c8718643c' }
];

async function checkAndSync() {
  console.log('Checking Day 34 video status...');
  console.log('='.repeat(40));
  
  let allComplete = true;
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
      
      // Also try to insert if not exists
      await pool.query(`
        INSERT INTO heygen_videos (id, day_of_year, phase, age_category, archetype, heygen_video_id, video_url, status, created_at, updated_at)
        VALUES (gen_random_uuid(), 34, $1, 'adult', 'storyteller', $2, $3, 'completed', NOW(), NOW())
        ON CONFLICT DO NOTHING
      `, [video.phase, video.videoId, url]).catch(() => {});
      
      completed++;
    } else if (status === 'processing' || status === 'pending') {
      console.log(`⏳ ${video.phase}: ${status}`);
      allComplete = false;
    } else {
      console.log(`❌ ${video.phase}: ${status} - ${data.data?.error?.message || ''}`);
      allComplete = false;
    }
  }
  
  console.log('');
  console.log(`Completed: ${completed}/5`);
  
  if (allComplete) {
    console.log('');
    console.log('✅ ALL DAY 34 VIDEOS READY!');
    console.log('Refresh thedailylesson.com to see them');
  } else {
    console.log('');
    console.log('Not all videos ready yet. Run again in 1-2 minutes.');
  }
  
  await pool.end();
}

checkAndSync().catch(console.error);
