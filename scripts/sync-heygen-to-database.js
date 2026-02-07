#!/usr/bin/env node
/**
 * HEYGEN → DATABASE SYNC SCRIPT
 * 
 * Polls HeyGen for completed videos and updates the Neon database
 * so videos appear on thedailylesson.com
 * 
 * Run: node scripts/sync-heygen-to-database.js
 */

import { config } from 'dotenv';
import pg from 'pg';

config();
config({ path: '.env.local' });

const { Pool } = pg;

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const DATABASE_URL = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;

if (!DATABASE_URL) {
  console.error('❌ DATABASE_URL not set');
  process.exit(1);
}

const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// Parse video title to extract day, phase
function parseTitle(title) {
  // Format: "Day 1 - HOOK" or "Day 14 - WISDOM"
  const match = title?.match(/Day\s+(\d+)\s*-\s*(\w+)/i);
  if (!match) return null;
  return {
    day: parseInt(match[1]),
    phase: match[2].toLowerCase()
  };
}

// Get list of recent videos from HeyGen
async function getHeyGenVideos(limit = 100) {
  console.log(`📥 Fetching ${limit} recent videos from HeyGen...`);
  
  const response = await fetch(`https://api.heygen.com/v1/video.list?limit=${limit}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  
  const data = await response.json();
  
  if (data.error) {
    console.error('HeyGen API error:', data.error);
    return [];
  }
  
  return data.data?.videos || [];
}

// Get video status with URL
async function getVideoStatus(videoId) {
  const response = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  
  return response.json();
}

// Update heygen_videos table
async function updateHeyGenVideosTable(video) {
  const parsed = parseTitle(video.title);
  if (!parsed) {
    console.log(`  ⚠️ Could not parse title: ${video.title}`);
    return false;
  }
  
  const { day, phase } = parsed;
  
  try {
    // Upsert into heygen_videos
    await pool.query(`
      INSERT INTO heygen_videos (
        day_of_year, phase, age_category, archetype, language,
        heygen_video_id, video_url, status, created_at, updated_at
      ) VALUES (
        $1, $2, 'adult', 'storyteller', 'en',
        $3, $4, 'completed', NOW(), NOW()
      )
      ON CONFLICT (day_of_year, phase, age_category, archetype, language)
      DO UPDATE SET 
        video_url = $4,
        heygen_video_id = $3,
        status = 'completed',
        updated_at = NOW()
    `, [day, phase, video.video_id, video.video_url]);
    
    return true;
  } catch (err) {
    console.log(`  ❌ DB error for Day ${day} ${phase}:`, err.message);
    return false;
  }
}

// Update kelly_lesson_assets table (legacy compatibility)
async function updateKellyLessonAssets(video) {
  const parsed = parseTitle(video.title);
  if (!parsed) return false;
  
  const { day, phase } = parsed;
  
  try {
    await pool.query(`
      UPDATE kelly_lesson_assets 
      SET video_url = $1, video_source = 'heygen', updated_at = NOW()
      WHERE day_number = $2 AND phase = $3 AND age_group = 'adult'
    `, [video.video_url, day, phase]);
    
    return true;
  } catch (err) {
    // Table might not exist or row might not exist - that's OK
    return false;
  }
}

async function main() {
  console.log('🔄 HeyGen → Database Sync Script');
  console.log('================================\n');
  
  // Get recent videos from HeyGen
  const videos = await getHeyGenVideos(200);
  console.log(`Found ${videos.length} videos in HeyGen\n`);
  
  let synced = 0;
  let skipped = 0;
  let errors = 0;
  
  for (const video of videos) {
    // Only process completed videos
    if (video.status !== 'completed') {
      skipped++;
      continue;
    }
    
    // Skip if no video URL
    if (!video.video_url) {
      console.log(`  ⏳ ${video.title} - No URL yet (still processing?)`);
      skipped++;
      continue;
    }
    
    console.log(`  📹 ${video.title}`);
    
    // Update both tables
    const heygenResult = await updateHeyGenVideosTable(video);
    const kellyResult = await updateKellyLessonAssets(video);
    
    if (heygenResult) {
      console.log(`     ✅ Updated heygen_videos`);
      synced++;
    } else {
      errors++;
    }
    
    // Small delay to avoid rate limiting
    await new Promise(r => setTimeout(r, 100));
  }
  
  console.log('\n================================');
  console.log('📊 SYNC COMPLETE');
  console.log(`  ✅ Synced: ${synced}`);
  console.log(`  ⏭️ Skipped: ${skipped}`);
  console.log(`  ❌ Errors: ${errors}`);
  
  // Verify by checking database
  console.log('\n📋 Verifying database...');
  
  const result = await pool.query(`
    SELECT day_of_year, phase, status, 
           CASE WHEN video_url IS NOT NULL THEN 'YES' ELSE 'NO' END as has_url
    FROM heygen_videos 
    WHERE status = 'completed' AND video_url IS NOT NULL
    ORDER BY day_of_year, phase
    LIMIT 20
  `);
  
  console.log(`\nFound ${result.rowCount} completed videos in heygen_videos table:`);
  for (const row of result.rows) {
    console.log(`  Day ${row.day_of_year} / ${row.phase}: ${row.has_url}`);
  }
  
  await pool.end();
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
