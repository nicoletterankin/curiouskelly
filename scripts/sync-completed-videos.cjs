#!/usr/bin/env node
/**
 * SYNC COMPLETED HEYGEN VIDEOS TO DATABASE
 * 
 * Reads video IDs from batch progress, checks status, updates database
 * 
 * Run: node scripts/sync-completed-videos.cjs
 */

require('dotenv').config();

const fs = require('fs');
const { Pool } = require('pg');

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const DATABASE_URL = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;

const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

async function getVideoStatus(videoId) {
  const response = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  return response.json();
}

async function updateDatabase(day, phase, videoId, videoUrl) {
  try {
    // Check if heygen_videos table exists and has the right structure
    const result = await pool.query(`
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
      RETURNING id
    `, [day, phase, videoId, videoUrl]);
    
    return { success: true, id: result.rows[0]?.id };
  } catch (err) {
    // If conflict constraint doesn't exist, try without ON CONFLICT
    try {
      // First try to update
      const updateResult = await pool.query(`
        UPDATE heygen_videos 
        SET video_url = $4, heygen_video_id = $3, status = 'completed', updated_at = NOW()
        WHERE day_of_year = $1 AND phase = $2 AND age_category = 'adult' AND archetype = 'storyteller'
        RETURNING id
      `, [day, phase, videoId, videoUrl]);
      
      if (updateResult.rowCount === 0) {
        // Insert new row
        await pool.query(`
          INSERT INTO heygen_videos (
            day_of_year, phase, age_category, archetype, language,
            heygen_video_id, video_url, status, created_at, updated_at
          ) VALUES (
            $1, $2, 'adult', 'storyteller', 'en', $3, $4, 'completed', NOW(), NOW()
          )
        `, [day, phase, videoId, videoUrl]);
      }
      return { success: true };
    } catch (err2) {
      return { success: false, error: err2.message };
    }
  }
}

async function main() {
  console.log('🔄 Syncing Completed HeyGen Videos to Database');
  console.log('==============================================\n');
  
  // Read progress file
  let progress;
  try {
    progress = JSON.parse(fs.readFileSync('heygen-batch-progress.json', 'utf8'));
  } catch (err) {
    console.error('❌ Could not read heygen-batch-progress.json:', err.message);
    process.exit(1);
  }
  
  console.log(`📋 Found ${progress.videos.length} videos in progress file\n`);
  
  let synced = 0;
  let pending = 0;
  let errors = 0;
  
  for (const video of progress.videos) {
    process.stdout.write(`Day ${video.day.toString().padStart(2)} / ${video.phase.padEnd(6)}: `);
    
    // Get status from HeyGen
    const status = await getVideoStatus(video.videoId);
    
    if (status.data?.status === 'completed' && status.data?.video_url) {
      // Update database
      const dbResult = await updateDatabase(
        video.day,
        video.phase,
        video.videoId,
        status.data.video_url
      );
      
      if (dbResult.success) {
        console.log(`✅ Synced (${status.data.video_url.substring(0, 40)}...)`);
        synced++;
      } else {
        console.log(`❌ DB Error: ${dbResult.error}`);
        errors++;
      }
    } else if (status.data?.status === 'processing' || status.data?.status === 'pending') {
      console.log(`⏳ Still processing...`);
      pending++;
    } else {
      console.log(`❓ Unknown status: ${status.data?.status || status.error?.message || 'no response'}`);
      errors++;
    }
    
    // Small delay between API calls
    await new Promise(r => setTimeout(r, 200));
  }
  
  console.log('\n==============================================');
  console.log('📊 SYNC SUMMARY');
  console.log(`  ✅ Synced to DB: ${synced}`);
  console.log(`  ⏳ Still processing: ${pending}`);
  console.log(`  ❌ Errors: ${errors}`);
  
  // Verify database
  console.log('\n📋 Database verification:');
  const dbCheck = await pool.query(`
    SELECT COUNT(*) as total,
           COUNT(video_url) as with_url
    FROM heygen_videos 
    WHERE status = 'completed'
  `);
  console.log(`  Total completed: ${dbCheck.rows[0].total}`);
  console.log(`  With video URL: ${dbCheck.rows[0].with_url}`);
  
  await pool.end();
  console.log('\n✅ Done!');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
