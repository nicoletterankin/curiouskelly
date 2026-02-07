/**
 * Sync all processing videos from HeyGen
 * Checks status and updates video_url when complete
 */

const { neon } = require('@neondatabase/serverless');
require('dotenv').config();

const DATABASE_URL = 'postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require';
const sql = neon(DATABASE_URL);

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';

async function syncAll() {
  console.log('🔄 Syncing all processing videos...\n');
  
  const processing = await sql`
    SELECT id, heygen_video_id, day_of_year, phase, age_category, archetype
    FROM heygen_videos 
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    ORDER BY day_of_year, age_category, phase
  `;
  
  console.log(`Found ${processing.length} videos in processing state\n`);
  
  let completed = 0, failed = 0, stillProcessing = 0;
  
  for (const video of processing) {
    try {
      const response = await fetch(
        `https://api.heygen.com/v1/video_status.get?video_id=${video.heygen_video_id}`,
        { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
      );
      const data = await response.json();
      
      const key = `Day${video.day_of_year}/${video.age_category}/${video.phase}`;
      
      if (data.data?.status === 'completed' && data.data?.video_url) {
        await sql`
          UPDATE heygen_videos 
          SET status = 'completed', 
              video_url = ${data.data.video_url},
              updated_at = NOW()
          WHERE id = ${video.id}
        `;
        console.log(`✅ ${key} - COMPLETED`);
        completed++;
      } else if (data.data?.status === 'failed') {
        await sql`
          UPDATE heygen_videos 
          SET status = 'failed', 
              error_message = ${data.data.error || 'Unknown error'},
              updated_at = NOW()
          WHERE id = ${video.id}
        `;
        console.log(`❌ ${key} - FAILED: ${data.data.error || 'unknown'}`);
        failed++;
      } else if (data.data?.status === 'pending' || data.data?.status === 'processing') {
        stillProcessing++;
        // Don't log every pending one to reduce noise
      } else {
        console.log(`? ${key} - Unknown status: ${data.data?.status}`);
      }
    } catch (e) {
      console.log(`⚠️ Error checking ${video.heygen_video_id}: ${e.message}`);
    }
    
    // Rate limit
    await new Promise(r => setTimeout(r, 150));
  }
  
  console.log(`\n════════════════════════════════════════════════════════════`);
  console.log(`📊 SYNC COMPLETE:`);
  console.log(`   Completed: ${completed}`);
  console.log(`   Failed: ${failed}`);
  console.log(`   Still processing: ${stillProcessing}`);
  console.log(`════════════════════════════════════════════════════════════`);
  
  // Show current totals
  const totals = await sql`
    SELECT status, COUNT(*) as count 
    FROM heygen_videos 
    GROUP BY status 
    ORDER BY count DESC
  `;
  console.log(`\n📈 DATABASE TOTALS:`);
  totals.forEach(r => console.log(`   ${r.status}: ${r.count}`));
}

syncAll().catch(console.error);
