/**
 * Poll HeyGen for completed videos and update database
 */
const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';

async function checkVideo(videoId) {
  const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  return (await res.json()).data;
}

async function poll() {
  console.log('Polling HeyGen for completed videos...\n');
  
  // Get all processing videos
  const processing = await sql`
    SELECT id, heygen_video_id, day_of_year, phase, age_category
    FROM heygen_videos 
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    LIMIT 100
  `;
  
  console.log(`Checking ${processing.length} videos...`);
  
  let completed = 0;
  let failed = 0;
  let stillProcessing = 0;
  
  for (const v of processing) {
    try {
      const hg = await checkVideo(v.heygen_video_id);
      
      if (hg.status === 'completed' && hg.video_url) {
        await sql`
          UPDATE heygen_videos SET
            status = 'completed',
            video_url = ${hg.video_url},
            duration_seconds = ${hg.duration || null},
            thumbnail_url = ${hg.thumbnail_url || null},
            completed_at = NOW(),
            updated_at = NOW()
          WHERE id = ${v.id}
        `;
        console.log(`✅ Day ${v.day_of_year} ${v.phase} ${v.age_category} - COMPLETED`);
        completed++;
      } else if (hg.status === 'failed') {
        await sql`
          UPDATE heygen_videos SET
            status = 'failed',
            error_message = ${hg.error?.message || JSON.stringify(hg.error)},
            updated_at = NOW()
          WHERE id = ${v.id}
        `;
        console.log(`❌ Day ${v.day_of_year} ${v.phase} ${v.age_category} - FAILED: ${hg.error?.message}`);
        failed++;
      } else {
        stillProcessing++;
      }
      
      await new Promise(r => setTimeout(r, 150)); // Rate limit
    } catch (e) {
      console.log(`Error checking ${v.heygen_video_id}: ${e.message}`);
    }
  }
  
  console.log(`\n=== POLL RESULTS ===`);
  console.log(`Completed: ${completed}`);
  console.log(`Failed: ${failed}`);
  console.log(`Still processing: ${stillProcessing}`);
}

poll().catch(console.error);
