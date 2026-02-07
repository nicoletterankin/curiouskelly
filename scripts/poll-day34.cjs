const { neon } = require('@neondatabase/serverless');

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function pollDay34() {
  console.log('=== POLLING DAY 34 VIDEOS ===\n');
  
  const processing = await sql`
    SELECT id, heygen_video_id, phase, age_category
    FROM heygen_videos
    WHERE day_of_year = 34 AND status = 'processing' AND heygen_video_id IS NOT NULL
  `;
  
  console.log(`Found ${processing.length} Day 34 videos still processing\n`);
  
  let completed = 0;
  let stillProcessing = 0;
  
  for (const v of processing) {
    try {
      const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${v.heygen_video_id}`, {
        headers: { 'X-Api-Key': HEYGEN_API_KEY }
      });
      const data = await res.json();
      const status = data.data?.status || data.status;
      
      if (status === 'completed' && data.data?.video_url) {
        console.log(`✅ COMPLETED: ${v.age_category} ${v.phase}`);
        
        // Update database
        await sql`
          UPDATE heygen_videos 
          SET status = 'completed', video_url = ${data.data.video_url}, completed_at = NOW()
          WHERE id = ${v.id}
        `;
        
        // Sync to kelly_lesson_assets
        await sql`
          UPDATE kelly_lesson_assets
          SET video_url = ${data.data.video_url}, video_source = 'heygen', updated_at = NOW()
          WHERE day_number = 34 AND phase = ${v.phase} AND age_group = ${v.age_category}
        `;
        
        completed++;
      } else if (status === 'failed') {
        console.log(`❌ FAILED: ${v.age_category} ${v.phase} - ${data.data?.error?.message || 'unknown'}`);
        await sql`
          UPDATE heygen_videos 
          SET status = 'failed', error_message = ${data.data?.error?.message || 'unknown'}
          WHERE id = ${v.id}
        `;
      } else {
        console.log(`⏳ Processing: ${v.age_category} ${v.phase}`);
        stillProcessing++;
      }
      
      await new Promise(r => setTimeout(r, 200));
    } catch (e) {
      console.log(`⚠️ Error: ${v.age_category} ${v.phase}: ${e.message}`);
    }
  }
  
  console.log(`\n=== RESULTS ===`);
  console.log(`Newly completed: ${completed}`);
  console.log(`Still processing: ${stillProcessing}`);
}

pollDay34().catch(console.error);
