const { neon } = require('@neondatabase/serverless');

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function checkVideo(videoId) {
  const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  return res.json();
}

async function bulkSync() {
  // Get ALL processing videos
  const processing = await sql`
    SELECT id, heygen_video_id, day_of_year, phase, age_category
    FROM heygen_videos
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
  `;
  
  console.log(`Found ${processing.length} videos to check\n`);
  
  let completed = 0;
  let failed = 0;
  let stillProcessing = 0;
  
  const batchSize = 50;
  for (let i = 0; i < processing.length; i += batchSize) {
    const batch = processing.slice(i, i + batchSize);
    console.log(`Checking batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(processing.length/batchSize)}...`);
    
    for (const v of batch) {
      try {
        const result = await checkVideo(v.heygen_video_id);
        const data = result.data || result;
        
        if (data.status === 'completed' && data.video_url) {
          await sql`
            UPDATE heygen_videos 
            SET status = 'completed', video_url = ${data.video_url}, completed_at = NOW()
            WHERE id = ${v.id}
          `;
          completed++;
        } else if (data.status === 'failed' || data.error) {
          await sql`
            UPDATE heygen_videos 
            SET status = 'failed', error_message = ${data.error?.message || 'Unknown error'}
            WHERE id = ${v.id}
          `;
          failed++;
        } else {
          stillProcessing++;
        }
        
        // Rate limit - 5 req/sec
        await new Promise(r => setTimeout(r, 200));
      } catch (e) {
        // Skip errors, continue
      }
    }
    
    console.log(`  Progress: ${completed} completed, ${failed} failed, ${stillProcessing} still processing`);
  }
  
  console.log(`\n=== BULK SYNC COMPLETE ===`);
  console.log(`Completed: ${completed}`);
  console.log(`Failed: ${failed}`);
  console.log(`Still processing: ${stillProcessing}`);
  
  // Now sync to kelly_lesson_assets
  if (completed > 0) {
    console.log(`\nSyncing to kelly_lesson_assets...`);
    const allCompleted = await sql`
      SELECT day_of_year, phase, age_category, archetype, video_url, heygen_video_id, script
      FROM heygen_videos
      WHERE status = 'completed' AND video_url IS NOT NULL
    `;
    
    let synced = 0;
    for (const v of allCompleted) {
      const existing = await sql`
        SELECT id FROM kelly_lesson_assets
        WHERE day_number = ${v.day_of_year} 
        AND phase = ${v.phase} 
        AND age_group = ${v.age_category}
        LIMIT 1
      `;
      
      if (existing.length > 0) {
        await sql`
          UPDATE kelly_lesson_assets
          SET video_url = ${v.video_url}, video_source = 'heygen', video_id = ${v.heygen_video_id}
          WHERE day_number = ${v.day_of_year} AND phase = ${v.phase} AND age_group = ${v.age_category}
        `;
      } else {
        await sql`
          INSERT INTO kelly_lesson_assets (day_number, phase, age_group, video_url, video_source, video_id, script_text)
          VALUES (${v.day_of_year}, ${v.phase}, ${v.age_category}, ${v.video_url}, 'heygen', ${v.heygen_video_id}, ${v.script})
        `;
      }
      synced++;
    }
    console.log(`Synced ${synced} videos to kelly_lesson_assets`);
  }
}

bulkSync().catch(console.error);
