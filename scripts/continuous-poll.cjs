const { neon } = require('@neondatabase/serverless');

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function checkVideo(videoId) {
  const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  return res.json();
}

async function pollBatch() {
  const processing = await sql`
    SELECT id, heygen_video_id, day_of_year, phase, age_category
    FROM heygen_videos
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    LIMIT 50
  `;
  
  if (processing.length === 0) {
    console.log('No processing videos to check');
    return { completed: 0, failed: 0 };
  }
  
  let completed = 0;
  let failed = 0;
  
  for (const v of processing) {
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
        console.log(`✅ Day ${v.day_of_year} ${v.phase} ${v.age_category}`);
      } else if (data.status === 'failed' || data.error) {
        await sql`
          UPDATE heygen_videos 
          SET status = 'failed', error_message = ${data.error?.message || 'Unknown error'}
          WHERE id = ${v.id}
        `;
        failed++;
        console.log(`❌ Day ${v.day_of_year} ${v.phase} ${v.age_category}: ${data.error?.message || 'failed'}`);
      }
      
      // Rate limit
      await new Promise(r => setTimeout(r, 200));
    } catch (e) {
      console.log(`⚠️ Error checking ${v.heygen_video_id}: ${e.message}`);
    }
  }
  
  return { completed, failed };
}

async function syncToAssets() {
  const completed = await sql`
    SELECT day_of_year, phase, age_category, archetype, video_url, heygen_video_id, script
    FROM heygen_videos
    WHERE status = 'completed' AND video_url IS NOT NULL
  `;
  
  let synced = 0;
  for (const v of completed) {
    // Check if already exists
    const existing = await sql`
      SELECT id FROM kelly_lesson_assets
      WHERE day_number = ${v.day_of_year} 
      AND phase = ${v.phase} 
      AND age_group = ${v.age_category}
      LIMIT 1
    `;
    
    if (existing.length > 0) {
      // Update existing
      await sql`
        UPDATE kelly_lesson_assets
        SET video_url = ${v.video_url}, video_source = 'heygen', video_id = ${v.heygen_video_id}
        WHERE day_number = ${v.day_of_year} AND phase = ${v.phase} AND age_group = ${v.age_category}
      `;
    } else {
      // Insert new
      await sql`
        INSERT INTO kelly_lesson_assets (day_number, phase, age_group, video_url, video_source, video_id, script_text)
        VALUES (${v.day_of_year}, ${v.phase}, ${v.age_category}, ${v.video_url}, 'heygen', ${v.heygen_video_id}, ${v.script})
      `;
    }
    synced++;
  }
  
  return synced;
}

async function run() {
  const iterations = parseInt(process.argv[2]) || 10;
  const interval = parseInt(process.argv[3]) || 30; // seconds
  
  console.log(`Running ${iterations} poll cycles with ${interval}s intervals...\n`);
  
  let totalCompleted = 0;
  let totalFailed = 0;
  
  for (let i = 1; i <= iterations; i++) {
    console.log(`\n=== CYCLE ${i}/${iterations} @ ${new Date().toISOString().slice(11,19)} ===`);
    
    // Quick status
    const status = await sql`
      SELECT status, COUNT(*) as count FROM heygen_videos GROUP BY status
    `;
    for (const s of status) {
      console.log(`${s.status}: ${s.count}`);
    }
    
    // Poll for completions
    const { completed, failed } = await pollBatch();
    totalCompleted += completed;
    totalFailed += failed;
    
    if (completed > 0) {
      console.log(`\nSyncing to kelly_lesson_assets...`);
      const synced = await syncToAssets();
      console.log(`Synced ${synced} videos to assets`);
    }
    
    console.log(`\nProgress: +${completed} completed, +${failed} failed`);
    console.log(`Session totals: ${totalCompleted} completed, ${totalFailed} failed`);
    
    if (i < iterations) {
      console.log(`\nWaiting ${interval}s...`);
      await new Promise(r => setTimeout(r, interval * 1000));
    }
  }
  
  console.log(`\n=== FINAL SUMMARY ===`);
  console.log(`Completed: ${totalCompleted}`);
  console.log(`Failed: ${totalFailed}`);
}

run().catch(console.error);
