const { neon } = require('@neondatabase/serverless');

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function findMissed() {
  console.log('=== FINDING MISSED COMPLETIONS ===\n');
  
  // Get completed videos from HeyGen's video.list
  const listRes = await fetch('https://api.heygen.com/v1/video.list?limit=100', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const listData = await listRes.json();
  
  const heygenCompleted = (listData.data?.videos || []).filter(v => v.status === 'completed');
  console.log(`HeyGen reports ${heygenCompleted.length} completed videos (from API list)\n`);
  
  let missedCount = 0;
  let alreadyRecorded = 0;
  
  for (const hv of heygenCompleted) {
    // Check if this video is in our database as completed
    const inDb = await sql`
      SELECT id, status FROM heygen_videos 
      WHERE heygen_video_id = ${hv.video_id}
      LIMIT 1
    `;
    
    if (inDb.length === 0) {
      // Not in our database at all - might be from before our tracking
      console.log(`⚠️ ${hv.video_id.slice(0, 8)}... not in our database`);
    } else if (inDb[0].status !== 'completed') {
      // In database but not marked completed - update it!
      console.log(`🔄 Updating ${hv.video_id.slice(0, 8)}... from ${inDb[0].status} to completed`);
      
      // Get the full video status to get URL
      const statusRes = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${hv.video_id}`, {
        headers: { 'X-Api-Key': HEYGEN_API_KEY }
      });
      const statusData = await statusRes.json();
      
      if (statusData.data?.video_url) {
        await sql`
          UPDATE heygen_videos
          SET status = 'completed', video_url = ${statusData.data.video_url}, completed_at = NOW()
          WHERE heygen_video_id = ${hv.video_id}
        `;
        missedCount++;
      }
    } else {
      alreadyRecorded++;
    }
    
    await new Promise(r => setTimeout(r, 100)); // Rate limit
  }
  
  console.log(`\n=== RESULTS ===`);
  console.log(`Already recorded: ${alreadyRecorded}`);
  console.log(`Newly updated: ${missedCount}`);
}

findMissed().catch(console.error);
