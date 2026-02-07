const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';

async function checkQueue() {
  console.log('=== HEYGEN ACCOUNT STATUS ===\n');
  
  // Check recent videos
  const listRes = await fetch('https://api.heygen.com/v1/video.list?limit=20', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const listData = await listRes.json();
  
  if (listData.data?.videos) {
    console.log('Recent videos in HeyGen:');
    for (const v of listData.data.videos.slice(0, 10)) {
      console.log(`  ${v.status.padEnd(12)} ${v.video_id.slice(0, 8)}... created: ${new Date(v.created_at * 1000).toISOString().slice(0, 16)}`);
    }
  }
  
  // Get status distribution
  const statusCounts = {};
  for (const v of listData.data?.videos || []) {
    statusCounts[v.status] = (statusCounts[v.status] || 0) + 1;
  }
  
  console.log('\nStatus distribution (last 20):');
  for (const [status, count] of Object.entries(statusCounts)) {
    console.log(`  ${status}: ${count}`);
  }
  
  // Check a specific processing video to see its position
  const { neon } = require('@neondatabase/serverless');
  const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');
  
  const sample = await sql`
    SELECT heygen_video_id, day_of_year, phase, created_at
    FROM heygen_videos
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 1
  `;
  
  if (sample.length > 0) {
    console.log('\nChecking random processing video:');
    const v = sample[0];
    console.log(`  Day ${v.day_of_year} ${v.phase}`);
    console.log(`  Submitted: ${new Date(v.created_at).toISOString()}`);
    
    const statusRes = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${v.heygen_video_id}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    });
    const statusData = await statusRes.json();
    
    console.log(`  HeyGen status: ${statusData.data?.status || statusData.status || 'unknown'}`);
    if (statusData.data?.video_url) {
      console.log(`  Video URL: ${statusData.data.video_url.slice(0, 60)}...`);
    }
    if (statusData.data?.error) {
      console.log(`  Error: ${JSON.stringify(statusData.data.error)}`);
    }
  }
}

checkQueue().catch(console.error);
