const { neon } = require('@neondatabase/serverless');

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function debug() {
  // Get some processing videos from different batches
  const samples = await sql`
    SELECT id, heygen_video_id, day_of_year, phase, age_category, created_at
    FROM heygen_videos
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    ORDER BY created_at ASC
    LIMIT 5
  `;
  
  console.log('=== CHECKING OLDEST PROCESSING VIDEOS ===\n');
  
  for (const v of samples) {
    console.log(`Day ${v.day_of_year} ${v.phase} ${v.age_category}`);
    console.log(`  Created: ${new Date(v.created_at).toISOString()}`);
    console.log(`  HeyGen ID: ${v.heygen_video_id}`);
    
    try {
      const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${v.heygen_video_id}`, {
        headers: { 'X-Api-Key': HEYGEN_API_KEY }
      });
      const data = await res.json();
      
      console.log(`  HeyGen Status: ${data.data?.status || data.status || 'unknown'}`);
      if (data.data?.video_url) {
        console.log(`  Video URL: ${data.data.video_url.substring(0, 60)}...`);
      }
      if (data.data?.error || data.error) {
        console.log(`  ERROR: ${JSON.stringify(data.data?.error || data.error)}`);
      }
      console.log('');
    } catch (e) {
      console.log(`  Fetch error: ${e.message}\n`);
    }
    
    await new Promise(r => setTimeout(r, 200));
  }
  
  // Also check a newer one
  const newest = await sql`
    SELECT id, heygen_video_id, day_of_year, phase, age_category, created_at
    FROM heygen_videos
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    ORDER BY created_at DESC
    LIMIT 3
  `;
  
  console.log('=== CHECKING NEWEST PROCESSING VIDEOS ===\n');
  
  for (const v of newest) {
    console.log(`Day ${v.day_of_year} ${v.phase} ${v.age_category}`);
    console.log(`  Created: ${new Date(v.created_at).toISOString()}`);
    
    try {
      const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${v.heygen_video_id}`, {
        headers: { 'X-Api-Key': HEYGEN_API_KEY }
      });
      const data = await res.json();
      
      console.log(`  HeyGen Status: ${data.data?.status || data.status || 'unknown'}`);
      if (data.data?.video_url) {
        console.log(`  Video URL exists!`);
      }
      if (data.data?.error || data.error) {
        console.log(`  ERROR: ${JSON.stringify(data.data?.error || data.error)}`);
      }
      console.log('');
    } catch (e) {
      console.log(`  Fetch error: ${e.message}\n`);
    }
    
    await new Promise(r => setTimeout(r, 200));
  }
}

debug().catch(console.error);
