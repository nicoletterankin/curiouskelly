/**
 * Check actual HeyGen status for our "processing" videos
 */
const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';

async function checkStatus(videoId) {
  const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY }
  });
  const data = await res.json();
  return data.data;
}

async function main() {
  console.log('Checking HeyGen status for sample of "processing" videos...\n');
  
  // Get 20 sample processing videos
  const samples = await sql`
    SELECT heygen_video_id, day_of_year, phase, age_category
    FROM heygen_videos 
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 20
  `;
  
  const statusCounts = {};
  
  for (const s of samples) {
    const hgStatus = await checkStatus(s.heygen_video_id);
    const status = hgStatus?.status || 'error';
    statusCounts[status] = (statusCounts[status] || 0) + 1;
    
    const emoji = status === 'completed' ? '✅' : status === 'failed' ? '❌' : status === 'waiting' ? '⏳' : '❓';
    console.log(`${emoji} Day ${s.day_of_year} ${s.phase} (${s.age_category}): ${status}`);
    
    if (status === 'failed' && hgStatus?.error) {
      console.log(`   Error: ${hgStatus.error.message || JSON.stringify(hgStatus.error)}`);
    }
    
    await new Promise(r => setTimeout(r, 200)); // Rate limit
  }
  
  console.log('\n═══════════════════════════════════════');
  console.log('HEYGEN STATUS SUMMARY:');
  Object.entries(statusCounts).forEach(([status, count]) => {
    console.log(`  ${status}: ${count}`);
  });
}

main().catch(console.error);
