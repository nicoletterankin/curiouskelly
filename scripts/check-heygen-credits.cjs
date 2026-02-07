const { neon } = require('@neondatabase/serverless');

const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';

async function checkCredits() {
  try {
    // Try to get quota from the video list endpoint which includes quota info
    const res = await fetch('https://api.heygen.com/v1/video.list?limit=1', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    });
    
    const data = await res.json();
    
    if (data.data) {
      console.log('=== HEYGEN API STATUS ===');
      console.log('API is working');
      console.log('Total videos in HeyGen account:', data.data?.videos?.length || 'N/A');
    }
    
    // Check our database for credit usage estimate
    const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');
    
    const stats = await sql`
      SELECT 
        status,
        COUNT(*) as count,
        SUM(COALESCE(duration_seconds, 60) / 60.0) as estimated_minutes
      FROM heygen_videos
      GROUP BY status
      ORDER BY count DESC
    `;
    
    console.log('\n=== DATABASE STATUS ===');
    let totalMinutes = 0;
    for (const row of stats) {
      const mins = parseFloat(row.estimated_minutes) || 0;
      totalMinutes += mins;
      console.log(`${row.status}: ${row.count} videos (~${mins.toFixed(1)} min)`);
    }
    
    console.log('\n=== CREDIT ESTIMATE ===');
    console.log(`Total videos submitted: ${stats.reduce((a, b) => a + parseInt(b.count), 0)}`);
    console.log(`Estimated minutes used: ${totalMinutes.toFixed(1)}`);
    console.log(`Started with: 668.5 credits (minutes)`);
    console.log(`Estimated remaining: ${(668.5 - totalMinutes).toFixed(1)} credits`);
    
  } catch (e) {
    console.error('Error:', e.message);
  }
}

checkCredits();
