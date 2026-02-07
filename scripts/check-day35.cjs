const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function check() {
  // Day 35 = February 4, 2026 = TODAY
  const day35Heygen = await sql`
    SELECT status, phase, age_category, video_url IS NOT NULL as has_url
    FROM heygen_videos
    WHERE day_of_year = 35
    ORDER BY age_category, phase
  `;
  
  console.log('=== DAY 35 (Feb 4, 2026) - HEYGEN STATUS ===');
  for (const v of day35Heygen) {
    const icon = v.status === 'completed' ? '✅' : '⏳';
    const url = v.has_url ? '🔗' : '  ';
    console.log(`${icon}${url} ${v.status.padEnd(12)} ${v.age_category.padEnd(8)} ${v.phase}`);
  }
  
  // Count by status
  const completed = day35Heygen.filter(v => v.status === 'completed').length;
  const processing = day35Heygen.filter(v => v.status === 'processing').length;
  console.log(`\nSummary: ${completed} completed, ${processing} processing`);
  
  // Check kelly_lesson_assets for Day 35
  const assets = await sql`
    SELECT phase, age_group, 
           video_url IS NOT NULL as has_video, 
           audio_url IS NOT NULL as has_audio,
           video_url
    FROM kelly_lesson_assets
    WHERE day_number = 35 AND video_url IS NOT NULL
    ORDER BY age_group, phase
  `;
  
  console.log('\n=== DAY 35 KELLY_LESSON_ASSETS (videos only) ===');
  for (const a of assets) {
    console.log(`🎬 ${a.age_group.padEnd(10)} ${a.phase.padEnd(8)} ${a.video_url?.slice(0, 50)}...`);
  }
  console.log(`\nTotal videos in assets for Day 35: ${assets.length}`);
}

check().catch(console.error);
