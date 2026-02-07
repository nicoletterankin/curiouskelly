const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function check() {
  // Today's videos (Day 34)
  const day34 = await sql`
    SELECT status, phase, age_category, video_url
    FROM heygen_videos
    WHERE day_of_year = 34
    ORDER BY age_category, phase
  `;
  
  console.log('=== TODAY (Day 34) HEYGEN STATUS ===');
  for (const v of day34) {
    const hasUrl = v.video_url ? '✅' : '⏳';
    console.log(`${hasUrl} ${v.status.padEnd(12)} ${v.age_category.padEnd(8)} ${v.phase}`);
  }
  
  // Check kelly_lesson_assets for Day 34
  const assets = await sql`
    SELECT phase, age_group, video_url IS NOT NULL as has_video, audio_url IS NOT NULL as has_audio
    FROM kelly_lesson_assets
    WHERE day_number = 34
    ORDER BY age_group, phase
  `;
  
  console.log('\n=== KELLY_LESSON_ASSETS Day 34 ===');
  console.log('(🎬=video, 🔊=audio)');
  for (const a of assets) {
    const video = a.has_video ? '🎬' : '  ';
    const audio = a.has_audio ? '🔊' : '  ';
    console.log(`${video}${audio} ${a.age_group.padEnd(10)} ${a.phase}`);
  }
  
  // Summary
  const videoCount = assets.filter(a => a.has_video).length;
  const audioCount = assets.filter(a => a.has_audio).length;
  console.log(`\nDay 34 Summary: ${videoCount} videos, ${audioCount} audio`);
}

check().catch(console.error);
