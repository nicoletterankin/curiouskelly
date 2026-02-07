const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function check() {
  console.log('=== CHECKING KELLY_LESSON_ASSETS (v0 data source) ===\n');
  
  // Check total and status breakdown
  const stats = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(CASE WHEN audio_url IS NOT NULL THEN 1 END) as has_audio,
      COUNT(CASE WHEN video_url IS NOT NULL THEN 1 END) as has_video,
      COUNT(CASE WHEN audio_url IS NOT NULL AND video_url IS NULL THEN 1 END) as ready_for_lipsync
    FROM kelly_lesson_assets
  `;
  
  console.log('Total records:', stats[0].total);
  console.log('Has audio:', stats[0].has_audio);
  console.log('Has video:', stats[0].has_video);
  console.log('Ready for lipsync (has audio, no video):', stats[0].ready_for_lipsync);
  
  // Check video sources
  console.log('\n=== VIDEO SOURCES ===');
  const sources = await sql`
    SELECT video_source, COUNT(*) as count 
    FROM kelly_lesson_assets 
    WHERE video_url IS NOT NULL
    GROUP BY video_source
  `;
  sources.forEach(s => console.log(`  ${s.video_source || 'null'}: ${s.count}`));
}

check().catch(console.error);
