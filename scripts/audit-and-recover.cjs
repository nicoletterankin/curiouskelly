/**
 * Audit database state and prepare for recovery
 */
const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function audit() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║           HEYGEN VIDEO DATABASE AUDIT                      ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
  
  // Status breakdown
  console.log('📊 STATUS BREAKDOWN:');
  const statuses = await sql`
    SELECT status, COUNT(*) as count 
    FROM heygen_videos 
    GROUP BY status 
    ORDER BY count DESC
  `;
  statuses.forEach(s => console.log(`   ${s.status}: ${s.count}`));
  
  // Failed videos - error breakdown
  console.log('\n❌ FAILED VIDEO ERRORS:');
  const failed = await sql`
    SELECT error_message, COUNT(*) as count 
    FROM heygen_videos 
    WHERE status = 'failed'
    GROUP BY error_message
    ORDER BY count DESC
    LIMIT 10
  `;
  failed.forEach(f => console.log(`   ${f.count}x: ${f.error_message?.substring(0, 80) || 'null'}`));
  
  // Processing videos - how long stuck?
  console.log('\n⏳ PROCESSING VIDEOS (potentially stuck):');
  const processing = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(CASE WHEN heygen_video_id IS NOT NULL THEN 1 END) as has_video_id,
      COUNT(CASE WHEN heygen_video_id IS NULL THEN 1 END) as no_video_id
    FROM heygen_videos 
    WHERE status = 'processing'
  `;
  console.log(`   Total processing: ${processing[0].total}`);
  console.log(`   With HeyGen video ID: ${processing[0].has_video_id}`);
  console.log(`   Without HeyGen video ID: ${processing[0].no_video_id}`);
  
  // Completed videos
  console.log('\n✅ COMPLETED VIDEOS:');
  const completed = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(CASE WHEN video_url IS NOT NULL THEN 1 END) as has_url
    FROM heygen_videos 
    WHERE status = 'completed'
  `;
  console.log(`   Total completed: ${completed[0].total}`);
  console.log(`   With video URL: ${completed[0].has_url}`);
  
  // Sample a processing video to check its HeyGen status
  console.log('\n🔍 SAMPLE PROCESSING VIDEO:');
  const sample = await sql`
    SELECT heygen_video_id, day_of_year, phase, age_category, created_at
    FROM heygen_videos 
    WHERE status = 'processing' AND heygen_video_id IS NOT NULL
    LIMIT 1
  `;
  if (sample[0]) {
    console.log(`   Video ID: ${sample[0].heygen_video_id}`);
    console.log(`   Day: ${sample[0].day_of_year}, Phase: ${sample[0].phase}, Age: ${sample[0].age_category}`);
    console.log(`   Created: ${sample[0].created_at}`);
  }
  
  console.log('\n════════════════════════════════════════════════════════════');
}

audit().catch(console.error);
