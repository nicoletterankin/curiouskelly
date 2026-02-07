const { neon } = require('@neondatabase/serverless');

const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function checkStatus() {
  console.log('=== HEYGEN VIDEO STATUS (ep-fragrant-scene) ===\n');
  
  // Status breakdown
  const status = await sql`SELECT status, COUNT(*) as count FROM heygen_videos GROUP BY status ORDER BY count DESC`;
  console.log('Status breakdown:');
  status.forEach(r => console.log(`  ${r.status}: ${r.count}`));
  
  // Completed with URLs
  const completed = await sql`SELECT COUNT(*) as count FROM heygen_videos WHERE status = 'completed' AND video_url IS NOT NULL`;
  console.log(`\nVideos ready to serve: ${completed[0].count}`);
  
  // Day coverage
  const dayRange = await sql`SELECT MIN(day_of_year) as min_day, MAX(day_of_year) as max_day FROM heygen_videos WHERE status = 'completed'`;
  console.log(`Day range with completed videos: Day ${dayRange[0].min_day} to Day ${dayRange[0].max_day}`);
  
  // Count by day
  const byDay = await sql`
    SELECT day_of_year, COUNT(*) as count 
    FROM heygen_videos 
    WHERE status = 'completed' AND video_url IS NOT NULL
    GROUP BY day_of_year 
    ORDER BY day_of_year
  `;
  console.log('\nCompleted videos by day:');
  byDay.forEach(r => console.log(`  Day ${r.day_of_year}: ${r.count} videos`));
  
  // Recent completions
  const recent = await sql`
    SELECT day_of_year, phase, age_category, archetype, updated_at 
    FROM heygen_videos 
    WHERE status = 'completed' 
    ORDER BY updated_at DESC 
    LIMIT 5
  `;
  console.log('\nMost recently completed:');
  recent.forEach(r => console.log(`  Day ${r.day_of_year} ${r.phase} (${r.age_category}/${r.archetype}) - ${r.updated_at}`));
  
  // Processing count
  const processing = await sql`SELECT COUNT(*) as count FROM heygen_videos WHERE status = 'processing'`;
  console.log(`\nCurrently processing: ${processing[0].count}`);
}

checkStatus().catch(console.error);
