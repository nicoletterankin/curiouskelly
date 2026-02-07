const { neon } = require('@neondatabase/serverless');

const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function check() {
  // Target: 365 days x 3 ages x 5 phases = 5,475 videos
  const TARGET = 365 * 3 * 5;
  
  // Count by status
  const statusCounts = await sql`
    SELECT status, COUNT(*) as count
    FROM heygen_videos
    GROUP BY status
  `;
  
  console.log('=== HEYGEN VIDEOS STATUS ===');
  let total = 0;
  for (const s of statusCounts) {
    console.log(`${s.status}: ${s.count}`);
    total += parseInt(s.count);
  }
  console.log(`Total in DB: ${total}`);
  console.log(`Target: ${TARGET}`);
  console.log(`Coverage: ${((total / TARGET) * 100).toFixed(1)}%`);
  
  // Coverage by day
  const dayCoverage = await sql`
    SELECT 
      day_of_year,
      COUNT(*) as total,
      COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
      COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing
    FROM heygen_videos
    GROUP BY day_of_year
    ORDER BY day_of_year
  `;
  
  console.log('\n=== DAYS WITH 15/15 COMPLETED ===');
  const fullDays = dayCoverage.filter(d => parseInt(d.completed) === 15);
  console.log(`${fullDays.length} days fully complete`);
  if (fullDays.length <= 20) {
    console.log('Days:', fullDays.map(d => d.day_of_year).join(', '));
  }
  
  // Missing days
  const allDays = new Set(dayCoverage.map(d => parseInt(d.day_of_year)));
  const missingDays = [];
  for (let d = 1; d <= 365; d++) {
    if (!allDays.has(d)) missingDays.push(d);
  }
  
  console.log('\n=== MISSING DAYS ===');
  console.log(`${missingDays.length} days not submitted`);
  if (missingDays.length <= 30) {
    console.log('Missing:', missingDays.join(', '));
  }
  
  // Coverage by age
  const ageCoverage = await sql`
    SELECT 
      age_category,
      COUNT(*) as total,
      COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
    FROM heygen_videos
    GROUP BY age_category
    ORDER BY total DESC
  `;
  
  console.log('\n=== BY AGE CATEGORY ===');
  for (const a of ageCoverage) {
    console.log(`${a.age_category}: ${a.completed}/${a.total} completed`);
  }
  
  // kelly_lesson_assets coverage
  const assetCoverage = await sql`
    SELECT 
      COUNT(*) as total,
      COUNT(CASE WHEN video_url IS NOT NULL THEN 1 END) as has_video,
      COUNT(CASE WHEN audio_url IS NOT NULL AND video_url IS NULL THEN 1 END) as audio_only,
      COUNT(DISTINCT day_number) as days_with_any
    FROM kelly_lesson_assets
  `;
  
  console.log('\n=== KELLY_LESSON_ASSETS ===');
  const a = assetCoverage[0];
  console.log(`Total records: ${a.total}`);
  console.log(`With video: ${a.has_video}`);
  console.log(`Audio only (ready for Fal): ${a.audio_only}`);
  console.log(`Days with any asset: ${a.days_with_any}`);
}

check().catch(console.error);
