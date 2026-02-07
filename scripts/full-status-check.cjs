const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function check() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║     ZERO-TRUST DATABASE VERIFICATION - ep-fragrant-scene   ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
  
  // Status breakdown
  const status = await sql`SELECT status, COUNT(*) as count FROM heygen_videos GROUP BY status ORDER BY count DESC`;
  console.log('📊 STATUS BREAKDOWN:');
  status.forEach(r => console.log(`   ${r.status.padEnd(15)} ${r.count}`));
  
  // Day coverage
  const days = await sql`
    SELECT MIN(day_of_year) as min_day, MAX(day_of_year) as max_day, COUNT(DISTINCT day_of_year) as unique_days
    FROM heygen_videos WHERE status = 'completed' AND video_url IS NOT NULL
  `;
  console.log(`\n📅 DAY COVERAGE:`);
  console.log(`   Range: Day ${days[0].min_day} to Day ${days[0].max_day}`);
  console.log(`   Unique days with videos: ${days[0].unique_days}`);
  
  // Total ready to serve
  const ready = await sql`SELECT COUNT(*) as count FROM heygen_videos WHERE status = 'completed' AND video_url IS NOT NULL`;
  console.log(`\n✅ VIDEOS READY TO SERVE: ${ready[0].count}`);
  
  // Processing count
  const processing = await sql`SELECT COUNT(*) as count FROM heygen_videos WHERE status = 'processing'`;
  console.log(`⏳ CURRENTLY PROCESSING: ${processing[0].count}`);
  
  // Check for days without any videos in range 1-90
  const missing = await sql`
    WITH all_days AS (SELECT generate_series(1, 90) as day)
    SELECT day FROM all_days 
    WHERE day NOT IN (SELECT DISTINCT day_of_year FROM heygen_videos WHERE status = 'completed')
    ORDER BY day
  `;
  if (missing.length > 0) {
    console.log(`\n❌ MISSING DAYS (1-90): ${missing.map(m => m.day).join(', ')}`);
  } else {
    console.log('\n✓ All days 1-90 have at least one completed video');
  }
  
  // Videos per day (for days 1-60)
  const perDay = await sql`
    SELECT day_of_year, COUNT(*) as count 
    FROM heygen_videos 
    WHERE status = 'completed' AND video_url IS NOT NULL AND day_of_year <= 60
    GROUP BY day_of_year 
    ORDER BY day_of_year
  `;
  console.log('\n📊 VIDEOS PER DAY (1-60):');
  let dayGroups = {};
  perDay.forEach(r => {
    if (!dayGroups[r.count]) dayGroups[r.count] = [];
    dayGroups[r.count].push(r.day_of_year);
  });
  Object.keys(dayGroups).sort((a,b) => b-a).forEach(count => {
    console.log(`   ${count} videos: Days ${dayGroups[count].length > 10 ? dayGroups[count].slice(0,5).join(', ') + '...' : dayGroups[count].join(', ')}`);
  });
  
  // Age/archetype coverage for Day 34
  const day34 = await sql`
    SELECT age_category, archetype, phase, status 
    FROM heygen_videos 
    WHERE day_of_year = 34 
    ORDER BY age_category, archetype, phase
  `;
  console.log('\n📋 DAY 34 COVERAGE (Sample):');
  day34.forEach(r => console.log(`   ${r.age_category}/${r.archetype}/${r.phase}: ${r.status}`));
  
  // Check HeyGen URL validity
  const urlSample = await sql`
    SELECT video_url FROM heygen_videos 
    WHERE status = 'completed' AND video_url IS NOT NULL 
    LIMIT 5
  `;
  console.log('\n🔗 SAMPLE VIDEO URLS:');
  urlSample.forEach((r, i) => {
    const url = r.video_url;
    const isHeyGen = url.includes('heygen.ai') || url.includes('files2.heygen');
    console.log(`   ${i+1}. ${isHeyGen ? '✓ HeyGen' : '? Unknown'}: ${url.substring(0, 70)}...`);
  });
  
  // Credits estimate
  const totalCompleted = parseInt(ready[0].count);
  const creditsUsed = Math.ceil(totalCompleted * 1.5); // Rough estimate
  console.log(`\n💰 ESTIMATED CREDITS USED: ~${creditsUsed} (${totalCompleted} videos × ~1.5 credits/video)`);
  
  console.log('\n════════════════════════════════════════════════════════════');
}

check().catch(console.error);
