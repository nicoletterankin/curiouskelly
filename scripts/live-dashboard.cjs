/**
 * Live dashboard showing video generation progress
 */

const { neon } = require('@neondatabase/serverless');
require('dotenv').config();

const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');
const HEYGEN_API_KEY = 'sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E';

async function dashboard() {
  console.clear();
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║           CURIOUS KELLY VIDEO GENERATION DASHBOARD         ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
  
  // Get status counts
  const status = await sql`SELECT status, COUNT(*) as count FROM heygen_videos GROUP BY status ORDER BY count DESC`;
  let completed = 0, processing = 0, failed = 0;
  status.forEach(s => {
    if (s.status === 'completed') completed = parseInt(s.count);
    if (s.status === 'processing') processing = parseInt(s.count);
    if (s.status === 'failed') failed = parseInt(s.count);
  });
  
  // Day coverage
  const coverage = await sql`
    SELECT 
      COUNT(DISTINCT day_of_year) FILTER (WHERE status = 'completed') as completed_days,
      COUNT(DISTINCT day_of_year) FILTER (WHERE status = 'processing') as processing_days
    FROM heygen_videos
  `;
  
  // Age breakdown
  const ageStats = await sql`
    SELECT age_category, 
           COUNT(*) FILTER (WHERE status = 'completed') as completed,
           COUNT(*) FILTER (WHERE status = 'processing') as processing
    FROM heygen_videos
    WHERE status IN ('completed', 'processing')
    GROUP BY age_category
  `;
  
  // Credits
  let credits = 'checking...';
  try {
    const res = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY }
    });
    const data = await res.json();
    credits = data.data?.remaining_quota || 'unknown';
  } catch (e) {}
  
  // Display
  console.log('📊 VIDEO STATUS:');
  console.log(`   ✅ Completed:  ${completed.toLocaleString()}`);
  console.log(`   ⏳ Processing: ${processing.toLocaleString()}`);
  console.log(`   ❌ Failed:     ${failed.toLocaleString()}`);
  console.log(`   📈 Total active: ${(completed + processing).toLocaleString()}\n`);
  
  console.log('📅 DAY COVERAGE:');
  console.log(`   Days with completed videos: ${coverage[0]?.completed_days || 0}`);
  console.log(`   Days being processed: ${coverage[0]?.processing_days || 0}\n`);
  
  console.log('🎭 AGE BREAKDOWN:');
  ageStats.forEach(a => {
    console.log(`   ${a.age_category.padEnd(8)}: ${a.completed} completed, ${a.processing} processing`);
  });
  
  console.log(`\n💰 HEYGEN CREDITS: ${credits.toLocaleString()}`);
  
  // Progress bar
  const total = 5475; // 365 days × 5 phases × 3 ages
  const progress = Math.round((completed / total) * 100);
  const bar = '█'.repeat(Math.floor(progress / 2)) + '░'.repeat(50 - Math.floor(progress / 2));
  console.log(`\n📈 PROGRESS TO 365 DAYS: [${bar}] ${progress}%`);
  console.log(`   ${completed} / ${total} videos\n`);
  
  console.log('════════════════════════════════════════════════════════════');
  console.log(`Last updated: ${new Date().toLocaleTimeString()}`);
}

dashboard().catch(console.error);
