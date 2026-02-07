const { neon } = require('@neondatabase/serverless');

const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function check() {
  // Recent completions
  const recent = await sql`
    SELECT status, day_of_year, phase, age_category, 
           created_at, completed_at, error_message
    FROM heygen_videos
    WHERE completed_at IS NOT NULL OR status = 'failed'
    ORDER BY COALESCE(completed_at, created_at) DESC
    LIMIT 20
  `;
  
  console.log('=== RECENT COMPLETED/FAILED ===');
  for (const v of recent) {
    const time = v.completed_at || v.created_at;
    console.log(`${v.status.toUpperCase()}: Day ${v.day_of_year} ${v.phase} ${v.age_category}`);
    if (v.error_message) console.log(`  Error: ${v.error_message.substring(0, 80)}`);
  }
  
  // Check for credit-related failures
  const creditFails = await sql`
    SELECT COUNT(*) as count
    FROM heygen_videos
    WHERE status = 'failed' 
    AND (error_message ILIKE '%credit%' OR error_message ILIKE '%quota%' OR error_message ILIKE '%balance%')
  `;
  
  console.log('\n=== CREDIT-RELATED FAILURES ===');
  console.log('Count:', creditFails[0].count);
  
  // Jobs by creation time
  const timeline = await sql`
    SELECT 
      DATE_TRUNC('hour', created_at) as hour,
      COUNT(*) as submitted,
      COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
      COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
    FROM heygen_videos
    GROUP BY DATE_TRUNC('hour', created_at)
    ORDER BY hour DESC
    LIMIT 10
  `;
  
  console.log('\n=== SUBMISSION TIMELINE ===');
  for (const t of timeline) {
    console.log(`${new Date(t.hour).toISOString().slice(0,13)}h: ${t.submitted} submitted, ${t.completed} completed, ${t.failed} failed`);
  }
}

check().catch(console.error);
