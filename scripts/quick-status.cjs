const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function check() {
  const result = await sql`SELECT status, COUNT(*) as count FROM heygen_videos GROUP BY status ORDER BY count DESC`;
  console.log('\n=== HEYGEN VIDEOS STATUS ===');
  result.forEach(r => console.log(`  ${r.status}: ${r.count}`));
  
  const total = result.reduce((sum, r) => sum + parseInt(r.count), 0);
  console.log(`  TOTAL: ${total}`);
}
check().catch(console.error);
