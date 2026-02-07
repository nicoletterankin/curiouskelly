/**
 * Reset failed/processing videos and start fresh with correct voice
 * 
 * This script:
 * 1. Deletes all non-completed heygen_videos records
 * 2. Keeps the 414 completed videos
 * 3. Prepares for fresh generation with correct voice
 */
const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

async function reset() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║           RESET NON-COMPLETED VIDEOS                       ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
  
  // Check current state
  console.log('📊 BEFORE RESET:');
  const before = await sql`
    SELECT status, COUNT(*) as count 
    FROM heygen_videos 
    GROUP BY status 
    ORDER BY count DESC
  `;
  before.forEach(s => console.log(`   ${s.status}: ${s.count}`));
  
  // Count what we'll keep
  const completed = await sql`
    SELECT COUNT(*) as count FROM heygen_videos WHERE status = 'completed'
  `;
  console.log(`\n✅ Keeping ${completed[0].count} completed videos`);
  
  // Delete non-completed
  const args = process.argv.slice(2);
  if (args.includes('--execute')) {
    console.log('\n🗑️  Deleting non-completed videos...');
    
    const result = await sql`
      DELETE FROM heygen_videos 
      WHERE status != 'completed'
      RETURNING id
    `;
    
    console.log(`   Deleted ${result.length} records`);
    
    // Verify
    console.log('\n📊 AFTER RESET:');
    const after = await sql`
      SELECT status, COUNT(*) as count 
      FROM heygen_videos 
      GROUP BY status 
      ORDER BY count DESC
    `;
    after.forEach(s => console.log(`   ${s.status}: ${s.count}`));
    
    console.log('\n✅ Reset complete. Ready for fresh generation with correct voice.');
  } else {
    console.log('\n⚠️  DRY RUN - No changes made');
    console.log('   Run with --execute to actually reset');
    
    const toDelete = await sql`
      SELECT COUNT(*) as count FROM heygen_videos WHERE status != 'completed'
    `;
    console.log(`   Would delete: ${toDelete[0].count} records`);
  }
}

reset().catch(console.error);
