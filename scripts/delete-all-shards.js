import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function deleteAllShards() {
  console.log('🔍 Checking current shard count...\n');

  // Get current count
  const { count: beforeCount, error: countError } = await supabase
    .from('lesson_shards')
    .select('*', { count: 'exact', head: true });

  if (countError) {
    console.error('Error counting shards:', countError);
    return;
  }

  console.log(`📊 Current shard count: ${beforeCount}`);
  console.log('\n⚠️  WARNING: About to delete ALL lesson_shards');
  console.log('This is safe because learn.html will fall back to lesson_atoms.');
  console.log('\n🗑️  Deleting all shards...\n');

  // Delete all shards
  const { error: deleteError } = await supabase
    .from('lesson_shards')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000'); // Delete all (using a condition that's always true)

  if (deleteError) {
    console.error('❌ Error deleting shards:', deleteError);
    return;
  }

  // Verify deletion
  const { count: afterCount, error: verifyError } = await supabase
    .from('lesson_shards')
    .select('*', { count: 'exact', head: true });

  if (verifyError) {
    console.error('Error verifying deletion:', verifyError);
    return;
  }

  console.log('✅ Deletion complete!');
  console.log(`Before: ${beforeCount} shards`);
  console.log(`After: ${afterCount} shards`);
  console.log(`Deleted: ${beforeCount - afterCount} shards`);
  console.log('\n💡 Next step: Verify fallback to lesson_atoms works, then regenerate shards for fixed lessons.');
}

deleteAllShards().catch(console.error);

