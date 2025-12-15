/**
 * Check lesson_visuals table structure
 */

const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function main() {
  console.log('Checking lesson_visuals table...\n');
  
  const { data, error } = await supabase
    .from('lesson_visuals')
    .select('*')
    .limit(10);
  
  if (error) {
    console.log('❌ Error:', error.message);
    console.log('\nTable may not exist. Need to create it or use different approach.');
    return;
  }
  
  console.log(`Found ${data?.length || 0} records\n`);
  
  if (data?.length) {
    console.log('Sample record:');
    console.log(JSON.stringify(data[0], null, 2));
    
    console.log('\n\nAll columns:');
    Object.keys(data[0]).forEach(key => {
      console.log(`  - ${key}`);
    });
  }
}

main().catch(console.error);
