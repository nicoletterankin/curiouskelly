/**
 * Check current video URLs in kelly_motion_library
 */
require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

async function main() {
  const { data, error } = await supabase
    .from('kelly_motion_library')
    .select('persona, phase, video_url, avatar_key')
    .eq('status', 'completed')
    .limit(5);
  
  if (error) {
    console.error('Error:', error.message);
    return;
  }
  
  console.log('Sample video URLs in database:');
  data.forEach(row => {
    const urlDomain = row.video_url?.includes('supabase.co') ? 'SUPABASE ✅' : 
                      row.video_url?.includes('heygen.ai') ? 'HEYGEN ❌' : 'OTHER';
    console.log(`  ${row.persona}/${row.phase} (${row.avatar_key}): ${urlDomain}`);
    console.log(`    ${row.video_url?.substring(0, 80)}...`);
  });
  
  // Count how many are Supabase vs HeyGen
  const { data: allData } = await supabase
    .from('kelly_motion_library')
    .select('video_url')
    .eq('status', 'completed');
  
  const supabaseCount = allData?.filter(r => r.video_url?.includes('supabase.co')).length || 0;
  const heygenCount = allData?.filter(r => r.video_url?.includes('heygen.ai')).length || 0;
  
  console.log(`\nTotal: ${allData?.length || 0}`);
  console.log(`  Supabase URLs: ${supabaseCount}`);
  console.log(`  HeyGen URLs: ${heygenCount}`);
}

main().catch(console.error);
