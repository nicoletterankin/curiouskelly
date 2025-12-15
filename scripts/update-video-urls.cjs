/**
 * Update Kelly Motion Library with Supabase Storage URLs
 * 
 * The videos are already uploaded to Supabase storage.
 * This script just updates the database records with permanent URLs.
 * 
 * Run: node scripts/update-video-urls.cjs
 */

require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const BUCKET_NAME = 'kelly-videos';

if (!SUPABASE_SERVICE_KEY) {
  console.error('❌ SUPABASE_SERVICE_ROLE_KEY not found');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false }
});

async function main() {
  console.log('===========================================');
  console.log('🔄 UPDATING KELLY MOTION LIBRARY URLs');
  console.log('===========================================\n');
  
  // Get all records that still have HeyGen URLs
  const { data: clips, error: fetchError } = await supabase
    .from('kelly_motion_library')
    .select('id, persona, age_bucket, phase, video_url')
    .eq('status', 'completed')
    .not('video_url', 'is', null);
  
  if (fetchError) {
    console.error('❌ Error fetching clips:', fetchError.message);
    return;
  }
  
  console.log(`Found ${clips.length} clips to check\n`);
  
  let updated = 0;
  let alreadyDone = 0;
  let errors = 0;
  
  for (const clip of clips) {
    const { id, persona, age_bucket, phase, video_url } = clip;
    
    // Skip if already pointing to Supabase
    if (video_url.includes('supabase.co/storage')) {
      alreadyDone++;
      continue;
    }
    
    // Generate the Supabase storage URL
    const storagePath = `motion/${persona}/${age_bucket}/${phase}.mp4`;
    const { data: { publicUrl } } = supabase.storage
      .from(BUCKET_NAME)
      .getPublicUrl(storagePath);
    
    // Update the record
    const { error: updateError } = await supabase
      .from('kelly_motion_library')
      .update({ video_url: publicUrl })
      .eq('id', id);
    
    if (updateError) {
      console.log(`❌ ${persona}/${phase}: ${updateError.message}`);
      errors++;
    } else {
      updated++;
    }
  }
  
  console.log('\n===========================================');
  console.log('SUMMARY');
  console.log('===========================================');
  console.log(`✅ Updated: ${updated}`);
  console.log(`⏭️ Already done: ${alreadyDone}`);
  console.log(`❌ Errors: ${errors}`);
  
  // Verify by checking one URL
  if (updated > 0) {
    console.log('\n🔍 Verifying a sample URL...');
    const { data: sample } = await supabase
      .from('kelly_motion_library')
      .select('persona, phase, video_url')
      .eq('status', 'completed')
      .limit(1)
      .single();
    
    if (sample) {
      console.log(`Sample: ${sample.persona}/${sample.phase}`);
      console.log(`URL: ${sample.video_url}`);
    }
  }
  
  console.log('\n🎉 Done!');
}

main().catch(console.error);
