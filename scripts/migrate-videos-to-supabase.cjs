/**
 * Migrate Kelly Videos from HeyGen to Supabase Storage
 * 
 * This script:
 * 1. Creates the kelly-videos bucket if it doesn't exist
 * 2. Downloads completed videos from HeyGen (before URLs expire)
 * 3. Uploads them to Supabase storage
 * 4. Updates the database with permanent URLs
 * 
 * Run: node scripts/migrate-videos-to-supabase.cjs
 */

require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

// Configuration
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const BUCKET_NAME = 'kelly-videos';

// Validate environment
if (!SUPABASE_SERVICE_KEY) {
  console.error('❌ SUPABASE_SERVICE_ROLE_KEY not found in environment!');
  console.error('   Please set this in your .env file');
  console.error('   You can find it in Supabase Dashboard > Settings > API > Service Role Key');
  process.exit(1);
}

// Create Supabase admin client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false }
});

// Download file from URL
function downloadFile(url) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    
    protocol.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        // Follow redirect
        return downloadFile(response.headers.location).then(resolve).catch(reject);
      }
      
      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode}`));
        return;
      }
      
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.on('end', () => resolve(Buffer.concat(chunks)));
      response.on('error', reject);
    }).on('error', reject);
  });
}

async function ensureBucket() {
  console.log('\n📦 Checking storage bucket...');
  
  // First, try to list files in the bucket to see if it exists
  const { data: files, error: listFilesError } = await supabase.storage
    .from(BUCKET_NAME)
    .list('', { limit: 1 });
  
  if (!listFilesError) {
    console.log('✅ Bucket "kelly-videos" already exists');
    return true;
  }
  
  // Bucket doesn't exist or we can't access it, try to create it
  console.log('📦 Creating bucket "kelly-videos"...');
  
  // Try creating with minimal options first
  const { data, error: createError } = await supabase.storage.createBucket(BUCKET_NAME, {
    public: true
  });
  
  if (createError) {
    // If it already exists, that's fine
    if (createError.message.includes('already exists') || createError.message.includes('duplicate')) {
      console.log('✅ Bucket already exists');
      return true;
    }
    console.error('❌ Error creating bucket:', createError.message);
    console.error('   You may need to create the bucket manually in Supabase Dashboard:');
    console.error('   1. Go to Storage > New bucket');
    console.error('   2. Name: "kelly-videos"');
    console.error('   3. Enable "Public bucket"');
    return false;
  }
  
  console.log('✅ Bucket created successfully');
  return true;
}

async function getCompletedMotionClips() {
  console.log('\n📊 Fetching completed motion clips...');
  
  const { data, error } = await supabase
    .from('kelly_motion_library')
    .select('*')
    .eq('status', 'completed')
    .not('video_url', 'is', null)
    .order('completed_at', { ascending: false });
  
  if (error) {
    console.error('❌ Error fetching clips:', error.message);
    return [];
  }
  
  console.log(`✅ Found ${data.length} completed clips`);
  return data;
}

async function migrateClip(clip) {
  const { id, avatar_key, persona, age_bucket, phase, video_url } = clip;
  
  // Skip if already migrated (URL points to our Supabase storage)
  if (video_url.includes('supabase.co/storage')) {
    return { status: 'skipped', reason: 'Already in Supabase' };
  }
  
  // Skip if URL is expired (HeyGen URLs expire)
  if (!video_url || !video_url.startsWith('http')) {
    return { status: 'skipped', reason: 'Invalid URL' };
  }
  
  try {
    // Download the video
    console.log(`  ⬇️ Downloading ${persona}/${phase}...`);
    const videoBuffer = await downloadFile(video_url);
    
    if (!videoBuffer || videoBuffer.length < 1000) {
      return { status: 'error', reason: 'Download failed or file too small' };
    }
    
    // Generate storage path: motion/{persona}/{age_bucket}/{phase}.mp4
    const storagePath = `motion/${persona}/${age_bucket}/${phase}.mp4`;
    
    // Upload to Supabase storage
    console.log(`  ⬆️ Uploading to ${storagePath}...`);
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from(BUCKET_NAME)
      .upload(storagePath, videoBuffer, {
        contentType: 'video/mp4',
        upsert: true
      });
    
    if (uploadError) {
      return { status: 'error', reason: uploadError.message };
    }
    
    // Get public URL
    const { data: { publicUrl } } = supabase.storage
      .from(BUCKET_NAME)
      .getPublicUrl(storagePath);
    
    // Update database with new URL (only video_url column)
    const { error: updateError } = await supabase
      .from('kelly_motion_library')
      .update({ video_url: publicUrl })
      .eq('id', id);
    
    if (updateError) {
      return { status: 'error', reason: `Upload OK but DB update failed: ${updateError.message}` };
    }
    
    return { status: 'success', url: publicUrl };
    
  } catch (e) {
    return { status: 'error', reason: e.message };
  }
}

async function main() {
  console.log('===========================================');
  console.log('🎬 KELLY VIDEO MIGRATION TO SUPABASE');
  console.log('===========================================');
  console.log('Supabase URL:', SUPABASE_URL);
  console.log('');
  
  // Step 1: Ensure bucket exists
  const bucketReady = await ensureBucket();
  if (!bucketReady) {
    console.error('❌ Could not ensure bucket exists. Aborting.');
    process.exit(1);
  }
  
  // Step 2: Get all completed clips
  const clips = await getCompletedMotionClips();
  
  if (clips.length === 0) {
    console.log('⚠️ No clips to migrate');
    return;
  }
  
  // Step 3: Migrate each clip
  console.log('\n🚀 Migrating videos...');
  
  const results = {
    success: 0,
    skipped: 0,
    error: 0,
    errors: []
  };
  
  // Process in batches to avoid overwhelming the API
  const BATCH_SIZE = 5;
  for (let i = 0; i < clips.length; i += BATCH_SIZE) {
    const batch = clips.slice(i, i + BATCH_SIZE);
    const batchNum = Math.floor(i / BATCH_SIZE) + 1;
    const totalBatches = Math.ceil(clips.length / BATCH_SIZE);
    
    console.log(`\n📦 Batch ${batchNum}/${totalBatches}`);
    
    const batchResults = await Promise.all(batch.map(clip => migrateClip(clip)));
    
    batchResults.forEach((result, idx) => {
      const clip = batch[idx];
      if (result.status === 'success') {
        results.success++;
        console.log(`  ✅ ${clip.persona}/${clip.phase} → ${result.url.split('/').pop()}`);
      } else if (result.status === 'skipped') {
        results.skipped++;
        console.log(`  ⏭️ ${clip.persona}/${clip.phase} (${result.reason})`);
      } else {
        results.error++;
        results.errors.push({ clip: `${clip.persona}/${clip.phase}`, error: result.reason });
        console.log(`  ❌ ${clip.persona}/${clip.phase}: ${result.reason}`);
      }
    });
    
    // Small delay between batches
    if (i + BATCH_SIZE < clips.length) {
      await new Promise(r => setTimeout(r, 1000));
    }
  }
  
  // Summary
  console.log('\n===========================================');
  console.log('MIGRATION COMPLETE');
  console.log('===========================================');
  console.log(`✅ Success: ${results.success}`);
  console.log(`⏭️ Skipped: ${results.skipped}`);
  console.log(`❌ Errors: ${results.error}`);
  
  if (results.errors.length > 0) {
    console.log('\nErrors:');
    results.errors.slice(0, 10).forEach(e => {
      console.log(`  - ${e.clip}: ${e.error}`);
    });
    if (results.errors.length > 10) {
      console.log(`  ... and ${results.errors.length - 10} more`);
    }
  }
  
  console.log('\n🎉 Done!');
}

main().catch(console.error);
