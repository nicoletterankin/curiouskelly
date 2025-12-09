#!/usr/bin/env npx tsx
/**
 * 🚀 UPLOAD DAY 1 HD VIDEOS TO SUPABASE STORAGE
 * 
 * This script uploads the generated Day 1 HD videos to Supabase Storage
 * in the correct directory structure that learn.html expects.
 * 
 * SOURCE: generated-videos/golden/*.mp4
 * TARGET: kelly-videos bucket
 *   kelly-videos/day-001/{archetype}/{phase}.mp4
 * 
 * URL Pattern: https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-videos/day-001/explorer/hook.mp4
 * 
 * Usage:
 *   npx tsx scripts/kelly-video-factory/upload-day1-videos.ts
 *   npx tsx scripts/kelly-video-factory/upload-day1-videos.ts --dry-run
 */

import 'dotenv/config';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  
  // Source directory with generated videos
  SOURCE_DIR: path.join(process.cwd(), 'generated-videos', 'golden'),
  
  // Alternative source (4K versions)
  ALTERNATE_SOURCE_DIR: path.join(process.cwd(), 'generated-videos', 'golden-lesson'),
  
  // Supabase storage bucket
  BUCKET_NAME: 'kelly-videos',
  
  // Day number
  DAY_NUMBER: 1,
};

// =============================================================================
// TYPES
// =============================================================================

interface VideoMapping {
  sourceFile: string;
  targetPath: string;
  archetype: string;
  phase: string;
}

// =============================================================================
// SUPABASE CLIENT
// =============================================================================

let supabase: SupabaseClient;

function getSupabase(): SupabaseClient {
  if (!supabase) {
    supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
  }
  return supabase;
}

// =============================================================================
// VIDEO MAPPING
// =============================================================================

/**
 * Map source video files to target storage paths
 * Source format: day_001_Hook_The_Explorer_golden.mp4
 * Target format: day-001/explorer/hook.mp4
 */
function mapVideoFiles(): VideoMapping[] {
  const mappings: VideoMapping[] = [];
  
  // Check if source directory exists
  if (!fs.existsSync(CONFIG.SOURCE_DIR)) {
    console.error(`❌ Source directory not found: ${CONFIG.SOURCE_DIR}`);
    return mappings;
  }
  
  // Read all .mp4 files
  const files = fs.readdirSync(CONFIG.SOURCE_DIR).filter(f => f.endsWith('.mp4') && f.startsWith('day_001'));
  
  for (const file of files) {
    // Parse filename: day_001_Hook_The_Explorer_golden.mp4
    const match = file.match(/day_(\d+)_(\w+)_(The_\w+)_golden\.mp4/);
    
    if (!match) {
      console.warn(`⚠️ Skipping unrecognized file: ${file}`);
      continue;
    }
    
    const [, dayStr, phase, archetype] = match;
    
    // Normalize archetype: "The_Explorer" -> "explorer"
    const normalizedArchetype = archetype.replace(/^The_/, '').toLowerCase();
    
    // Normalize phase: "Hook" -> "hook"
    const normalizedPhase = phase.toLowerCase();
    
    const targetPath = `day-${dayStr}/${normalizedArchetype}/${normalizedPhase}.mp4`;
    
    mappings.push({
      sourceFile: path.join(CONFIG.SOURCE_DIR, file),
      targetPath,
      archetype: normalizedArchetype,
      phase: normalizedPhase,
    });
  }
  
  return mappings;
}

/**
 * Map alternate source (4K versions from golden-lesson folder)
 * Source format: day_001_Hook_The_Explorer/final_4k.mp4
 */
function mapAlternateVideos(): VideoMapping[] {
  const mappings: VideoMapping[] = [];
  
  if (!fs.existsSync(CONFIG.ALTERNATE_SOURCE_DIR)) {
    return mappings;
  }
  
  const folders = fs.readdirSync(CONFIG.ALTERNATE_SOURCE_DIR).filter(f => 
    fs.statSync(path.join(CONFIG.ALTERNATE_SOURCE_DIR, f)).isDirectory() &&
    f.startsWith('day_001')
  );
  
  for (const folder of folders) {
    const videoPath = path.join(CONFIG.ALTERNATE_SOURCE_DIR, folder, 'final_4k.mp4');
    
    if (!fs.existsSync(videoPath)) {
      continue;
    }
    
    // Parse folder name: day_001_Hook_The_Explorer
    const match = folder.match(/day_(\d+)_(\w+)_(The_\w+)/);
    
    if (!match) {
      continue;
    }
    
    const [, dayStr, phase, archetype] = match;
    const normalizedArchetype = archetype.replace(/^The_/, '').toLowerCase();
    const normalizedPhase = phase.toLowerCase();
    
    const targetPath = `day-${dayStr}/${normalizedArchetype}/${normalizedPhase}.mp4`;
    
    mappings.push({
      sourceFile: videoPath,
      targetPath,
      archetype: normalizedArchetype,
      phase: normalizedPhase,
    });
  }
  
  return mappings;
}

// =============================================================================
// UPLOAD FUNCTIONS
// =============================================================================

async function ensureBucketExists(): Promise<boolean> {
  const sb = getSupabase();
  
  // Check if bucket exists
  const { data: buckets, error } = await sb.storage.listBuckets();
  
  if (error) {
    console.error('❌ Error listing buckets:', error.message);
    return false;
  }
  
  const bucketExists = buckets?.some(b => b.name === CONFIG.BUCKET_NAME);
  
  if (!bucketExists) {
    console.log(`📦 Creating bucket: ${CONFIG.BUCKET_NAME}`);
    const { error: createError } = await sb.storage.createBucket(CONFIG.BUCKET_NAME, {
      public: true,
      fileSizeLimit: 100 * 1024 * 1024, // 100MB
    });
    
    if (createError) {
      console.error('❌ Error creating bucket:', createError.message);
      return false;
    }
    
    console.log('✅ Bucket created');
  } else {
    console.log(`✅ Bucket exists: ${CONFIG.BUCKET_NAME}`);
  }
  
  return true;
}

async function uploadVideo(mapping: VideoMapping, dryRun: boolean): Promise<boolean> {
  const sb = getSupabase();
  
  console.log(`\n📹 ${mapping.archetype}/${mapping.phase}`);
  console.log(`   Source: ${path.basename(mapping.sourceFile)}`);
  console.log(`   Target: ${mapping.targetPath}`);
  
  if (dryRun) {
    console.log('   [DRY RUN] Would upload');
    return true;
  }
  
  // Read file
  const fileBuffer = fs.readFileSync(mapping.sourceFile);
  const fileSizeMB = fileBuffer.length / (1024 * 1024);
  console.log(`   Size: ${fileSizeMB.toFixed(2)} MB`);
  
  // Upload to Supabase Storage
  const { error } = await sb.storage
    .from(CONFIG.BUCKET_NAME)
    .upload(mapping.targetPath, fileBuffer, {
      contentType: 'video/mp4',
      upsert: true, // Overwrite if exists
    });
  
  if (error) {
    console.error(`   ❌ Upload failed: ${error.message}`);
    return false;
  }
  
  // Get public URL
  const { data: urlData } = sb.storage
    .from(CONFIG.BUCKET_NAME)
    .getPublicUrl(mapping.targetPath);
  
  console.log(`   ✅ Uploaded: ${urlData.publicUrl}`);
  
  return true;
}

async function updateDatabaseUrls(mappings: VideoMapping[], dryRun: boolean): Promise<void> {
  const sb = getSupabase();
  
  console.log('\n📝 Updating database URLs...\n');
  
  // Get Day 1 lesson ID
  const { data: lesson, error: lessonError } = await sb
    .from('core_lessons')
    .select('id')
    .eq('day_number', CONFIG.DAY_NUMBER)
    .single();
  
  if (lessonError || !lesson) {
    console.error('❌ Failed to get Day 1 lesson:', lessonError?.message);
    return;
  }
  
  for (const mapping of mappings) {
    // Construct the full public URL
    const publicUrl = `${CONFIG.SUPABASE_URL}/storage/v1/object/public/${CONFIG.BUCKET_NAME}/${mapping.targetPath}`;
    
    // Map archetype folder name to database archetype
    const archetypeMap: Record<string, string> = {
      'explorer': 'The Explorer',
      'rebel': 'The Rebel',
      'scientist': 'The Scientist',
    };
    
    // Map phase to database phase
    const phaseMap: Record<string, string> = {
      'hook': 'Hook',
      'fact1': 'Fact1',
      'fact2': 'Fact2',
      'fact3': 'Fact3',
      'wisdom': 'Wisdom',
    };
    
    const dbArchetype = archetypeMap[mapping.archetype] || mapping.archetype;
    const dbPhase = phaseMap[mapping.phase] || mapping.phase;
    
    console.log(`   ${dbArchetype} - ${dbPhase}`);
    
    if (dryRun) {
      console.log(`      [DRY RUN] Would update hd_video_url to: ${publicUrl}`);
      continue;
    }
    
    // Update lesson_atoms table
    const { error: updateError } = await sb
      .from('lesson_atoms')
      .update({ hd_video_url: publicUrl })
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', dbArchetype)
      .eq('phase', dbPhase);
    
    if (updateError) {
      console.error(`      ❌ Update failed: ${updateError.message}`);
    } else {
      console.log(`      ✅ Updated`);
    }
  }
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const useAlternate = args.includes('--4k');
  
  console.log('\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log('║  🚀 UPLOAD DAY 1 HD VIDEOS TO SUPABASE                               ║');
  console.log('╚' + '═'.repeat(70) + '╝');
  
  if (dryRun) {
    console.log('\n⚠️  DRY RUN MODE - No changes will be made\n');
  }
  
  // Validate API keys
  if (!CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_KEY) {
    console.error('❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
    process.exit(1);
  }
  
  // Ensure bucket exists
  const bucketReady = await ensureBucketExists();
  if (!bucketReady) {
    process.exit(1);
  }
  
  // Map video files
  console.log('\n📂 Scanning video files...\n');
  
  let mappings = mapVideoFiles();
  
  if (mappings.length === 0 || useAlternate) {
    console.log('   Trying alternate source (4K versions)...');
    mappings = mapAlternateVideos();
  }
  
  if (mappings.length === 0) {
    console.error('❌ No video files found to upload');
    console.log('\nExpected files in:');
    console.log(`   ${CONFIG.SOURCE_DIR}`);
    console.log('   Format: day_001_Hook_The_Explorer_golden.mp4\n');
    process.exit(1);
  }
  
  console.log(`   Found ${mappings.length} videos to upload\n`);
  
  // Upload videos
  console.log('═'.repeat(72));
  console.log('📤 UPLOADING VIDEOS');
  console.log('═'.repeat(72));
  
  let successCount = 0;
  let failCount = 0;
  
  for (const mapping of mappings) {
    const success = await uploadVideo(mapping, dryRun);
    if (success) {
      successCount++;
    } else {
      failCount++;
    }
  }
  
  // Update database URLs
  console.log('\n' + '═'.repeat(72));
  console.log('📝 UPDATING DATABASE');
  console.log('═'.repeat(72));
  
  await updateDatabaseUrls(mappings, dryRun);
  
  // Summary
  console.log('\n' + '═'.repeat(72));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(72));
  console.log(`   ✅ Uploaded: ${successCount}`);
  console.log(`   ❌ Failed: ${failCount}`);
  console.log(`   📦 Bucket: ${CONFIG.BUCKET_NAME}`);
  console.log(`   🔗 Base URL: ${CONFIG.SUPABASE_URL}/storage/v1/object/public/${CONFIG.BUCKET_NAME}/day-001/`);
  
  if (dryRun) {
    console.log('\n⚠️  This was a DRY RUN. Re-run without --dry-run to actually upload.\n');
  }
  
  console.log('\n🎉 Done!\n');
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});


