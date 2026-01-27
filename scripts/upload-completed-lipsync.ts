#!/usr/bin/env npx tsx
/**
 * 📤 UPLOAD COMPLETED LIPSYNC VIDEOS TO SUPABASE
 * 
 * Finds lip-synced videos in the kelly-pipeline/videos/lipsync folder,
 * uploads them to Supabase Storage, and updates kelly_lesson_assets table.
 * 
 * Usage:
 *   npx tsx scripts/upload-completed-lipsync.ts
 *   npx tsx scripts/upload-completed-lipsync.ts --day 14
 *   npx tsx scripts/upload-completed-lipsync.ts --dry-run
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  // Video source directory (WSL path translated to Windows)
  VIDEO_DIR: 'C:\\Users\\user\\kelly-pipeline\\videos\\lipsync',
  BUCKET_NAME: 'kelly-videos',
  STORAGE_PREFIX: 'lipsync',
};

// Phase mapping: filename -> database phase
const PHASE_MAP: Record<string, string> = {
  'welcome': 'hook',
  'hook': 'hook',
  'mainContent': 'story',
  'story': 'story',
  'wisdomMoment': 'wisdom',
  'wisdom': 'wisdom',
  'wonder': 'wonder',
  'action': 'action',
};

// Age bucket normalization
const AGE_BUCKET_MAP: Record<string, number> = {
  '2-5': 5,
  '6-12': 8,
  '13-17': 16,
  '18-35': 35,
  '36-60': 50,
  '61+': 70,
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// VIDEO FILE DISCOVERY
// =============================================================================

interface VideoFile {
  filepath: string;
  filename: string;
  ageBucket: string;
  ageGroup: number;
  phase: string;
  dbPhase: string;
  dayNumber: number;
}

function parseVideoFilename(filename: string, dayNumber: number): VideoFile | null {
  // Pattern: {ageBucket}-{phase}.mp4 (e.g., "13-17-welcome.mp4")
  const match = filename.match(/^(\d+-\d+|\d+\+?)-(\w+)\.mp4$/);
  if (!match) {
    console.log(`   ⚠️ Skipping unrecognized filename: ${filename}`);
    return null;
  }

  const [, ageBucket, phase] = match;
  const dbPhase = PHASE_MAP[phase];
  
  if (!dbPhase) {
    console.log(`   ⚠️ Unknown phase "${phase}" in: ${filename}`);
    return null;
  }

  const ageGroup = AGE_BUCKET_MAP[ageBucket];
  if (!ageGroup) {
    console.log(`   ⚠️ Unknown age bucket "${ageBucket}" in: ${filename}`);
    return null;
  }

  return {
    filepath: '',
    filename,
    ageBucket,
    ageGroup,
    phase,
    dbPhase,
    dayNumber,
  };
}

function findVideoFiles(videoDir: string, dayNumber: number): VideoFile[] {
  const videos: VideoFile[] = [];
  
  if (!fs.existsSync(videoDir)) {
    console.error(`❌ Video directory not found: ${videoDir}`);
    return videos;
  }

  const files = fs.readdirSync(videoDir);
  
  for (const file of files) {
    if (!file.endsWith('.mp4')) continue;
    
    const filepath = path.join(videoDir, file);
    const stats = fs.statSync(filepath);
    
    // Skip directories
    if (stats.isDirectory()) continue;
    
    const parsed = parseVideoFilename(file, dayNumber);
    if (parsed) {
      parsed.filepath = filepath;
      videos.push(parsed);
    }
  }

  return videos;
}

// =============================================================================
// UPLOAD FUNCTIONS
// =============================================================================

async function uploadVideo(video: VideoFile, dryRun: boolean): Promise<string | null> {
  const fileBuffer = fs.readFileSync(video.filepath);
  const fileSizeMB = (fileBuffer.length / 1024 / 1024).toFixed(2);
  
  // Storage path: lipsync/2026/en/day-{N}/{phase}-age{X}-en.mp4
  const year = new Date().getFullYear();
  const storagePath = `${CONFIG.STORAGE_PREFIX}/${year}/en/day-${video.dayNumber.toString().padStart(3, '0')}/${video.dbPhase}-age${video.ageGroup}-en.mp4`;
  
  console.log(`   📤 Uploading ${fileSizeMB} MB to ${storagePath}...`);
  
  if (dryRun) {
    console.log(`   ✅ (DRY RUN) Would upload to: ${storagePath}`);
    return `https://placeholder.supabase.co/storage/v1/object/public/${CONFIG.BUCKET_NAME}/${storagePath}`;
  }

  const { data, error } = await supabase.storage
    .from(CONFIG.BUCKET_NAME)
    .upload(storagePath, fileBuffer, {
      contentType: 'video/mp4',
      upsert: true,
    });

  if (error) {
    console.error(`   ❌ Upload failed:`, error.message);
    return null;
  }

  // Get public URL
  const { data: urlData } = supabase.storage
    .from(CONFIG.BUCKET_NAME)
    .getPublicUrl(storagePath);

  console.log(`   ✅ Uploaded: ${urlData.publicUrl}`);
  return urlData.publicUrl;
}

async function updateDatabase(
  video: VideoFile,
  videoUrl: string,
  dryRun: boolean
): Promise<boolean> {
  console.log(`   📝 Updating kelly_lesson_assets: day=${video.dayNumber}, phase=${video.dbPhase}, age=${video.ageGroup}`);
  
  if (dryRun) {
    console.log(`   ✅ (DRY RUN) Would update status to 'complete'`);
    return true;
  }

  // Update the kelly_lesson_assets table
  const { error, count } = await supabase
    .from('kelly_lesson_assets')
    .update({
      video_url: videoUrl,
      video_source: 'sadtalker',
      status: 'complete',
      updated_at: new Date().toISOString(),
    })
    .eq('day_number', video.dayNumber)
    .eq('phase', video.dbPhase)
    .eq('age_group', video.ageGroup)
    .eq('language', 'en');

  if (error) {
    console.error(`   ❌ Database update failed:`, error.message);
    return false;
  }

  console.log(`   ✅ Database updated (status: complete)`);
  return true;
}

// =============================================================================
// MAIN PIPELINE
// =============================================================================

interface UploadStats {
  found: number;
  uploaded: number;
  failed: number;
  skipped: number;
}

async function runUploadPipeline(dayNumber: number, dryRun: boolean): Promise<UploadStats> {
  const stats: UploadStats = { found: 0, uploaded: 0, failed: 0, skipped: 0 };
  
  console.log('\n' + '═'.repeat(72));
  console.log(`  📤 LIPSYNC VIDEO UPLOAD - Day ${dayNumber}`);
  console.log('═'.repeat(72));
  
  // Find videos
  console.log(`\n📂 Scanning: ${CONFIG.VIDEO_DIR}`);
  const videos = findVideoFiles(CONFIG.VIDEO_DIR, dayNumber);
  stats.found = videos.length;
  
  if (videos.length === 0) {
    console.log('   ⚠️ No MP4 files found to upload');
    return stats;
  }
  
  console.log(`   ✓ Found ${videos.length} video(s)\n`);
  
  // Process each video
  for (let i = 0; i < videos.length; i++) {
    const video = videos[i];
    console.log(`\n[${i + 1}/${videos.length}] ${video.filename}`);
    console.log(`   Age: ${video.ageBucket} → ${video.ageGroup}`);
    console.log(`   Phase: ${video.phase} → ${video.dbPhase}`);
    
    // Upload
    const videoUrl = await uploadVideo(video, dryRun);
    if (!videoUrl) {
      stats.failed++;
      continue;
    }
    
    // Update database
    const success = await updateDatabase(video, videoUrl, dryRun);
    if (success) {
      stats.uploaded++;
    } else {
      stats.failed++;
    }
  }
  
  return stats;
}

// =============================================================================
// STATUS REPORT
// =============================================================================

async function getRegistryStatus(): Promise<Record<string, number>> {
  const { data, error } = await supabase
    .from('kelly_lesson_assets')
    .select('status')
    .then(result => {
      if (result.error) throw result.error;
      const counts: Record<string, number> = {};
      for (const row of result.data || []) {
        counts[row.status] = (counts[row.status] || 0) + 1;
      }
      return { data: counts, error: null };
    });

  if (error) {
    console.error('Failed to get registry status:', error);
    return {};
  }

  return data;
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  let dayNumber = new Date().getDate(); // Default to today's day of month
  let dryRun = false;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--day':
        dayNumber = parseInt(args[++i]);
        break;
      case '--dry-run':
        dryRun = true;
        break;
      case '--help':
        console.log(`
📤 Upload Completed Lipsync Videos

Usage:
  npx tsx scripts/upload-completed-lipsync.ts [options]

Options:
  --day <number>   Day number to assign (default: today's date)
  --dry-run        Show what would happen without uploading
  --help           Show this help

Examples:
  npx tsx scripts/upload-completed-lipsync.ts
  npx tsx scripts/upload-completed-lipsync.ts --day 14
  npx tsx scripts/upload-completed-lipsync.ts --dry-run
`);
        process.exit(0);
    }
  }

  // Check prerequisites
  if (!CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_KEY) {
    console.error('❌ Missing Supabase credentials. Set PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY');
    process.exit(1);
  }

  console.log('\n🚀 KELLY LIPSYNC VIDEO UPLOADER');
  console.log('━'.repeat(72));
  console.log(`Day Number: ${dayNumber}`);
  console.log(`Dry Run: ${dryRun}`);
  console.log(`Source: ${CONFIG.VIDEO_DIR}`);
  console.log(`Target: ${CONFIG.BUCKET_NAME}/${CONFIG.STORAGE_PREFIX}/`);
  console.log('━'.repeat(72));

  // Run upload
  const stats = await runUploadPipeline(dayNumber, dryRun);

  // Get registry status
  console.log('\n' + '═'.repeat(72));
  console.log('  📊 REGISTRY STATUS');
  console.log('═'.repeat(72));
  
  const status = await getRegistryStatus();
  for (const [state, count] of Object.entries(status).sort()) {
    const emoji = state === 'complete' ? '✅' : state === 'audio_ready' ? '🔊' : state === 'pending' ? '⏳' : '❓';
    console.log(`   ${emoji} ${state}: ${count}`);
  }

  // Final report
  console.log('\n' + '═'.repeat(72));
  console.log('  📋 UPLOAD REPORT');
  console.log('═'.repeat(72));
  console.log(`   Found:    ${stats.found} videos`);
  console.log(`   Uploaded: ${stats.uploaded} videos`);
  console.log(`   Failed:   ${stats.failed} videos`);
  console.log(`   Skipped:  ${stats.skipped} videos`);
  console.log('═'.repeat(72));

  if (dryRun) {
    console.log('\n💡 This was a DRY RUN. Run without --dry-run to actually upload.');
  }
}

main().catch(console.error);
