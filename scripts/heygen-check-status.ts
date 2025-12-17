#!/usr/bin/env npx tsx
/**
 * HEYGEN STATUS CHECKER
 * 
 * Checks the status of HeyGen video generation jobs.
 * Can check individual videos or all pending videos in a day manifest.
 * 
 * Usage:
 *   npx tsx scripts/heygen-check-status.ts <video_id>
 *   npx tsx scripts/heygen-check-status.ts --day 351
 *   npx tsx scripts/heygen-check-status.ts --day 351 --poll
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

interface VideoStatus {
  video_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  video_url?: string;
  thumbnail_url?: string;
  duration?: number;
  error?: string;
}

interface ManifestVideo {
  video_id: string;
  status: string;
  video_url?: string;
  phases: string[];
  total_scenes: number;
  estimated_duration: number;
  submitted: string;
}

interface DayManifest {
  day: number;
  generated: string;
  updated: string;
  videos: { [archetype: string]: ManifestVideo };
}

// ═══════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

async function checkVideoStatus(videoId: string): Promise<VideoStatus> {
  const response = await fetch(
    `https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`,
    { headers: { 'X-Api-Key': HEYGEN_API_KEY } }
  );
  
  const result = await response.json();
  
  return {
    video_id: videoId,
    status: result.data?.status || 'unknown',
    video_url: result.data?.video_url,
    thumbnail_url: result.data?.thumbnail_url,
    duration: result.data?.duration,
    error: result.data?.error,
  };
}

// ═══════════════════════════════════════════════════════════════════
// MANIFEST FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

function loadManifest(day: number): DayManifest | null {
  const manifestPath = path.join(process.cwd(), 'generated-videos', `day-${day}-manifest.json`);
  
  if (!fs.existsSync(manifestPath)) {
    return null;
  }
  
  return JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
}

function saveManifest(manifest: DayManifest): void {
  const manifestPath = path.join(process.cwd(), 'generated-videos', `day-${manifest.day}-manifest.json`);
  manifest.updated = new Date().toISOString();
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}

// ═══════════════════════════════════════════════════════════════════
// DISPLAY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

function displayStatus(archetype: string, status: VideoStatus): void {
  const icon = 
    status.status === 'completed' ? '✅' :
    status.status === 'failed' ? '❌' :
    status.status === 'processing' ? '⏳' : '⏸️';
  
  console.log(`\n${icon} ${archetype.toUpperCase()}`);
  console.log(`   Video ID: ${status.video_id}`);
  console.log(`   Status: ${status.status}`);
  
  if (status.status === 'completed') {
    console.log(`   Duration: ${status.duration}s`);
    console.log(`   Video: ${status.video_url}`);
    if (status.thumbnail_url) {
      console.log(`   Thumbnail: ${status.thumbnail_url}`);
    }
  } else if (status.status === 'failed') {
    console.log(`   Error: ${status.error || 'Unknown error'}`);
  }
}

function displaySummary(results: { archetype: string; status: VideoStatus }[]): void {
  const completed = results.filter(r => r.status.status === 'completed').length;
  const failed = results.filter(r => r.status.status === 'failed').length;
  const pending = results.filter(r => ['pending', 'processing'].includes(r.status.status)).length;
  
  console.log('\n════════════════════════════════════════════════════════════════');
  console.log(`📊 SUMMARY: ${completed} completed, ${pending} pending, ${failed} failed`);
  console.log('════════════════════════════════════════════════════════════════');
  
  if (completed > 0) {
    console.log('\n✅ Completed videos:');
    results
      .filter(r => r.status.status === 'completed')
      .forEach(r => console.log(`   ${r.archetype}: ${r.status.video_url}`));
  }
  
  if (failed > 0) {
    console.log('\n❌ Failed videos:');
    results
      .filter(r => r.status.status === 'failed')
      .forEach(r => console.log(`   ${r.archetype}: ${r.status.error}`));
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════

async function checkSingleVideo(videoId: string): Promise<void> {
  console.log(`\n🔍 Checking video: ${videoId}`);
  
  const status = await checkVideoStatus(videoId);
  displayStatus('Video', status);
}

async function checkDayManifest(day: number, poll: boolean): Promise<void> {
  console.log(`\n📋 Checking Day ${day} manifest...`);
  
  const manifest = loadManifest(day);
  
  if (!manifest) {
    console.error(`❌ No manifest found for Day ${day}`);
    console.log(`   Expected: generated-videos/day-${day}-manifest.json`);
    return;
  }
  
  const archetypes = Object.keys(manifest.videos);
  console.log(`   Found ${archetypes.length} video(s): ${archetypes.join(', ')}`);
  
  const checkAll = async (): Promise<{ archetype: string; status: VideoStatus }[]> => {
    const results: { archetype: string; status: VideoStatus }[] = [];
    
    for (const archetype of archetypes) {
      const video = manifest.videos[archetype];
      const status = await checkVideoStatus(video.video_id);
      
      // Update manifest
      manifest.videos[archetype].status = status.status;
      if (status.video_url) {
        manifest.videos[archetype].video_url = status.video_url;
      }
      
      results.push({ archetype, status });
      displayStatus(archetype, status);
    }
    
    saveManifest(manifest);
    return results;
  };
  
  if (poll) {
    console.log('\n⏳ Polling mode - checking every 30 seconds until all complete...');
    console.log('   Press Ctrl+C to stop\n');
    
    let allComplete = false;
    let iteration = 0;
    
    while (!allComplete) {
      iteration++;
      console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
      console.log(`   Check #${iteration} - ${new Date().toLocaleTimeString()}`);
      console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
      
      const results = await checkAll();
      displaySummary(results);
      
      const pending = results.filter(r => ['pending', 'processing'].includes(r.status.status));
      
      if (pending.length === 0) {
        allComplete = true;
        console.log('\n🎉 All videos complete!');
      } else {
        console.log(`\n⏳ ${pending.length} still processing... waiting 30s`);
        await new Promise(r => setTimeout(r, 30000));
      }
    }
  } else {
    const results = await checkAll();
    displaySummary(results);
  }
}

async function main(): Promise<void> {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🔍 HEYGEN STATUS CHECKER                                      ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  const args = process.argv.slice(2);
  
  // Check for --day argument
  const dayIndex = args.indexOf('--day');
  if (dayIndex !== -1 && args[dayIndex + 1]) {
    const day = parseInt(args[dayIndex + 1]);
    const poll = args.includes('--poll');
    await checkDayManifest(day, poll);
    return;
  }
  
  // Check for direct video ID
  const videoId = args.find(a => !a.startsWith('--'));
  if (videoId) {
    await checkSingleVideo(videoId);
    return;
  }
  
  // Show usage
  console.log('\nUsage:');
  console.log('  npx tsx scripts/heygen-check-status.ts <video_id>');
  console.log('  npx tsx scripts/heygen-check-status.ts --day 351');
  console.log('  npx tsx scripts/heygen-check-status.ts --day 351 --poll');
  console.log('');
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
