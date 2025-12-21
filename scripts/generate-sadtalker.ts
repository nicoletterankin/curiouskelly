#!/usr/bin/env npx tsx
/**
 * 🎬 SADTALKER VIDEO GENERATION
 * 
 * Generates talking head videos using SadTalker via fal.ai.
 * This is the fallback pipeline when HeyGen is unavailable.
 * 
 * Usage:
 *   npx tsx scripts/generate-sadtalker.ts --day=354
 *   npx tsx scripts/generate-sadtalker.ts --days=355,356
 *   npx tsx scripts/generate-sadtalker.ts --days=auto  # Next 3 days without videos
 */

import 'dotenv/config';
import { fal } from '@fal-ai/client';
import { createClient } from '@supabase/supabase-js';

// Configure fal.ai
fal.config({ credentials: process.env.FAL_KEY! });

// Supabase client
const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY!;
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Kelly reference image (verified working photorealistic image)
const KELLY_IMAGE = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/photorealistic-test/kelly_1765361262640.png';

const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];

// =============================================================================
// ARGUMENTS
// =============================================================================

function parseArgs(): { days: number[] } {
  const args = process.argv.slice(2);
  let days: number[] = [];
  
  for (const arg of args) {
    if (arg.startsWith('--day=')) {
      days = [parseInt(arg.split('=')[1], 10)];
    } else if (arg.startsWith('--days=')) {
      const value = arg.split('=')[1];
      if (value === 'auto') {
        // Will be resolved below
        days = [];
      } else {
        days = value.split(',').map(d => parseInt(d.trim(), 10));
      }
    }
  }
  
  return { days };
}

function getTodayDayNumber(): number {
  const startDate = new Date('2025-01-01');
  const today = new Date();
  const diffTime = today.getTime() - startDate.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return Math.min(365, Math.max(1, diffDays));
}

async function getNextDaysWithoutVideos(count: number = 3): Promise<number[]> {
  const today = getTodayDayNumber();
  const candidates = [today + 1, today + 2, today + 3].filter(d => d <= 365);
  const needsGeneration: number[] = [];
  
  for (const day of candidates) {
    const { count: videoCount } = await supabase
      .from('kelly_video_assets')
      .select('*', { count: 'exact', head: true })
      .eq('lesson_day', day)
      .eq('status', 'validated');
    
    if ((videoCount || 0) < PHASES.length) {
      needsGeneration.push(day);
      if (needsGeneration.length >= count) break;
    }
  }
  
  return needsGeneration;
}

// =============================================================================
// AUDIO LOOKUP
// =============================================================================

async function getAudioUrls(day: number): Promise<Record<string, string>> {
  const paddedDay = String(day).padStart(3, '0');
  const audioUrls: Record<string, string> = {};
  
  // List audio files from storage
  const { data: files } = await supabase.storage
    .from('kelly-templates')
    .list(`heygen/audio`, { search: `day_${day}_` });
  
  if (!files || files.length === 0) {
    console.log(`   ⚠️  No audio files found for day ${day}`);
    return audioUrls;
  }
  
  // Get the most recent audio file for each phase
  const phaseFiles: Record<string, { name: string; timestamp: number }> = {};
  
  for (const file of files) {
    for (const phase of PHASES) {
      if (file.name.includes(`day_${day}_${phase}_`)) {
        // Extract timestamp from filename: day_354_hook_1766200432881.mp3
        const match = file.name.match(/_(\d+)\.mp3$/);
        const timestamp = match ? parseInt(match[1], 10) : 0;
        
        if (!phaseFiles[phase] || timestamp > phaseFiles[phase].timestamp) {
          phaseFiles[phase] = { name: file.name, timestamp };
        }
      }
    }
  }
  
  // Convert to public URLs
  for (const [phase, file] of Object.entries(phaseFiles)) {
    const { data } = supabase.storage
      .from('kelly-templates')
      .getPublicUrl(`heygen/audio/${file.name}`);
    audioUrls[phase] = data.publicUrl;
  }
  
  return audioUrls;
}

// =============================================================================
// VIDEO GENERATION
// =============================================================================

async function generateVideo(day: number, phase: string, audioUrl: string): Promise<string | null> {
  console.log(`\n📽️  Day ${day} - Phase: ${phase}`);
  console.log(`   Audio: ${audioUrl.substring(audioUrl.lastIndexOf('/') + 1)}`);
  console.log('   ⏳ Processing with SadTalker...');
  
  try {
    const result = await fal.subscribe('fal-ai/sadtalker', {
      input: {
        source_image_url: KELLY_IMAGE,
        driven_audio_url: audioUrl,
        still: false, // Allow head motion
        enhancer: 'gfpgan', // Face enhancement
        preprocess: 'crop', // Crop to face
      },
      logs: false,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') {
          process.stdout.write('.');
        }
      }
    });
    
    const videoUrl = (result as any)?.video?.url || (result as any)?.data?.video?.url;
    
    if (!videoUrl) {
      console.log('\n   ❌ No video URL in response');
      return null;
    }
    
    console.log('\n   ✅ Video generated!');
    
    // Download and upload to Supabase
    console.log('   📤 Uploading to Supabase...');
    const response = await fetch(videoUrl);
    const buffer = Buffer.from(await response.arrayBuffer());
    
    const storagePath = `production/day_${String(day).padStart(3, '0')}/day_${day}_${phase}_explorer_sadtalker.mp4`;
    
    const { error: uploadError } = await supabase.storage
      .from('kelly-videos')
      .upload(storagePath, buffer, { 
        contentType: 'video/mp4', 
        upsert: true 
      });
    
    if (uploadError) {
      console.log(`   ⚠️  Upload error: ${uploadError.message}`);
      // Still return the fal.ai URL as backup
      return videoUrl;
    }
    
    const { data: urlData } = supabase.storage
      .from('kelly-videos')
      .getPublicUrl(storagePath);
    
    const publicUrl = urlData.publicUrl;
    console.log(`   📁 Uploaded: ${storagePath}`);
    
    // Register in kelly_video_assets
    await supabase.from('kelly_video_assets').upsert({
      lesson_day: day,
      phase,
      template: 'The Explorer',
      age_bucket: 'adult',
      public_url: publicUrl,
      storage_path: storagePath,
      status: 'validated',
      quality_tier: 'standard',
      language: 'en',
    }, {
      onConflict: 'lesson_day,phase,template,age_bucket,language',
    });
    
    return publicUrl;
    
  } catch (error) {
    console.log(`\n   ❌ Error: ${(error as Error).message}`);
    return null;
  }
}

// =============================================================================
// MAIN
// =============================================================================

async function generateDay(day: number): Promise<{ success: number; failed: number }> {
  console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  console.log(`📅 DAY ${day}`);
  console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  
  // Get audio URLs for this day
  const audioUrls = await getAudioUrls(day);
  
  if (Object.keys(audioUrls).length === 0) {
    console.log(`\n⚠️  No audio found for day ${day}. Run audio generation first:`);
    console.log(`   npx tsx scripts/daily-generation-engine.ts --days=${day} --audio-only`);
    return { success: 0, failed: PHASES.length };
  }
  
  // Check which phases already have videos
  const { data: existingVideos } = await supabase
    .from('kelly_video_assets')
    .select('phase')
    .eq('lesson_day', day)
    .eq('status', 'validated');
  
  const existingPhases = new Set((existingVideos || []).map(v => v.phase));
  
  let success = 0;
  let failed = 0;
  
  for (const phase of PHASES) {
    if (existingPhases.has(phase)) {
      console.log(`\n⏭️  Phase ${phase}: already exists, skipping`);
      success++;
      continue;
    }
    
    const audioUrl = audioUrls[phase];
    if (!audioUrl) {
      console.log(`\n⚠️  No audio for phase: ${phase}`);
      failed++;
      continue;
    }
    
    const videoUrl = await generateVideo(day, phase, audioUrl);
    
    if (videoUrl) {
      success++;
    } else {
      failed++;
    }
    
    // Small delay between phases
    await new Promise(r => setTimeout(r, 1000));
  }
  
  return { success, failed };
}

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║         🎬 SADTALKER VIDEO GENERATION                        ║
╚══════════════════════════════════════════════════════════════╝
`);

  if (!process.env.FAL_KEY) {
    console.error('❌ FAL_KEY not set in environment');
    process.exit(1);
  }

  let { days } = parseArgs();
  
  // Handle auto mode
  if (days.length === 0) {
    console.log('🔍 Auto mode: finding days without videos...');
    days = await getNextDaysWithoutVideos(3);
    
    if (days.length === 0) {
      console.log('✅ All upcoming days have videos!');
      return;
    }
  }
  
  console.log(`Target Days: ${days.join(', ')}`);
  console.log(`Kelly Image: ${KELLY_IMAGE}`);
  console.log(`Phases: ${PHASES.length}`);

  const results: Record<number, { success: number; failed: number }> = {};
  
  for (const day of days) {
    results[day] = await generateDay(day);
  }

  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                        📊 SUMMARY                            ║
╚══════════════════════════════════════════════════════════════╝
`);

  let totalSuccess = 0;
  let totalFailed = 0;
  
  for (const [day, result] of Object.entries(results)) {
    const icon = result.failed === 0 ? '✅' : result.success > 0 ? '🟡' : '❌';
    console.log(`  ${icon} Day ${day}: ${result.success}/${PHASES.length} videos`);
    totalSuccess += result.success;
    totalFailed += result.failed;
  }

  console.log(`
Total: ${totalSuccess} success, ${totalFailed} failed

✨ Next steps:
   1. Test locally: http://localhost:3000/learn.html?day=${days[0]}
   2. Sync to database: npx tsx scripts/sync-bucket-to-database.ts
   3. Verify ready: npx tsx scripts/verify-day-ready.ts --days=${days.join(',')}
`);
}

main().catch(console.error);



