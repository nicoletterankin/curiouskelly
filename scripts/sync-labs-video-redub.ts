#!/usr/bin/env npx tsx
/**
 * 🎬 SYNC LABS VIDEO RE-DUB
 * 
 * Uses EXISTING HeyGen videos (with Kling motion baked in) as the base,
 * then re-dubs with new audio via Sync Labs for future days.
 * 
 * KEY INSIGHT: Sync Labs lipsync-2 can do VIDEO-TO-VIDEO, preserving
 * the natural motion from HeyGen's Kling treatment.
 * 
 * This keeps Kelly consistent across all videos!
 * 
 * Pipeline:
 *   1. Use a completed HeyGen video as motion reference
 *   2. Generate new ElevenLabs audio for the new day's script
 *   3. Apply Sync Labs lipsync-2 to redub the video
 * 
 * Usage:
 *   npx tsx scripts/sync-labs-video-redub.ts --day 352 --reference-day 351
 *   npx tsx scripts/sync-labs-video-redub.ts --day 352 --only scientist,explorer
 *   npx tsx scripts/sync-labs-video-redub.ts --day 352 --dry-run
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const CONFIG = {
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const ARCHETYPES = [
  'scientist', 'explorer', 'rebel', 'architect',
  'diplomat', 'empath', 'macgyver', 'mystic',
  'provider', 'storyteller', 'strategist', 'survivor'
] as const;

type Archetype = typeof ARCHETYPES[number];

const OUTPUT_DIR = path.join(process.cwd(), 'generated-videos', 'sync-labs-redub');

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

interface LessonPhase {
  script: string;
  duration?: number;
}

interface Lesson {
  meta: { day: number; topic: string };
  phases: Record<string, LessonPhase>;
  phaseOrder: string[];
}

interface HeyGenManifestVideo {
  video_id: string;
  status: string;
  video_url?: string;
  phases: string[];
}

interface HeyGenManifest {
  day: number;
  videos: Record<string, HeyGenManifestVideo>;
}

interface RedubResult {
  archetype: Archetype;
  success: boolean;
  referenceVideoUrl?: string;
  newAudioUrl?: string;
  redubVideoUrl?: string;
  error?: string;
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function loadLesson(day: number): Lesson {
  const lessonPath = path.join(process.cwd(), 'public', 'lessons', `day-${day}.json`);
  if (!fs.existsSync(lessonPath)) {
    throw new Error(`Lesson file not found: ${lessonPath}`);
  }
  return JSON.parse(fs.readFileSync(lessonPath, 'utf-8'));
}

function loadHeyGenManifest(day: number): HeyGenManifest | null {
  const manifestPath = path.join(process.cwd(), 'generated-videos', `day-${day}-manifest.json`);
  if (!fs.existsSync(manifestPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
}

function combineScripts(lesson: Lesson): string {
  const scripts: string[] = [];
  for (const phaseName of lesson.phaseOrder) {
    const phase = lesson.phases[phaseName];
    if (phase?.script) {
      scripts.push(phase.script);
    }
  }
  return scripts.join(' ');
}

function getCompletedVideos(manifest: HeyGenManifest): Map<Archetype, string> {
  const completed = new Map<Archetype, string>();
  
  for (const [archetype, video] of Object.entries(manifest.videos)) {
    if (video.status === 'completed' && video.video_url) {
      completed.set(archetype as Archetype, video.video_url);
    }
  }
  
  return completed;
}

// ═══════════════════════════════════════════════════════════════════
// PIPELINE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

async function generateAudio(text: string, filename: string): Promise<{ buffer: Buffer; localPath: string }> {
  console.log(`   🎤 Generating audio (${text.length} chars)...`);
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.85,
          style: 0.2,
          use_speaker_boost: true,
        },
      }),
    }
  );
  
  if (!response.ok) {
    throw new Error(`ElevenLabs error: ${response.status}`);
  }
  
  const buffer = Buffer.from(await response.arrayBuffer());
  const localPath = path.join(OUTPUT_DIR, filename);
  fs.writeFileSync(localPath, buffer);
  
  console.log(`      ✅ Audio saved (${(buffer.length / 1024).toFixed(1)} KB)`);
  return { buffer, localPath };
}

async function uploadAudio(supabase: any, buffer: Buffer, filename: string): Promise<string> {
  const { error } = await supabase.storage
    .from('kelly-templates')
    .upload(`sync-labs-redub/${filename}`, buffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });
  
  if (error) {
    throw new Error(`Supabase upload failed: ${error.message}`);
  }
  
  const { data } = supabase.storage
    .from('kelly-templates')
    .getPublicUrl(`sync-labs-redub/${filename}`);
  
  return data.publicUrl;
}

async function redubWithSyncLabs(videoUrl: string, audioUrl: string): Promise<string> {
  console.log(`   🚀 Re-dubbing with Sync Labs lipsync-2...`);
  console.log(`      📹 Base video (HeyGen motion): ${videoUrl.substring(0, 50)}...`);
  console.log(`      🔊 New audio: ${audioUrl.substring(0, 50)}...`);
  
  const response = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'lipsync-2',
      input: [
        { type: 'video', url: videoUrl },
        { type: 'audio', url: audioUrl },
      ],
    }),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Sync Labs error: ${response.status} - ${errorText}`);
  }
  
  const job = await response.json();
  console.log(`      ⏳ Job ${job.id} - polling...`);
  
  // Poll for completion
  for (let i = 0; i < 60; i++) {
    await sleep(5000);
    
    const statusResponse = await fetch(`https://api.sync.so/v2/generate/${job.id}`, {
      headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
    });
    
    const status = await statusResponse.json();
    
    if (status.status === 'COMPLETED') {
      const videoUrl = status.output?.[0]?.url || status.outputUrl;
      console.log(`      ✅ Re-dub complete!`);
      return videoUrl;
    }
    
    if (status.status === 'FAILED' || status.status === 'REJECTED') {
      throw new Error(`Sync Labs job failed: ${status.error || status.message}`);
    }
    
    if (i % 6 === 0) {
      process.stdout.write('.');
    }
  }
  
  throw new Error('Sync Labs job timed out');
}

// ═══════════════════════════════════════════════════════════════════
// MAIN PIPELINE
// ═══════════════════════════════════════════════════════════════════

async function redubArchetype(
  supabase: any,
  targetDay: number,
  archetype: Archetype,
  referenceVideoUrl: string,
  lesson: Lesson
): Promise<RedubResult> {
  console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  console.log(`🎯 ${archetype.toUpperCase()} - Day ${targetDay} (using HeyGen motion base)`);
  console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  
  try {
    // 1. Generate new audio for target day
    const script = combineScripts(lesson);
    const audioFilename = `day${targetDay}_${archetype}_${Date.now()}.mp3`;
    const { buffer: audioBuffer } = await generateAudio(script, audioFilename);
    
    // 2. Upload audio to get public URL
    const audioUrl = await uploadAudio(supabase, audioBuffer, audioFilename);
    console.log(`      ☁️ Audio URL: ${audioUrl.substring(0, 50)}...`);
    
    // 3. Re-dub the HeyGen video with new audio
    const redubVideoUrl = await redubWithSyncLabs(referenceVideoUrl, audioUrl);
    
    console.log(`   🎉 SUCCESS - ${archetype}`);
    
    return {
      archetype,
      success: true,
      referenceVideoUrl,
      newAudioUrl: audioUrl,
      redubVideoUrl,
    };
    
  } catch (error: any) {
    console.log(`   ❌ FAILED - ${archetype}: ${error.message}`);
    return {
      archetype,
      success: false,
      error: error.message,
    };
  }
}

async function runRedubBatch(
  targetDay: number,
  referenceDay: number,
  archetypesToRun: Archetype[],
  dryRun: boolean
): Promise<void> {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 SYNC LABS VIDEO RE-DUB                                     ║');
  console.log('║  Uses HeyGen motion base • Re-dubs with new audio             ║');
  console.log('║  Kelly stays CONSISTENT across all videos!                    ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');
  
  // Validate API keys
  const keys = {
    ELEVENLABS: !!CONFIG.ELEVENLABS_API_KEY,
    SYNC_LABS: !!CONFIG.SYNC_LABS_API_KEY,
    SUPABASE: !!CONFIG.SUPABASE_URL && !!CONFIG.SUPABASE_KEY,
  };
  
  console.log('🔑 API Keys:');
  Object.entries(keys).forEach(([name, valid]) => {
    console.log(`   ${valid ? '✅' : '❌'} ${name}`);
  });
  
  if (!keys.ELEVENLABS || !keys.SYNC_LABS || !keys.SUPABASE) {
    console.error('\n❌ Missing required API keys');
    process.exit(1);
  }
  
  // Load reference manifest (completed HeyGen videos)
  const referenceManifest = loadHeyGenManifest(referenceDay);
  if (!referenceManifest) {
    console.error(`\n❌ No HeyGen manifest found for reference day ${referenceDay}`);
    console.log(`   Expected: generated-videos/day-${referenceDay}-manifest.json`);
    process.exit(1);
  }
  
  const completedVideos = getCompletedVideos(referenceManifest);
  console.log(`\n📹 Reference Day ${referenceDay}: ${completedVideos.size} completed videos`);
  
  // Check which archetypes we can process
  const processable: { archetype: Archetype; videoUrl: string }[] = [];
  const missing: Archetype[] = [];
  
  for (const archetype of archetypesToRun) {
    const videoUrl = completedVideos.get(archetype);
    if (videoUrl) {
      processable.push({ archetype, videoUrl });
    } else {
      missing.push(archetype);
    }
  }
  
  console.log(`   ✅ Can process: ${processable.map(p => p.archetype).join(', ')}`);
  if (missing.length > 0) {
    console.log(`   ⚠️ Missing reference videos: ${missing.join(', ')}`);
  }
  
  // Load target lesson
  const targetLesson = loadLesson(targetDay);
  console.log(`\n📋 Target Day ${targetDay}: "${targetLesson.meta?.topic}"`);
  console.log(`   Phases: ${targetLesson.phaseOrder?.join(', ')}`);
  
  if (dryRun) {
    console.log('\n🔍 DRY RUN - Would redub:');
    processable.forEach(p => console.log(`   • ${p.archetype}`));
    console.log('\nRun without --dry-run to generate.');
    return;
  }
  
  if (processable.length === 0) {
    console.error('\n❌ No archetypes available to process');
    process.exit(1);
  }
  
  // Setup
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
  
  // Process videos
  const results: RedubResult[] = [];
  
  for (const { archetype, videoUrl } of processable) {
    const result = await redubArchetype(supabase, targetDay, archetype, videoUrl, targetLesson);
    results.push(result);
    await sleep(2000);
  }
  
  // Save manifest
  const manifest = {
    targetDay,
    referenceDay,
    generated: new Date().toISOString(),
    pipeline: 'sync-labs-redub',
    videos: Object.fromEntries(results.map(r => [r.archetype, r])),
  };
  
  const manifestPath = path.join(OUTPUT_DIR, `day-${targetDay}-redub-manifest.json`);
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  // Summary
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log('\n');
  console.log('═'.repeat(64));
  console.log('📊 RE-DUB COMPLETE');
  console.log('═'.repeat(64));
  console.log(`   ✅ Successful: ${successful.length}`);
  console.log(`   ❌ Failed: ${failed.length}`);
  console.log(`   📁 Manifest: ${manifestPath}`);
  
  if (successful.length > 0) {
    console.log('\n   🎬 Re-dubbed Videos (HeyGen motion + new audio):');
    successful.forEach(r => {
      console.log(`      ${r.archetype}: ${r.redubVideoUrl?.substring(0, 60)}...`);
    });
  }
  
  if (missing.length > 0) {
    console.log('\n   ⚠️ Skipped (no reference video):');
    missing.forEach(a => console.log(`      ${a}`));
  }
  
  console.log('═'.repeat(64));
}

// ═══════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════

function parseArgs(): { day: number; referenceDay: number; dryRun: boolean; only: Archetype[] } {
  const args = process.argv.slice(2);
  
  let day: number | undefined;
  let referenceDay = 351; // Default to Day 351
  let dryRun = false;
  let only: Archetype[] = [];
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      day = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--reference-day' && args[i + 1]) {
      referenceDay = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--dry-run') {
      dryRun = true;
    } else if (args[i] === '--only' && args[i + 1]) {
      only = args[i + 1].split(',').map(s => s.trim() as Archetype);
      i++;
    }
  }
  
  if (!day) {
    console.log('');
    console.log('Usage:');
    console.log('  npx tsx scripts/sync-labs-video-redub.ts --day 352');
    console.log('  npx tsx scripts/sync-labs-video-redub.ts --day 352 --reference-day 351');
    console.log('  npx tsx scripts/sync-labs-video-redub.ts --day 352 --only scientist,explorer');
    console.log('  npx tsx scripts/sync-labs-video-redub.ts --day 352 --dry-run');
    console.log('');
    console.log('The --reference-day flag specifies which completed HeyGen videos to use');
    console.log('as the motion base. Default is 351.');
    console.log('');
    process.exit(1);
  }
  
  return { day, referenceDay, dryRun, only };
}

async function main(): Promise<void> {
  const { day, referenceDay, dryRun, only } = parseArgs();
  
  const archetypesToRun = only.length > 0 
    ? only.filter(a => ARCHETYPES.includes(a))
    : [...ARCHETYPES];
  
  await runRedubBatch(day, referenceDay, archetypesToRun, dryRun);
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});
