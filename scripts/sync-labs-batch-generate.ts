#!/usr/bin/env npx tsx
/**
 * 🚀 SYNC LABS BATCH GENERATOR
 * 
 * Bypasses HeyGen queue using Sync Labs + Replicate + ElevenLabs pipeline.
 * Generates Kelly videos for all 12 archetypes using existing motion library images.
 * 
 * Pipeline:
 *   1. Use Kelly motion library images (already uploaded to HeyGen, but we can regenerate)
 *   2. Generate ElevenLabs audio from lesson scripts
 *   3. Create base video with Wav2Lip (Replicate) 
 *   4. Enhance with Sync Labs lipsync-2 (95% accuracy)
 * 
 * Usage:
 *   npx tsx scripts/sync-labs-batch-generate.ts --day 351
 *   npx tsx scripts/sync-labs-batch-generate.ts --day 351 --only explorer,mystic,provider
 *   npx tsx scripts/sync-labs-batch-generate.ts --day 351 --dry-run
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const CONFIG = {
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
};

const ARCHETYPES = [
  'scientist', 'explorer', 'rebel', 'architect',
  'diplomat', 'empath', 'macgyver', 'mystic',
  'provider', 'storyteller', 'strategist', 'survivor'
] as const;

type Archetype = typeof ARCHETYPES[number];

const OUTPUT_DIR = path.join(process.cwd(), 'generated-videos', 'sync-labs-production');

// Archetype-specific image prompts
const ARCHETYPE_PROMPTS: Record<Archetype, string> = {
  scientist: 'kelly, curious scientist expression, analytical thoughtful gaze, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with visible catchlights, modern classroom with scientific posters, cinematic lighting, 4K UHD',
  explorer: 'kelly, adventurous explorer expression, excited curious eyes, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes sparkling with wonder, map and globe in background, cinematic lighting, 4K UHD',
  rebel: 'kelly, confident rebel expression, determined yet warm smile, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with bold energy, modern creative space, cinematic lighting, 4K UHD',
  architect: 'kelly, structured architect expression, focused planning look, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with precision, blueprints in background, cinematic lighting, 4K UHD',
  diplomat: 'kelly, warm diplomatic expression, open welcoming smile, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with empathy, peaceful meeting room, cinematic lighting, 4K UHD',
  empath: 'kelly, deeply caring empath expression, gentle understanding eyes, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes radiating compassion, cozy warm space, cinematic lighting, 4K UHD',
  macgyver: 'kelly, clever macgyver expression, problem-solving twinkle in eyes, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with creativity, workshop tools in background, cinematic lighting, 4K UHD',
  mystic: 'kelly, wise mystic expression, knowing serene smile, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with depth, soft mystical light, cinematic lighting, 4K UHD',
  provider: 'kelly, nurturing provider expression, protective caring look, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with warmth, homey comfortable space, cinematic lighting, 4K UHD',
  storyteller: 'kelly, engaging storyteller expression, animated expressive face, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes twinkling with tales, library books in background, cinematic lighting, 4K UHD',
  strategist: 'kelly, strategic thinker expression, calculating yet friendly, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with sharp intelligence, chess board in background, cinematic lighting, 4K UHD',
  survivor: 'kelly, resilient survivor expression, determined hopeful gaze, wearing soft powder blue crewneck sweater, long wavy chestnut brown hair, warm brown eyes with strength, natural outdoor light, cinematic lighting, 4K UHD',
};

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

interface GenerationResult {
  archetype: Archetype;
  success: boolean;
  imageUrl?: string;
  audioUrl?: string;
  baseVideoUrl?: string;
  premiumVideoUrl?: string;
  duration?: number;
  error?: string;
}

interface BatchManifest {
  day: number;
  generated: string;
  pipeline: 'sync-labs';
  videos: Record<string, GenerationResult>;
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

function combineScripts(lesson: Lesson): string {
  // Combine all phase scripts into one for the full video
  const scripts: string[] = [];
  for (const phaseName of lesson.phaseOrder) {
    const phase = lesson.phases[phaseName];
    if (phase?.script) {
      scripts.push(phase.script);
    }
  }
  return scripts.join(' ');
}

// ═══════════════════════════════════════════════════════════════════
// PIPELINE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

async function generateImage(replicate: Replicate, archetype: Archetype): Promise<string> {
  console.log(`   📸 Generating Kelly image for ${archetype}...`);
  
  const output = await replicate.run(
    "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
    {
      input: {
        prompt: ARCHETYPE_PROMPTS[archetype],
        hf_lora: CONFIG.KELLY_LORA_URL,
        lora_scale: 0.85,
        num_outputs: 1,
        aspect_ratio: "16:9",
        output_format: "png",
        guidance_scale: 3.5,
        output_quality: 100,
        num_inference_steps: 35,
      }
    }
  );
  
  const imageUrl = Array.isArray(output) ? String(output[0]) : String(output);
  console.log(`      ✅ Image ready`);
  return imageUrl;
}

async function generateAudio(text: string, filename: string): Promise<{ localPath: string; buffer: Buffer }> {
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
  return { localPath, buffer };
}

async function uploadAudio(supabase: any, buffer: Buffer, filename: string): Promise<string> {
  const { error } = await supabase.storage
    .from('kelly-templates')
    .upload(`sync-labs-production/${filename}`, buffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });
  
  if (error) {
    throw new Error(`Supabase upload failed: ${error.message}`);
  }
  
  const { data } = supabase.storage
    .from('kelly-templates')
    .getPublicUrl(`sync-labs-production/${filename}`);
  
  return data.publicUrl;
}

async function generateBaseVideo(replicate: Replicate, imageUrl: string, audioBuffer: Buffer): Promise<string> {
  console.log(`   🎬 Generating base video with Wav2Lip...`);
  
  const audioDataUri = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  const output = await replicate.run(
    "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
    {
      input: {
        face: imageUrl,
        audio: audioDataUri,
        fps: 25,
        pads: "0 10 0 0",
        smooth: true,
        resize_factor: 1,
      }
    }
  );
  
  const videoUrl = typeof output === 'string' ? output : String(output);
  console.log(`      ✅ Base video ready`);
  return videoUrl;
}

async function enhanceWithSyncLabs(baseVideoUrl: string, audioUrl: string): Promise<string> {
  console.log(`   🚀 Enhancing with Sync Labs lipsync-2...`);
  
  const response = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'lipsync-2',
      input: [
        { type: 'video', url: baseVideoUrl },
        { type: 'audio', url: audioUrl },
      ],
    }),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Sync Labs error: ${response.status} - ${errorText}`);
  }
  
  const job = await response.json();
  console.log(`      ⏳ Job ${job.id} - polling for completion...`);
  
  // Poll for completion
  for (let i = 0; i < 60; i++) {
    await sleep(5000);
    
    const statusResponse = await fetch(`https://api.sync.so/v2/generate/${job.id}`, {
      headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
    });
    
    const status = await statusResponse.json();
    
    if (status.status === 'COMPLETED') {
      const videoUrl = status.output?.[0]?.url || status.outputUrl;
      console.log(`      ✅ Premium video ready (95% lip-sync)`);
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

async function generateArchetypeVideo(
  replicate: Replicate,
  supabase: any,
  day: number,
  archetype: Archetype,
  lesson: Lesson
): Promise<GenerationResult> {
  console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  console.log(`🎯 ${archetype.toUpperCase()} - Day ${day}`);
  console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  
  try {
    // 1. Generate image
    const imageUrl = await generateImage(replicate, archetype);
    
    // 2. Generate audio
    const script = combineScripts(lesson);
    const audioFilename = `day${day}_${archetype}_${Date.now()}.mp3`;
    const { buffer: audioBuffer } = await generateAudio(script, audioFilename);
    
    // 3. Upload audio to get public URL
    const audioUrl = await uploadAudio(supabase, audioBuffer, audioFilename);
    console.log(`      ☁️ Audio URL: ${audioUrl.substring(0, 50)}...`);
    
    // 4. Generate base video
    const baseVideoUrl = await generateBaseVideo(replicate, imageUrl, audioBuffer);
    
    // 5. Enhance with Sync Labs
    const premiumVideoUrl = await enhanceWithSyncLabs(baseVideoUrl, audioUrl);
    
    console.log(`   🎉 SUCCESS - ${archetype}`);
    
    return {
      archetype,
      success: true,
      imageUrl,
      audioUrl,
      baseVideoUrl,
      premiumVideoUrl,
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

async function runBatch(
  day: number,
  archetypesToRun: Archetype[],
  dryRun: boolean
): Promise<void> {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🚀 SYNC LABS BATCH GENERATOR                                  ║');
  console.log('║  No queues • 95% lip-sync • ~2 min per video                  ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');
  
  // Validate API keys
  const keys = {
    REPLICATE: !!CONFIG.REPLICATE_API_TOKEN,
    ELEVENLABS: !!CONFIG.ELEVENLABS_API_KEY,
    SYNC_LABS: !!CONFIG.SYNC_LABS_API_KEY,
    SUPABASE: !!CONFIG.SUPABASE_URL && !!CONFIG.SUPABASE_KEY,
  };
  
  console.log('🔑 API Keys:');
  Object.entries(keys).forEach(([name, valid]) => {
    console.log(`   ${valid ? '✅' : '❌'} ${name}`);
  });
  
  if (!keys.REPLICATE || !keys.ELEVENLABS || !keys.SYNC_LABS || !keys.SUPABASE) {
    console.error('\n❌ Missing required API keys');
    process.exit(1);
  }
  
  // Load lesson
  const lesson = loadLesson(day);
  console.log(`\n📋 Day ${day}: "${lesson.meta?.topic}"`);
  console.log(`   Phases: ${lesson.phaseOrder?.join(', ')}`);
  console.log(`   Archetypes to generate: ${archetypesToRun.length}`);
  
  if (dryRun) {
    console.log('\n🔍 DRY RUN - Would generate:');
    archetypesToRun.forEach(a => console.log(`   • ${a}`));
    console.log('\nRun without --dry-run to generate.');
    return;
  }
  
  // Setup
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
  
  // Generate videos
  const results: GenerationResult[] = [];
  
  for (const archetype of archetypesToRun) {
    const result = await generateArchetypeVideo(replicate, supabase, day, archetype, lesson);
    results.push(result);
    
    // Small delay between videos
    await sleep(2000);
  }
  
  // Save manifest
  const manifest: BatchManifest = {
    day,
    generated: new Date().toISOString(),
    pipeline: 'sync-labs',
    videos: {},
  };
  
  results.forEach(r => {
    manifest.videos[r.archetype] = r;
  });
  
  const manifestPath = path.join(OUTPUT_DIR, `day-${day}-sync-labs-manifest.json`);
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  // Summary
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log('\n');
  console.log('═'.repeat(64));
  console.log('📊 BATCH COMPLETE');
  console.log('═'.repeat(64));
  console.log(`   ✅ Successful: ${successful.length}`);
  console.log(`   ❌ Failed: ${failed.length}`);
  console.log(`   📁 Manifest: ${manifestPath}`);
  
  if (successful.length > 0) {
    console.log('\n   🎬 Premium Videos (95% lip-sync):');
    successful.forEach(r => {
      console.log(`      ${r.archetype}: ${r.premiumVideoUrl?.substring(0, 60)}...`);
    });
  }
  
  if (failed.length > 0) {
    console.log('\n   ❌ Failed:');
    failed.forEach(r => {
      console.log(`      ${r.archetype}: ${r.error}`);
    });
  }
  
  console.log('═'.repeat(64));
}

// ═══════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════

function parseArgs(): { day: number; dryRun: boolean; only: Archetype[] } {
  const args = process.argv.slice(2);
  
  let day: number | undefined;
  let dryRun = false;
  let only: Archetype[] = [];
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      day = parseInt(args[i + 1]);
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
    console.log('  npx tsx scripts/sync-labs-batch-generate.ts --day 351');
    console.log('  npx tsx scripts/sync-labs-batch-generate.ts --day 351 --dry-run');
    console.log('  npx tsx scripts/sync-labs-batch-generate.ts --day 351 --only explorer,mystic,provider');
    console.log('');
    process.exit(1);
  }
  
  return { day, dryRun, only };
}

async function main(): Promise<void> {
  const { day, dryRun, only } = parseArgs();
  
  const archetypesToRun = only.length > 0 
    ? only.filter(a => ARCHETYPES.includes(a))
    : [...ARCHETYPES];
  
  await runBatch(day, archetypesToRun, dryRun);
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});
