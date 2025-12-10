#!/usr/bin/env npx tsx
/**
 * 🎬 DATABASE-DRIVEN KELLY VIDEO PIPELINE
 * 
 * This pipeline fetches prompts from Supabase, logs all generation runs,
 * and enables transparency for Learner Commons.
 * 
 * KEY DIFFERENCES FROM HARDCODED PIPELINE:
 * 1. All prompts come from kelly_prompts table (editable without code changes)
 * 2. Generation runs are logged in kelly_generation_runs (audit trail)
 * 3. Assets are tracked in kelly_generated_assets (reuse capability)
 * 4. Learner Commons can display "how this video was made"
 * 
 * USAGE:
 *   npx tsx db-prompt-pipeline.ts --day 1 --archetype "The Explorer" --phase Hook
 *   npx tsx db-prompt-pipeline.ts --day 1 --dry-run
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // API Keys
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  ELEVENLABS_KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  
  // Supabase
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  
  // Storage
  STORAGE_BUCKET: 'kelly-videos',
  
  // Kelly LoRA
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.90,
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'golden-lesson-hd'),
  
  // Archetypes and phases
  ARCHETYPES: ['The Explorer', 'The Rebel', 'The Scientist'] as const,
  PHASES: ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'] as const,
};

type Archetype = typeof CONFIG.ARCHETYPES[number];
type Phase = typeof CONFIG.PHASES[number];

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
// PROMPT FETCHING FROM DATABASE
// =============================================================================

interface KellyPrompts {
  identity: string;
  expression: string;
  gesture: string;
  motion: string;
  voice: { text: string; settings: { stability: number; similarity_boost: number; style: number; speed: number } };
  background: string;
  style: string;
  negative: string;
}

interface ComposedPrompts {
  imagePrompt: string;
  motionPrompt: string;
  negativePrompt: string;
  voiceSettings: { stability: number; similarity_boost: number; style: number; speed: number };
  rawPrompts: KellyPrompts;
}

async function fetchPrompts(archetype: Archetype, phase: Phase): Promise<ComposedPrompts> {
  // Get raw prompts for logging/transparency
  const { data: rawPrompts, error: rawError } = await getSupabase().rpc('get_kelly_prompts', {
    p_archetype: archetype,
    p_phase: phase
  });
  
  if (rawError) {
    throw new Error(`Failed to fetch prompts: ${rawError.message}`);
  }
  
  // Get composed image prompt (combines identity + expression + gesture + background + style)
  const { data: composedData, error: composeError } = await getSupabase().rpc('compose_kelly_image_prompt', {
    p_archetype: archetype,
    p_phase: phase
  });
  
  if (composeError) {
    throw new Error(`Failed to compose prompt: ${composeError.message}`);
  }
  
  const prompts = rawPrompts as KellyPrompts;
  
  console.log(`  📚 Fetched v2 prompts from database for ${archetype}/${phase}`);
  console.log(`  🎭 Archetype energy: ${prompts.voice?.text?.substring(0, 50)}...`);
  
  return {
    imagePrompt: composedData as string,
    motionPrompt: prompts.motion || '',
    negativePrompt: prompts.negative || '',
    voiceSettings: prompts.voice?.settings || { stability: 0.5, similarity_boost: 0.85, style: 0.2, speed: 1.0 },
    rawPrompts: prompts
  };
}

// Legacy function for backward compatibility
function composeImagePrompt(prompts: KellyPrompts): string {
  return [
    prompts.identity,
    prompts.expression,
    prompts.gesture,
    prompts.background,
    prompts.style
  ].filter(Boolean).join(', ');
}

function hashPrompts(prompts: KellyPrompts): string {
  const content = JSON.stringify(prompts);
  return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
}

// =============================================================================
// GENERATION RUN LOGGING
// =============================================================================

interface GenerationRun {
  id: string;
  day_number: number;
  archetype: string;
  phase: string;
  status: string;
  prompts_snapshot: KellyPrompts;
}

async function createGenerationRun(dayNumber: number, archetype: Archetype, phase: Phase, prompts: KellyPrompts): Promise<string> {
  const { data, error } = await getSupabase()
    .from('kelly_generation_runs')
    .insert({
      day_number: dayNumber,
      archetype,
      phase,
      status: 'pending',
      prompts_snapshot: prompts
    })
    .select('id')
    .single();
  
  if (error) {
    throw new Error(`Failed to create generation run: ${error.message}`);
  }
  
  console.log(`  📝 Created generation run: ${data.id}`);
  return data.id;
}

async function updateGenerationStatus(runId: string, status: string, updates: Record<string, any> = {}): Promise<void> {
  const { error } = await getSupabase()
    .from('kelly_generation_runs')
    .update({ 
      status, 
      ...updates,
      ...(status === 'completed' || status === 'failed' ? { completed_at: new Date().toISOString() } : {})
    })
    .eq('id', runId);
  
  if (error) {
    console.error(`  ⚠️ Failed to update run status: ${error.message}`);
  }
}

// =============================================================================
// ASSET MANAGEMENT
// =============================================================================

async function saveAsset(
  assetType: 'source_image' | 'motion_video' | 'lipsync_video' | 'audio',
  promptHash: string,
  prompts: KellyPrompts,
  archetype: Archetype,
  phase: Phase,
  dayNumber: number | null,
  fileUrl: string,
  metadata: Record<string, any>
): Promise<string> {
  const { data, error } = await getSupabase()
    .from('kelly_generated_assets')
    .upsert({
      asset_type: assetType,
      prompt_hash: promptHash,
      prompts_used: prompts,
      archetype,
      phase,
      day_number: dayNumber,
      file_url: fileUrl,
      file_size_bytes: metadata.size,
      duration_seconds: metadata.duration,
      generation_cost_usd: metadata.cost,
      generation_time_ms: metadata.time_ms,
      model_used: metadata.model,
      model_version: metadata.version,
      generation_params: metadata.params
    }, {
      onConflict: 'prompt_hash,asset_type,day_number'
    })
    .select('id')
    .single();
  
  if (error) {
    console.error(`  ⚠️ Failed to save asset: ${error.message}`);
    return '';
  }
  
  return data.id;
}

async function findReusableAsset(
  assetType: 'source_image' | 'motion_video',
  promptHash: string
): Promise<{ id: string; file_url: string } | null> {
  const { data, error } = await getSupabase()
    .from('kelly_generated_assets')
    .select('id, file_url')
    .eq('asset_type', assetType)
    .eq('prompt_hash', promptHash)
    .is('day_number', null) // Only reusable assets (not day-specific)
    .single();
  
  if (error || !data) return null;
  
  // Update reuse count
  await getSupabase()
    .from('kelly_generated_assets')
    .update({ 
      reuse_count: getSupabase().rpc('increment', { x: 1 }),
      last_reused_at: new Date().toISOString()
    })
    .eq('id', data.id);
  
  console.log(`  ♻️ Reusing existing ${assetType}: ${data.id}`);
  return data;
}

// =============================================================================
// STEP 1: AUDIO GENERATION (ElevenLabs)
// =============================================================================

async function generateAudio(
  scriptText: string, 
  prompts: ComposedPrompts,
  dayNumber: number,
  archetype: Archetype,
  phase: Phase
): Promise<{ url: string; localPath: string; duration: number }> {
  console.log('  🎤 Generating audio (ElevenLabs)...');
  console.log(`  🎙️ Voice settings: stability=${prompts.voiceSettings.stability}, style=${prompts.voiceSettings.style}, speed=${prompts.voiceSettings.speed}`);
  
  const voiceSettings = prompts.voiceSettings;
  
  const startTime = Date.now();
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.ELEVENLABS_KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text: scriptText,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: voiceSettings.stability,
          similarity_boost: voiceSettings.similarity_boost,
          style: voiceSettings.style,
          use_speaker_boost: true
        }
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`ElevenLabs API error: ${response.status} - ${await response.text()}`);
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const duration = scriptText.split(' ').length * 0.3; // Estimate
  
  // Save locally
  const outputDir = path.join(CONFIG.OUTPUT_DIR, `day_${String(dayNumber).padStart(3, '0')}_${phase}_${archetype.replace(/\s+/g, '_')}`);
  fs.mkdirSync(outputDir, { recursive: true });
  const localPath = path.join(outputDir, 'audio.mp3');
  fs.writeFileSync(localPath, audioBuffer);
  
  // Upload to Supabase - use same path format as videos for consistency
  const archetypeSlug = archetype.replace('The ', '').toLowerCase();
  const storagePath = `day-${String(dayNumber).padStart(3, '0')}/${archetypeSlug}/${phase.toLowerCase()}_audio.mp3`;
  
  console.log(`  ☁️ Uploading audio to: ${storagePath}`);
  const { error: uploadError } = await getSupabase().storage.from(CONFIG.STORAGE_BUCKET).upload(storagePath, audioBuffer, {
    contentType: 'audio/mpeg',
    upsert: true
  });
  
  if (uploadError) {
    console.error(`  ⚠️ Audio upload failed: ${uploadError.message}`);
    throw new Error(`Audio upload failed: ${uploadError.message}`);
  }
  
  const { data: urlData } = getSupabase().storage.from(CONFIG.STORAGE_BUCKET).getPublicUrl(storagePath);
  console.log(`  ✅ Audio uploaded: ${urlData.publicUrl}`);
  
  const timeMs = Date.now() - startTime;
  console.log(`  ✅ Audio: ${(audioBuffer.length / 1024).toFixed(1)} KB, ~${duration.toFixed(1)}s (${timeMs}ms)`);
  
  return { url: urlData.publicUrl, localPath, duration };
}

// =============================================================================
// STEP 2: IMAGE GENERATION (Flux + Kelly LoRA)
// =============================================================================

async function generateImage(
  prompts: ComposedPrompts,
  promptHash: string,
  archetype: Archetype,
  phase: Phase,
  dayNumber: number
): Promise<{ url: string; localPath: string }> {
  console.log('  🎨 Generating image (Flux + LoRA)...');
  
  // Use the pre-composed image prompt from Supabase
  const fullPrompt = prompts.imagePrompt;
  console.log(`  📝 Prompt: "${fullPrompt.substring(0, 100)}..."`);
  
  const startTime = Date.now();
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  const output = await replicate.run(
    "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
    {
      input: {
        prompt: fullPrompt,
        hf_lora: CONFIG.KELLY_LORA_URL,
        lora_scale: CONFIG.LORA_SCALE,
        num_outputs: 1,
        aspect_ratio: "16:9",
        output_format: "png",
        guidance_scale: 3.5,
        output_quality: 100,
        num_inference_steps: 35,
        disable_safety_checker: true,
        negative_prompt: prompts.negativePrompt,
      }
    }
  );

  const imageUrl = Array.isArray(output) ? String(output[0]) : String(output);
  
  // Download and save locally
  const outputDir = path.join(CONFIG.OUTPUT_DIR, `day_${String(dayNumber).padStart(3, '0')}_${phase}_${archetype.replace(/\s+/g, '_')}`);
  fs.mkdirSync(outputDir, { recursive: true });
  const localPath = path.join(outputDir, 'source_image.png');
  
  const imageResponse = await fetch(imageUrl);
  const imageBuffer = Buffer.from(await imageResponse.arrayBuffer());
  fs.writeFileSync(localPath, imageBuffer);
  
  const timeMs = Date.now() - startTime;
  console.log(`  ✅ Image: ${(imageBuffer.length / 1024 / 1024).toFixed(2)} MB (${timeMs}ms)`);
  
  // Save asset record
  await saveAsset('source_image', promptHash, prompts.rawPrompts, archetype, phase, null, imageUrl, {
    size: imageBuffer.length,
    time_ms: timeMs,
    model: 'flux-dev-lora',
    params: { lora_scale: CONFIG.LORA_SCALE, guidance_scale: 3.5 }
  });
  
  return { url: imageUrl, localPath };
}

// =============================================================================
// STEP 3: MOTION VIDEO (MiniMax Video-01)
// =============================================================================

async function generateMotionVideo(
  imageUrl: string,
  prompts: ComposedPrompts,
  promptHash: string,
  archetype: Archetype,
  phase: Phase,
  dayNumber: number
): Promise<{ url: string; localPath: string }> {
  console.log('  🎬 Generating motion (MiniMax Video-01)...');
  console.log(`  📝 Motion: "${prompts.motionPrompt.substring(0, 60)}..."`);
  
  const startTime = Date.now();
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  const prediction = await replicate.predictions.create({
    version: '5aa835260ff7f40f4069c41185f72036accf99e29957bb4a3b3a911f3b6c1912',
    input: {
      prompt: `${prompts.motionPrompt}. Natural fluid movement, warm genuine energy.`,
      first_frame_image: imageUrl,
      prompt_optimizer: true,
    },
  });

  // Poll for completion
  let attempts = 0;
  const maxAttempts = 120;
  
  while (attempts < maxAttempts) {
    const status = await replicate.predictions.get(prediction.id);
    
    if (status.status === 'succeeded') {
      const videoUrl = Array.isArray(status.output) ? status.output[0] : status.output;
      
      // Download and save locally
      const outputDir = path.join(CONFIG.OUTPUT_DIR, `day_${String(dayNumber).padStart(3, '0')}_${phase}_${archetype.replace(/\s+/g, '_')}`);
      const localPath = path.join(outputDir, 'motion_video.mp4');
      
      const videoResponse = await fetch(String(videoUrl));
      const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
      fs.writeFileSync(localPath, videoBuffer);
      
      const timeMs = Date.now() - startTime;
      console.log(`  ✅ Motion: ${(videoBuffer.length / 1024).toFixed(0)} KB (${(timeMs/1000).toFixed(0)}s)`);
      
      // Save asset record
      await saveAsset('motion_video', promptHash, prompts.rawPrompts, archetype, phase, null, String(videoUrl), {
        size: videoBuffer.length,
        time_ms: timeMs,
        model: 'minimax/video-01'
      });
      
      return { url: String(videoUrl), localPath };
    }
    
    if (status.status === 'failed') {
      throw new Error(`MiniMax failed: ${status.error}`);
    }
    
    process.stdout.write(`\r  ⏳ MiniMax: ${status.status} (${attempts * 5}s)`);
    await new Promise(r => setTimeout(r, 5000));
    attempts++;
  }
  
  throw new Error('MiniMax timed out');
}

// =============================================================================
// STEP 4: LIP-SYNC (Sync Labs lipsync-2)
// =============================================================================

async function applyLipSync(
  motionVideoUrl: string,
  audioUrl: string,
  prompts: ComposedPrompts,
  archetype: Archetype,
  phase: Phase,
  dayNumber: number
): Promise<{ url: string; localPath: string }> {
  console.log('  👄 Applying lip-sync (Sync Labs lipsync-2)...');
  
  const startTime = Date.now();
  
  const response = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'lipsync-2',
      input: [
        { type: 'video', url: motionVideoUrl },
        { type: 'audio', url: audioUrl }
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`Sync Labs API error: ${response.status}`);
  }

  const job = await response.json();
  console.log(`  ⏳ Sync Labs job: ${job.id}`);
  
  // Poll for completion
  let attempts = 0;
  const maxAttempts = 120;
  
  while (attempts < maxAttempts) {
    const statusRes = await fetch(`https://api.sync.so/v2/generate/${job.id}`, {
      headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY }
    });
    const status = await statusRes.json();
    
    if (status.status === 'COMPLETED') {
      const outputUrl = status.output?.[0]?.url || status.outputUrl || status.output;
      
      // Download and save locally
      const outputDir = path.join(CONFIG.OUTPUT_DIR, `day_${String(dayNumber).padStart(3, '0')}_${phase}_${archetype.replace(/\s+/g, '_')}`);
      const localPath = path.join(outputDir, 'final_hd.mp4');
      
      const videoResponse = await fetch(outputUrl);
      const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
      fs.writeFileSync(localPath, videoBuffer);
      
      const timeMs = Date.now() - startTime;
      console.log(`\n  ✅ Lip-sync: ${(videoBuffer.length / 1024 / 1024).toFixed(2)} MB (${(timeMs/1000).toFixed(0)}s)`);
      
      // Save asset record (day-specific, not reusable)
      await saveAsset('lipsync_video', '', prompts.rawPrompts, archetype, phase, dayNumber, outputUrl, {
        size: videoBuffer.length,
        time_ms: timeMs,
        model: 'sync-labs/lipsync-2'
      });
      
      return { url: outputUrl, localPath };
    }
    
    if (status.status === 'FAILED' || status.status === 'REJECTED') {
      throw new Error(`Sync Labs failed: ${status.error || status.message}`);
    }
    
    process.stdout.write(`\r  ⏳ Sync Labs: ${status.status} (${attempts * 5}s)`);
    await new Promise(r => setTimeout(r, 5000));
    attempts++;
  }
  
  throw new Error('Sync Labs timed out');
}

// =============================================================================
// STEP 5: UPLOAD & UPDATE DATABASE
// =============================================================================

async function uploadAndUpdateDatabase(
  localPath: string,
  dayNumber: number,
  archetype: Archetype,
  phase: Phase
): Promise<string> {
  console.log('  ☁️ Uploading final video to Supabase...');
  
  const fileBuffer = fs.readFileSync(localPath);
  // Use consistent path format: day-001/explorer/hook.mp4
  const archetypeSlug = archetype.replace('The ', '').toLowerCase();
  const storagePath = `day-${String(dayNumber).padStart(3, '0')}/${archetypeSlug}/${phase.toLowerCase()}.mp4`;
  
  console.log(`  📁 Storage path: ${storagePath}`);
  const { error: uploadError } = await getSupabase().storage.from(CONFIG.STORAGE_BUCKET).upload(storagePath, fileBuffer, {
    contentType: 'video/mp4',
    upsert: true
  });
  
  if (uploadError) {
    console.error(`  ⚠️ Video upload failed: ${uploadError.message}`);
    throw new Error(`Video upload failed: ${uploadError.message}`);
  }
  
  const { data: urlData } = getSupabase().storage.from(CONFIG.STORAGE_BUCKET).getPublicUrl(storagePath);
  
  // Update lesson_atoms
  const { data: lesson } = await getSupabase()
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();
  
  if (lesson) {
    await getSupabase()
      .from('lesson_atoms')
      .update({ hd_video_url: urlData.publicUrl })
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', archetype)
      .eq('phase', phase);
  }
  
  console.log(`  ✅ Uploaded: ${urlData.publicUrl}`);
  return urlData.publicUrl;
}

// =============================================================================
// MAIN PIPELINE
// =============================================================================

async function generateGoldenVideo(
  dayNumber: number,
  archetype: Archetype,
  phase: Phase,
  options: { dryRun?: boolean; force?: boolean } = {}
): Promise<void> {
  const startTime = Date.now();
  
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`📅 Day ${dayNumber} | ${archetype} | ${phase}`);
  console.log('═'.repeat(70));
  
  // Step 0: Fetch prompts from database
  const prompts = await fetchPrompts(archetype, phase);
  const promptHash = hashPrompts(prompts.rawPrompts);
  console.log(`  🔑 Prompt hash: ${promptHash}`);
  
  if (options.dryRun) {
    console.log('\n  🔍 [DRY RUN] Would generate with these prompts:');
    console.log(`  📸 Image: "${prompts.imagePrompt.substring(0, 80)}..."`);
    console.log(`  🎬 Motion: "${prompts.motionPrompt.substring(0, 60)}..."`);
    console.log(`  🎙️ Voice: ${JSON.stringify(prompts.voiceSettings)}`);
    console.log(`  🚫 Negative: "${prompts.negativePrompt.substring(0, 50)}..."`);
    return;
  }
  
  // Create generation run for audit
  const runId = await createGenerationRun(dayNumber, archetype, phase, prompts.rawPrompts);
  
  try {
    // Get script from lesson_atoms
    const { data: lesson } = await getSupabase()
      .from('core_lessons')
      .select('id')
      .eq('day_number', dayNumber)
      .single();
    
    if (!lesson) throw new Error(`No lesson found for day ${dayNumber}`);
    
    const { data: atom } = await getSupabase()
      .from('lesson_atoms')
      .select('content')
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', archetype)
      .eq('phase', phase)
      .single();
    
    if (!atom?.content?.script) throw new Error(`No script found for ${archetype}/${phase}`);
    
    const scriptText = atom.content.script;
    console.log(`  📜 Script: "${scriptText.substring(0, 50)}..."`);
    
    // Step 1: Audio (always unique per day)
    await updateGenerationStatus(runId, 'generating_audio');
    const audio = await generateAudio(scriptText, prompts, dayNumber, archetype, phase);
    
    // Step 2: Image
    await updateGenerationStatus(runId, 'generating_image');
    const image = await generateImage(prompts, promptHash, archetype, phase, dayNumber);
    
    // Step 3: Motion
    await updateGenerationStatus(runId, 'generating_motion');
    const motion = await generateMotionVideo(image.url, prompts, promptHash, archetype, phase, dayNumber);
    
    // Step 4: Lip-sync
    await updateGenerationStatus(runId, 'generating_lipsync');
    const lipsync = await applyLipSync(motion.url, audio.url, prompts, archetype, phase, dayNumber);
    
    // Step 5: Upload
    await updateGenerationStatus(runId, 'uploading');
    const finalUrl = await uploadAndUpdateDatabase(lipsync.localPath, dayNumber, archetype, phase);
    
    // Complete
    const totalTime = Date.now() - startTime;
    await updateGenerationStatus(runId, 'completed', {
      total_time_ms: totalTime,
      total_cost_usd: 0.15 // Estimate
    });
    
    console.log(`\n  ✅ COMPLETE in ${(totalTime/1000).toFixed(0)}s`);
    console.log(`  📁 ${finalUrl}`);
    
  } catch (error: any) {
    await updateGenerationStatus(runId, 'failed', { error_message: error.message });
    console.error(`\n  ❌ FAILED: ${error.message}`);
    throw error;
  }
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  
  let dayNumber: number | undefined;
  let archetype: Archetype | undefined;
  let phase: Phase | undefined;
  let dryRun = false;
  let force = false;
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--day': dayNumber = parseInt(args[++i]); break;
      case '--archetype': archetype = args[++i] as Archetype; break;
      case '--phase': phase = args[++i] as Phase; break;
      case '--dry-run': dryRun = true; break;
      case '--force': force = true; break;
      case '--help':
        console.log(`
🎬 Database-Driven Kelly Video Pipeline

USAGE:
  npx tsx db-prompt-pipeline.ts --day 1 --archetype "The Explorer" --phase Hook
  npx tsx db-prompt-pipeline.ts --day 1  # All 15 videos
  npx tsx db-prompt-pipeline.ts --day 1 --dry-run  # Preview prompts

OPTIONS:
  --day <number>        Day number (1-365)
  --archetype <name>    "The Explorer", "The Rebel", or "The Scientist"
  --phase <name>        "Hook", "Fact1", "Fact2", "Fact3", or "Wisdom"
  --dry-run             Preview prompts without generating
  --force               Regenerate even if exists
        `);
        process.exit(0);
    }
  }
  
  if (!dayNumber) {
    console.error('❌ --day is required');
    process.exit(1);
  }
  
  console.log('\n🔑 Checking API Keys...');
  console.log(`   REPLICATE: ${CONFIG.REPLICATE_API_TOKEN ? '✅' : '❌'}`);
  console.log(`   ELEVENLABS: ${CONFIG.ELEVENLABS_API_KEY ? '✅' : '❌'}`);
  console.log(`   SYNC_LABS: ${CONFIG.SYNC_LABS_API_KEY ? '✅' : '❌'}`);
  console.log(`   SUPABASE: ${CONFIG.SUPABASE_URL ? '✅' : '❌'}`);
  
  const archetypes = archetype ? [archetype] : CONFIG.ARCHETYPES;
  const phases = phase ? [phase] : CONFIG.PHASES;
  
  for (const a of archetypes) {
    for (const p of phases) {
      await generateGoldenVideo(dayNumber, a as Archetype, p as Phase, { dryRun, force });
    }
  }
  
  console.log('\n✅ All done!');
}

main().catch(console.error);

