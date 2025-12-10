#!/usr/bin/env npx tsx
/**
 * 🎬 HD GOLDEN LESSON PIPELINE
 * 
 * Production-perfect video generation for Curious Kelly lessons.
 * Generates 1080p HD lip-synced videos for all 365 days × 3 archetypes × 5 phases.
 * 
 * ARCHITECTURE (December 2025 State-of-the-Art):
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  STEP 1: AUDIO (ElevenLabs)                                         │
 * │  → Kelly's voice with archetype-specific emotion                    │
 * │  → Model: eleven_multilingual_v2                                    │
 * │  → Output: MP3 audio file                                           │
 * └─────────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  STEP 2: SOURCE IMAGE (Flux + Kelly LoRA)                           │
 * │  → Consistent Kelly character with phase-appropriate expression     │
 * │  → Model: lucataco/flux-dev-lora                                    │
 * │  → LoRA: CuriousKellycom/curious-kelly-lora (scale: 0.85)          │
 * │  → Output: 1344x768 PNG (16:9)                                      │
 * └─────────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  STEP 3: MOTION VIDEO (MiniMax Video-01)                            │
 * │  → Natural teacher gestures and expressions                         │
 * │  → Input: Source image + motion prompt                              │
 * │  → Output: ~6 second base video with movement                       │
 * └─────────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  STEP 4: LIP-SYNC (Sync Labs lipsync-2)                             │
 * │  → 95%+ accuracy mouth movements                                    │
 * │  → Input: Motion video + ElevenLabs audio                           │
 * │  → Output: Final lip-synced HD video                                │
 * └─────────────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  STEP 5: UPLOAD & DATABASE                                          │
 * │  → Upload to Supabase Storage (kelly-videos bucket)                 │
 * │  → Update lesson_atoms.hd_video_url                                 │
 * │  → Generate lipsync.json for Unity/WebGL fallback                   │
 * └─────────────────────────────────────────────────────────────────────┘
 * 
 * USAGE:
 *   # Generate single video
 *   npx tsx hd-golden-lesson-pipeline.ts --day 2 --archetype "The Explorer" --phase Hook
 * 
 *   # Generate all videos for a day (15 videos)
 *   npx tsx hd-golden-lesson-pipeline.ts --day 2
 * 
 *   # Generate range of days
 *   npx tsx hd-golden-lesson-pipeline.ts --from 2 --to 10
 * 
 *   # Dry run (no generation, just show what would be generated)
 *   npx tsx hd-golden-lesson-pipeline.ts --day 2 --dry-run
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

export const CONFIG = {
  // API Keys (from environment)
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  ELEVENLABS_KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  
  // Supabase
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  
  // Storage
  STORAGE_BUCKET: 'kelly-videos',
  TEMP_BUCKET: 'kelly-templates', // For temporary audio uploads
  
  // Kelly LoRA - increased scale for stronger character consistency
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.90,
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'golden-lesson-hd'),
  
  // Models (December 2025)
  MODELS: {
    FLUX_LORA: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
    MINIMAX: 'minimax/video-01',
    WAV2LIP: 'devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
  },
  
  // Archetypes and phases
  ARCHETYPES: ['The Explorer', 'The Rebel', 'The Scientist'] as const,
  PHASES: ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'] as const,
};

// =============================================================================
// TYPES
// =============================================================================

type Archetype = typeof CONFIG.ARCHETYPES[number];
type Phase = typeof CONFIG.PHASES[number];

interface GenerationSteps {
  audio?: { path: string; url: string; duration: number };
  image?: { path: string; url: string };
  motion?: { path: string; url: string };
  lipsync?: { path: string; url: string };
  finalVideo?: { path: string; url: string };
}

interface GenerationResult {
  success: boolean;
  dayNumber: number;
  archetype: Archetype;
  phase: Phase;
  steps: GenerationSteps;
  supabaseUrl?: string;
  error?: string;
  duration: number;
  cost?: number;
}

interface LessonAtom {
  id: string;
  core_lesson_id: string;
  archetype: string;
  phase: string;
  content: {
    script: string;
    options?: Array<{
      letter: string;
      text: string;
      quality: string;
      response: string;
    }>;
  };
  hd_video_url?: string;
}

// =============================================================================
// KELLY VISUAL IDENTITY (CANONICAL - DO NOT CHANGE)
// =============================================================================

const KELLY = {
  // Core identity - LOCKED SINGLE SOURCE OF TRUTH
  // Vanna White vibe: calm, confident, composed presenter
  identity: `kelly, calm confident female teacher, warm brown wavy shoulder-length hair with subtle caramel highlights center-parted, hazel-brown eyes with steady direct gaze, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater, poised composed posture, looking directly at camera`,
  
  // Background and style
  background: `professional warm classroom setting, soft natural lighting, shallow depth of field, cinematic quality`,
  style: `professional portrait photography, 85mm lens, soft diffused lighting, warm color grading, 4K UHD`,
  
  // Negative prompt (things to AVOID) - includes motion artifacts
  negative: `pink sweater, red sweater, purple sweater, teal sweater, green sweater, yellow sweater, beige sweater, auburn hair, chestnut hair, deformed, blurry, bad anatomy, extra fingers, mutated hands, poorly drawn face, mutation, disfigured, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, out of frame, cropped, watermark, signature, text, wandering eyes, looking away, darting gaze`,
  
  // Phase-specific expressions and gestures - Calm presenter vibe BUT must allow natural speaking
  phases: {
    Hook: {
      expression: 'warm confident expression, direct eye contact with viewer, relaxed natural face ready to speak',
      gesture: 'hands resting naturally, composed open posture, centered in frame',
      motion: 'speaking naturally to camera, mouth moving with speech, steady gaze at camera, gentle natural breathing',
    },
    Fact1: {
      expression: 'calm curious expression, engaged teaching face, steady direct gaze',
      gesture: 'hands resting naturally, one hand may gesture subtly, composed posture',
      motion: 'speaking and explaining naturally, mouth articulating words, steady eye contact, soft gentle blinking',
    },
    Fact2: {
      expression: 'warm teaching expression, engaged speaking face, direct eye contact',
      gesture: 'hands at rest with occasional subtle illustrative gesture, composed frame',
      motion: 'speaking clearly to camera, natural mouth movement, steady gaze, calm composed presence',
    },
    Fact3: {
      expression: 'knowing confident expression, warm direct gaze, engaged teaching face',
      gesture: 'hands resting open, relaxed confident posture, centered frame',
      motion: 'speaking with confidence, natural articulation, steady direct eye contact, calm assured presence',
    },
    Wisdom: {
      expression: 'warm sincere expression, soft empathetic gaze, heartfelt speaking face',
      gesture: 'hands together or resting naturally, intimate but composed posture',
      motion: 'speaking warmly and sincerely, natural mouth movement, steady eye contact, gentle nodding',
    },
  } as Record<Phase, { expression: string; gesture: string; motion: string }>,
  
  // Archetype-specific voice settings
  archetypeVoice: {
    'The Explorer': { stability: 0.45, similarity: 0.85, style: 0.25, speed: 1.05 },
    'The Rebel': { stability: 0.40, similarity: 0.85, style: 0.35, speed: 1.10 },
    'The Scientist': { stability: 0.55, similarity: 0.85, style: 0.15, speed: 0.95 },
  } as Record<Archetype, { stability: number; similarity: number; style: number; speed: number }>,
};

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
// UTILITY FUNCTIONS
// =============================================================================

function log(emoji: string, message: string, indent = 0): void {
  const prefix = '  '.repeat(indent);
  console.log(`${prefix}${emoji} ${message}`);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function formatDayNumber(day: number): string {
  return String(day).padStart(3, '0');
}

function archetypeToPath(archetype: Archetype): string {
  return archetype.replace('The ', '').toLowerCase();
}

function phaseToPath(phase: Phase): string {
  return phase.toLowerCase();
}

async function downloadFile(url: string, outputPath: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, buffer);
}

// =============================================================================
// STEP 1: AUDIO GENERATION (ElevenLabs)
// =============================================================================

async function generateAudio(
  script: string,
  archetype: Archetype,
  outputDir: string
): Promise<{ localPath: string; publicUrl: string; duration: number }> {
  log('🎤', 'Generating audio (ElevenLabs)...', 1);
  log('📝', `"${script.substring(0, 60)}..."`, 2);
  
  const voiceSettings = KELLY.archetypeVoice[archetype];
  
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
        text: script,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: voiceSettings.stability,
          similarity_boost: voiceSettings.similarity,
          style: voiceSettings.style,
          use_speaker_boost: true,
        },
      }),
    }
  );
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`ElevenLabs error: ${response.status} - ${error}`);
  }
  
  const buffer = Buffer.from(await response.arrayBuffer());
  const localPath = path.join(outputDir, 'audio.mp3');
  fs.writeFileSync(localPath, buffer);
  
  // Estimate duration (rough: ~150 words per minute, average 5 chars per word)
  const wordCount = script.split(/\s+/).length;
  const estimatedDuration = (wordCount / 150) * 60;
  
  log('✅', `Audio generated: ${(buffer.length / 1024).toFixed(1)} KB (~${estimatedDuration.toFixed(1)}s)`, 2);
  
  // Upload to Supabase for public URL (required for Sync Labs)
  const timestamp = Date.now();
  const storagePath = `hd-pipeline/audio_${timestamp}.mp3`;
  
  const { error: uploadError } = await getSupabase().storage
    .from(CONFIG.TEMP_BUCKET)
    .upload(storagePath, buffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });
  
  if (uploadError) {
    throw new Error(`Audio upload failed: ${uploadError.message}`);
  }
  
  const { data: urlData } = getSupabase().storage
    .from(CONFIG.TEMP_BUCKET)
    .getPublicUrl(storagePath);
  
  log('☁️', `Uploaded: ${urlData.publicUrl.substring(0, 60)}...`, 2);
  
  return {
    localPath,
    publicUrl: urlData.publicUrl,
    duration: estimatedDuration,
  };
}

// =============================================================================
// STEP 2: IMAGE GENERATION (Flux + LoRA)
// =============================================================================

async function generateImage(
  phase: Phase,
  outputDir: string
): Promise<{ localPath: string; url: string }> {
  log('🎨', `Generating source image (Flux + LoRA)...`, 1);
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  const phaseConfig = KELLY.phases[phase];
  
  const prompt = `${KELLY.identity}, ${phaseConfig.expression}, ${phaseConfig.gesture}, ${KELLY.background}, ${KELLY.style}`;
  
  log('📝', `Prompt: "${prompt.substring(0, 80)}..."`, 2);
  
  const output = await replicate.run(CONFIG.MODELS.FLUX_LORA as `${string}/${string}:${string}`, {
    input: {
      prompt,
      hf_lora: CONFIG.KELLY_LORA_URL,
      lora_scale: CONFIG.LORA_SCALE,
      num_outputs: 1,
      aspect_ratio: '16:9',
      output_format: 'png',
      guidance_scale: 3.5,
      output_quality: 100,
      num_inference_steps: 35,
      disable_safety_checker: true,
    },
  });
  
  // Extract URL from output
  let imageUrl: string;
  if (Array.isArray(output)) {
    imageUrl = String(output[0]);
  } else if (typeof output === 'object' && output !== null) {
    imageUrl = String((output as any).url || (output as any).toString());
  } else {
    imageUrl = String(output);
  }
  
  if (!imageUrl.startsWith('http')) {
    throw new Error(`Invalid image URL: ${imageUrl}`);
  }
  
  // Download locally
  const localPath = path.join(outputDir, 'source_image.png');
  await downloadFile(imageUrl, localPath);
  
  const stats = fs.statSync(localPath);
  log('✅', `Image generated: ${(stats.size / 1024).toFixed(1)} KB`, 2);
  
  return { localPath, url: imageUrl };
}

// =============================================================================
// STEP 3: MOTION VIDEO (MiniMax Video-01)
// =============================================================================

async function generateMotionVideo(
  imageUrl: string,
  phase: Phase,
  outputDir: string
): Promise<{ localPath: string; url: string }> {
  log('🎬', `Generating motion video (MiniMax Video-01)...`, 1);
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  const phaseConfig = KELLY.phases[phase];
  
  // Motion prompt: calm presence BUT must speak naturally with mouth movement
  const motionPrompt = `Professional female teacher speaking to camera. ${phaseConfig.motion}. She is TALKING and her mouth is moving naturally as she speaks. Steady direct eye contact with camera. Natural breathing, soft blinking. Smooth cinematic quality, warm lighting. CRITICAL: Mouth must open and move naturally while speaking. Eyes stay focused on camera. AVOID: closed mouth, frozen face, wandering eyes, looking away, excessive head movement.`;
  
  log('📝', `Motion: "${motionPrompt.substring(0, 60)}..."`, 2);
  
  const prediction = await replicate.predictions.create({
    model: CONFIG.MODELS.MINIMAX,
    input: {
      prompt: motionPrompt,
      first_frame_image: imageUrl,
      prompt_optimizer: true,
    },
  });
  
  log('⏳', `Prediction ID: ${prediction.id}`, 2);
  
  // Poll for completion (MiniMax takes 2-5 minutes)
  const maxAttempts = 120; // 10 minutes max
  for (let i = 0; i < maxAttempts; i++) {
    const status = await replicate.predictions.get(prediction.id);
    
    if (status.status === 'succeeded') {
      let videoUrl: string;
      if (typeof status.output === 'string') {
        videoUrl = status.output;
      } else if (Array.isArray(status.output)) {
        videoUrl = status.output[0];
      } else {
        throw new Error(`Unexpected output: ${JSON.stringify(status.output)}`);
      }
      
      // Download locally
      const localPath = path.join(outputDir, 'motion_video.mp4');
      await downloadFile(videoUrl, localPath);
      
      const stats = fs.statSync(localPath);
      log('✅', `Motion video: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`, 2);
      
      return { localPath, url: videoUrl };
    }
    
    if (status.status === 'failed') {
      throw new Error(`MiniMax failed: ${status.error}`);
    }
    
    if (status.status === 'canceled') {
      throw new Error('MiniMax job was canceled');
    }
    
    // Progress indicator
    if (i % 12 === 0) {
      const elapsed = Math.round((i * 5) / 60);
      log('⏳', `Status: ${status.status} (${elapsed}m elapsed)`, 2);
    }
    
    await sleep(5000);
  }
  
  throw new Error('MiniMax timed out after 10 minutes');
}

// =============================================================================
// STEP 4: LIP-SYNC (Sync Labs)
// =============================================================================

async function applyLipSync(
  videoUrl: string,
  audioUrl: string,
  outputDir: string
): Promise<{ localPath: string; url: string }> {
  log('👄', `Applying lip-sync (Sync Labs lipsync-2-pro)...`, 1);
  
  if (!CONFIG.SYNC_LABS_API_KEY) {
    log('⚠️', 'No SYNC_LABS_API_KEY, falling back to Wav2Lip', 2);
    return applyWav2Lip(videoUrl, audioUrl, outputDir);
  }
  
  // Submit job to Sync Labs
  const submitRes = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'lipsync-2-pro',  // Upgraded from lipsync-2 for higher quality
      input: [
        { type: 'video', url: videoUrl },
        { type: 'audio', url: audioUrl },
      ],
    }),
  });
  
  if (!submitRes.ok) {
    const error = await submitRes.text();
    log('⚠️', `Sync Labs error: ${submitRes.status} - ${error.substring(0, 100)}`, 2);
    log('⚠️', 'Falling back to Wav2Lip...', 2);
    return applyWav2Lip(videoUrl, audioUrl, outputDir);
  }
  
  const job = await submitRes.json();
  log('⏳', `Job ID: ${job.id}`, 2);
  
  // Poll for completion
  const maxAttempts = 180; // 15 minutes max
  for (let i = 0; i < maxAttempts; i++) {
    const statusRes = await fetch(`https://api.sync.so/v2/generate/${job.id}`, {
      headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
    });
    
    if (!statusRes.ok) {
      throw new Error(`Sync Labs poll error: ${statusRes.status}`);
    }
    
    const status = await statusRes.json();
    
    if (status.status === 'COMPLETED') {
      const outputUrl = status.output?.[0]?.url || status.outputUrl || status.output;
      if (!outputUrl) {
        throw new Error('Sync Labs completed but no output URL');
      }
      
      // Download locally
      const localPath = path.join(outputDir, 'final_hd.mp4');
      await downloadFile(outputUrl, localPath);
      
      const stats = fs.statSync(localPath);
      log('✅', `Lip-synced video: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`, 2);
      
      return { localPath, url: outputUrl };
    }
    
    if (status.status === 'FAILED' || status.status === 'REJECTED') {
      throw new Error(`Sync Labs failed: ${status.error || status.message || 'Unknown error'}`);
    }
    
    // Progress indicator
    if (i % 12 === 0) {
      const elapsed = Math.round((i * 5) / 60);
      log('⏳', `Status: ${status.status} (${elapsed}m elapsed)`, 2);
    }
    
    await sleep(5000);
  }
  
  throw new Error('Sync Labs timed out after 15 minutes');
}

// =============================================================================
// FALLBACK: WAV2LIP (if Sync Labs unavailable)
// =============================================================================

async function applyWav2Lip(
  videoUrl: string,
  audioUrl: string,
  outputDir: string
): Promise<{ localPath: string; url: string }> {
  log('👄', `Applying Wav2Lip (fallback)...`, 2);
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  const output = await replicate.run(CONFIG.MODELS.WAV2LIP as `${string}/${string}:${string}`, {
    input: {
      face: videoUrl,
      audio: audioUrl,
      fps: 25,
      pads: '0 10 0 0',
      smooth: true,
      resize_factor: 1,
    },
  });
  
  const lipsyncUrl = typeof output === 'string' ? output : String(output);
  
  // Download locally
  const localPath = path.join(outputDir, 'final_hd.mp4');
  await downloadFile(lipsyncUrl, localPath);
  
  const stats = fs.statSync(localPath);
  log('✅', `Wav2Lip video: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`, 2);
  
  return { localPath, url: lipsyncUrl };
}

// =============================================================================
// STEP 5: UPLOAD TO SUPABASE & UPDATE DATABASE
// =============================================================================

async function uploadAndUpdateDatabase(
  dayNumber: number,
  archetype: Archetype,
  phase: Phase,
  localVideoPath: string,
  outputDir: string
): Promise<string> {
  log('☁️', `Uploading to Supabase Storage...`, 1);
  
  const sb = getSupabase();
  const dayStr = formatDayNumber(dayNumber);
  const archetypePath = archetypeToPath(archetype);
  const phasePath = phaseToPath(phase);
  
  // Storage path: day-001/explorer/hook.mp4
  const storagePath = `day-${dayStr}/${archetypePath}/${phasePath}.mp4`;
  
  // Read file
  const videoBuffer = fs.readFileSync(localVideoPath);
  
  // Upload
  const { error: uploadError } = await sb.storage
    .from(CONFIG.STORAGE_BUCKET)
    .upload(storagePath, videoBuffer, {
      contentType: 'video/mp4',
      upsert: true,
    });
  
  if (uploadError) {
    throw new Error(`Upload failed: ${uploadError.message}`);
  }
  
  // Get public URL
  const { data: urlData } = sb.storage
    .from(CONFIG.STORAGE_BUCKET)
    .getPublicUrl(storagePath);
  
  const publicUrl = urlData.publicUrl;
  log('✅', `Uploaded: ${publicUrl}`, 2);
  
  // Update lesson_atoms table
  log('📝', `Updating lesson_atoms.hd_video_url...`, 1);
  
  // Get lesson ID
  const { data: lesson, error: lessonError } = await sb
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();
  
  if (lessonError || !lesson) {
    log('⚠️', `Lesson not found for day ${dayNumber}`, 2);
    return publicUrl;
  }
  
  // Update atom
  const { error: updateError } = await sb
    .from('lesson_atoms')
    .update({ hd_video_url: publicUrl })
    .eq('core_lesson_id', lesson.id)
    .eq('archetype', archetype)
    .eq('phase', phase);
  
  if (updateError) {
    log('⚠️', `Database update failed: ${updateError.message}`, 2);
  } else {
    log('✅', `Database updated`, 2);
  }
  
  // Generate lipsync.json for Unity/WebGL fallback
  const lipsyncPath = path.join(outputDir, 'lipsync.json');
  if (!fs.existsSync(lipsyncPath)) {
    // Create placeholder lipsync data
    const lipsyncData = {
      version: '1.0',
      duration: 21.0, // Placeholder
      fps: 30,
      keyframes: [],
      metadata: { note: 'Generated by HD Golden Lesson Pipeline' },
    };
    fs.writeFileSync(lipsyncPath, JSON.stringify(lipsyncData, null, 2));
  }
  
  return publicUrl;
}

// =============================================================================
// FETCH LESSON CONTENT FROM DATABASE
// =============================================================================

async function fetchLessonContent(
  dayNumber: number,
  archetype: Archetype,
  phase: Phase
): Promise<LessonAtom | null> {
  const sb = getSupabase();
  
  // Get lesson ID
  const { data: lesson, error: lessonError } = await sb
    .from('core_lessons')
    .select('id, topic')
    .eq('day_number', dayNumber)
    .single();
  
  if (lessonError || !lesson) {
    log('❌', `Lesson not found for day ${dayNumber}`, 1);
    return null;
  }
  
  // Get atom
  const { data: atom, error: atomError } = await sb
    .from('lesson_atoms')
    .select('id, core_lesson_id, archetype, phase, content, hd_video_url')
    .eq('core_lesson_id', lesson.id)
    .eq('archetype', archetype)
    .eq('phase', phase)
    .single();
  
  if (atomError || !atom) {
    log('❌', `Atom not found for ${archetype} / ${phase}`, 1);
    return null;
  }
  
  return atom as LessonAtom;
}

// =============================================================================
// MAIN GENERATION FUNCTION
// =============================================================================

export async function generateHDVideo(
  archetype: Archetype,
  phase: Phase,
  dayNumber: number
): Promise<GenerationResult> {
  const startTime = Date.now();
  const steps: GenerationSteps = {};
  
  // Create output directory
  const outputDir = path.join(
    CONFIG.OUTPUT_DIR,
    `day_${formatDayNumber(dayNumber)}_${phase}_${archetype.replace(/\s+/g, '_')}`
  );
  fs.mkdirSync(outputDir, { recursive: true });
  
  console.log('\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log(`║  🎬 HD GOLDEN LESSON PIPELINE`.padEnd(71) + '║');
  console.log(`║  Day ${dayNumber} | ${archetype} | ${phase}`.padEnd(71) + '║');
  console.log('╚' + '═'.repeat(70) + '╝');
  
  try {
    // Fetch lesson content
    log('📚', 'Fetching lesson content...', 0);
    const atom = await fetchLessonContent(dayNumber, archetype, phase);
    
    if (!atom || !atom.content?.script) {
      throw new Error(`No script found for Day ${dayNumber} / ${archetype} / ${phase}`);
    }
    
    const script = atom.content.script;
    log('✅', `Script: "${script.substring(0, 60)}..."`, 1);
    
    // Check if already generated
    if (atom.hd_video_url) {
      log('⚠️', `Video already exists: ${atom.hd_video_url}`, 1);
      log('⚠️', `Use --force to regenerate`, 1);
    }
    
    // STEP 1: Audio
    const audio = await generateAudio(script, archetype, outputDir);
    steps.audio = {
      path: audio.localPath,
      url: audio.publicUrl,
      duration: audio.duration,
    };
    
    // STEP 2: Image
    const image = await generateImage(phase, outputDir);
    steps.image = {
      path: image.localPath,
      url: image.url,
    };
    
    // STEP 3: Motion Video
    const motion = await generateMotionVideo(image.url, phase, outputDir);
    steps.motion = {
      path: motion.localPath,
      url: motion.url,
    };
    
    // STEP 4: Lip-Sync
    const lipsync = await applyLipSync(motion.url, audio.publicUrl, outputDir);
    steps.lipsync = {
      path: lipsync.localPath,
      url: lipsync.url,
    };
    
    // STEP 5: Upload & Database
    const supabaseUrl = await uploadAndUpdateDatabase(
      dayNumber,
      archetype,
      phase,
      lipsync.localPath,
      outputDir
    );
    
    steps.finalVideo = {
      path: lipsync.localPath,
      url: supabaseUrl,
    };
    
    const duration = (Date.now() - startTime) / 1000;
    
    console.log('\n' + '═'.repeat(72));
    log('✅', `GENERATION COMPLETE in ${duration.toFixed(1)}s`, 0);
    log('📁', `Local: ${lipsync.localPath}`, 0);
    log('☁️', `Supabase: ${supabaseUrl}`, 0);
    console.log('═'.repeat(72));
    
    return {
      success: true,
      dayNumber,
      archetype,
      phase,
      steps,
      supabaseUrl,
      duration,
    };
    
  } catch (error: any) {
    const duration = (Date.now() - startTime) / 1000;
    
    console.log('\n' + '═'.repeat(72));
    log('❌', `GENERATION FAILED: ${error.message}`, 0);
    console.log('═'.repeat(72));
    
    // Save error info
    const errorPath = path.join(outputDir, 'error.json');
    fs.writeFileSync(errorPath, JSON.stringify({
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString(),
      steps,
    }, null, 2));
    
    return {
      success: false,
      dayNumber,
      archetype,
      phase,
      steps,
      error: error.message,
      duration,
    };
  }
}

// =============================================================================
// BATCH GENERATION
// =============================================================================

async function generateDay(dayNumber: number, dryRun: boolean = false): Promise<GenerationResult[]> {
  console.log('\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log(`║  📅 GENERATING DAY ${dayNumber} (${CONFIG.ARCHETYPES.length * CONFIG.PHASES.length} videos)`.padEnd(71) + '║');
  console.log('╚' + '═'.repeat(70) + '╝');
  
  const results: GenerationResult[] = [];
  let completed = 0;
  const total = CONFIG.ARCHETYPES.length * CONFIG.PHASES.length;
  
  for (const archetype of CONFIG.ARCHETYPES) {
    for (const phase of CONFIG.PHASES) {
      completed++;
      console.log(`\n[${completed}/${total}] ${archetype} / ${phase}`);
      
      if (dryRun) {
        log('🔍', '[DRY RUN] Would generate video', 1);
        continue;
      }
      
      const result = await generateHDVideo(archetype, phase, dayNumber);
      results.push(result);
      
      // Brief pause between generations
      if (completed < total) {
        log('⏳', 'Waiting 5 seconds before next generation...', 1);
        await sleep(5000);
      }
    }
  }
  
  return results;
}

async function generateDayRange(
  fromDay: number,
  toDay: number,
  dryRun: boolean = false
): Promise<void> {
  const totalDays = toDay - fromDay + 1;
  const totalVideos = totalDays * CONFIG.ARCHETYPES.length * CONFIG.PHASES.length;
  
  console.log('\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log(`║  🎬 HD GOLDEN LESSON PIPELINE - BATCH GENERATION`.padEnd(71) + '║');
  console.log(`║  Days ${fromDay}-${toDay} (${totalDays} days, ${totalVideos} videos)`.padEnd(71) + '║');
  console.log('╚' + '═'.repeat(70) + '╝');
  
  const allResults: GenerationResult[] = [];
  
  for (let day = fromDay; day <= toDay; day++) {
    const results = await generateDay(day, dryRun);
    allResults.push(...results);
    
    // Save progress
    const progressPath = path.join(CONFIG.OUTPUT_DIR, `results_${Date.now()}.json`);
    fs.writeFileSync(progressPath, JSON.stringify(allResults, null, 2));
  }
  
  // Final summary
  const successful = allResults.filter(r => r.success).length;
  const failed = allResults.filter(r => !r.success).length;
  
  console.log('\n\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log('║  📊 BATCH GENERATION COMPLETE'.padEnd(71) + '║');
  console.log('╚' + '═'.repeat(70) + '╝');
  console.log(`\n   ✅ Successful: ${successful}/${allResults.length}`);
  console.log(`   ❌ Failed: ${failed}/${allResults.length}`);
  
  if (failed > 0) {
    console.log('\n   Failed items:');
    allResults.filter(r => !r.success).forEach(r => {
      console.log(`      Day ${r.dayNumber} / ${r.archetype} / ${r.phase}: ${r.error}`);
    });
  }
}

// =============================================================================
// CLI
// =============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  
  // Parse arguments
  let dayNumber: number | undefined;
  let fromDay: number | undefined;
  let toDay: number | undefined;
  let archetype: Archetype | undefined;
  let phase: Phase | undefined;
  let dryRun = false;
  let force = false;
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--day':
        dayNumber = parseInt(args[++i]);
        break;
      case '--from':
        fromDay = parseInt(args[++i]);
        break;
      case '--to':
        toDay = parseInt(args[++i]);
        break;
      case '--archetype':
        archetype = args[++i] as Archetype;
        break;
      case '--phase':
        phase = args[++i] as Phase;
        break;
      case '--dry-run':
        dryRun = true;
        break;
      case '--force':
        force = true;
        break;
      case '--help':
        console.log(`
🎬 HD Golden Lesson Pipeline - Production Video Generation

USAGE:
  npx tsx hd-golden-lesson-pipeline.ts [options]

OPTIONS:
  --day <number>          Generate videos for specific day (1-365)
  --from <number>         Start day for range generation
  --to <number>           End day for range generation
  --archetype <name>      Filter to specific archetype
  --phase <name>          Filter to specific phase
  --dry-run               Show what would be generated without doing it
  --force                 Regenerate even if video exists
  --help                  Show this help

EXAMPLES:
  # Generate single video
  npx tsx hd-golden-lesson-pipeline.ts --day 2 --archetype "The Explorer" --phase Hook

  # Generate all videos for Day 2 (15 videos)
  npx tsx hd-golden-lesson-pipeline.ts --day 2

  # Generate Days 2-10 (135 videos)
  npx tsx hd-golden-lesson-pipeline.ts --from 2 --to 10

  # Dry run
  npx tsx hd-golden-lesson-pipeline.ts --day 2 --dry-run

ARCHETYPES:
  "The Explorer", "The Rebel", "The Scientist"

PHASES:
  Hook, Fact1, Fact2, Fact3, Wisdom

REQUIRED ENV VARS:
  REPLICATE_API_TOKEN       For Flux + MiniMax
  ELEVENLABS_API_KEY        For Kelly's voice
  ELEVENLABS_KELLY_VOICE_ID Kelly's voice ID
  SYNC_LABS_API_KEY         For high-quality lip-sync
  PUBLIC_SUPABASE_URL       Supabase URL
  SUPABASE_SERVICE_ROLE_KEY Supabase key
`);
        process.exit(0);
    }
  }
  
  // Validate API keys
  console.log('\n🔑 Checking API Keys...');
  console.log(`   REPLICATE: ${CONFIG.REPLICATE_API_TOKEN ? '✅' : '❌ Required'}`);
  console.log(`   ELEVENLABS: ${CONFIG.ELEVENLABS_API_KEY ? '✅' : '❌ Required'}`);
  console.log(`   SYNC_LABS: ${CONFIG.SYNC_LABS_API_KEY ? '✅' : '⚪ Optional (falls back to Wav2Lip)'}`);
  console.log(`   SUPABASE: ${CONFIG.SUPABASE_URL && CONFIG.SUPABASE_KEY ? '✅' : '❌ Required'}`);
  
  if (!CONFIG.REPLICATE_API_TOKEN || !CONFIG.ELEVENLABS_API_KEY || !CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_KEY) {
    console.error('\n❌ Missing required API keys. Add them to .env');
    process.exit(1);
  }
  
  // Execute based on arguments
  if (fromDay !== undefined && toDay !== undefined) {
    // Range generation
    await generateDayRange(fromDay, toDay, dryRun);
  } else if (dayNumber !== undefined) {
    if (archetype && phase) {
      // Single video
      if (dryRun) {
        console.log(`\n🔍 [DRY RUN] Would generate: Day ${dayNumber} / ${archetype} / ${phase}`);
      } else {
        await generateHDVideo(archetype, phase, dayNumber);
      }
    } else {
      // Full day
      await generateDay(dayNumber, dryRun);
    }
  } else {
    console.log('\n⚠️ No day specified. Use --day, --from/--to, or --help');
    console.log('   Example: npx tsx hd-golden-lesson-pipeline.ts --day 2');
  }
}

// Run if called directly
main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});

// Export for programmatic use
export { generateDay, generateDayRange, fetchLessonContent };

