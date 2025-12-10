#!/usr/bin/env npx tsx
/**
 * 🏭 UNIFIED LESSON FACTORY
 * 
 * The complete orchestrator for generating ALL lesson assets:
 * - Visual Plans (Gemini)
 * - Infographics (Imagen/Flux Pro)
 * - Option Card Images (Imagen)
 * - Kelly Source Images (Flux + LoRA)
 * - Kelly Response Images (Flux + LoRA)
 * - Motion Videos (MiniMax)
 * - Audio (ElevenLabs)
 * - Lipsync (Sync Labs lipsync-2-pro)
 * - Response Videos
 * - Supabase Upload
 * - Cloudflare R2 Backup
 * 
 * Architecture: SEED + EXPANSION
 * - Seeds: 210 base assets per day
 * - Expansion: 54× for videos, 18× for images
 * - Full scale: ~3,795 assets per day
 * 
 * Usage:
 *   npx tsx scripts/lesson-factory/unified-factory.ts --day 1
 *   npx tsx scripts/lesson-factory/unified-factory.ts --day 1 --seeds-only
 *   npx tsx scripts/lesson-factory/unified-factory.ts --range 1-7
 *   npx tsx scripts/lesson-factory/unified-factory.ts --day 1 --dry-run
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
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
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  GEMINI_API_KEY: process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY!,
  
  // Supabase
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  
  // Cloudflare R2
  R2_ACCOUNT_ID: process.env.CLOUDFLARE_ACCOUNT_ID,
  R2_ACCESS_KEY: process.env.CLOUDFLARE_R2_ACCESS_KEY_ID,
  R2_SECRET_KEY: process.env.CLOUDFLARE_R2_SECRET_ACCESS_KEY,
  R2_BUCKET: process.env.KELLY_ASSETS_BUCKET || 'kelly-assets',
  R2_CDN_URL: process.env.KELLY_ASSETS_CDN_URL || 'https://assets.curiouskelly.com',
  
  // Kelly Voice
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  
  // Kelly LoRA
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.90,
  
  // Storage buckets
  VIDEO_BUCKET: 'kelly-videos',
  VISUAL_BUCKET: 'lesson-visuals',
  TEMP_BUCKET: 'kelly-templates',
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-assets', 'unified-factory'),
  
  // Models
  MODELS: {
    FLUX_LORA: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
    FLUX_PRO: 'black-forest-labs/flux-1.1-pro',
    MINIMAX: 'minimax/video-01',
    WAV2LIP: 'devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
  },
  
  // Expansion dimensions
  LANGUAGES: ['en', 'es', 'fr'] as const,
  AGE_BUCKETS: ['5-7', '8-12', '13-17', '18-35', '36-60', '61+'] as const,
  TONES: ['playful', 'conversational', 'reflective'] as const,
  ARCHETYPES: ['The Explorer', 'The Rebel', 'The Scientist'] as const,
  PHASES: ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'] as const,
};

// =============================================================================
// TYPES
// =============================================================================

type Language = typeof CONFIG.LANGUAGES[number];
type AgeBucket = typeof CONFIG.AGE_BUCKETS[number];
type Tone = typeof CONFIG.TONES[number];
type Archetype = typeof CONFIG.ARCHETYPES[number];
type Phase = typeof CONFIG.PHASES[number];

interface SeedAssets {
  dayNumber: number;
  archetype: Archetype;
  
  // Videos (51 per day)
  mainVideos: Map<Phase, VideoAsset>;
  responseVideos: Map<string, VideoAsset>; // key: `${phase}_${option}`
  
  // Images
  infographics: Map<Phase, ImageAsset>;
  optionCards: Map<string, ImageAsset>; // key: `${phase}_${option}`
  kellySourceImages: Map<Phase, ImageAsset>;
  kellyResponseImages: Map<string, ImageAsset>;
  backgrounds: Map<Phase, ImageAsset>;
  
  // Metadata
  thumbnail?: ImageAsset;
  socialShare?: ImageAsset;
}

interface VideoAsset {
  localPath: string;
  supabaseUrl?: string;
  r2Url?: string;
  audioDuration: number;
}

interface ImageAsset {
  localPath: string;
  supabaseUrl?: string;
  r2Url?: string;
  width: number;
  height: number;
}

interface GenerationProgress {
  dayNumber: number;
  startTime: Date;
  
  // Counts
  totalAssets: number;
  completedAssets: number;
  failedAssets: number;
  
  // Costs
  totalCost: number;
  
  // Status
  currentStep: string;
  errors: string[];
}

// =============================================================================
// KELLY VISUAL IDENTITY (LOCKED)
// =============================================================================

const KELLY = {
  // Core identity - NEVER CHANGE
  identity: `kelly, calm confident female teacher, warm brown wavy shoulder-length hair with subtle caramel highlights center-parted, hazel-brown eyes with steady direct gaze, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater`,
  
  background: `professional warm classroom setting, soft natural lighting, shallow depth of field, cinematic quality`,
  style: `professional portrait photography, 85mm lens, soft diffused lighting, warm color grading, 4K UHD`,
  
  negative: `pink sweater, red sweater, purple sweater, teal sweater, green sweater, yellow sweater, auburn hair, chestnut hair, deformed, blurry, bad anatomy, extra fingers, mutated hands, poorly drawn face, mutation, disfigured, low quality, wandering eyes, looking away`,
  
  // Gaze directions for spatial awareness
  gaze: {
    camera: 'looking directly at camera, warm engaged expression',
    diagram: 'gaze directed up and to her right as if looking at a diagram above, head slightly tilted up, interested expression',
    options: 'gaze directed to her right as if acknowledging content there, inviting expression, slight turn of head',
    down: 'gaze directed slightly downward, thoughtful reflective expression',
  },
  
  // Gestures for interaction
  gestures: {
    none: 'hands resting naturally, composed posture',
    point: 'right hand raised with index finger extended, pointing gesture toward upper right, teaching pose',
    pushRail: 'right arm extended toward right side of frame, palm down, pushing gesture as if sliding a panel',
    pullContent: 'right arm extended to side, fingers curved as if grasping, pulling motion toward center',
    openPalm: 'both hands open, palms facing viewer, welcoming inclusive gesture',
    handsHeart: 'both hands placed gently over heart, sincere warm expression',
  },
  
  // Phase-specific configurations
  phases: {
    Hook: {
      expression: 'warm confident expression, direct eye contact with viewer',
      gesture: 'handsResting naturally, composed open posture',
      gaze: 'camera',
      motion: 'speaking naturally to camera, steady gaze, gentle natural breathing',
    },
    Fact1: {
      expression: 'calm curious expression, engaged teaching face',
      gesture: 'one hand may gesture subtly',
      gaze: 'camera', // Will look at diagram at key moments
      motion: 'speaking and explaining naturally, steady eye contact',
    },
    Fact2: {
      expression: 'warm teaching expression, engaged speaking face',
      gesture: 'occasional subtle illustrative gesture',
      gaze: 'camera',
      motion: 'speaking clearly to camera, natural mouth movement',
    },
    Fact3: {
      expression: 'knowing confident expression, warm direct gaze',
      gesture: 'hands resting open, relaxed confident posture',
      gaze: 'camera',
      motion: 'speaking with confidence, natural articulation',
    },
    Wisdom: {
      expression: 'warm sincere expression, soft empathetic gaze',
      gesture: 'hands together or resting naturally',
      gaze: 'camera',
      motion: 'speaking warmly and sincerely, gentle nodding',
    },
  } as Record<Phase, { expression: string; gesture: string; gaze: string; motion: string }>,
  
  // Response expressions by quality
  responseExpressions: {
    best: 'genuinely delighted expression, eyes crinkled with authentic joy, subtle forward lean',
    good: 'warm supportive expression, gentle approving smile, open accepting posture',
    redirect: 'thoughtful understanding expression, compassionate gaze, patient composed posture',
  },
  
  // Voice settings by archetype - ALL 10 archetypes must be defined!
  voiceSettings: {
    'The Explorer': { stability: 0.45, similarity: 0.85, style: 0.25, speed: 1.05 },
    'The Rebel': { stability: 0.40, similarity: 0.85, style: 0.35, speed: 1.10 },
    'The Scientist': { stability: 0.55, similarity: 0.85, style: 0.15, speed: 0.95 },
    'The Architect': { stability: 0.50, similarity: 0.85, style: 0.20, speed: 1.00 },
    'The Diplomat': { stability: 0.55, similarity: 0.85, style: 0.20, speed: 0.98 },
    'The Empath': { stability: 0.48, similarity: 0.85, style: 0.30, speed: 0.95 },
    'The MacGyver': { stability: 0.42, similarity: 0.85, style: 0.25, speed: 1.08 },
    'The Mystic': { stability: 0.52, similarity: 0.85, style: 0.35, speed: 0.92 },
    'The Storyteller': { stability: 0.45, similarity: 0.85, style: 0.40, speed: 1.02 },
    'The Survivor': { stability: 0.50, similarity: 0.85, style: 0.25, speed: 1.00 },
  } as Record<Archetype, { stability: number; similarity: number; style: number; speed: number }>,
};

// =============================================================================
// CLIENTS
// =============================================================================

let supabase: SupabaseClient;
let replicate: Replicate;
let r2Client: S3Client | null = null;

function getSupabase(): SupabaseClient {
  if (!supabase) {
    supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
  }
  return supabase;
}

function getReplicate(): Replicate {
  if (!replicate) {
    replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  }
  return replicate;
}

function getR2Client(): S3Client | null {
  if (!r2Client && CONFIG.R2_ACCOUNT_ID && CONFIG.R2_ACCESS_KEY && CONFIG.R2_SECRET_KEY) {
    r2Client = new S3Client({
      region: 'auto',
      endpoint: `https://${CONFIG.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
      credentials: {
        accessKeyId: CONFIG.R2_ACCESS_KEY,
        secretAccessKey: CONFIG.R2_SECRET_KEY,
      },
    });
  }
  return r2Client;
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

function formatDay(day: number): string {
  return String(day).padStart(3, '0');
}

function archetypeToPath(archetype: Archetype): string {
  return archetype.replace('The ', '').toLowerCase();
}

function phaseToPath(phase: Phase): string {
  return phase.toLowerCase();
}

function calculateHash(buffer: Buffer): string {
  return crypto.createHash('sha256').update(buffer).digest('hex').substring(0, 16);
}

async function downloadFile(url: string, outputPath: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed: ${response.status}`);
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, buffer);
}

// =============================================================================
// STEP 1: FETCH LESSON DATA
// =============================================================================

interface LessonData {
  id: string;
  dayNumber: number;
  topic: string;
  universalTruth: string;
  atoms: LessonAtom[];
}

interface LessonAtom {
  id: string;
  archetype: Archetype;
  phase: Phase;
  script: string;
  options?: Array<{
    letter: string;
    text: string;
    quality: 'best' | 'good' | 'redirect';
    response: string;
  }>;
}

async function fetchLessonData(dayNumber: number): Promise<LessonData | null> {
  log('📚', `Fetching lesson data for Day ${dayNumber}...`);
  
  const sb = getSupabase();
  
  // Get core lesson
  const { data: lesson, error: lessonError } = await sb
    .from('core_lessons')
    .select('id, day_number, topic, universal_truth')
    .eq('day_number', dayNumber)
    .single();
  
  if (lessonError || !lesson) {
    log('❌', `Lesson not found: ${lessonError?.message || 'No data'}`, 1);
    return null;
  }
  
  // Get atoms
  const { data: atoms, error: atomsError } = await sb
    .from('lesson_atoms')
    .select('id, archetype, phase, content')
    .eq('core_lesson_id', lesson.id);
  
  if (atomsError || !atoms) {
    log('❌', `Atoms not found: ${atomsError?.message || 'No data'}`, 1);
    return null;
  }
  
  log('✅', `Found ${atoms.length} atoms for "${lesson.topic}"`, 1);
  
  return {
    id: lesson.id,
    dayNumber: lesson.day_number,
    topic: lesson.topic,
    universalTruth: lesson.universal_truth,
    atoms: atoms.map(a => ({
      id: a.id,
      archetype: a.archetype as Archetype,
      phase: a.phase as Phase,
      script: a.content?.script || '',
      options: a.content?.options || [],
    })),
  };
}

// =============================================================================
// STEP 2: GENERATE INFOGRAPHICS (Flux Pro)
// =============================================================================

async function generateInfographic(
  topic: string,
  phase: Phase,
  archetype: Archetype,
  outputDir: string
): Promise<ImageAsset | null> {
  log('🎨', `Generating infographic for ${phase}...`, 2);
  
  const rep = getReplicate();
  
  // Build infographic prompt based on phase
  const infographicPrompts: Record<Phase, string> = {
    Hook: `Educational infographic introducing "${topic}". Eye-catching visual with key concept visualization. Clean modern design, warm inviting colors, professional educational style. 8K resolution, no text clutter.`,
    Fact1: `Educational diagram explaining the first key concept of "${topic}". Split-scene comparison or process visualization. Clear labels, scientific accuracy, photorealistic elements. 8K, clean typography.`,
    Fact2: `Educational infographic showing deeper details of "${topic}". Before/after or cause-effect visualization. Data visualization elements, clean design. 8K resolution.`,
    Fact3: `Educational diagram connecting all concepts of "${topic}". Process flow or system diagram. Clear visual hierarchy, educational illustration style. 8K resolution.`,
    Wisdom: `Inspirational educational visual summarizing "${topic}". Synthesis of learning, achievement visualization. Warm aspirational mood, clean modern design. 8K resolution.`,
  };
  
  const prompt = infographicPrompts[phase];
  
  try {
    const output = await rep.run(CONFIG.MODELS.FLUX_PRO, {
      input: {
        prompt,
        aspect_ratio: '16:9',
        output_format: 'png',
        output_quality: 100,
        safety_tolerance: 2,
      },
    });
    
    const imageUrl = typeof output === 'string' ? output : String(output);
    const localPath = path.join(outputDir, 'infographics', `${phaseToPath(phase)}.png`);
    await downloadFile(imageUrl, localPath);
    
    log('✅', `Infographic saved: ${path.basename(localPath)}`, 3);
    
    return {
      localPath,
      width: 1920,
      height: 1080,
    };
  } catch (error: any) {
    log('❌', `Infographic failed: ${error.message}`, 3);
    return null;
  }
}

// =============================================================================
// STEP 3: GENERATE OPTION CARD IMAGES (512x512)
// =============================================================================

async function generateOptionCard(
  optionText: string,
  optionLetter: string,
  quality: 'best' | 'good' | 'redirect',
  phase: Phase,
  topic: string,
  outputDir: string
): Promise<ImageAsset | null> {
  log('🃏', `Generating option card ${optionLetter}...`, 2);
  
  const rep = getReplicate();
  
  // Determine border glow based on quality
  const borderStyle = quality === 'best' 
    ? 'subtle green glow border indicating correct answer'
    : 'neutral clean border';
  
  const prompt = `Educational choice card for "${optionText}". Visual representation of the concept, simple icon or illustration. ${borderStyle}. Clean modern design, high contrast, 512x512, easy to tap on mobile. Educational infographic style, no text except small label.`;
  
  try {
    const output = await rep.run(CONFIG.MODELS.FLUX_PRO, {
      input: {
        prompt,
        aspect_ratio: '1:1',
        output_format: 'png',
        output_quality: 100,
      },
    });
    
    const imageUrl = typeof output === 'string' ? output : String(output);
    const localPath = path.join(outputDir, 'options', `${phaseToPath(phase)}_option_${optionLetter.toLowerCase()}.png`);
    await downloadFile(imageUrl, localPath);
    
    log('✅', `Option card saved: ${path.basename(localPath)}`, 3);
    
    return {
      localPath,
      width: 512,
      height: 512,
    };
  } catch (error: any) {
    log('❌', `Option card failed: ${error.message}`, 3);
    return null;
  }
}

// =============================================================================
// STEP 4: GENERATE KELLY SOURCE IMAGES (Flux + LoRA)
// =============================================================================

async function generateKellySourceImage(
  phase: Phase,
  gaze: keyof typeof KELLY.gaze,
  gesture: keyof typeof KELLY.gestures,
  outputDir: string
): Promise<ImageAsset | null> {
  log('👩', `Generating Kelly source image (${phase}, gaze: ${gaze})...`, 2);
  
  const rep = getReplicate();
  const phaseConfig = KELLY.phases[phase];
  
  const prompt = `${KELLY.identity}, ${phaseConfig.expression}, ${KELLY.gaze[gaze]}, ${KELLY.gestures[gesture]}, ${KELLY.background}, ${KELLY.style}`;
  
  try {
    const output = await rep.run(CONFIG.MODELS.FLUX_LORA as `${string}/${string}:${string}`, {
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
    
    let imageUrl: string;
    if (Array.isArray(output)) {
      imageUrl = String(output[0]);
    } else {
      imageUrl = String(output);
    }
    
    const localPath = path.join(outputDir, 'kelly-source', `${phaseToPath(phase)}_${gaze}.png`);
    await downloadFile(imageUrl, localPath);
    
    log('✅', `Kelly source saved: ${path.basename(localPath)}`, 3);
    
    return {
      localPath,
      width: 1344,
      height: 768,
    };
  } catch (error: any) {
    log('❌', `Kelly source failed: ${error.message}`, 3);
    return null;
  }
}

// =============================================================================
// STEP 5: GENERATE KELLY RESPONSE IMAGES
// =============================================================================

async function generateKellyResponseImage(
  quality: 'best' | 'good' | 'redirect',
  outputDir: string,
  identifier: string
): Promise<ImageAsset | null> {
  log('👩', `Generating Kelly response image (${quality})...`, 2);
  
  const rep = getReplicate();
  const expression = KELLY.responseExpressions[quality];
  
  const prompt = `${KELLY.identity}, ${expression}, ${KELLY.gaze.camera}, ${KELLY.background}, ${KELLY.style}`;
  
  try {
    const output = await rep.run(CONFIG.MODELS.FLUX_LORA as `${string}/${string}:${string}`, {
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
      },
    });
    
    let imageUrl: string;
    if (Array.isArray(output)) {
      imageUrl = String(output[0]);
    } else {
      imageUrl = String(output);
    }
    
    const localPath = path.join(outputDir, 'kelly-responses', `${identifier}_${quality}.png`);
    await downloadFile(imageUrl, localPath);
    
    log('✅', `Kelly response saved: ${path.basename(localPath)}`, 3);
    
    return {
      localPath,
      width: 1344,
      height: 768,
    };
  } catch (error: any) {
    log('❌', `Kelly response failed: ${error.message}`, 3);
    return null;
  }
}

// =============================================================================
// STEP 6: GENERATE AUDIO (ElevenLabs)
// =============================================================================

async function generateAudio(
  script: string,
  archetype: Archetype,
  outputDir: string,
  identifier: string
): Promise<{ localPath: string; publicUrl: string; duration: number } | null> {
  log('🎤', `Generating audio...`, 2);
  
  const voiceSettings = KELLY.voiceSettings[archetype];
  
  try {
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
      throw new Error(`ElevenLabs: ${response.status}`);
    }
    
    const buffer = Buffer.from(await response.arrayBuffer());
    const localPath = path.join(outputDir, 'audio', `${identifier}.mp3`);
    fs.mkdirSync(path.dirname(localPath), { recursive: true });
    fs.writeFileSync(localPath, buffer);
    
    // Estimate duration
    const wordCount = script.split(/\s+/).length;
    const duration = (wordCount / 150) * 60;
    
    // Upload for public URL
    const sb = getSupabase();
    const storagePath = `factory-temp/audio_${Date.now()}_${identifier}.mp3`;
    
    await sb.storage.from(CONFIG.TEMP_BUCKET).upload(storagePath, buffer, {
      contentType: 'audio/mpeg',
      upsert: true,
    });
    
    const { data: urlData } = sb.storage.from(CONFIG.TEMP_BUCKET).getPublicUrl(storagePath);
    
    log('✅', `Audio saved: ${(buffer.length / 1024).toFixed(1)} KB (~${duration.toFixed(1)}s)`, 3);
    
    return {
      localPath,
      publicUrl: urlData.publicUrl,
      duration,
    };
  } catch (error: any) {
    log('❌', `Audio failed: ${error.message}`, 3);
    return null;
  }
}

// =============================================================================
// STEP 7: GENERATE MOTION VIDEO (MiniMax)
// =============================================================================

async function generateMotionVideo(
  imageUrl: string,
  phase: Phase,
  outputDir: string,
  identifier: string
): Promise<{ localPath: string; url: string } | null> {
  log('🎬', `Generating motion video (MiniMax)...`, 2);
  
  const rep = getReplicate();
  const phaseConfig = KELLY.phases[phase];
  
  const motionPrompt = `Professional female teacher speaking to camera. ${phaseConfig.motion}. She is TALKING and her mouth is moving naturally as she speaks. Steady direct eye contact with camera. Natural breathing, soft blinking. Smooth cinematic quality, warm lighting. CRITICAL: Mouth must open and move naturally while speaking.`;
  
  try {
    const prediction = await rep.predictions.create({
      model: CONFIG.MODELS.MINIMAX,
      input: {
        prompt: motionPrompt,
        first_frame_image: imageUrl,
        prompt_optimizer: true,
      },
    });
    
    log('⏳', `Prediction ID: ${prediction.id}`, 3);
    
    // Poll for completion
    const maxAttempts = 120;
    for (let i = 0; i < maxAttempts; i++) {
      const status = await rep.predictions.get(prediction.id);
      
      if (status.status === 'succeeded') {
        let videoUrl: string;
        if (typeof status.output === 'string') {
          videoUrl = status.output;
        } else if (Array.isArray(status.output)) {
          videoUrl = status.output[0];
        } else {
          throw new Error('Unexpected output format');
        }
        
        const localPath = path.join(outputDir, 'motion', `${identifier}.mp4`);
        await downloadFile(videoUrl, localPath);
        
        log('✅', `Motion video saved`, 3);
        return { localPath, url: videoUrl };
      }
      
      if (status.status === 'failed') {
        throw new Error(`MiniMax failed: ${status.error}`);
      }
      
      if (i % 12 === 0) {
        log('⏳', `Status: ${status.status}`, 3);
      }
      
      await sleep(5000);
    }
    
    throw new Error('MiniMax timed out');
  } catch (error: any) {
    log('❌', `Motion video failed: ${error.message}`, 3);
    return null;
  }
}

// =============================================================================
// STEP 8: APPLY LIPSYNC (Sync Labs lipsync-2-pro)
// =============================================================================

async function applyLipsync(
  videoUrl: string,
  audioUrl: string,
  outputDir: string,
  identifier: string
): Promise<{ localPath: string; url: string } | null> {
  log('👄', `Applying lipsync (Sync Labs lipsync-2-pro)...`, 2);
  
  if (!CONFIG.SYNC_LABS_API_KEY) {
    log('⚠️', 'No SYNC_LABS_API_KEY, falling back to Wav2Lip', 3);
    return applyWav2Lip(videoUrl, audioUrl, outputDir, identifier);
  }
  
  try {
    // Submit job
    const submitRes = await fetch('https://api.sync.so/v2/generate', {
      method: 'POST',
      headers: {
        'x-api-key': CONFIG.SYNC_LABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'lipsync-2-pro', // ✅ CONFIRMED PREMIUM MODEL
        input: [
          { type: 'video', url: videoUrl },
          { type: 'audio', url: audioUrl },
        ],
      }),
    });
    
    if (!submitRes.ok) {
      const error = await submitRes.text();
      log('⚠️', `Sync Labs error: ${error.substring(0, 100)}`, 3);
      return applyWav2Lip(videoUrl, audioUrl, outputDir, identifier);
    }
    
    const job = await submitRes.json();
    log('⏳', `Job ID: ${job.id}`, 3);
    
    // Poll for completion
    const maxAttempts = 180;
    for (let i = 0; i < maxAttempts; i++) {
      const statusRes = await fetch(`https://api.sync.so/v2/generate/${job.id}`, {
        headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
      });
      
      const status = await statusRes.json();
      
      if (status.status === 'COMPLETED') {
        const outputUrl = status.output?.[0]?.url || status.outputUrl || status.output;
        
        const localPath = path.join(outputDir, 'final', `${identifier}.mp4`);
        await downloadFile(outputUrl, localPath);
        
        log('✅', `Lipsync complete`, 3);
        return { localPath, url: outputUrl };
      }
      
      if (status.status === 'FAILED' || status.status === 'REJECTED') {
        throw new Error(`Sync Labs failed: ${status.error || status.message}`);
      }
      
      if (i % 12 === 0) {
        log('⏳', `Status: ${status.status}`, 3);
      }
      
      await sleep(5000);
    }
    
    throw new Error('Sync Labs timed out');
  } catch (error: any) {
    log('❌', `Lipsync failed: ${error.message}`, 3);
    return null;
  }
}

async function applyWav2Lip(
  videoUrl: string,
  audioUrl: string,
  outputDir: string,
  identifier: string
): Promise<{ localPath: string; url: string } | null> {
  const rep = getReplicate();
  
  try {
    const output = await rep.run(CONFIG.MODELS.WAV2LIP as `${string}/${string}:${string}`, {
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
    const localPath = path.join(outputDir, 'final', `${identifier}.mp4`);
    await downloadFile(lipsyncUrl, localPath);
    
    log('✅', `Wav2Lip complete`, 3);
    return { localPath, url: lipsyncUrl };
  } catch (error: any) {
    log('❌', `Wav2Lip failed: ${error.message}`, 3);
    return null;
  }
}

// =============================================================================
// STEP 9: UPLOAD TO SUPABASE
// =============================================================================

async function uploadToSupabase(
  localPath: string,
  bucket: string,
  storagePath: string
): Promise<string | null> {
  const sb = getSupabase();
  const buffer = fs.readFileSync(localPath);
  const contentType = localPath.endsWith('.mp4') ? 'video/mp4' : 'image/png';
  
  const { error } = await sb.storage.from(bucket).upload(storagePath, buffer, {
    contentType,
    upsert: true,
  });
  
  if (error) {
    log('❌', `Supabase upload failed: ${error.message}`, 3);
    return null;
  }
  
  const { data } = sb.storage.from(bucket).getPublicUrl(storagePath);
  return data.publicUrl;
}

// =============================================================================
// STEP 10: BACKUP TO CLOUDFLARE R2
// =============================================================================

async function backupToR2(
  localPath: string,
  r2Key: string
): Promise<string | null> {
  const client = getR2Client();
  if (!client) {
    log('⚠️', 'R2 not configured, skipping backup', 3);
    return null;
  }
  
  try {
    const buffer = fs.readFileSync(localPath);
    const contentType = localPath.endsWith('.mp4') ? 'video/mp4' : 'image/png';
    
    await client.send(new PutObjectCommand({
      Bucket: CONFIG.R2_BUCKET,
      Key: r2Key,
      Body: buffer,
      ContentType: contentType,
      CacheControl: 'public, max-age=31536000, immutable',
      Metadata: {
        'uploaded-at': new Date().toISOString(),
        'file-hash': calculateHash(buffer),
      },
    }));
    
    return `${CONFIG.R2_CDN_URL}/${r2Key}`;
  } catch (error: any) {
    log('❌', `R2 backup failed: ${error.message}`, 3);
    return null;
  }
}

// =============================================================================
// MAIN SEED GENERATION
// =============================================================================

async function generateDaySeeds(
  dayNumber: number,
  archetype: Archetype,
  dryRun: boolean = false
): Promise<{ success: boolean; assets: number; cost: number }> {
  const dayStr = formatDay(dayNumber);
  const archetypePath = archetypeToPath(archetype);
  
  console.log('\n' + '═'.repeat(70));
  console.log(`🏭 UNIFIED LESSON FACTORY - Day ${dayNumber} / ${archetype}`);
  console.log('═'.repeat(70));
  
  if (dryRun) {
    log('🔍', '[DRY RUN] Would generate seed assets', 0);
    return { success: true, assets: 0, cost: 0 };
  }
  
  // Create output directory
  const outputDir = path.join(CONFIG.OUTPUT_DIR, `day-${dayStr}`, archetypePath);
  fs.mkdirSync(outputDir, { recursive: true });
  
  // Fetch lesson data
  const lesson = await fetchLessonData(dayNumber);
  if (!lesson) {
    return { success: false, assets: 0, cost: 0 };
  }
  
  // Get atoms for this archetype
  const archetypeAtoms = lesson.atoms.filter(a => a.archetype === archetype);
  if (archetypeAtoms.length === 0) {
    log('❌', `No atoms found for ${archetype}`, 1);
    return { success: false, assets: 0, cost: 0 };
  }
  
  let assetsGenerated = 0;
  let totalCost = 0;
  
  // Generate assets for each phase
  for (const phase of CONFIG.PHASES) {
    const atom = archetypeAtoms.find(a => a.phase === phase);
    if (!atom) {
      log('⚠️', `No atom for ${phase}`, 1);
      continue;
    }
    
    console.log(`\n📍 Phase: ${phase}`);
    console.log('─'.repeat(50));
    
    // 1. Generate infographic
    const infographic = await generateInfographic(lesson.topic, phase, archetype, outputDir);
    if (infographic) {
      assetsGenerated++;
      totalCost += 0.04;
    }
    await sleep(2000);
    
    // 2. Generate Kelly source image
    const kellySource = await generateKellySourceImage(phase, 'camera', 'none', outputDir);
    if (kellySource) {
      assetsGenerated++;
      totalCost += 0.04;
    }
    await sleep(2000);
    
    // 3. Generate audio
    const audio = await generateAudio(atom.script, archetype, outputDir, `${phaseToPath(phase)}_main`);
    if (audio) {
      assetsGenerated++;
      totalCost += 0.02;
    }
    await sleep(1000);
    
    // 4. Generate motion video (if we have source image)
    if (kellySource && audio) {
      // Need to upload source image for MiniMax
      const imageUrl = await uploadToSupabase(
        kellySource.localPath,
        CONFIG.TEMP_BUCKET,
        `factory-temp/source_${Date.now()}.png`
      );
      
      if (imageUrl) {
        const motion = await generateMotionVideo(imageUrl, phase, outputDir, `${phaseToPath(phase)}_main`);
        if (motion) {
          assetsGenerated++;
          totalCost += 0.12;
          
          // 5. Apply lipsync
          const lipsync = await applyLipsync(motion.url, audio.publicUrl, outputDir, `${phaseToPath(phase)}_main`);
          if (lipsync) {
            assetsGenerated++;
            totalCost += 0.20;
            
            // Upload to Supabase
            const storagePath = `day-${dayStr}/${archetypePath}/${phaseToPath(phase)}.mp4`;
            const supabaseUrl = await uploadToSupabase(lipsync.localPath, CONFIG.VIDEO_BUCKET, storagePath);
            
            if (supabaseUrl) {
              log('☁️', `Uploaded to Supabase: ${storagePath}`, 2);
              
              // Backup to R2
              const r2Url = await backupToR2(lipsync.localPath, `videos/${storagePath}`);
              if (r2Url) {
                log('☁️', `Backed up to R2`, 2);
              }
              
              // Update database
              const sb = getSupabase();
              await sb.from('lesson_atoms')
                .update({ hd_video_url: supabaseUrl })
                .eq('id', atom.id);
              log('📝', `Database updated`, 2);
            }
          }
        }
      }
    }
    
    // 6. Generate option cards and response videos (for phases with options)
    if (atom.options && atom.options.length > 0 && phase !== 'Wisdom') {
      for (const option of atom.options) {
        // Option card
        const optionCard = await generateOptionCard(
          option.text,
          option.letter,
          option.quality,
          phase,
          lesson.topic,
          outputDir
        );
        if (optionCard) {
          assetsGenerated++;
          totalCost += 0.02;
        }
        await sleep(1000);
        
        // Kelly response image
        const responseImage = await generateKellyResponseImage(
          option.quality,
          outputDir,
          `${phaseToPath(phase)}_${option.letter.toLowerCase()}`
        );
        if (responseImage) {
          assetsGenerated++;
          totalCost += 0.04;
        }
        await sleep(1000);
        
        // Response audio
        const responseAudio = await generateAudio(
          option.response,
          archetype,
          outputDir,
          `${phaseToPath(phase)}_response_${option.letter.toLowerCase()}`
        );
        if (responseAudio) {
          assetsGenerated++;
          totalCost += 0.02;
        }
        await sleep(1000);
        
        // Response video (if we have image and audio)
        if (responseImage && responseAudio) {
          const imageUrl = await uploadToSupabase(
            responseImage.localPath,
            CONFIG.TEMP_BUCKET,
            `factory-temp/response_${Date.now()}.png`
          );
          
          if (imageUrl) {
            const motion = await generateMotionVideo(
              imageUrl,
              phase,
              outputDir,
              `${phaseToPath(phase)}_response_${option.letter.toLowerCase()}`
            );
            
            if (motion) {
              assetsGenerated++;
              totalCost += 0.12;
              
              const lipsync = await applyLipsync(
                motion.url,
                responseAudio.publicUrl,
                outputDir,
                `${phaseToPath(phase)}_response_${option.letter.toLowerCase()}`
              );
              
              if (lipsync) {
                assetsGenerated++;
                totalCost += 0.20;
                
                // Upload
                const storagePath = `day-${dayStr}/${archetypePath}/${phaseToPath(phase)}_response_${option.letter.toLowerCase()}.mp4`;
                const supabaseUrl = await uploadToSupabase(lipsync.localPath, CONFIG.VIDEO_BUCKET, storagePath);
                
                if (supabaseUrl) {
                  log('☁️', `Response video uploaded: ${storagePath}`, 2);
                  await backupToR2(lipsync.localPath, `videos/${storagePath}`);
                }
              }
            }
          }
        }
      }
    }
    
    // Brief pause between phases
    await sleep(3000);
  }
  
  console.log('\n' + '═'.repeat(70));
  log('✅', `Day ${dayNumber} / ${archetype} COMPLETE`, 0);
  log('📊', `Assets generated: ${assetsGenerated}`, 0);
  log('💰', `Estimated cost: $${totalCost.toFixed(2)}`, 0);
  console.log('═'.repeat(70));
  
  return {
    success: true,
    assets: assetsGenerated,
    cost: totalCost,
  };
}

// =============================================================================
// CLI
// =============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  
  let dayNumber: number | undefined;
  let fromDay: number | undefined;
  let toDay: number | undefined;
  let archetype: Archetype | undefined;
  let seedsOnly = true; // Default to seeds only
  let dryRun = false;
  
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
      case '--seeds-only':
        seedsOnly = true;
        break;
      case '--full-expansion':
        seedsOnly = false;
        break;
      case '--dry-run':
        dryRun = true;
        break;
      case '--help':
        console.log(`
🏭 UNIFIED LESSON FACTORY

Generates ALL lesson assets: videos, images, audio, with Supabase + R2 backup.

USAGE:
  npx tsx scripts/lesson-factory/unified-factory.ts [options]

OPTIONS:
  --day <number>        Generate for specific day
  --from <number>       Start day for range
  --to <number>         End day for range
  --archetype <name>    Filter to specific archetype
  --seeds-only          Generate seed templates only (default)
  --full-expansion      Generate all language/age/tone variants
  --dry-run             Show what would be generated
  --help                Show this help

EXAMPLES:
  # Generate Day 1 seeds for all archetypes
  npx tsx scripts/lesson-factory/unified-factory.ts --day 1

  # Generate Day 1 seeds for Explorer only
  npx tsx scripts/lesson-factory/unified-factory.ts --day 1 --archetype "The Explorer"

  # Generate Days 1-7
  npx tsx scripts/lesson-factory/unified-factory.ts --from 1 --to 7

  # Dry run
  npx tsx scripts/lesson-factory/unified-factory.ts --day 1 --dry-run

ARCHETYPES:
  "The Explorer", "The Rebel", "The Scientist"

SEED ASSETS PER DAY (per archetype):
  - 5 infographics
  - 5 Kelly source images
  - 5 main videos
  - 12 option cards
  - 12 Kelly response images
  - 12 response videos
  ≈ 51 assets per archetype × 3 = 153 per day

REQUIRED ENV VARS:
  REPLICATE_API_TOKEN
  ELEVENLABS_API_KEY
  SYNC_LABS_API_KEY
  PUBLIC_SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  
OPTIONAL (for R2 backup):
  CLOUDFLARE_ACCOUNT_ID
  CLOUDFLARE_R2_ACCESS_KEY_ID
  CLOUDFLARE_R2_SECRET_ACCESS_KEY
  KELLY_ASSETS_BUCKET
  KELLY_ASSETS_CDN_URL
`);
        process.exit(0);
    }
  }
  
  // Validate API keys
  console.log('\n🔑 Checking API Keys...');
  console.log(`   REPLICATE: ${CONFIG.REPLICATE_API_TOKEN ? '✅' : '❌'}`);
  console.log(`   ELEVENLABS: ${CONFIG.ELEVENLABS_API_KEY ? '✅' : '❌'}`);
  console.log(`   SYNC_LABS: ${CONFIG.SYNC_LABS_API_KEY ? '✅ (lipsync-2-pro)' : '⚠️ (Wav2Lip fallback)'}`);
  console.log(`   SUPABASE: ${CONFIG.SUPABASE_URL && CONFIG.SUPABASE_KEY ? '✅' : '❌'}`);
  console.log(`   R2 BACKUP: ${CONFIG.R2_ACCOUNT_ID ? '✅' : '⚠️ (disabled)'}`);
  
  if (!CONFIG.REPLICATE_API_TOKEN || !CONFIG.ELEVENLABS_API_KEY || !CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_KEY) {
    console.error('\n❌ Missing required API keys');
    process.exit(1);
  }
  
  // Execute
  const archetypes = archetype ? [archetype] : [...CONFIG.ARCHETYPES];
  let totalAssets = 0;
  let totalCost = 0;
  
  if (fromDay !== undefined && toDay !== undefined) {
    // Range
    for (let day = fromDay; day <= toDay; day++) {
      for (const arch of archetypes) {
        const result = await generateDaySeeds(day, arch, dryRun);
        totalAssets += result.assets;
        totalCost += result.cost;
      }
    }
  } else if (dayNumber !== undefined) {
    // Single day
    for (const arch of archetypes) {
      const result = await generateDaySeeds(dayNumber, arch, dryRun);
      totalAssets += result.assets;
      totalCost += result.cost;
    }
  } else {
    console.log('\n⚠️ No day specified. Use --day, --from/--to, or --help');
  }
  
  // Summary
  if (totalAssets > 0) {
    console.log('\n\n' + '█'.repeat(70));
    console.log('📊 FACTORY RUN COMPLETE');
    console.log('█'.repeat(70));
    console.log(`   Total assets: ${totalAssets}`);
    console.log(`   Estimated cost: $${totalCost.toFixed(2)}`);
    console.log('█'.repeat(70));
  }
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});

