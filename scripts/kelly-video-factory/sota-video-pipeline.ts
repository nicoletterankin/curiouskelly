/**
 * 🚀 SOTA (State-of-the-Art) Kelly Video Pipeline
 * 
 * The ultimate API-based video generation pipeline for creating
 * the best digital human teacher on the planet.
 * 
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  TIER 1: SYNC LABS (Premium Lip-Sync)                          │
 * │  Input: Base video/image + ElevenLabs audio                    │
 * │  Output: 4K lip-synced video with natural mouth movements      │
 * │  Quality: 95%+ accuracy                                        │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  TIER 2: HEDRA/LIVEPORTRAIT (Full Face Animation)              │
 * │  Input: Kelly image + audio                                    │
 * │  Output: Full facial expressions, not just lips                │
 * │  Quality: Eyes, brows, head movement                           │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  TIER 3: OMNIHUMAN (Full Body, Next-Gen)                       │
 * │  Input: Reference image + audio                                │
 * │  Output: Full body animation with gestures                     │
 * │  Quality: Industry-leading realism                             │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * Usage:
 *   npx tsx scripts/kelly-video-factory/sota-video-pipeline.ts --tier sync --text "Hello!"
 *   npx tsx scripts/kelly-video-factory/sota-video-pipeline.ts --tier hedra --pose excited
 *   npx tsx scripts/kelly-video-factory/sota-video-pipeline.ts --tier omnihuman --full-body
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION - API KEYS & ENDPOINTS
// =============================================================================

const CONFIG = {
  // Replicate
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  
  // ElevenLabs
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  
  // Sync Labs (Premium Lip-Sync)
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY,
  SYNC_LABS_API_URL: 'https://api.sync.so/v2',
  
  // fal.ai (OmniHuman, SadTalker fallback)
  FAL_KEY: process.env.FAL_KEY,
  
  // Hedra API
  HEDRA_API_KEY: process.env.HEDRA_API_KEY,
  
  // Kelly LoRA
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'sota'),
};

// =============================================================================
// KELLY VISUAL IDENTITY (Canonical - Do Not Change)
// =============================================================================

const KELLY = {
  // Core identity prompt
  identity: `kelly, friendly approachable teacher, intelligent warmth, genuine smile lines, natural beauty, woman with long wavy chestnut brown hair with subtle highlights and warm brown eyes with visible catchlights, wearing soft powder blue crewneck sweater`,
  
  // Cinematic style
  style: `cinematic lighting, shallow depth of field, 85mm lens, professional color grading, soft diffused lighting, 4K UHD`,
  
  // Negative prompt (things to avoid)
  negative: `pink sweater, red sweater, beige sweater, teal sweater, green sweater, yellow sweater, deformed, blurry, bad anatomy, extra fingers, mutated hands, poorly drawn face, mutation, disfigured, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, out of frame, cropped, watermark, signature, text`,
  
  // Pose variations for different emotional states
  poses: {
    excited: {
      prompt: 'eyes sparkling with genuine excitement and wonder, natural joyful expression with teeth showing, hands gesturing expressively, warm modern classroom environment',
      emotion: 'excited',
      headMotion: 'slight_nod',
    },
    curious: {
      prompt: 'head slightly tilted with one eyebrow raised in genuine curiosity, warm inviting smile, cozy study room with warm wood tones',
      emotion: 'curious',
      headMotion: 'tilt_right',
    },
    explaining: {
      prompt: 'animated expression while explaining something fascinating, hands positioned as if holding an invisible concept, leaning slightly forward',
      emotion: 'engaged',
      headMotion: 'forward_lean',
    },
    thoughtful: {
      prompt: 'contemplative expression with a soft knowing smile, chin resting gently on hand, gazing slightly off-camera',
      emotion: 'thoughtful',
      headMotion: 'slow_turn',
    },
    heartfelt: {
      prompt: 'hand placed gently over heart, eyes filled with genuine warmth and sincerity, soft empathetic smile',
      emotion: 'sincere',
      headMotion: 'gentle_nod',
    },
    welcome: {
      prompt: 'arms open in welcoming gesture, genuine warm smile, standing in beautiful sunlit setting',
      emotion: 'welcoming',
      headMotion: 'slight_bow',
    },
    celebrating: {
      prompt: 'genuinely proud and delighted expression, subtle clapping or thumbs up gesture, eyes crinkled with authentic happiness',
      emotion: 'proud',
      headMotion: 'excited_nod',
    },
  } as Record<string, { prompt: string; emotion: string; headMotion: string }>,
};

// =============================================================================
// TIER 1: SYNC LABS - Premium Lip-Sync (Best Quality)
// =============================================================================

interface SyncLabsConfig {
  model: 'lipsync-2' | 'lipsync-2-pro' | 'wav2lip++';
  maxCredits?: number;
  webhookUrl?: string;
}

async function generateWithSyncLabs(
  videoUrl: string,
  audioUrl: string,
  config: SyncLabsConfig = { model: 'lipsync-2' }
): Promise<string> {
  console.log('\n🎬 SYNC LABS - Premium Lip-Sync (95%+ accuracy)');
  console.log('━'.repeat(60));
  
  if (!CONFIG.SYNC_LABS_API_KEY) {
    throw new Error('SYNC_LABS_API_KEY not configured. Sign up at https://sync.so');
  }
  
  const videoStr = String(videoUrl);
  const audioStr = String(audioUrl);
  
  console.log(`   Model: ${config.model}`);
  console.log(`   Video Input: ${videoStr.substring(0, 60)}...`);
  console.log(`   Audio Input: ${audioStr.startsWith('http') ? audioStr.substring(0, 60) + '...' : 'data URI'}`);
  
  // Sync Labs API v2 format
  const payload = {
    model: config.model,
    input: [
      {
        type: 'video',
        url: videoStr,
      },
      {
        type: 'audio',
        url: audioStr,
      }
    ],
    ...(config.maxCredits && { maxCredits: config.maxCredits }),
    ...(config.webhookUrl && { webhookUrl: config.webhookUrl }),
  };
  
  console.log(`   🚀 Submitting job to Sync Labs...`);
  
  // Create lip-sync job
  const response = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Sync Labs API error: ${response.status} - ${error}`);
  }
  
  const job = await response.json();
  console.log(`   Job ID: ${job.id}`);
  console.log(`   Status: ${job.status}`);
  
  // Poll for completion
  const result = await pollSyncLabsJob(job.id);
  console.log(`   ✅ Complete!`);
  
  return result;
}

async function pollSyncLabsJob(jobId: string, maxAttempts = 180): Promise<string> {
  console.log(`   ⏳ Waiting for Sync Labs processing...`);
  
  for (let i = 0; i < maxAttempts; i++) {
    const response = await fetch(`https://api.sync.so/v2/generate/${jobId}`, {
      headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY! },
    });
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Sync Labs poll error: ${response.status} - ${error}`);
    }
    
    const job = await response.json();
    
    if (job.status === 'COMPLETED') {
      // Extract output URL from the response
      const outputUrl = job.output?.[0]?.url || job.outputUrl || job.output;
      if (outputUrl) {
        return outputUrl;
      }
      throw new Error('Sync Labs completed but no output URL found');
    }
    
    if (job.status === 'FAILED' || job.status === 'REJECTED') {
      throw new Error(`Sync Labs job failed: ${job.error || job.message || 'Unknown error'}`);
    }
    
    // Show progress
    if (i % 6 === 0) { // Every 30 seconds
      console.log(`      Status: ${job.status} (${Math.round(i * 5 / 60)}m elapsed)`);
    }
    process.stdout.write('.');
    await sleep(5000);
  }
  
  throw new Error('Sync Labs job timed out after 15 minutes');
}

// =============================================================================
// TIER 2: HEDRA/LIVEPORTRAIT - Full Facial Animation
// =============================================================================

async function generateWithHedra(
  imageUrl: string,
  audioUrl: string,
  emotion: string = 'neutral'
): Promise<string> {
  console.log('\n🎭 HEDRA - Character-1 Full Face Animation');
  console.log('━'.repeat(60));
  
  if (!CONFIG.HEDRA_API_KEY) {
    console.log('   ⚠️ HEDRA_API_KEY not configured, falling back to LivePortrait');
    return generateWithLivePortrait(imageUrl, audioUrl);
  }
  
  // Hedra API call
  const response = await fetch('https://api.hedra.com/v1/characters/generate', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${CONFIG.HEDRA_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_url: imageUrl,
      audio_url: audioUrl,
      emotion: emotion,
      aspect_ratio: '16:9',
    }),
  });
  
  if (!response.ok) {
    console.log('   ⚠️ Hedra API error, falling back to LivePortrait');
    return generateWithLivePortrait(imageUrl, audioUrl);
  }
  
  const result = await response.json();
  return result.video_url;
}

async function generateWithLivePortrait(
  imageUrl: string,
  audioPath: string
): Promise<string> {
  console.log('\n🎭 LIVEPORTRAIT - Audio-Driven Animation');
  console.log('━'.repeat(60));
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  // Convert local audio file to data URI if it's a local path
  let audioInput: string;
  if (audioPath.startsWith('http')) {
    audioInput = audioPath;
  } else if (fs.existsSync(audioPath)) {
    const audioBuffer = fs.readFileSync(audioPath);
    audioInput = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
    console.log(`   📤 Converting audio to data URI (${(audioBuffer.length / 1024).toFixed(1)} KB)`);
  } else {
    throw new Error(`Audio file not found: ${audioPath}`);
  }
  
  console.log(`   🎬 Running LivePortrait...`);
  
  // LivePortrait model on Replicate
  const output = await replicate.run(
    "fofr/live-portrait:067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146",
    {
      input: {
        source_image: imageUrl,
        driving_audio: audioInput,
        live_portrait_dsize: 512,
        live_portrait_scale: 2.3,
        video_frame_load_cap: 128,
        live_portrait_lip_zero: true,
        live_portrait_relative: true,
        live_portrait_vx_ratio: 0,
        live_portrait_vy_ratio: -0.12,
        live_portrait_stitching: true,
        live_portrait_eye_retargeting: true,
        live_portrait_lip_retargeting: true,
      }
    }
  );
  
  const videoUrl = typeof output === 'string' ? output : (output as any)?.output;
  console.log(`   ✅ Complete: ${videoUrl}`);
  
  return videoUrl;
}

// =============================================================================
// TIER 3: OMNIHUMAN / HALLO2 - Full Body Animation (Next-Gen)
// =============================================================================

async function generateWithOmniHuman(
  imageUrl: string,
  audioUrl: string,
  fullBody: boolean = false
): Promise<string> {
  console.log('\n🧬 OMNIHUMAN - Next-Gen Digital Human');
  console.log('━'.repeat(60));
  
  if (!CONFIG.FAL_KEY) {
    console.log('   ⚠️ FAL_KEY not configured, falling back to Hallo2');
    return generateWithHallo2(imageUrl, audioUrl);
  }
  
  // Import fal.ai client dynamically
  const { fal } = await import('@fal-ai/client');
  fal.config({ credentials: CONFIG.FAL_KEY });
  
  console.log(`   Full Body: ${fullBody}`);
  console.log(`   Processing...`);
  
  const result = await fal.subscribe('fal-ai/omnihuman', {
    input: {
      image_url: imageUrl,
      audio_url: audioUrl,
      video_length: 'auto',
      resolution: '1080p',
    },
    logs: true,
    onQueueUpdate: (update: any) => {
      if (update.status === 'IN_PROGRESS') {
        process.stdout.write('.');
      }
    },
  });
  
  const videoUrl = (result as any)?.video?.url;
  console.log(`\n   ✅ Complete: ${videoUrl}`);
  
  return videoUrl;
}

async function generateWithHallo2(
  imageUrl: string,
  audioUrl: string
): Promise<string> {
  console.log('\n🎬 HALLO2 - Audio-Driven Portrait Animation');
  console.log('━'.repeat(60));
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  // Check for Hallo2 model availability
  try {
    const output = await replicate.run(
      "fudan-generative-vision/hallo2:4a1f98214e31c4a8e3d78c8c8c82c45f8c7f8e7c7d6e5f4a3b2c1d0e9f8a7b6c",
      {
        input: {
          source_image: imageUrl,
          driving_audio: audioUrl,
          pose_weight: 1.0,
          face_weight: 1.0,
          lip_weight: 1.0,
          face_expand_ratio: 1.2,
        }
      }
    );
    
    const videoUrl = typeof output === 'string' ? output : (output as any)?.output;
    console.log(`   ✅ Complete: ${videoUrl}`);
    return videoUrl;
    
  } catch (error: any) {
    console.log(`   ⚠️ Hallo2 unavailable: ${error.message}`);
    console.log('   Falling back to enhanced SadTalker...');
    return generateWithEnhancedSadTalker(imageUrl, audioUrl);
  }
}

// =============================================================================
// FALLBACK: WAV2LIP (Reliable baseline)
// =============================================================================

async function generateWithEnhancedSadTalker(
  imageUrl: string,
  audioPath: string
): Promise<string> {
  console.log('\n👄 WAV2LIP - Reliable Lip-Sync');
  console.log('━'.repeat(60));
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  // Convert local audio file to data URI if it's a local path
  let audioInput: string;
  if (audioPath.startsWith('http')) {
    audioInput = audioPath;
  } else if (fs.existsSync(audioPath)) {
    const audioBuffer = fs.readFileSync(audioPath);
    audioInput = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
    console.log(`   📤 Converting audio to data URI (${(audioBuffer.length / 1024).toFixed(1)} KB)`);
  } else {
    throw new Error(`Audio file not found: ${audioPath}`);
  }
  
  console.log(`   🎬 Running Wav2Lip...`);
  
  // Wav2Lip - proven reliable model
  const output = await replicate.run(
    "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
    {
      input: {
        face: imageUrl,
        audio: audioInput,
        fps: 25,
        pads: "0 10 0 0",
        smooth: true,
        resize_factor: 1,
      }
    }
  );
  
  const videoUrl = typeof output === 'string' ? output : (output as any);
  console.log(`   ✅ Complete: ${videoUrl}`);
  
  return videoUrl;
}

// =============================================================================
// IMAGE GENERATION - Kelly with LoRA
// =============================================================================

async function generateKellyImage(
  pose: string = 'excited',
  aspectRatio: string = '16:9'
): Promise<string> {
  console.log('\n🎨 Generating Kelly Image with LoRA');
  console.log('━'.repeat(60));
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  const poseConfig = KELLY.poses[pose] || KELLY.poses.excited;
  
  const fullPrompt = `${KELLY.identity}, ${poseConfig.prompt}, ${KELLY.style}`;
  
  console.log(`   Pose: ${pose}`);
  console.log(`   Prompt: ${fullPrompt.substring(0, 80)}...`);
  
  const output = await replicate.run(
    "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
    {
      input: {
        prompt: fullPrompt,
        hf_lora: CONFIG.KELLY_LORA_URL,
        lora_scale: 0.85,
        num_outputs: 1,
        aspect_ratio: aspectRatio,
        output_format: "png",
        guidance_scale: 3.5,
        output_quality: 100,
        num_inference_steps: 35,  // Higher for better quality
        disable_safety_checker: true,
      }
    }
  );
  
  // Extract string URL from output (SDK returns various types)
  let imageUrl: string;
  if (Array.isArray(output)) {
    imageUrl = String(output[0]);
  } else if (typeof output === 'object' && output !== null) {
    // Handle FileOutput object
    imageUrl = String((output as any).url || (output as any).toString());
  } else {
    imageUrl = String(output);
  }
  
  // Ensure we have a valid URL string
  if (!imageUrl.startsWith('http')) {
    throw new Error(`Invalid image URL returned: ${imageUrl}`);
  }
  
  console.log(`   ✅ Image generated: ${imageUrl}`);
  
  return imageUrl;
}

// =============================================================================
// AUDIO GENERATION - Kelly's Voice
// =============================================================================

interface AudioResult {
  localPath: string;
  publicUrl: string;
  buffer: Buffer;
}

async function generateKellyAudio(text: string): Promise<AudioResult> {
  console.log('\n🎤 Generating Kelly Audio (ElevenLabs)');
  console.log('━'.repeat(60));
  console.log(`   Text: "${text.substring(0, 60)}${text.length > 60 ? '...' : ''}"`);
  
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
          style: 0.2,  // Slight expressiveness
          use_speaker_boost: true,
        },
      }),
    }
  );
  
  if (!response.ok) {
    throw new Error(`ElevenLabs API error: ${response.status}`);
  }
  
  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const audioFileName = `kelly_audio_${Date.now()}.mp3`;
  const audioPath = path.join(CONFIG.OUTPUT_DIR, audioFileName);
  
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(audioPath, audioBuffer);
  
  console.log(`   ✅ Audio saved: ${audioPath} (${(audioBuffer.length / 1024).toFixed(1)} KB)`);
  
  // Upload to Supabase for public URL (required for Sync Labs)
  let publicUrl = audioPath;
  
  try {
    const { createClient } = await import('@supabase/supabase-js');
    const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    
    if (supabaseUrl && supabaseKey) {
      const supabase = createClient(supabaseUrl, supabaseKey);
      
      const { error } = await supabase.storage
        .from('kelly-templates')
        .upload(`sota-pipeline/${audioFileName}`, audioBuffer, {
          contentType: 'audio/mpeg',
          upsert: true,
        });
      
      if (!error) {
        const { data } = supabase.storage
          .from('kelly-templates')
          .getPublicUrl(`sota-pipeline/${audioFileName}`);
        
        publicUrl = data.publicUrl;
        console.log(`   ☁️ Uploaded to Supabase: ${publicUrl.substring(0, 60)}...`);
      }
    }
  } catch (e) {
    console.log(`   ⚠️ Supabase upload skipped`);
  }
  
  return {
    localPath: audioPath,
    publicUrl,
    buffer: audioBuffer,
  };
}

// =============================================================================
// VIDEO UPSCALING & POST-PROCESSING
// =============================================================================

async function upscaleVideo(videoUrl: string, scale: number = 4): Promise<string> {
  console.log('\n🔍 Upscaling Video to 4K');
  console.log('━'.repeat(60));
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  const output = await replicate.run(
    "lucataco/real-esrgan-video:c23768236472c41b7a121ee735c8073e29080c02d343419c4b7f0e56e045cb4d",
    {
      input: {
        video_path: videoUrl,
        scale: scale,
        face_enhance: true,  // Use GFPGAN on faces
      }
    }
  );
  
  const upscaledUrl = typeof output === 'string' ? output : (output as any)?.output;
  console.log(`   ✅ Upscaled: ${upscaledUrl}`);
  
  return upscaledUrl;
}

async function enhanceFaces(videoUrl: string): Promise<string> {
  console.log('\n✨ Enhancing Faces (CodeFormer)');
  console.log('━'.repeat(60));
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  const output = await replicate.run(
    "sczhou/codeformer:7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56",
    {
      input: {
        image: videoUrl,
        upscale: 2,
        face_upsample: true,
        background_enhance: true,
        codeformer_fidelity: 0.7,  // Balance between enhancement and identity
      }
    }
  );
  
  const enhancedUrl = typeof output === 'string' ? output : (output as any)?.output;
  console.log(`   ✅ Enhanced: ${enhancedUrl}`);
  
  return enhancedUrl;
}

// =============================================================================
// MAIN PIPELINE ORCHESTRATOR
// =============================================================================

type PipelineTier = 'sync' | 'hedra' | 'omnihuman' | 'best-available';

interface PipelineOptions {
  tier: PipelineTier;
  text: string;
  pose?: string;
  fullBody?: boolean;
  upscale?: boolean;
  existingImage?: string;
  existingAudio?: string;
}

interface PipelineResult {
  success: boolean;
  tier: string;
  imageUrl?: string;
  audioUrl?: string;
  videoUrl?: string;
  finalVideoUrl?: string;
  duration?: number;
  error?: string;
}

async function runSOTAPipeline(options: PipelineOptions): Promise<PipelineResult> {
  const startTime = Date.now();
  
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  🚀 SOTA KELLY VIDEO PIPELINE                                ║');
  console.log('║  Making Kelly the best digital human teacher on the planet   ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`  Tier: ${options.tier.toUpperCase()}`);
  console.log(`  Pose: ${options.pose || 'excited'}`);
  console.log(`  Text: "${options.text.substring(0, 50)}${options.text.length > 50 ? '...' : ''}"`);
  console.log(`  Upscale: ${options.upscale ? 'Yes (4K)' : 'No'}`);
  
  try {
    // Step 1: Generate or use existing image
    let imageUrl = options.existingImage;
    if (!imageUrl) {
      imageUrl = await generateKellyImage(options.pose || 'excited');
    }
    
    // Step 2: Generate or use existing audio
    let audioResult: AudioResult;
    if (options.existingAudio) {
      // Use existing audio
      const existingBuffer = options.existingAudio.startsWith('http') 
        ? Buffer.from('') 
        : (fs.existsSync(options.existingAudio) ? fs.readFileSync(options.existingAudio) : Buffer.from(''));
      audioResult = {
        localPath: options.existingAudio,
        publicUrl: options.existingAudio,
        buffer: existingBuffer,
      };
    } else {
      audioResult = await generateKellyAudio(options.text);
    }
    
    // For Wav2Lip/local models, use local path or data URI
    // For Sync Labs, use public URL
    const audioForLocal = audioResult.localPath;
    const audioForSyncLabs = audioResult.publicUrl;
    
    // Step 3: Generate video based on tier
    let videoUrl: string;
    let actualTier = options.tier;
    
    switch (options.tier) {
      case 'sync':
        // Sync Labs needs a base video first, generate with Wav2Lip then enhance
        console.log('\n   🌟 Using Sync Labs (95% accuracy)...');
        const syncBaseVideo = await generateWithEnhancedSadTalker(imageUrl, audioForLocal);
        videoUrl = await generateWithSyncLabs(syncBaseVideo, audioForSyncLabs, {
          model: 'lipsync-2',
        });
        break;
        
      case 'hedra':
        videoUrl = await generateWithHedra(
          imageUrl,
          audioForLocal,
          KELLY.poses[options.pose || 'excited']?.emotion || 'neutral'
        );
        break;
        
      case 'omnihuman':
        videoUrl = await generateWithOmniHuman(imageUrl, audioForLocal, options.fullBody);
        break;
        
      case 'best-available':
      default:
        // Try in order of preference - Sync Labs is now our primary tier!
        if (CONFIG.SYNC_LABS_API_KEY) {
          console.log('\n   🌟 Using Sync Labs (95% accuracy)...');
          // First generate base video with Wav2Lip
          const baseVideo = await generateWithEnhancedSadTalker(imageUrl, audioForLocal);
          // Then enhance with Sync Labs (needs public URL for audio)
          videoUrl = await generateWithSyncLabs(baseVideo, audioForSyncLabs, {
            model: 'lipsync-2',
          });
          actualTier = 'sync-labs';
        } else if (CONFIG.HEDRA_API_KEY) {
          videoUrl = await generateWithHedra(imageUrl, audioForLocal);
          actualTier = 'hedra';
        } else if (CONFIG.FAL_KEY) {
          videoUrl = await generateWithOmniHuman(imageUrl, audioForLocal);
          actualTier = 'omnihuman';
        } else {
          // Fallback to Wav2Lip only
          videoUrl = await generateWithEnhancedSadTalker(imageUrl, audioForLocal);
          actualTier = 'wav2lip';
        }
        break;
    }
    
    // Step 4: Optional upscaling
    let finalVideoUrl = videoUrl;
    if (options.upscale && videoUrl) {
      finalVideoUrl = await upscaleVideo(videoUrl, 4);
    }
    
    const duration = (Date.now() - startTime) / 1000;
    
    return {
      success: true,
      tier: actualTier,
      imageUrl,
      audioUrl: audioResult.publicUrl,
      videoUrl,
      finalVideoUrl,
      duration,
    };
    
  } catch (error: any) {
    return {
      success: false,
      tier: options.tier,
      error: error.message,
      duration: (Date.now() - startTime) / 1000,
    };
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function downloadFile(url: string, filepath: string): Promise<void> {
  const response = await fetch(url);
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(filepath, buffer);
}

// =============================================================================
// CLI INTERFACE
// =============================================================================

async function main() {
  // Parse CLI arguments
  const args = process.argv.slice(2);
  
  const options: PipelineOptions = {
    tier: 'best-available',
    text: "Hello! I'm Kelly, and I'm so excited to learn with you today. Let's discover something amazing together!",
    pose: 'excited',
    upscale: false,
  };
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--tier':
        options.tier = args[++i] as PipelineTier;
        break;
      case '--text':
        options.text = args[++i];
        break;
      case '--pose':
        options.pose = args[++i];
        break;
      case '--full-body':
        options.fullBody = true;
        break;
      case '--upscale':
        options.upscale = true;
        break;
      case '--image':
        options.existingImage = args[++i];
        break;
      case '--audio':
        options.existingAudio = args[++i];
        break;
      case '--help':
        console.log(`
SOTA Kelly Video Pipeline - Make Kelly the best digital human teacher

Usage:
  npx tsx sota-video-pipeline.ts [options]

Options:
  --tier <tier>       Pipeline tier: sync, hedra, omnihuman, best-available (default)
  --text <text>       Text for Kelly to speak
  --pose <pose>       Kelly's pose: excited, curious, explaining, thoughtful, heartfelt, welcome, celebrating
  --full-body         Enable full body animation (omnihuman only)
  --upscale           Upscale output to 4K
  --image <url>       Use existing image URL
  --audio <url>       Use existing audio URL
  --help              Show this help

Tiers (in order of quality):
  sync        - Sync Labs lipsync-2-pro (95%+ lip accuracy, 4K)
  hedra       - Hedra Character-1 (full face animation)
  omnihuman   - OmniHuman (full body, next-gen)
  best-available - Auto-select best available based on API keys

Examples:
  npx tsx sota-video-pipeline.ts --tier sync --text "Welcome to today's lesson!"
  npx tsx sota-video-pipeline.ts --tier hedra --pose curious --upscale
  npx tsx sota-video-pipeline.ts --tier omnihuman --full-body

API Keys Required (add to .env):
  REPLICATE_API_TOKEN   - Required for all tiers
  ELEVENLABS_API_KEY    - Required for audio generation
  SYNC_LABS_API_KEY     - For Sync Labs tier (https://sync.so)
  HEDRA_API_KEY         - For Hedra tier (https://hedra.com)
  FAL_KEY               - For OmniHuman tier (https://fal.ai)
        `);
        process.exit(0);
    }
  }
  
  // Validate required keys
  console.log('\n🔑 Checking API Keys...');
  console.log(`   REPLICATE: ${CONFIG.REPLICATE_API_TOKEN ? '✅' : '❌ Required'}`);
  console.log(`   ELEVENLABS: ${CONFIG.ELEVENLABS_API_KEY ? '✅' : '❌ Required'}`);
  console.log(`   SYNC_LABS: ${CONFIG.SYNC_LABS_API_KEY ? '✅' : '⚪ Optional'}`);
  console.log(`   HEDRA: ${CONFIG.HEDRA_API_KEY ? '✅' : '⚪ Optional'}`);
  console.log(`   FAL: ${CONFIG.FAL_KEY ? '✅' : '⚪ Optional'}`);
  
  if (!CONFIG.REPLICATE_API_TOKEN || !CONFIG.ELEVENLABS_API_KEY) {
    console.error('\n❌ Missing required API keys. Add them to .env');
    process.exit(1);
  }
  
  // Run pipeline
  const result = await runSOTAPipeline(options);
  
  // Print results
  console.log('\n');
  console.log('═'.repeat(64));
  
  if (result.success) {
    console.log('✅ PIPELINE COMPLETE');
    console.log('═'.repeat(64));
    console.log(`   Tier Used: ${result.tier}`);
    console.log(`   Duration: ${result.duration?.toFixed(1)}s`);
    console.log('');
    console.log('   📁 Output:');
    console.log(`      Image: ${result.imageUrl}`);
    console.log(`      Audio: ${result.audioUrl}`);
    console.log(`      Video: ${result.videoUrl}`);
    if (result.finalVideoUrl !== result.videoUrl) {
      console.log(`      Final (4K): ${result.finalVideoUrl}`);
    }
  } else {
    console.log('❌ PIPELINE FAILED');
    console.log('═'.repeat(64));
    console.log(`   Error: ${result.error}`);
    console.log(`   Duration: ${result.duration?.toFixed(1)}s`);
  }
  
  console.log('═'.repeat(64));
}

// Export for programmatic use
export {
  runSOTAPipeline,
  generateKellyImage,
  generateKellyAudio,
  generateWithSyncLabs,
  generateWithHedra,
  generateWithLivePortrait,
  generateWithOmniHuman,
  generateWithHallo2,
  generateWithEnhancedSadTalker,
  upscaleVideo,
  enhanceFaces,
  CONFIG,
  KELLY,
};

// Run if called directly
main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});

