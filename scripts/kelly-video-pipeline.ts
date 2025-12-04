/**
 * 🎬 Kelly Video Production Pipeline
 * 
 * High-quality lip-sync using:
 * 1. Kelly LoRA + Flux for perfect Kelly frames
 * 2. Wav2Lip HD for 95%+ accurate lip-sync
 * 3. RealESRGAN for 4K upscaling
 * 
 * Usage:
 *   npx ts-node scripts/kelly-video-pipeline.ts
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { fal } from '@fal-ai/client';
import * as fs from 'fs';
import * as path from 'path';

// Configuration
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const FAL_KEY = process.env.FAL_KEY;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

// Kelly LoRA configuration (from your existing setup)
const KELLY_LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

const OUTPUT_DIR = path.join(process.cwd(), 'test-output', 'production');
const KELLY_IMAGE_PATH = 'public/kelly/poses/kelly_welcome.png';

const TEST_TEXT = "Hello! I'm Kelly, your learning companion. Today we're going to explore something truly fascinating. Are you ready to discover the wonders of our world together?";

interface PipelineResult {
  audioPath: string;
  baseImagePath: string;
  lipSyncVideoPath: string;
  finalVideoPath: string;
  metrics: {
    totalTime: number;
    audioTime: number;
    imageTime: number;
    lipSyncTime: number;
    upscaleTime: number;
  };
}

/**
 * Step 1: Generate high-quality TTS audio
 */
async function generateAudio(text: string): Promise<{ path: string; duration: number }> {
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📢 STEP 1: Generating Kelly\'s voice...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const startTime = Date.now();

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY!,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.85,
          style: 0.1,
          use_speaker_boost: true,
        },
      }),
    }
  );

  if (!response.ok) throw new Error(`TTS error: ${response.status}`);

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const audioPath = path.join(OUTPUT_DIR, 'kelly-voice.mp3');
  fs.writeFileSync(audioPath, audioBuffer);
  
  const duration = (Date.now() - startTime) / 1000;
  console.log(`   ✅ Audio generated: ${(audioBuffer.length / 1024).toFixed(1)} KB`);
  console.log(`   ⏱️  Time: ${duration.toFixed(1)}s`);
  
  return { path: audioPath, duration };
}

/**
 * Step 2: Generate Kelly image with Flux + LoRA (optional - use existing if good)
 */
async function generateKellyImage(replicate: Replicate): Promise<{ path: string; duration: number }> {
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🎨 STEP 2: Preparing Kelly image...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const startTime = Date.now();

  // Check if we have a good source image
  if (fs.existsSync(KELLY_IMAGE_PATH)) {
    const stats = fs.statSync(KELLY_IMAGE_PATH);
    if (stats.size > 500000) { // > 500KB = high quality
      console.log(`   ✅ Using existing Kelly image: ${KELLY_IMAGE_PATH}`);
      console.log(`   📐 Size: ${(stats.size / 1024).toFixed(1)} KB`);
      return { path: KELLY_IMAGE_PATH, duration: 0 };
    }
  }

  // Generate fresh Kelly image with LoRA
  console.log('   🎨 Generating fresh Kelly image with Flux + LoRA...');
  
  try {
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: "curious_kelly, professional photo portrait, warm friendly smile, looking at camera, soft studio lighting, high resolution, 4k, detailed face, clear skin, photorealistic",
          hf_lora: KELLY_LORA_URL,
          lora_scale: 0.9,
          num_outputs: 1,
          aspect_ratio: "1:1",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28,
        }
      }
    );

    const imageUrl = Array.isArray(output) ? output[0] : output;
    if (typeof imageUrl === 'string') {
      const response = await fetch(imageUrl);
      const buffer = Buffer.from(await response.arrayBuffer());
      const imagePath = path.join(OUTPUT_DIR, 'kelly-generated.png');
      fs.writeFileSync(imagePath, buffer);
      
      const duration = (Date.now() - startTime) / 1000;
      console.log(`   ✅ Generated Kelly image: ${(buffer.length / 1024).toFixed(1)} KB`);
      console.log(`   ⏱️  Time: ${duration.toFixed(1)}s`);
      return { path: imagePath, duration };
    }
  } catch (error: any) {
    console.log(`   ⚠️ LoRA generation failed: ${error.message}`);
    console.log('   📷 Falling back to existing image');
  }

  return { path: KELLY_IMAGE_PATH, duration: 0 };
}

/**
 * Step 3: Apply Wav2Lip HD for accurate lip-sync
 */
async function applyWav2Lip(
  replicate: Replicate,
  imagePath: string,
  audioPath: string
): Promise<{ path: string; duration: number }> {
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('👄 STEP 3: Applying Wav2Lip HD lip-sync...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const startTime = Date.now();

  // Convert files to base64 data URLs
  const imageBuffer = fs.readFileSync(imagePath);
  const audioBuffer = fs.readFileSync(audioPath);
  
  const imageBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
  const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;

  console.log('   📤 Uploading to Replicate...');
  console.log(`   📷 Image: ${(imageBuffer.length / 1024).toFixed(1)} KB`);
  console.log(`   🔊 Audio: ${(audioBuffer.length / 1024).toFixed(1)} KB`);

  // Try Wav2Lip models in order of quality
  const wav2lipModels = [
    {
      name: 'Wav2Lip (devxpy)',
      model: "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
      input: (face: string, audio: string) => ({
        face,
        audio,
        fps: 30,
        pads: "0 10 0 0",
        smooth: true,
        resize_factor: 1,
      }),
    },
    {
      name: 'Video Retalking',
      model: "chenxwh/video-retalking:db5a650c807b007dc5f9e5abe27c53e1b62880d1f94d218d27ce7fa802711d67",
      input: (face: string, audio: string) => ({
        face,
        input_audio: audio,
      }),
    },
  ];

  for (const model of wav2lipModels) {
    console.log(`\n   🔄 Trying ${model.name}...`);
    
    try {
      const output = await replicate.run(model.model as `${string}/${string}:${string}`, {
        input: model.input(imageBase64, audioBase64),
      });

      const videoUrl = typeof output === 'string' ? output : (output as any)?.output;
      
      if (videoUrl && typeof videoUrl === 'string') {
        console.log('   📥 Downloading video...');
        const response = await fetch(videoUrl);
        const videoBuffer = Buffer.from(await response.arrayBuffer());
        const videoPath = path.join(OUTPUT_DIR, 'kelly-lipsync.mp4');
        fs.writeFileSync(videoPath, videoBuffer);
        
        const duration = (Date.now() - startTime) / 1000;
        console.log(`   ✅ Lip-sync video: ${(videoBuffer.length / 1024).toFixed(1)} KB`);
        console.log(`   ⏱️  Time: ${duration.toFixed(1)}s`);
        return { path: videoPath, duration };
      }
    } catch (error: any) {
      console.log(`   ❌ ${model.name} failed: ${error.message?.substring(0, 50)}`);
    }
  }

  // Fallback to fal.ai SadTalker if Replicate fails
  console.log('\n   🔄 Falling back to fal.ai SadTalker...');
  return await fallbackToSadTalker(imagePath, audioPath, startTime);
}

/**
 * Fallback: Use fal.ai SadTalker
 */
async function fallbackToSadTalker(
  imagePath: string,
  audioPath: string,
  startTime: number
): Promise<{ path: string; duration: number }> {
  // Upload to fal.ai
  const imageBuffer = fs.readFileSync(imagePath);
  const audioBuffer = fs.readFileSync(audioPath);
  
  const imageUrl = await fal.storage.upload(new Blob([imageBuffer], { type: 'image/png' }));
  const audioUrl = await fal.storage.upload(new Blob([audioBuffer], { type: 'audio/mpeg' }));

  const result = await fal.subscribe('fal-ai/sadtalker', {
    input: {
      source_image_url: imageUrl,
      driven_audio_url: audioUrl,
      enhancer: 'gfpgan',
      preprocess: 'full',
      still_mode: false,
      expression_scale: 1.0,
      face_model_resolution: '512',
    },
    logs: false,
  });

  const videoUrl = (result as any)?.video?.url || (result as any)?.data?.video?.url;
  if (!videoUrl) throw new Error('No video from SadTalker');

  const response = await fetch(videoUrl);
  const videoBuffer = Buffer.from(await response.arrayBuffer());
  const videoPath = path.join(OUTPUT_DIR, 'kelly-lipsync.mp4');
  fs.writeFileSync(videoPath, videoBuffer);
  
  const duration = (Date.now() - startTime) / 1000;
  console.log(`   ✅ Lip-sync video (SadTalker): ${(videoBuffer.length / 1024).toFixed(1)} KB`);
  return { path: videoPath, duration };
}

/**
 * Step 4: Upscale to 4K
 */
async function upscaleTo4K(videoPath: string): Promise<{ path: string; duration: number }> {
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🔍 STEP 4: Upscaling to 4K...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const startTime = Date.now();

  const videoBuffer = fs.readFileSync(videoPath);
  const videoUrl = await fal.storage.upload(new Blob([videoBuffer], { type: 'video/mp4' }));

  console.log('   🔄 Running RealESRGAN 4x upscaler...');

  const result = await fal.subscribe('fal-ai/video-upscaler', {
    input: {
      video_url: videoUrl,
      scale: 4,
    },
    logs: false,
    onQueueUpdate: (update: any) => {
      if (update.status === 'IN_PROGRESS') {
        process.stdout.write('.');
      }
    },
  });

  console.log('');

  const upscaledUrl = (result as any)?.video?.url || (result as any)?.data?.video?.url;
  if (!upscaledUrl) throw new Error('No upscaled video URL');

  const response = await fetch(upscaledUrl);
  const upscaledBuffer = Buffer.from(await response.arrayBuffer());
  const finalPath = path.join(OUTPUT_DIR, 'kelly-4k-final.mp4');
  fs.writeFileSync(finalPath, upscaledBuffer);
  
  const duration = (Date.now() - startTime) / 1000;
  console.log(`   ✅ 4K video: ${(upscaledBuffer.length / 1024 / 1024).toFixed(2)} MB`);
  console.log(`   ⏱️  Time: ${duration.toFixed(1)}s`);
  
  return { path: finalPath, duration };
}

/**
 * Main pipeline
 */
async function runPipeline(): Promise<PipelineResult> {
  console.log('');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 KELLY VIDEO PRODUCTION PIPELINE                          ║');
  console.log('║  High-Quality Lip-Sync with LoRA + Wav2Lip + 4K Upscale     ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('');

  // Validate environment
  if (!ELEVENLABS_API_KEY) throw new Error('ELEVENLABS_API_KEY required');
  if (!FAL_KEY) throw new Error('FAL_KEY required');
  
  const hasReplicate = !!REPLICATE_API_TOKEN;
  console.log(`✅ ElevenLabs: Configured`);
  console.log(`✅ fal.ai: Configured`);
  console.log(`${hasReplicate ? '✅' : '⚠️'} Replicate: ${hasReplicate ? 'Configured' : 'Not configured (will use fal.ai)'}`);

  // Initialize
  const replicate = hasReplicate ? new Replicate({ auth: REPLICATE_API_TOKEN }) : null;
  
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const totalStart = Date.now();
  const metrics = { totalTime: 0, audioTime: 0, imageTime: 0, lipSyncTime: 0, upscaleTime: 0 };

  // Step 1: Audio
  const audio = await generateAudio(TEST_TEXT);
  metrics.audioTime = audio.duration;

  // Step 2: Image
  const image = replicate 
    ? await generateKellyImage(replicate)
    : { path: KELLY_IMAGE_PATH, duration: 0 };
  metrics.imageTime = image.duration;

  // Step 3: Lip-sync
  const lipSync = replicate
    ? await applyWav2Lip(replicate, image.path, audio.path)
    : await fallbackToSadTalker(image.path, audio.path, Date.now());
  metrics.lipSyncTime = lipSync.duration;

  // Step 4: Upscale
  const final = await upscaleTo4K(lipSync.path);
  metrics.upscaleTime = final.duration;

  metrics.totalTime = (Date.now() - totalStart) / 1000;

  return {
    audioPath: audio.path,
    baseImagePath: image.path,
    lipSyncVideoPath: lipSync.path,
    finalVideoPath: final.path,
    metrics,
  };
}

/**
 * Main
 */
async function main() {
  try {
    const result = await runPipeline();

    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════╗');
    console.log('║  ✅ PIPELINE COMPLETE                                        ║');
    console.log('╚══════════════════════════════════════════════════════════════╝');
    console.log('');
    console.log('📊 Performance Metrics:');
    console.log(`   Audio generation:  ${result.metrics.audioTime.toFixed(1)}s`);
    console.log(`   Image preparation: ${result.metrics.imageTime.toFixed(1)}s`);
    console.log(`   Lip-sync:          ${result.metrics.lipSyncTime.toFixed(1)}s`);
    console.log(`   4K Upscale:        ${result.metrics.upscaleTime.toFixed(1)}s`);
    console.log(`   ─────────────────────────`);
    console.log(`   Total:             ${(result.metrics.totalTime / 60).toFixed(1)} minutes`);
    console.log('');
    console.log('📁 Output Files:');
    console.log(`   ${result.audioPath}`);
    console.log(`   ${result.lipSyncVideoPath}`);
    console.log(`   ${result.finalVideoPath} ← FINAL 4K VIDEO`);
    console.log('');

    // Open final video
    const { exec } = require('child_process');
    exec(`start "" "${result.finalVideoPath}"`);

  } catch (error) {
    console.error('\n❌ Pipeline failed:', error);
    process.exit(1);
  }
}

main();

