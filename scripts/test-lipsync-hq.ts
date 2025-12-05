/**
 * High-Quality Lip-Sync Video Generation Test
 * 
 * Tests multiple models to find the best quality output:
 * - LivePortrait (best image preservation)
 * - Hedra Character-1 (high quality avatars)
 * - LatentSync (newer, better quality)
 * - SadTalker (baseline)
 * 
 * Usage:
 *   npx ts-node scripts/test-lipsync-hq.ts
 */

import 'dotenv/config';
import { fal } from '@fal-ai/client';
import * as fs from 'fs';
import * as path from 'path';

const FAL_KEY = process.env.FAL_KEY;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

const TEST_TEXT = "Hello! I'm Kelly, and I'm so excited to learn with you today! Let's discover something amazing together.";
const OUTPUT_DIR = path.join(process.cwd(), 'test-output');

// Use high-res Kelly image
const KELLY_IMAGE_PATH = 'public/kelly/poses/kelly_welcome.png';

interface ModelConfig {
  name: string;
  endpoint: string;
  inputFormat: (imageUrl: string, audioUrl: string) => Record<string, any>;
  extractVideo: (result: any) => string | null;
}

const MODELS: ModelConfig[] = [
  {
    name: 'LivePortrait',
    endpoint: 'fal-ai/liveportrait',
    inputFormat: (imageUrl, audioUrl) => ({
      image_url: imageUrl,
      video_url: audioUrl, // LivePortrait can use audio as driving
      dsize: 1024, // Higher resolution
      scale: 2.3,
      vx_ratio: 0,
      vy_ratio: -0.125,
      flag_relative: true,
      flag_pasteback: true,
      flag_do_crop: true,
      flag_lip_sync: true,
    }),
    extractVideo: (result) => result?.video?.url || result?.data?.video?.url,
  },
  {
    name: 'Hedra Character-1',
    endpoint: 'fal-ai/hedra/character-1',
    inputFormat: (imageUrl, audioUrl) => ({
      image_url: imageUrl,
      audio_url: audioUrl,
      aspect_ratio: '1:1',
    }),
    extractVideo: (result) => result?.video?.url || result?.data?.video?.url || result?.video,
  },
  {
    name: 'LatentSync',
    endpoint: 'fal-ai/latentsync',
    inputFormat: (imageUrl, audioUrl) => ({
      image_url: imageUrl,
      audio_url: audioUrl,
      guidance_scale: 1.5,
      inference_steps: 25,
      seed: 42,
    }),
    extractVideo: (result) => result?.video?.url || result?.data?.video?.url,
  },
  {
    name: 'SyncLipsync',
    endpoint: 'fal-ai/sync-lipsync',
    inputFormat: (imageUrl, audioUrl) => ({
      image_url: imageUrl,
      audio_url: audioUrl,
      sync_mode: 'cut_off', // More accurate sync
    }),
    extractVideo: (result) => result?.video?.url || result?.data?.video?.url || result?.video,
  },
  {
    name: 'SadTalker (Enhanced)',
    endpoint: 'fal-ai/sadtalker',
    inputFormat: (imageUrl, audioUrl) => ({
      source_image_url: imageUrl,
      driven_audio_url: audioUrl,
      enhancer: 'gfpgan', // Face enhancement for better quality
      preprocess: 'full', // Full face processing
      still_mode: false,
      expression_scale: 1.0,
      pose_style: 0, // Natural pose
      ref_pose_video_url: null,
      batch_size: 2, // Better quality
      face_model_resolution: '512', // Higher resolution
    }),
    extractVideo: (result) => result?.video?.url || result?.data?.video?.url || (typeof result === 'string' ? result : null),
  },
];

/**
 * Upload file to fal.ai storage
 */
async function uploadToFal(filePath: string, contentType: string): Promise<string> {
  console.log(`   📤 Uploading ${path.basename(filePath)}...`);
  const fileBuffer = fs.readFileSync(filePath);
  const blob = new Blob([fileBuffer], { type: contentType });
  const url = await fal.storage.upload(blob);
  console.log(`   ✅ Uploaded: ${url.substring(0, 60)}...`);
  return url;
}

/**
 * Generate TTS audio with highest quality settings
 */
async function generateHighQualityAudio(text: string): Promise<string> {
  console.log('\n🎤 Generating high-quality TTS audio...');
  
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
          similarity_boost: 0.8, // Higher for better voice match
          style: 0.0,
          use_speaker_boost: true,
        },
        output_format: 'mp3_44100_128', // Higher quality audio
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`TTS API error: ${response.status}`);
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const audioPath = path.join(OUTPUT_DIR, 'kelly-speech-hq.mp3');
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   ✅ Audio saved: ${audioPath} (${audioBuffer.length} bytes)`);
  
  return audioPath;
}

/**
 * Try a single model
 */
async function tryModel(
  model: ModelConfig,
  imageUrl: string,
  audioUrl: string
): Promise<{ success: boolean; videoPath?: string; error?: string; duration?: number }> {
  console.log(`\n🎬 Trying ${model.name}...`);
  const startTime = Date.now();

  try {
    const input = model.inputFormat(imageUrl, audioUrl);
    console.log(`   Input params:`, JSON.stringify(input, null, 2).substring(0, 200));

    const result = await fal.subscribe(model.endpoint, {
      input,
      logs: false,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') {
          console.log(`   ⏳ Processing...`);
        }
      },
    });

    const duration = (Date.now() - startTime) / 1000;
    console.log(`   ⏱️ Completed in ${duration.toFixed(1)}s`);
    console.log(`   Result:`, JSON.stringify(result, null, 2).substring(0, 300));

    const videoUrl = model.extractVideo(result);
    
    if (videoUrl) {
      // Download video
      console.log(`   📥 Downloading video...`);
      const videoResponse = await fetch(videoUrl);
      if (videoResponse.ok) {
        const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
        const safeName = model.name.toLowerCase().replace(/[^a-z0-9]/g, '-');
        const videoPath = path.join(OUTPUT_DIR, `kelly-${safeName}.mp4`);
        fs.writeFileSync(videoPath, videoBuffer);
        console.log(`   ✅ Saved: ${videoPath} (${(videoBuffer.length / 1024).toFixed(1)} KB)`);
        return { success: true, videoPath, duration };
      }
    }

    return { success: false, error: 'No video URL in response', duration };
  } catch (error: any) {
    const duration = (Date.now() - startTime) / 1000;
    console.log(`   ❌ Failed (${duration.toFixed(1)}s): ${error.message?.substring(0, 100)}`);
    return { success: false, error: error.message, duration };
  }
}

/**
 * Main function
 */
async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🎬 HIGH-QUALITY LIP-SYNC TEST');
  console.log('   Testing multiple models for best quality');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  // Check API keys
  if (!FAL_KEY) {
    console.error('❌ FAL_KEY not set!');
    process.exit(1);
  }
  if (!ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY not set!');
    process.exit(1);
  }
  console.log('✅ API keys configured');

  // Create output directory
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Check Kelly image
  if (!fs.existsSync(KELLY_IMAGE_PATH)) {
    console.error(`❌ Image not found: ${KELLY_IMAGE_PATH}`);
    process.exit(1);
  }

  // Get image dimensions
  const imageStats = fs.statSync(KELLY_IMAGE_PATH);
  console.log(`📷 Source image: ${KELLY_IMAGE_PATH} (${(imageStats.size / 1024).toFixed(1)} KB)`);

  // Generate audio
  const audioPath = await generateHighQualityAudio(TEST_TEXT);

  // Upload to fal.ai
  console.log('\n📤 Uploading assets to fal.ai...');
  const imageUrl = await uploadToFal(KELLY_IMAGE_PATH, 'image/png');
  const audioUrl = await uploadToFal(audioPath, 'audio/mpeg');

  // Try each model
  const results: Array<{ model: string; success: boolean; path?: string; error?: string; duration?: number }> = [];

  for (const model of MODELS) {
    const result = await tryModel(model, imageUrl, audioUrl);
    results.push({
      model: model.name,
      success: result.success,
      path: result.videoPath,
      error: result.error,
      duration: result.duration,
    });
  }

  // Summary
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   📊 RESULTS SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  console.log('✅ Successful:');
  for (const r of successful) {
    console.log(`   ${r.model}: ${r.path} (${r.duration?.toFixed(1)}s)`);
  }

  if (failed.length > 0) {
    console.log('\n❌ Failed:');
    for (const r of failed) {
      console.log(`   ${r.model}: ${r.error?.substring(0, 50)}`);
    }
  }

  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`   Generated ${successful.length} videos in test-output/`);
  console.log('   Compare them to find the best quality!');
  console.log('═══════════════════════════════════════════════════════════════');
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});


