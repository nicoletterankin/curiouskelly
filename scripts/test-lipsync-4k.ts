/**
 * 4K Lip-Sync Video Generation
 * 
 * 1. Generate lip-sync with SadTalker
 * 2. Upscale to 4K using fal.ai's Video Upscaler (RealESRGAN)
 * 
 * Usage:
 *   npx ts-node scripts/test-lipsync-4k.ts
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
const KELLY_IMAGE_PATH = 'public/kelly/poses/kelly_welcome.png';

/**
 * Upload file to fal.ai storage
 */
async function uploadToFal(filePath: string, contentType: string): Promise<string> {
  console.log(`   📤 Uploading ${path.basename(filePath)}...`);
  const fileBuffer = fs.readFileSync(filePath);
  const blob = new Blob([fileBuffer], { type: contentType });
  const url = await fal.storage.upload(blob);
  console.log(`   ✅ Uploaded`);
  return url;
}

/**
 * Generate TTS audio
 */
async function generateAudio(text: string): Promise<string> {
  console.log('\n🎤 Step 1: Generating TTS audio...');
  
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
          similarity_boost: 0.8,
          style: 0.0,
          use_speaker_boost: true,
        },
      }),
    }
  );

  if (!response.ok) throw new Error(`TTS error: ${response.status}`);

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const audioPath = path.join(OUTPUT_DIR, 'kelly-speech.mp3');
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   ✅ Audio: ${audioPath} (${(audioBuffer.length / 1024).toFixed(1)} KB)`);
  
  return audioPath;
}

/**
 * Generate lip-sync video with SadTalker (best quality settings)
 */
async function generateLipSync(imageUrl: string, audioUrl: string): Promise<string> {
  console.log('\n🎬 Step 2: Generating lip-sync video...');
  console.log('   Using SadTalker with GFPGAN enhancement...');

  const result = await fal.subscribe('fal-ai/sadtalker', {
    input: {
      source_image_url: imageUrl,
      driven_audio_url: audioUrl,
      enhancer: 'gfpgan',           // Face enhancement
      preprocess: 'full',            // Full face processing
      still_mode: false,
      expression_scale: 1.0,
      pose_style: 0,
      batch_size: 2,
      face_model_resolution: '512', // Higher base resolution
    },
    logs: false,
    onQueueUpdate: (update: any) => {
      if (update.status === 'IN_PROGRESS') {
        process.stdout.write('.');
      }
    },
  });

  console.log('');
  
  const videoUrl = (result as any)?.video?.url || (result as any)?.data?.video?.url;
  if (!videoUrl) throw new Error('No video URL in response');

  // Download
  console.log('   📥 Downloading base video...');
  const videoResponse = await fetch(videoUrl);
  const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
  const videoPath = path.join(OUTPUT_DIR, 'kelly-lipsync-base.mp4');
  fs.writeFileSync(videoPath, videoBuffer);
  console.log(`   ✅ Base video: ${videoPath} (${(videoBuffer.length / 1024).toFixed(1)} KB)`);

  return videoUrl;
}

/**
 * Upscale video to 4K using RealESRGAN
 */
async function upscaleVideo(videoUrl: string): Promise<string> {
  console.log('\n🔍 Step 3: Upscaling to 4K...');
  console.log('   Using RealESRGAN video upscaler...');

  try {
    const result = await fal.subscribe('fal-ai/video-upscaler', {
      input: {
        video_url: videoUrl,
        scale: 4,  // 4x upscale for 4K
      },
      logs: false,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') {
          process.stdout.write('.');
        }
      },
    });

    console.log('');
    console.log('   Result:', JSON.stringify(result, null, 2).substring(0, 300));

    const upscaledUrl = (result as any)?.video?.url || (result as any)?.data?.video?.url || (result as any)?.output?.url;
    
    if (upscaledUrl) {
      console.log('   📥 Downloading 4K video...');
      const videoResponse = await fetch(upscaledUrl);
      const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
      const videoPath = path.join(OUTPUT_DIR, 'kelly-4k.mp4');
      fs.writeFileSync(videoPath, videoBuffer);
      console.log(`   ✅ 4K video: ${videoPath} (${(videoBuffer.length / 1024 / 1024).toFixed(2)} MB)`);
      return videoPath;
    }
  } catch (error: any) {
    console.log(`\n   ⚠️ Upscaler error: ${error.message}`);
    console.log('   Trying alternative upscaler...');
    
    // Try alternative upscaler endpoint
    try {
      const result = await fal.subscribe('fal-ai/esrgan', {
        input: {
          video_url: videoUrl,
          scale: 4,
        },
        logs: false,
      });
      
      const url = (result as any)?.video?.url || (result as any)?.output?.url;
      if (url) {
        const videoResponse = await fetch(url);
        const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
        const videoPath = path.join(OUTPUT_DIR, 'kelly-4k.mp4');
        fs.writeFileSync(videoPath, videoBuffer);
        console.log(`   ✅ 4K video: ${videoPath} (${(videoBuffer.length / 1024 / 1024).toFixed(2)} MB)`);
        return videoPath;
      }
    } catch (e: any) {
      console.log(`   ❌ Alternative upscaler also failed: ${e.message}`);
    }
  }

  return '';
}

/**
 * Main function
 */
async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🎬 4K LIP-SYNC VIDEO GENERATION');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  if (!FAL_KEY || !ELEVENLABS_API_KEY) {
    console.error('❌ Missing API keys!');
    process.exit(1);
  }

  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const startTime = Date.now();

  // Step 1: Generate audio
  const audioPath = await generateAudio(TEST_TEXT);

  // Upload assets
  console.log('\n📤 Uploading assets to fal.ai...');
  const imageUrl = await uploadToFal(KELLY_IMAGE_PATH, 'image/png');
  const audioUrl = await uploadToFal(audioPath, 'audio/mpeg');

  // Step 2: Generate lip-sync
  const baseVideoUrl = await generateLipSync(imageUrl, audioUrl);

  // Step 3: Upscale to 4K
  const finalPath = await upscaleVideo(baseVideoUrl);

  const totalTime = ((Date.now() - startTime) / 1000 / 60).toFixed(1);

  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`   ✅ COMPLETE! Total time: ${totalTime} minutes`);
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');
  console.log('   Generated files:');
  console.log('   📁 test-output/kelly-speech.mp3      - TTS audio');
  console.log('   📁 test-output/kelly-lipsync-base.mp4 - Base lip-sync');
  if (finalPath) {
    console.log('   📁 test-output/kelly-4k.mp4          - 4K upscaled video');
  }
  console.log('');
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});

