/**
 * Test script for fal.ai Omnihuman 1.5 lip-sync video generation
 * 
 * This uses ByteDance's Omnihuman model hosted on fal.ai
 * 
 * Usage:
 *   1. Get a FAL API key from https://fal.ai/dashboard/keys
 *   2. Set FAL_KEY in .env.local
 *   3. Run: npx ts-node scripts/test-fal-omnihuman.ts
 */

import 'dotenv/config';
import { fal } from '@fal-ai/client';
import * as fs from 'fs';
import * as path from 'path';

// Configuration
const FAL_KEY = process.env.FAL_KEY;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

const TEST_TEXT = "Hello! I'm Kelly, and I'm so excited to learn with you today!";
const OUTPUT_DIR = path.join(process.cwd(), 'test-output');

// Image to use (you may need to host this publicly or use fal's upload)
const KELLY_IMAGE_PATH = 'public/kelly/poses/kelly_welcome.png';

/**
 * Upload a file to fal.ai storage (returns a public URL)
 */
async function uploadToFal(filePath: string, contentType: string): Promise<string> {
  console.log(`📤 Uploading ${path.basename(filePath)} to fal.ai...`);
  
  const fileBuffer = fs.readFileSync(filePath);
  const blob = new Blob([fileBuffer], { type: contentType });
  
  // @ts-ignore - fal.storage.upload exists
  const url = await fal.storage.upload(blob);
  console.log(`   ✅ Uploaded: ${url}`);
  return url;
}

/**
 * Generate TTS audio using ElevenLabs
 */
async function generateAudio(text: string): Promise<string> {
  console.log('\n📢 Generating TTS audio with ElevenLabs...');
  
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
          similarity_boost: 0.75,
        },
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`TTS API error: ${response.status}`);
  }

  const audioBuffer = await response.arrayBuffer();
  const audioPath = path.join(OUTPUT_DIR, 'kelly-speech.mp3');
  fs.writeFileSync(audioPath, Buffer.from(audioBuffer));
  console.log(`   ✅ Audio saved: ${audioPath} (${audioBuffer.byteLength} bytes)`);
  
  return audioPath;
}

/**
 * Generate lip-sync video using fal.ai Omnihuman 1.5
 */
async function generateVideo(imageUrl: string, audioUrl: string): Promise<string | null> {
  console.log('\n🎬 Generating lip-sync video with Omnihuman 1.5...');
  console.log(`   Image: ${imageUrl.substring(0, 50)}...`);
  console.log(`   Audio: ${audioUrl.substring(0, 50)}...`);

  try {
    // @ts-ignore - fal types may be incomplete
    const result = await fal.subscribe('fal-ai/omnihuman', {
      input: {
        image_url: imageUrl,
        audio_url: audioUrl,
      },
      logs: true,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS' && update.logs) {
          update.logs.forEach((log: any) => console.log(`   📊 ${log.message}`));
        } else {
          console.log(`   📊 Status: ${update.status}`);
        }
      },
    });

    console.log('\n   ✅ Generation complete!');
    console.log('   Result:', JSON.stringify(result, null, 2).substring(0, 500));

    // Extract video URL from result
    const videoUrl = result.data?.video?.url || result.data?.video_url || result.video?.url;
    
    if (videoUrl) {
      console.log(`\n📥 Downloading video from: ${videoUrl}`);
      
      const videoResponse = await fetch(videoUrl);
      if (videoResponse.ok) {
        const videoBuffer = await videoResponse.arrayBuffer();
        const videoPath = path.join(OUTPUT_DIR, 'kelly-talking.mp4');
        fs.writeFileSync(videoPath, Buffer.from(videoBuffer));
        console.log(`   ✅ Video saved: ${videoPath} (${videoBuffer.byteLength} bytes)`);
        return videoPath;
      }
    }

    return null;
  } catch (error: any) {
    console.error('   ❌ Video generation failed:', error.message);
    
    // Check if it's an API error with details
    if (error.body) {
      console.error('   API Response:', JSON.stringify(error.body, null, 2));
    }
    
    return null;
  }
}

/**
 * Try alternative fal.ai models for lip-sync
 */
async function tryAlternativeModels(imageUrl: string, audioUrl: string) {
  const models = [
    'fal-ai/sadtalker',
    'fal-ai/wav2lip', 
    'fal-ai/liveportrait',
    'fal-ai/sync-lipsync',
    'fal-ai/latent-sync',
    'fal-ai/hedra-character-1',
  ];

  console.log('\n🔍 Trying alternative lip-sync models...');

  for (const model of models) {
    console.log(`\n   Trying: ${model}`);
    try {
      // @ts-ignore
      const result = await fal.subscribe(model, {
        input: {
          image_url: imageUrl,
          audio_url: audioUrl,
          source_image_url: imageUrl, // Some models use this
          driven_audio_url: audioUrl,  // Some models use this
        },
        logs: false,
      });
      
      console.log(`   ✅ ${model} succeeded!`);
      console.log('   Result:', JSON.stringify(result, null, 2).substring(0, 300));
      return { model, result };
    } catch (error: any) {
      const msg = error.message || 'Unknown error';
      const status = error.status || '';
      console.log(`   ❌ Failed: ${status} ${msg.substring(0, 50)}`);
    }
  }

  return null;
}

/**
 * Main function
 */
async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🎬 FAL.AI OMNIHUMAN 1.5 LIP-SYNC TEST');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  // Check API keys
  if (!FAL_KEY) {
    console.error('❌ FAL_KEY not set!');
    console.log('   1. Go to https://fal.ai/dashboard/keys');
    console.log('   2. Create an API key');
    console.log('   3. Add FAL_KEY=your_key to .env.local');
    process.exit(1);
  }
  console.log('✅ FAL_KEY found:', FAL_KEY.substring(0, 10) + '...');

  if (!ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY not set!');
    process.exit(1);
  }
  console.log('✅ ELEVENLABS_API_KEY found:', ELEVENLABS_API_KEY.substring(0, 10) + '...');

  // Configure fal.ai client - credentials are set via FAL_KEY env var automatically

  // Create output directory
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Check Kelly image exists
  if (!fs.existsSync(KELLY_IMAGE_PATH)) {
    console.error(`❌ Kelly image not found: ${KELLY_IMAGE_PATH}`);
    process.exit(1);
  }

  // Step 1: Generate TTS audio
  const audioPath = await generateAudio(TEST_TEXT);

  // Step 2: Upload files to fal.ai
  const imageUrl = await uploadToFal(KELLY_IMAGE_PATH, 'image/png');
  const audioUrl = await uploadToFal(audioPath, 'audio/mpeg');

  // Step 3: Generate video with Omnihuman
  let videoPath = await generateVideo(imageUrl, audioUrl);

  // If Omnihuman failed, try alternatives
  if (!videoPath) {
    console.log('\n⚠️ Omnihuman failed, trying alternative models...');
    const result = await tryAlternativeModels(imageUrl, audioUrl);
    
    if (result) {
      console.log(`\n✅ Found working model: ${result.model}`);
      
      // Try to extract and save the video
      const videoUrl = result.result?.video?.url || result.result?.data?.video?.url;
      if (videoUrl) {
        const videoResponse = await fetch(videoUrl);
        if (videoResponse.ok) {
          const videoBuffer = await videoResponse.arrayBuffer();
          videoPath = path.join(OUTPUT_DIR, `kelly-talking-${result.model.split('/')[1]}.mp4`);
          fs.writeFileSync(videoPath, Buffer.from(videoBuffer));
          console.log(`   ✅ Video saved: ${videoPath}`);
        }
      }
    }
  }

  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  if (videoPath) {
    console.log(`   ✅ SUCCESS! Video saved to: ${videoPath}`);
  } else {
    console.log('   ⚠️ No video generated. You may need to:');
    console.log('      1. Check your fal.ai subscription');
    console.log('      2. Try a different model');
    console.log('      3. Use a different input image format');
  }
  console.log('═══════════════════════════════════════════════════════════════');
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});

