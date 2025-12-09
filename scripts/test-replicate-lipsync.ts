/**
 * Test script for Replicate lip-sync video generation
 * 
 * Uses SadTalker or other lip-sync models on Replicate
 * 
 * Usage:
 *   npx ts-node scripts/test-replicate-lipsync.ts
 * 
 * Prerequisites:
 *   - REPLICATE_API_TOKEN in .env.local
 *   - ELEVENLABS_API_KEY in .env.local
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

const TEST_TEXT = "Hello! I'm Kelly, and I'm so excited to learn with you today!";
const OUTPUT_DIR = path.join(process.cwd(), 'test-output');
const KELLY_IMAGE_PATH = 'public/kelly/poses/kelly_welcome.png';

/**
 * Convert file to base64 data URL
 */
function fileToDataUrl(filePath: string, mimeType: string): string {
  const buffer = fs.readFileSync(filePath);
  const base64 = buffer.toString('base64');
  return `data:${mimeType};base64,${base64}`;
}

/**
 * Generate TTS audio using ElevenLabs
 */
async function generateAudio(text: string): Promise<{ path: string; buffer: Buffer }> {
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

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const audioPath = path.join(OUTPUT_DIR, 'kelly-speech.mp3');
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   ✅ Audio saved: ${audioPath} (${audioBuffer.byteLength} bytes)`);
  
  return { path: audioPath, buffer: audioBuffer };
}

/**
 * Try SadTalker model on Replicate
 */
async function trySadTalker(replicate: Replicate, imageDataUrl: string, audioDataUrl: string): Promise<string | null> {
  console.log('\n🎭 Trying SadTalker model...');
  
  try {
    // SadTalker model on Replicate
    const output = await replicate.run(
      "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
      {
        input: {
          source_image: imageDataUrl,
          driven_audio: audioDataUrl,
          enhancer: "gfpgan", // Face enhancement
          preprocess: "crop", // Crop to face
          still_mode: false,
          use_ref_video: false,
        }
      }
    );
    
    console.log('   ✅ SadTalker output:', output);
    
    if (typeof output === 'string' && output.startsWith('http')) {
      return output;
    } else if (output && typeof output === 'object' && 'url' in output) {
      return (output as any).url;
    }
    
    return null;
  } catch (error: any) {
    console.log(`   ❌ SadTalker failed:`, error.message?.substring(0, 100));
    return null;
  }
}

/**
 * Try Wav2Lip model on Replicate
 */
async function tryWav2Lip(replicate: Replicate, imageDataUrl: string, audioDataUrl: string): Promise<string | null> {
  console.log('\n👄 Trying Wav2Lip model...');
  
  try {
    const output = await replicate.run(
      "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
      {
        input: {
          face: imageDataUrl,
          audio: audioDataUrl,
          pads: "0 10 0 0", // Padding around face
          smooth: true,
          fps: 25,
          resize_factor: 1,
        }
      }
    );
    
    console.log('   ✅ Wav2Lip output:', output);
    
    if (typeof output === 'string' && output.startsWith('http')) {
      return output;
    }
    
    return null;
  } catch (error: any) {
    console.log(`   ❌ Wav2Lip failed:`, error.message?.substring(0, 100));
    return null;
  }
}

/**
 * Try Lipsync model on Replicate
 */
async function tryLipsync(replicate: Replicate, imageUrl: string, audioUrl: string): Promise<string | null> {
  console.log('\n🗣️ Trying sync-lipsync model...');
  
  try {
    // Try the sync model which might accept URLs directly
    const output = await replicate.run(
      "zsxkib/video-retalking:a32a4a1f499a8f8c3a86a6a9cebecc38a9b57822a9f5ee8b0e7fc65a75c1e0b1",
      {
        input: {
          face: imageUrl,
          input_audio: audioUrl,
        }
      }
    );
    
    console.log('   ✅ Lipsync output:', output);
    return typeof output === 'string' ? output : null;
  } catch (error: any) {
    console.log(`   ❌ Lipsync failed:`, error.message?.substring(0, 100));
    return null;
  }
}

/**
 * Download video from URL
 */
async function downloadVideo(url: string, filename: string): Promise<string> {
  console.log(`\n📥 Downloading video...`);
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }
  
  const buffer = Buffer.from(await response.arrayBuffer());
  const outputPath = path.join(OUTPUT_DIR, filename);
  fs.writeFileSync(outputPath, buffer);
  
  console.log(`   ✅ Saved: ${outputPath} (${buffer.byteLength} bytes)`);
  return outputPath;
}

/**
 * Main function
 */
async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🎬 REPLICATE LIP-SYNC VIDEO TEST');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  // Check API keys
  if (!REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not set!');
    console.log('   Add REPLICATE_API_TOKEN=r8_xxx to .env.local');
    process.exit(1);
  }
  console.log('✅ REPLICATE_API_TOKEN found:', REPLICATE_API_TOKEN.substring(0, 10) + '...');

  if (!ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY not set!');
    process.exit(1);
  }
  console.log('✅ ELEVENLABS_API_KEY found:', ELEVENLABS_API_KEY.substring(0, 10) + '...');

  // Initialize Replicate
  const replicate = new Replicate({
    auth: REPLICATE_API_TOKEN,
  });

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
  const audio = await generateAudio(TEST_TEXT);

  // Step 2: Prepare data URLs for Replicate
  console.log('\n📦 Preparing data URLs...');
  const imageDataUrl = fileToDataUrl(KELLY_IMAGE_PATH, 'image/png');
  const audioDataUrl = fileToDataUrl(audio.path, 'audio/mpeg');
  console.log(`   Image: ${imageDataUrl.substring(0, 50)}... (${Math.round(imageDataUrl.length / 1024)}KB)`);
  console.log(`   Audio: ${audioDataUrl.substring(0, 50)}... (${Math.round(audioDataUrl.length / 1024)}KB)`);

  // Step 3: Try different lip-sync models
  let videoUrl: string | null = null;
  let modelUsed = '';

  // Try SadTalker first (generally best results)
  videoUrl = await trySadTalker(replicate, imageDataUrl, audioDataUrl);
  if (videoUrl) modelUsed = 'SadTalker';

  // If SadTalker failed, try Wav2Lip
  if (!videoUrl) {
    videoUrl = await tryWav2Lip(replicate, imageDataUrl, audioDataUrl);
    if (videoUrl) modelUsed = 'Wav2Lip';
  }

  // Step 4: Download the result if we got one
  if (videoUrl) {
    const outputPath = await downloadVideo(videoUrl, `kelly-talking-${modelUsed.toLowerCase()}.mp4`);
    
    console.log('');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`   ✅ SUCCESS! Video generated with ${modelUsed}`);
    console.log(`   📁 Output: ${outputPath}`);
    console.log('═══════════════════════════════════════════════════════════════');
  } else {
    console.log('');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('   ⚠️ No models succeeded. Possible issues:');
    console.log('   - Image format not compatible (try a JPEG)');
    console.log('   - Face not detected in image');
    console.log('   - Model temporarily unavailable');
    console.log('═══════════════════════════════════════════════════════════════');
  }
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});



