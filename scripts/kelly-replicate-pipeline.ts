/**
 * 🎬 Kelly Video Pipeline - Replicate Only
 * 
 * Uses only Replicate for:
 * 1. Kelly LoRA + Flux for base image
 * 2. SadTalker/Wav2Lip for lip-sync
 * 3. Real-ESRGAN for 4K upscaling
 * 
 * NO fal.ai dependencies
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
const KELLY_LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

const OUTPUT_DIR = path.join(process.cwd(), 'test-output', 'replicate');
const KELLY_IMAGE_PATH = 'public/kelly/poses/kelly_welcome.png';

const TEST_TEXT = "Hello! I'm Kelly, your learning companion. Today we're going to explore something truly fascinating together!";

/**
 * Convert file to data URL
 */
function fileToDataUrl(filePath: string, mimeType: string): string {
  const buffer = fs.readFileSync(filePath);
  return `data:${mimeType};base64,${buffer.toString('base64')}`;
}

/**
 * Step 1: Generate TTS
 */
async function generateAudio(text: string): Promise<string> {
  console.log('\n🎤 STEP 1: Generating Kelly\'s voice...');
  
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
        voice_settings: { stability: 0.5, similarity_boost: 0.85 },
      }),
    }
  );

  if (!response.ok) throw new Error(`TTS error: ${response.status}`);

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const audioPath = path.join(OUTPUT_DIR, 'kelly-voice.mp3');
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   ✅ Audio: ${(audioBuffer.length / 1024).toFixed(1)} KB`);
  
  return audioPath;
}

/**
 * Step 2: Generate Kelly image with LoRA (optional)
 */
async function generateKellyImage(replicate: Replicate): Promise<string> {
  console.log('\n🎨 STEP 2: Preparing Kelly image...');
  
  // Use existing high-quality image
  if (fs.existsSync(KELLY_IMAGE_PATH)) {
    const stats = fs.statSync(KELLY_IMAGE_PATH);
    console.log(`   ✅ Using existing: ${KELLY_IMAGE_PATH} (${(stats.size / 1024).toFixed(1)} KB)`);
    return KELLY_IMAGE_PATH;
  }

  // Generate with LoRA if needed
  console.log('   🎨 Generating with Flux + LoRA...');
  const output = await replicate.run(
    "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
    {
      input: {
        prompt: "curious_kelly, professional portrait, warm friendly smile, looking at camera, studio lighting, 4k, photorealistic",
        hf_lora: KELLY_LORA_URL,
        lora_scale: 0.9,
        aspect_ratio: "1:1",
        output_format: "png",
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
    console.log(`   ✅ Generated: ${(buffer.length / 1024).toFixed(1)} KB`);
    return imagePath;
  }

  return KELLY_IMAGE_PATH;
}

/**
 * Step 3: Apply lip-sync using Replicate models
 */
async function applyLipSync(
  replicate: Replicate,
  imagePath: string,
  audioPath: string
): Promise<string> {
  console.log('\n👄 STEP 3: Applying lip-sync...');
  
  const imageDataUrl = fileToDataUrl(imagePath, 'image/png');
  const audioDataUrl = fileToDataUrl(audioPath, 'audio/mpeg');

  // List of Replicate lip-sync models to try
  const models = [
    {
      name: 'SadTalker',
      model: "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
      input: {
        source_image: imageDataUrl,
        driven_audio: audioDataUrl,
        enhancer: "gfpgan",
        preprocess: "crop",
        still_mode: false,
      },
    },
    {
      name: 'SadTalker (alt)',
      model: "lucataco/sadtalker:85f79f4a1d369fc190998c3dbbf6e67a8b6bee9fcbae33ff6be3261aaaefd85e",
      input: {
        source_image: imageDataUrl,
        driven_audio: audioDataUrl,
        enhancer: "gfpgan", 
      },
    },
  ];

  for (const model of models) {
    console.log(`\n   🔄 Trying ${model.name}...`);
    
    try {
      const output = await replicate.run(model.model as `${string}/${string}:${string}`, {
        input: model.input,
      });

      console.log(`   📦 Output type: ${typeof output}`);
      
      // Handle different output formats
      let videoUrl: string | null = null;
      
      if (typeof output === 'string') {
        videoUrl = output;
      } else if (output && typeof output === 'object') {
        videoUrl = (output as any).output || (output as any).video || (output as any)[0];
      }

      if (videoUrl && typeof videoUrl === 'string' && videoUrl.startsWith('http')) {
        console.log(`   📥 Downloading video...`);
        const response = await fetch(videoUrl);
        const videoBuffer = Buffer.from(await response.arrayBuffer());
        const videoPath = path.join(OUTPUT_DIR, 'kelly-lipsync.mp4');
        fs.writeFileSync(videoPath, videoBuffer);
        console.log(`   ✅ ${model.name}: ${(videoBuffer.length / 1024).toFixed(1)} KB`);
        return videoPath;
      } else {
        console.log(`   ⚠️ Unexpected output:`, JSON.stringify(output).substring(0, 200));
      }
    } catch (error: any) {
      console.log(`   ❌ ${model.name} failed: ${error.message?.substring(0, 80)}`);
    }
  }

  throw new Error('All lip-sync models failed');
}

/**
 * Step 4: Upscale video using Real-ESRGAN
 */
async function upscaleVideo(replicate: Replicate, videoPath: string): Promise<string> {
  console.log('\n🔍 STEP 4: Upscaling to HD/4K...');
  
  // Read video and convert to data URL
  const videoBuffer = fs.readFileSync(videoPath);
  const videoDataUrl = `data:video/mp4;base64,${videoBuffer.toString('base64')}`;

  console.log(`   📤 Uploading ${(videoBuffer.length / 1024 / 1024).toFixed(2)} MB video...`);

  try {
    // Try video upscaler
    const output = await replicate.run(
      "lucataco/real-esrgan-video:c23768236472c41b7a121ee735c8073e29080c02d343419c4b7f0e56e045cb4d",
      {
        input: {
          video: videoDataUrl,
          scale: 4,
          face_enhance: true,
        }
      }
    );

    const outputUrl = typeof output === 'string' ? output : (output as any)?.output;
    
    if (outputUrl && typeof outputUrl === 'string') {
      console.log(`   📥 Downloading upscaled video...`);
      const response = await fetch(outputUrl);
      const upscaledBuffer = Buffer.from(await response.arrayBuffer());
      const finalPath = path.join(OUTPUT_DIR, 'kelly-4k.mp4');
      fs.writeFileSync(finalPath, upscaledBuffer);
      console.log(`   ✅ 4K video: ${(upscaledBuffer.length / 1024 / 1024).toFixed(2)} MB`);
      return finalPath;
    }
  } catch (error: any) {
    console.log(`   ⚠️ Video upscaler failed: ${error.message?.substring(0, 50)}`);
  }

  // Return original if upscaling fails
  console.log('   ⚠️ Returning base video (upscaling unavailable)');
  return videoPath;
}

/**
 * Main pipeline
 */
async function main() {
  console.log('');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 KELLY VIDEO PIPELINE (Replicate Only)                    ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');

  if (!REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN required');
    process.exit(1);
  }
  if (!ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY required');
    process.exit(1);
  }

  console.log('✅ Replicate: Configured');
  console.log('✅ ElevenLabs: Configured');

  const replicate = new Replicate({ auth: REPLICATE_API_TOKEN });

  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const startTime = Date.now();

  try {
    // Step 1: Audio
    const audioPath = await generateAudio(TEST_TEXT);

    // Step 2: Image
    const imagePath = await generateKellyImage(replicate);

    // Step 3: Lip-sync
    const lipSyncPath = await applyLipSync(replicate, imagePath, audioPath);

    // Step 4: Upscale (optional)
    const finalPath = await upscaleVideo(replicate, lipSyncPath);

    const totalTime = (Date.now() - startTime) / 1000;

    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════╗');
    console.log('║  ✅ PIPELINE COMPLETE                                        ║');
    console.log('╚══════════════════════════════════════════════════════════════╝');
    console.log(`   Total time: ${(totalTime / 60).toFixed(1)} minutes`);
    console.log('');
    console.log('📁 Output files:');
    console.log(`   ${audioPath}`);
    console.log(`   ${lipSyncPath}`);
    console.log(`   ${finalPath} ← FINAL`);

    // Open the video
    const { exec } = require('child_process');
    exec(`start "" "${finalPath}"`);

  } catch (error) {
    console.error('\n❌ Pipeline failed:', error);
    process.exit(1);
  }
}

main();



