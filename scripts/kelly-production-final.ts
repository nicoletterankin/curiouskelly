/**
 * 🎬 Kelly Production Pipeline - FINAL
 * 
 * Complete pipeline using Replicate:
 * 1. ElevenLabs TTS
 * 2. SadTalker lip-sync (via Replicate file API)
 * 3. Real-ESRGAN 4K upscaling
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

const OUTPUT_DIR = path.join(process.cwd(), 'test-output', 'production-final');
const KELLY_IMAGE_PATH = 'public/kelly/poses/kelly_welcome.png';

const TEST_TEXT = "Hello! I'm Kelly, and I'm so excited to learn with you today! Let's discover something truly amazing together. Are you ready?";

async function sleep(ms: number) {
  return new Promise(r => setTimeout(r, ms));
}

async function main() {
  console.log('');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 KELLY PRODUCTION PIPELINE - FINAL                        ║');
  console.log('║  SadTalker + Real-ESRGAN 4K                                  ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('');

  if (!REPLICATE_API_TOKEN || !ELEVENLABS_API_KEY) {
    console.error('❌ Missing API keys');
    process.exit(1);
  }

  const replicate = new Replicate({ auth: REPLICATE_API_TOKEN });
  
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const startTime = Date.now();

  // ═══════════════════════════════════════════════════════════════
  // STEP 1: Generate Audio
  // ═══════════════════════════════════════════════════════════════
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📢 STEP 1: Generating Kelly\'s voice...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  const audioResponse = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY!,
      },
      body: JSON.stringify({
        text: TEST_TEXT,
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

  if (!audioResponse.ok) throw new Error(`TTS failed: ${audioResponse.status}`);
  
  const audioBuffer = Buffer.from(await audioResponse.arrayBuffer());
  const audioPath = path.join(OUTPUT_DIR, 'kelly-voice.mp3');
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   ✅ Audio: ${(audioBuffer.length / 1024).toFixed(1)} KB`);

  // ═══════════════════════════════════════════════════════════════
  // STEP 2: Upload to Replicate
  // ═══════════════════════════════════════════════════════════════
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📤 STEP 2: Uploading to Replicate...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  const imageBuffer = fs.readFileSync(KELLY_IMAGE_PATH);
  console.log(`   📷 Image: ${(imageBuffer.length / 1024).toFixed(1)} KB`);

  const imageFile = await replicate.files.create(imageBuffer, {
    filename: 'kelly.png',
    content_type: 'image/png',
  });
  console.log(`   ✅ Image uploaded`);

  const audioFile = await replicate.files.create(audioBuffer, {
    filename: 'audio.mp3',
    content_type: 'audio/mpeg',
  });
  console.log(`   ✅ Audio uploaded`);

  // ═══════════════════════════════════════════════════════════════
  // STEP 3: SadTalker Lip-Sync
  // ═══════════════════════════════════════════════════════════════
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('👄 STEP 3: Generating lip-sync video...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  const sadtalkerStart = Date.now();
  
  const prediction = await replicate.predictions.create({
    version: "3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
    input: {
      source_image: imageFile.urls.get,
      driven_audio: audioFile.urls.get,
      enhancer: "gfpgan",  // Face enhancement
      preprocess: "crop",
      still_mode: false,
      expression_scale: 1.0,
    },
  });

  console.log(`   🔄 Processing (ID: ${prediction.id})...`);

  let result = prediction;
  let dots = 0;
  while (result.status !== 'succeeded' && result.status !== 'failed') {
    await sleep(3000);
    result = await replicate.predictions.get(prediction.id);
    dots++;
    process.stdout.write(dots % 10 === 0 ? '.' : '');
  }
  console.log('');

  if (result.status !== 'succeeded') {
    throw new Error(`SadTalker failed: ${result.error}`);
  }

  const sadtalkerTime = (Date.now() - sadtalkerStart) / 1000;
  console.log(`   ⏱️  Completed in ${sadtalkerTime.toFixed(0)}s`);

  // Download lip-sync video
  const lipSyncUrl = result.output as string;
  console.log(`   📥 Downloading...`);
  const lipSyncResponse = await fetch(lipSyncUrl);
  const lipSyncBuffer = Buffer.from(await lipSyncResponse.arrayBuffer());
  const lipSyncPath = path.join(OUTPUT_DIR, 'kelly-lipsync.mp4');
  fs.writeFileSync(lipSyncPath, lipSyncBuffer);
  console.log(`   ✅ Lip-sync video: ${(lipSyncBuffer.length / 1024).toFixed(1)} KB`);

  // ═══════════════════════════════════════════════════════════════
  // STEP 4: 4K Upscaling
  // ═══════════════════════════════════════════════════════════════
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🔍 STEP 4: Upscaling to 4K...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  const upscaleStart = Date.now();
  let finalPath = lipSyncPath;

  try {
    // Upload video to Replicate
    const videoFile = await replicate.files.create(lipSyncBuffer, {
      filename: 'video.mp4',
      content_type: 'video/mp4',
    });
    console.log(`   📤 Video uploaded for upscaling`);

    // Run Real-ESRGAN video upscaler
    const upscalePrediction = await replicate.predictions.create({
      version: "42e09a96c888025dde9e2a63fc1ed9bb8bb2a87d7d41f69dcbc68f62df6882d3",
      input: {
        video: videoFile.urls.get,
        scale: 4,
        face_enhance: true,
      },
    });

    console.log(`   🔄 Upscaling (ID: ${upscalePrediction.id})...`);

    let upscaleResult = upscalePrediction;
    dots = 0;
    while (upscaleResult.status !== 'succeeded' && upscaleResult.status !== 'failed') {
      await sleep(5000);
      upscaleResult = await replicate.predictions.get(upscalePrediction.id);
      dots++;
      process.stdout.write(dots % 6 === 0 ? '.' : '');
    }
    console.log('');

    if (upscaleResult.status === 'succeeded' && upscaleResult.output) {
      const upscaleUrl = upscaleResult.output as string;
      console.log(`   📥 Downloading 4K video...`);
      const upscaleResponse = await fetch(upscaleUrl);
      const upscaleBuffer = Buffer.from(await upscaleResponse.arrayBuffer());
      finalPath = path.join(OUTPUT_DIR, 'kelly-4k.mp4');
      fs.writeFileSync(finalPath, upscaleBuffer);
      console.log(`   ✅ 4K video: ${(upscaleBuffer.length / 1024 / 1024).toFixed(2)} MB`);
    } else {
      console.log(`   ⚠️ Upscaling failed, using base video`);
    }
  } catch (e: any) {
    console.log(`   ⚠️ Upscaling error: ${e.message?.substring(0, 50)}`);
    console.log(`   📁 Using base video instead`);
  }

  const upscaleTime = (Date.now() - upscaleStart) / 1000;
  console.log(`   ⏱️  Completed in ${upscaleTime.toFixed(0)}s`);

  // ═══════════════════════════════════════════════════════════════
  // COMPLETE
  // ═══════════════════════════════════════════════════════════════
  const totalTime = (Date.now() - startTime) / 1000;

  console.log('');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  ✅ PIPELINE COMPLETE                                        ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`   Total time: ${(totalTime / 60).toFixed(1)} minutes`);
  console.log('');
  console.log('📁 Output files:');
  console.log(`   ${audioPath}`);
  console.log(`   ${lipSyncPath}`);
  console.log(`   ${finalPath} ← FINAL`);
  console.log('');

  // Open the final video
  try {
    execSync(`start "" "${finalPath}"`, { stdio: 'ignore' });
  } catch (e) {
    // ignore
  }
}

main().catch(error => {
  console.error('\n❌ Pipeline failed:', error);
  process.exit(1);
});




