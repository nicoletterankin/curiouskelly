/**
 * Kelly Quick Lip-Sync - Use Existing Images
 * 
 * Takes an existing Kelly image and generates a lip-synced video.
 * No need to generate new images - uses your LoRA-generated assets.
 * 
 * Usage:
 *   npx tsx scripts/kelly-quick-lipsync.ts --image public/kelly/poses/kelly_welcome.png --text "Hello!"
 *   npx tsx scripts/kelly-quick-lipsync.ts --lesson 1 --text "Welcome to day one!"
 *   npx tsx scripts/kelly-quick-lipsync.ts --pose idle --text "Let me think..."
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIG
// =============================================================================

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN!;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY!;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
const OUTPUT_DIR = path.join(process.cwd(), 'generated-videos');

// Kelly image locations
const KELLY_IMAGES = {
  // Core poses
  poses: {
    idle: 'public/kelly/poses/kelly_idle.png',
    welcome: 'public/kelly/poses/kelly_welcome.png',
    listening: 'public/kelly/poses/kelly_listening.png',
    hint: 'public/kelly/poses/kelly_hint.png',
    clasp: 'public/kelly/poses/kelly_clasp.png',
    choice_left: 'public/kelly/poses/kelly_choice_left.png',
    choice_right: 'public/kelly/poses/kelly_choice_right.png',
  },
  
  // Get lesson image (e.g., hero image for a lesson day)
  getLessonImage: (dayNumber: number, type: 'hero' | 'guide-point' | 'reaction' = 'hero') => {
    const dayStr = dayNumber.toString().padStart(3, '0');
    return `public/kelly/lessons/${dayStr}/lesson-${dayNumber}-${type}.png`;
  },
  
  // Get phase image
  getPhaseImage: (dayNumber: number, phase: 'hook' | 'q1' | 'q2' | 'q3' | 'wisdom') => {
    const dayStr = dayNumber.toString().padStart(3, '0');
    return `public/kelly/phases/${dayStr}/${phase}.png`;
  },
};

// =============================================================================
// FUNCTIONS
// =============================================================================

async function generateAudio(text: string): Promise<Buffer> {
  console.log('🔊 Generating audio...');
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.4,
          use_speaker_boost: true,
        },
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`TTS error: ${response.status}`);
  }

  return Buffer.from(await response.arrayBuffer());
}

async function generateLipSyncVideo(
  replicate: Replicate,
  imageBuffer: Buffer,
  audioBuffer: Buffer
): Promise<Buffer> {
  console.log('🎬 Generating lip-sync video...');
  
  // Upload files to Replicate using their file API
  console.log('   📤 Uploading image to Replicate...');
  const imageFile = await replicate.files.create(imageBuffer, {
    filename: 'kelly.png',
    content_type: 'image/png',
  });
  console.log(`   ✅ Image uploaded: ${imageFile.urls.get.substring(0, 60)}...`);
  
  console.log('   📤 Uploading audio to Replicate...');
  const audioFile = await replicate.files.create(audioBuffer, {
    filename: 'audio.mp3',
    content_type: 'audio/mpeg',
  });
  console.log(`   ✅ Audio uploaded: ${audioFile.urls.get.substring(0, 60)}...`);
  
  // Run SadTalker with the uploaded URLs
  console.log('   🎭 Running SadTalker...');
  
  const prediction = await replicate.predictions.create({
    version: "3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
    input: {
      source_image: imageFile.urls.get,
      driven_audio: audioFile.urls.get,
      enhancer: "gfpgan",
      preprocess: "crop",
      still_mode: false,
    },
  });
  
  console.log(`   Prediction ID: ${prediction.id}`);
  console.log(`   Status: ${prediction.status}`);
  
  // Poll for completion
  let result = prediction;
  let dots = 0;
  while (result.status !== 'succeeded' && result.status !== 'failed') {
    await new Promise(r => setTimeout(r, 2000));
    result = await replicate.predictions.get(prediction.id);
    dots++;
    if (dots % 5 === 0) {
      process.stdout.write(`\r   ⏳ Processing... (${dots * 2}s)`);
    }
  }
  console.log('');
  
  if (result.status === 'failed') {
    throw new Error(`SadTalker failed: ${result.error || 'Unknown error'}`);
  }
  
  // Download the video
  if (!result.output || typeof result.output !== 'string') {
    throw new Error('No video URL in output');
  }
  
  console.log(`   📥 Downloading video...`);
  const videoResponse = await fetch(result.output);
  if (!videoResponse.ok) {
    throw new Error(`Failed to download video: ${videoResponse.status}`);
  }
  
  return Buffer.from(await videoResponse.arrayBuffer());
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log('   🎬 KELLY QUICK LIP-SYNC');
  console.log('═══════════════════════════════════════════════════════');
  
  // Parse args
  const args = process.argv.slice(2);
  let imagePath: string | undefined;
  let text = "Hello! I'm Kelly, and I'm excited to learn with you today!";
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--image' && args[i + 1]) {
      imagePath = args[i + 1];
      i++;
    } else if (args[i] === '--text' && args[i + 1]) {
      text = args[i + 1];
      i++;
    } else if (args[i] === '--lesson' && args[i + 1]) {
      const dayNum = parseInt(args[i + 1]);
      imagePath = KELLY_IMAGES.getLessonImage(dayNum, 'hero');
      i++;
    } else if (args[i] === '--pose' && args[i + 1]) {
      const pose = args[i + 1] as keyof typeof KELLY_IMAGES.poses;
      imagePath = KELLY_IMAGES.poses[pose] || KELLY_IMAGES.poses.idle;
      i++;
    }
  }
  
  // Default to welcome pose
  if (!imagePath) {
    imagePath = KELLY_IMAGES.poses.welcome;
  }
  
  // Resolve path
  const fullImagePath = path.resolve(process.cwd(), imagePath);
  
  console.log('');
  console.log(`📷 Image: ${imagePath}`);
  console.log(`💬 Text: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);
  
  // Check prerequisites
  if (!REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not set!');
    process.exit(1);
  }
  if (!ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY not set!');
    process.exit(1);
  }
  if (!fs.existsSync(fullImagePath)) {
    console.error(`❌ Image not found: ${fullImagePath}`);
    console.log('\nAvailable poses:');
    Object.entries(KELLY_IMAGES.poses).forEach(([name, posePath]) => {
      const exists = fs.existsSync(path.resolve(process.cwd(), posePath)) ? '✅' : '❌';
      console.log(`   ${exists} ${name}: ${posePath}`);
    });
    process.exit(1);
  }
  
  console.log('');
  
  // Create output dir
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  const replicate = new Replicate({ auth: REPLICATE_API_TOKEN });
  const startTime = Date.now();
  
  // Load image
  const imageBuffer = fs.readFileSync(fullImagePath);
  console.log(`   ✅ Image loaded (${(imageBuffer.byteLength / 1024).toFixed(1)} KB)`);
  
  // Generate audio
  const audioBuffer = await generateAudio(text);
  const audioPath = path.join(OUTPUT_DIR, `audio_${Date.now()}.mp3`);
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   ✅ Audio saved (${(audioBuffer.byteLength / 1024).toFixed(1)} KB)`);
  
  // Generate video
  const videoBuffer = await generateLipSyncVideo(replicate, imageBuffer, audioBuffer);
  
  const videoPath = path.join(OUTPUT_DIR, `kelly_lipsync_${Date.now()}.mp4`);
  fs.writeFileSync(videoPath, videoBuffer);
  
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  
  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`   ✅ SUCCESS! Video generated in ${duration}s`);
  console.log(`   📁 ${videoPath}`);
  console.log(`   📊 ${(videoBuffer.byteLength / 1024).toFixed(1)} KB`);
  console.log('═══════════════════════════════════════════════════════');
  
  // Try to open it
  try {
    const { exec } = await import('child_process');
    exec(`start "" "${videoPath}"`);
    console.log('\n   🎬 Opening video...');
  } catch {
    // Ignore if can't open
  }
}

main().catch(err => {
  console.error('\n❌ Error:', err.message || err);
  process.exit(1);
});
