/**
 * Direct Replicate lip-sync test with URL uploads
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

const OUTPUT_DIR = path.join(process.cwd(), 'test-output', 'direct');
const KELLY_IMAGE_PATH = 'public/kelly/poses/kelly_welcome.png';

async function main() {
  console.log('🎬 Direct Replicate Lip-Sync Test\n');

  const replicate = new Replicate({ auth: REPLICATE_API_TOKEN });

  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Step 1: Generate audio
  console.log('📢 Generating audio...');
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
        text: "Hello! I'm Kelly. Let's learn together!",
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.8 },
      }),
    }
  );
  const audioBuffer = Buffer.from(await audioResponse.arrayBuffer());
  const audioPath = path.join(OUTPUT_DIR, 'audio.mp3');
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   ✅ Audio: ${audioBuffer.length} bytes`);

  // Step 2: Upload files to Replicate using their file API
  console.log('\n📤 Uploading to Replicate...');
  
  const imageBuffer = fs.readFileSync(KELLY_IMAGE_PATH);
  
  // Use Replicate's file upload
  const imageFile = await replicate.files.create(imageBuffer, {
    filename: 'kelly.png',
    content_type: 'image/png',
  });
  console.log(`   Image uploaded: ${imageFile.urls.get}`);

  const audioFile = await replicate.files.create(audioBuffer, {
    filename: 'audio.mp3', 
    content_type: 'audio/mpeg',
  });
  console.log(`   Audio uploaded: ${audioFile.urls.get}`);

  // Step 3: Run SadTalker with URLs
  console.log('\n🎭 Running SadTalker...');
  
  const prediction = await replicate.predictions.create({
    version: "3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
    input: {
      source_image: imageFile.urls.get,
      driven_audio: audioFile.urls.get,
      enhancer: "gfpgan",
      preprocess: "crop",
    },
  });

  console.log(`   Prediction ID: ${prediction.id}`);
  console.log(`   Status: ${prediction.status}`);

  // Poll for completion
  let result = prediction;
  while (result.status !== 'succeeded' && result.status !== 'failed') {
    await new Promise(r => setTimeout(r, 2000));
    result = await replicate.predictions.get(prediction.id);
    console.log(`   Status: ${result.status}`);
  }

  if (result.status === 'succeeded') {
    console.log('\n✅ Success!');
    console.log(`   Output: ${result.output}`);
    
    // Download video
    if (result.output && typeof result.output === 'string') {
      const videoResponse = await fetch(result.output);
      const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
      const videoPath = path.join(OUTPUT_DIR, 'kelly-talking.mp4');
      fs.writeFileSync(videoPath, videoBuffer);
      console.log(`   📁 Saved: ${videoPath} (${videoBuffer.length} bytes)`);
      
      // Open it
      const { exec } = require('child_process');
      exec(`start "" "${videoPath}"`);
    }
  } else {
    console.log('\n❌ Failed:', result.error);
  }
}

main().catch(console.error);




