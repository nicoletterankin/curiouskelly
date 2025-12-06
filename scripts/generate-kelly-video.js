/**
 * 🎬 Kelly 4K Lipsync Video Generator
 * 
 * Direct script to generate talking videos of Kelly
 * Uses: ElevenLabs TTS + Replicate SadTalker
 * 
 * Usage:
 *   node scripts/generate-kelly-video.js
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

// Load env vars
require('dotenv').config();

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;

// Config
const CONFIG = {
  day: 1,
  phase: 'hook',
  text: "Hello! Today we're going to explore something amazing together. Are you ready to learn?",
  model: 'sadtalker',
  upscale: true,
};

// Lipsync models on Replicate
const LIPSYNC_MODELS = {
  sadtalker: 'cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376',
  sadtalker_alt: 'lucataco/sadtalker:85f79f4a1d369fc190998c3dbbf6e67a8b6bee9fcbae33ff6be3261aaaefd85e',
  liveportrait: 'fofr/live-portrait:067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146',
  wav2lip: 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
};

const UPSCALER_MODEL = 'lucataco/real-esrgan-video:c23768236472c41b7a121ee735c8073e29080c02d343419c4b7f0e56e045cb4d';

// Helper: Make HTTP request
function makeRequest(options, data = null) {
  return new Promise((resolve, reject) => {
    const protocol = options.protocol === 'http:' ? http : https;
    const req = protocol.request(options, (res) => {
      let body = [];
      res.on('data', chunk => body.push(chunk));
      res.on('end', () => {
        const buffer = Buffer.concat(body);
        if (res.headers['content-type']?.includes('application/json')) {
          try {
            resolve({ status: res.statusCode, data: JSON.parse(buffer.toString()) });
          } catch {
            resolve({ status: res.statusCode, data: buffer });
          }
        } else {
          resolve({ status: res.statusCode, data: buffer });
        }
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

// Step 1: Generate TTS with ElevenLabs
async function generateTTS(text) {
  console.log('\n🎙️ Step 1: Generating Kelly\'s voice with ElevenLabs...');
  console.log(`   Text: "${text.substring(0, 50)}..."`);
  
  const response = await makeRequest({
    hostname: 'api.elevenlabs.io',
    path: `/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
    method: 'POST',
    headers: {
      'Accept': 'audio/mpeg',
      'Content-Type': 'application/json',
      'xi-api-key': ELEVENLABS_API_KEY,
    }
  }, JSON.stringify({
    text,
    model_id: 'eleven_turbo_v2_5',
    voice_settings: {
      stability: 0.5,
      similarity_boost: 0.85,
      style: 0.0,
      use_speaker_boost: true,
    }
  }));
  
  if (response.status !== 200) {
    throw new Error(`ElevenLabs error: ${response.status} - ${JSON.stringify(response.data)}`);
  }
  
  console.log(`   ✅ Audio generated: ${(response.data.length / 1024).toFixed(1)} KB`);
  return response.data;
}

// Step 2: Load and encode image
async function loadImage(day, phase) {
  console.log(`\n📸 Step 2: Loading Kelly image for Day ${day} - ${phase}...`);
  
  const paddedDay = String(day).padStart(3, '0');
  const imagePath = path.join(__dirname, '..', 'public', 'kelly', 'phases', paddedDay, `${phase}.png`);
  
  if (!fs.existsSync(imagePath)) {
    throw new Error(`Image not found: ${imagePath}`);
  }
  
  const imageBuffer = fs.readFileSync(imagePath);
  console.log(`   ✅ Image loaded: ${(imageBuffer.length / 1024).toFixed(1)} KB`);
  
  return imageBuffer;
}

// Step 3: Generate lipsync video with Replicate
async function generateLipsync(imageBuffer, audioBuffer, modelKey = 'sadtalker') {
  console.log(`\n🎬 Step 3: Creating lipsync video with ${modelKey}...`);
  
  const model = LIPSYNC_MODELS[modelKey];
  if (!model) throw new Error(`Unknown model: ${modelKey}`);
  
  // Convert to base64
  const imageBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
  const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  console.log(`   Sending to Replicate...`);
  
  // Create prediction
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify({
    version: model.split(':')[1],
    input: {
      source_image: imageBase64,
      driven_audio: audioBase64,
      enhancer: 'gfpgan',
      preprocess: 'crop',
      still_mode: false,
      expression_scale: 1.0,
    }
  }));
  
  if (createResponse.status !== 201) {
    throw new Error(`Replicate create error: ${createResponse.status} - ${JSON.stringify(createResponse.data)}`);
  }
  
  const predictionId = createResponse.data.id;
  console.log(`   Prediction started: ${predictionId}`);
  
  // Poll for completion
  let attempts = 0;
  const maxAttempts = 120; // 2 minutes max
  
  while (attempts < maxAttempts) {
    await new Promise(r => setTimeout(r, 3000)); // Wait 3s between polls
    
    const statusResponse = await makeRequest({
      hostname: 'api.replicate.com',
      path: `/v1/predictions/${predictionId}`,
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      }
    });
    
    const status = statusResponse.data.status;
    process.stdout.write(`\r   Status: ${status} (${attempts * 3}s)...        `);
    
    if (status === 'succeeded') {
      console.log(`\n   ✅ Video generated!`);
      return statusResponse.data.output;
    } else if (status === 'failed') {
      throw new Error(`Replicate failed: ${statusResponse.data.error}`);
    }
    
    attempts++;
  }
  
  throw new Error('Timeout waiting for video generation');
}

// Step 4: Upscale video (optional)
async function upscaleVideo(videoUrl) {
  console.log(`\n✨ Step 4: Upscaling to 4K with Real-ESRGAN...`);
  
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify({
    version: UPSCALER_MODEL.split(':')[1],
    input: {
      video: videoUrl,
      scale: 4,
      face_enhance: true,
    }
  }));
  
  if (createResponse.status !== 201) {
    console.log(`   ⚠️ Upscale failed to start, returning original video`);
    return videoUrl;
  }
  
  const predictionId = createResponse.data.id;
  console.log(`   Upscale started: ${predictionId}`);
  
  // Poll for completion
  let attempts = 0;
  const maxAttempts = 180; // 9 minutes for 4K upscaling
  
  while (attempts < maxAttempts) {
    await new Promise(r => setTimeout(r, 3000));
    
    const statusResponse = await makeRequest({
      hostname: 'api.replicate.com',
      path: `/v1/predictions/${predictionId}`,
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      }
    });
    
    const status = statusResponse.data.status;
    process.stdout.write(`\r   Upscaling: ${status} (${attempts * 3}s)...        `);
    
    if (status === 'succeeded') {
      console.log(`\n   ✅ 4K video ready!`);
      return statusResponse.data.output;
    } else if (status === 'failed') {
      console.log(`\n   ⚠️ Upscale failed, returning original video`);
      return videoUrl;
    }
    
    attempts++;
  }
  
  console.log(`\n   ⚠️ Upscale timeout, returning original video`);
  return videoUrl;
}

// Step 5: Download and save video
async function downloadVideo(url, outputPath) {
  console.log(`\n💾 Step 5: Downloading video...`);
  
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const protocol = urlObj.protocol === 'http:' ? http : https;
    
    protocol.get(url, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301) {
        // Follow redirect
        return downloadVideo(res.headers.location, outputPath).then(resolve).catch(reject);
      }
      
      const file = fs.createWriteStream(outputPath);
      res.pipe(file);
      file.on('finish', () => {
        file.close();
        console.log(`   ✅ Saved to: ${outputPath}`);
        resolve(outputPath);
      });
    }).on('error', reject);
  });
}

// Main
async function main() {
  console.log('═'.repeat(60));
  console.log('🎬 KELLY 4K LIPSYNC VIDEO GENERATOR');
  console.log('═'.repeat(60));
  console.log(`Day: ${CONFIG.day}`);
  console.log(`Phase: ${CONFIG.phase}`);
  console.log(`Model: ${CONFIG.model}`);
  console.log(`4K Upscale: ${CONFIG.upscale}`);
  console.log('═'.repeat(60));
  
  // Check env vars
  if (!ELEVENLABS_API_KEY || ELEVENLABS_API_KEY === 'your_key_here') {
    throw new Error('ELEVENLABS_API_KEY not configured');
  }
  if (!REPLICATE_API_TOKEN || REPLICATE_API_TOKEN === 'your_replicate_token_here') {
    throw new Error('REPLICATE_API_TOKEN not configured');
  }
  
  const startTime = Date.now();
  
  try {
    // Generate audio
    const audioBuffer = await generateTTS(CONFIG.text);
    
    // Load image
    const imageBuffer = await loadImage(CONFIG.day, CONFIG.phase);
    
    // Generate lipsync video
    let videoUrl = await generateLipsync(imageBuffer, audioBuffer, CONFIG.model);
    
    // Upscale to 4K
    if (CONFIG.upscale && videoUrl) {
      videoUrl = await upscaleVideo(videoUrl);
    }
    
    // Download video
    const outputDir = path.join(__dirname, '..', 'generated-videos');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    const outputPath = path.join(outputDir, `kelly_day${CONFIG.day}_${CONFIG.phase}_${Date.now()}.mp4`);
    await downloadVideo(videoUrl, outputPath);
    
    const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
    
    console.log('\n' + '═'.repeat(60));
    console.log('✅ SUCCESS!');
    console.log('═'.repeat(60));
    console.log(`Total time: ${totalTime}s`);
    console.log(`Output: ${outputPath}`);
    console.log(`Video URL: ${videoUrl}`);
    console.log('═'.repeat(60));
    
  } catch (error) {
    console.error('\n❌ ERROR:', error.message);
    process.exit(1);
  }
}

main();

