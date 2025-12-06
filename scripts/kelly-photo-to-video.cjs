/**
 * 🎬 Kelly Photo-to-Video Generation
 * 
 * Uses Kelly's actual photos from lesson assets
 * to create character-consistent video templates.
 * 
 * This ensures the generated videos look exactly like Kelly!
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config();

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL;

// Kelly photos from lesson assets
const KELLY_PHOTOS = {
  day1_hook: 'https://hgamibxwwpkimqpvtqoa.supabase.co/storage/v1/object/public/kelly-lessons/lessons/day_1/hook.png',
  day1_q1: 'https://hgamibxwwpkimqpvtqoa.supabase.co/storage/v1/object/public/kelly-lessons/lessons/day_1/q1.png',
  day1_wisdom: 'https://hgamibxwwpkimqpvtqoa.supabase.co/storage/v1/object/public/kelly-lessons/lessons/day_1/wisdom.png',
  day5_hook: 'https://hgamibxwwpkimqpvtqoa.supabase.co/storage/v1/object/public/kelly-lessons/lessons/day_5/hook.png',
  day10_hook: 'https://hgamibxwwpkimqpvtqoa.supabase.co/storage/v1/object/public/kelly-lessons/lessons/day_10/hook.png',
};

// Image-to-video models
const I2V_MODELS = {
  svd: {
    version: 'stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438',
    name: 'Stable Video Diffusion',
    inputKey: 'input_image',
  },
  animatediff: {
    version: 'lucataco/animate-diff:beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f',
    name: 'AnimateDiff',
    inputKey: 'image',
  },
};

const outputDir = path.join(__dirname, '..', 'template-forge', 'kelly-i2v');
fs.mkdirSync(outputDir, { recursive: true });

function makeRequest(options, data = null) {
  return new Promise((resolve, reject) => {
    const protocol = options.protocol === 'http:' ? http : https;
    const req = protocol.request(options, (res) => {
      let body = [];
      res.on('data', chunk => body.push(chunk));
      res.on('end', () => {
        const buffer = Buffer.concat(body);
        if (res.headers['content-type']?.includes('application/json')) {
          try { resolve({ status: res.statusCode, data: JSON.parse(buffer.toString()) }); }
          catch { resolve({ status: res.statusCode, data: buffer }); }
        } else { resolve({ status: res.statusCode, data: buffer }); }
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

async function generateFromPhoto(photoUrl, model, motionPrompt) {
  console.log(`\n🎬 Generating video from Kelly photo...`);
  console.log(`   Photo: ${photoUrl.split('/').pop()}`);
  console.log(`   Model: ${model.name}`);
  console.log(`   Motion: ${motionPrompt || 'natural movement'}`);
  
  const version = model.version.split(':')[1];
  
  let input = {};
  
  if (model.name === 'Stable Video Diffusion') {
    input = {
      [model.inputKey]: photoUrl,
      motion_bucket_id: 127, // Higher = more motion
      fps: 7,
      cond_aug: 0.02,
      decoding_t: 7,
      video_length: '25_frames_with_svd_xt', // 25 frames for longer clip
    };
  } else if (model.name === 'AnimateDiff') {
    input = {
      [model.inputKey]: photoUrl,
      prompt: motionPrompt || 'a woman speaking naturally, subtle head movements, blinking',
      negative_prompt: 'blurry, low quality, distorted face',
      num_frames: 16,
      fps: 8,
    };
  }
  
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify({ version, input }));
  
  if (createResponse.status !== 201) {
    let errorMsg = JSON.stringify(createResponse.data);
    if (createResponse.data?.type === 'Buffer') {
      errorMsg = Buffer.from(createResponse.data.data).toString('utf8');
    }
    throw new Error(`Failed: ${createResponse.status} - ${errorMsg}`);
  }
  
  const predictionId = createResponse.data.id;
  console.log(`   Prediction: ${predictionId}`);
  
  // Poll for completion
  let attempts = 0;
  while (attempts < 120) {
    await new Promise(r => setTimeout(r, 3000));
    
    const statusResponse = await makeRequest({
      hostname: 'api.replicate.com',
      path: `/v1/predictions/${predictionId}`,
      method: 'GET',
      headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
    });
    
    const status = statusResponse.data.status;
    process.stdout.write(`\r   Status: ${status} (${attempts * 3}s)...      `);
    
    if (status === 'succeeded') {
      console.log('\n   ✅ Video generated!');
      return statusResponse.data.output;
    } else if (status === 'failed') {
      console.log('');
      throw new Error(`Failed: ${statusResponse.data.error}`);
    }
    
    attempts++;
  }
  
  throw new Error('Timeout');
}

async function downloadVideo(url, filename) {
  const outputPath = path.join(outputDir, filename);
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301) {
        return downloadVideo(res.headers.location, filename).then(resolve).catch(reject);
      }
      const file = fs.createWriteStream(outputPath);
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(outputPath); });
    }).on('error', reject);
  });
}

async function main() {
  console.log('═'.repeat(70));
  console.log('🎬 KELLY PHOTO-TO-VIDEO GENERATION');
  console.log('   Creating character-consistent templates from Kelly\'s photos');
  console.log('═'.repeat(70));
  
  const args = process.argv.slice(2);
  const modelArg = args.find(a => a.startsWith('--model='));
  const photoArg = args.find(a => a.startsWith('--photo='));
  
  const modelKey = modelArg ? modelArg.split('=')[1] : 'svd';
  const photoKey = photoArg ? photoArg.split('=')[1] : 'day1_hook';
  
  const model = I2V_MODELS[modelKey];
  const photoUrl = KELLY_PHOTOS[photoKey];
  
  if (!model) {
    console.log('Available models: svd, animatediff');
    return;
  }
  
  if (!photoUrl) {
    console.log('Available photos:', Object.keys(KELLY_PHOTOS).join(', '));
    return;
  }
  
  console.log(`\n🎯 Model: ${model.name}`);
  console.log(`📸 Photo: ${photoKey}`);
  
  const startTime = Date.now();
  
  try {
    const output = await generateFromPhoto(photoUrl, model, 'speaking naturally with subtle head movements');
    
    // Handle different output formats
    let videoUrl;
    if (typeof output === 'string') {
      videoUrl = output;
    } else if (Array.isArray(output)) {
      videoUrl = output[0];
    }
    
    const filename = `kelly_${photoKey}_${modelKey}_${Date.now()}.mp4`;
    const localPath = await downloadVideo(videoUrl, filename);
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    console.log('');
    console.log('═'.repeat(70));
    console.log('✅ KELLY PHOTO-TO-VIDEO SUCCESS!');
    console.log('═'.repeat(70));
    console.log(`   Time: ${duration}s`);
    console.log(`   File: ${localPath}`);
    console.log(`   URL: ${videoUrl}`);
    console.log('');
    console.log('🎯 Next: Apply V2V lipsync to this Kelly-consistent video');
    console.log('═'.repeat(70));
    
  } catch (error) {
    console.log('');
    console.log(`❌ FAILED: ${error.message}`);
  }
}

main();

