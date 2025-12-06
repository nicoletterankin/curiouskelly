/**
 * 🎬 KELLY LORA PIPELINE
 * 
 * THE CORRECT APPROACH:
 * 1. Generate Kelly image using Flux Dev + Kelly LoRA (character consistent!)
 * 2. Animate the image using image-to-video
 * 3. Apply V2V lipsync with ElevenLabs audio
 * 
 * This produces REAL KELLY, not AI slop.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config();

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

// THE KELLY LORA - This is what makes her REAL KELLY
const KELLY_LORA = 'huggingface.co/CuriousKellycom/curious-kelly-lora';
const KELLY_LORA_SCALE = 0.85;

// Models
const FLUX_DEV_LORA = 'black-forest-labs/flux-dev-lora';
const SVD_MODEL = 'stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438';
const WAV2LIP_MODEL = 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef';

const outputDir = path.join(__dirname, '..', 'template-forge', 'kelly-lora');
fs.mkdirSync(outputDir, { recursive: true });

// Template prompts - MUST include "kelly" trigger word
// IMPORTANT: Emphasize LIGHT BLUE sweater strongly to override LoRA defaults
const KELLY_TEMPLATES = {
  welcome: {
    name: 'Kelly Welcome',
    prompt: 'kelly, woman with long wavy brown hair and brown eyes, wearing light blue sweater, standing on sunlit forest path, arms open in welcoming gesture, warm genuine smile, full body shot, professional photography, 4K, blue sweater',
    aspect_ratio: '16:9',
  },
  explain: {
    name: 'Kelly Explain', 
    prompt: 'kelly, woman with long wavy brown hair and brown eyes, wearing light blue sweater, sitting in directors chair in studio with dark background, natural hand gestures while explaining, engaged expression, professional lighting, 4K, blue sweater',
    aspect_ratio: '16:9',
  },
  heartfelt: {
    name: 'Kelly Heartfelt',
    prompt: 'kelly, woman with long wavy brown hair and brown eyes, wearing light blue sweater, hand on heart, sincere warm emotional expression, soft golden lighting, close up portrait, 4K, blue sweater',
    aspect_ratio: '16:9',
  },
  curious: {
    name: 'Kelly Curious',
    prompt: 'kelly, woman with long wavy brown hair and brown eyes, wearing light blue sweater, tilting head thoughtfully with curious expression, examining something in hands, soft natural lighting, 4K, blue sweater',
    aspect_ratio: '16:9',
  },
  excited: {
    name: 'Kelly Excited',
    prompt: 'kelly, woman with long wavy brown hair and brown eyes, wearing light blue sweater, eyes wide with excitement, big joyful smile, hands raised in excitement, bright cheerful lighting, 4K, blue sweater',
    aspect_ratio: '16:9',
  },
};

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

async function runReplicate(modelVersion, input) {
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify({ version: modelVersion, input }));
  
  if (createResponse.status !== 201) {
    let errorMsg = JSON.stringify(createResponse.data);
    if (createResponse.data?.type === 'Buffer') {
      errorMsg = Buffer.from(createResponse.data.data).toString('utf8');
    }
    throw new Error(`Failed: ${createResponse.status} - ${errorMsg}`);
  }
  
  const predictionId = createResponse.data.id;
  console.log(`   Prediction: ${predictionId}`);
  
  // Poll
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
    process.stdout.write(`\r   Status: ${status} (${attempts * 3}s)...                    `);
    
    if (status === 'succeeded') {
      console.log('');
      return statusResponse.data.output;
    } else if (status === 'failed') {
      console.log('');
      throw new Error(`Failed: ${statusResponse.data.error}`);
    }
    attempts++;
  }
  throw new Error('Timeout');
}

// Step 1: Generate Kelly image using LoRA
async function generateKellyImage(template) {
  console.log('\n🎨 Step 1: Generating Kelly image with LoRA...');
  console.log(`   LoRA: ${KELLY_LORA}`);
  console.log(`   Scale: ${KELLY_LORA_SCALE}`);
  console.log(`   Prompt: "${template.prompt.substring(0, 60)}..."`);
  
  // Get model version
  const modelResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: `/v1/models/${FLUX_DEV_LORA}`,
    method: 'GET',
    headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
  });
  
  const version = modelResponse.data.latest_version.id;
  console.log(`   Model version: ${version.substring(0, 20)}...`);
  
  const output = await runReplicate(version, {
    prompt: template.prompt,
    lora_weights: KELLY_LORA,
    lora_scale: KELLY_LORA_SCALE,
    aspect_ratio: template.aspect_ratio || '16:9',
    output_format: 'png',
    output_quality: 100,
    num_inference_steps: 28,
    guidance: 3.5,
  });
  
  const imageUrl = Array.isArray(output) ? output[0] : output;
  console.log(`   ✅ Kelly image generated: ${imageUrl}`);
  return imageUrl;
}

// Step 2: Animate the image
async function animateImage(imageUrl) {
  console.log('\n🎬 Step 2: Animating Kelly image...');
  
  const output = await runReplicate(SVD_MODEL.split(':')[1], {
    input_image: imageUrl,
    video_length: '25_frames_with_svd_xt',
    fps: 8,
    motion_bucket_id: 80, // Subtle motion
    cond_aug: 0.02,
  });
  
  const videoUrl = Array.isArray(output) ? output[0] : output;
  console.log(`   ✅ Animation generated: ${videoUrl}`);
  return videoUrl;
}

// Step 3: Generate audio
async function generateAudio(text) {
  console.log('\n🎙️ Step 3: Generating Kelly voice...');
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
    voice_settings: { stability: 0.5, similarity_boost: 0.85 }
  }));
  
  if (response.status !== 200) throw new Error(`ElevenLabs error: ${response.status}`);
  console.log(`   ✅ Audio: ${(response.data.length / 1024).toFixed(1)}KB`);
  return response.data;
}

// Step 4: Apply V2V lipsync
async function applyLipsync(videoUrl, audioBuffer) {
  console.log('\n👄 Step 4: Applying V2V lipsync...');
  
  const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  const output = await runReplicate(WAV2LIP_MODEL.split(':')[1], {
    face: videoUrl,
    audio: audioBase64,
    fps: 25,
    smooth: true,
  });
  
  console.log(`   ✅ Lipsync applied: ${output}`);
  return output;
}

async function downloadFile(url, filepath) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301) {
        return downloadFile(res.headers.location, filepath).then(resolve).catch(reject);
      }
      const file = fs.createWriteStream(filepath);
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(filepath); });
    }).on('error', reject);
  });
}

async function main() {
  console.log('═'.repeat(70));
  console.log('🎬 KELLY LORA PIPELINE');
  console.log('   Generate REAL KELLY using trained LoRA');
  console.log('═'.repeat(70));
  
  const args = process.argv.slice(2);
  const templateKey = args[0] || 'explain';
  const scriptText = args[1] || 'Hello curious learner! Today we are going to discover something amazing about how our brains work.';
  
  const template = KELLY_TEMPLATES[templateKey];
  if (!template) {
    console.log('\nAvailable templates:', Object.keys(KELLY_TEMPLATES).join(', '));
    return;
  }
  
  console.log(`\n🎯 Template: ${template.name}`);
  console.log(`📝 Script: "${scriptText.substring(0, 50)}..."`);
  
  const startTime = Date.now();
  
  try {
    // Full pipeline
    const imageUrl = await generateKellyImage(template);
    const animationUrl = await animateImage(imageUrl);
    const audioBuffer = await generateAudio(scriptText);
    const finalVideoUrl = await applyLipsync(animationUrl, audioBuffer);
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    // Download final video
    const filename = `kelly_lora_${templateKey}_${Date.now()}.mp4`;
    const filepath = path.join(outputDir, filename);
    await downloadFile(finalVideoUrl, filepath);
    
    // Also download the Kelly image for face audit
    const imgFilename = `kelly_lora_${templateKey}_${Date.now()}.png`;
    const imgFilepath = path.join(outputDir, imgFilename);
    await downloadFile(imageUrl, imgFilepath);
    
    console.log('\n' + '═'.repeat(70));
    console.log('✅ KELLY LORA PIPELINE COMPLETE!');
    console.log('═'.repeat(70));
    console.log(`   Total time: ${duration}s`);
    console.log(`   Kelly image: ${imgFilepath}`);
    console.log(`   Final video: ${filepath}`);
    console.log(`   Video URL: ${finalVideoUrl}`);
    console.log('');
    console.log('🎯 Now run face audit on the image:');
    console.log(`   python kelly_face_audit.py ${imgFilepath}`);
    console.log('═'.repeat(70));
    
    // Save result
    fs.writeFileSync(
      path.join(outputDir, `kelly_lora_${templateKey}_result.json`),
      JSON.stringify({
        template: templateKey,
        prompt: template.prompt,
        script: scriptText,
        imageUrl,
        animationUrl,
        finalVideoUrl,
        localImage: imgFilepath,
        localVideo: filepath,
        duration,
        timestamp: new Date().toISOString(),
      }, null, 2)
    );
    
  } catch (error) {
    console.log('\n' + '═'.repeat(70));
    console.log(`❌ PIPELINE FAILED: ${error.message}`);
    console.log('═'.repeat(70));
  }
}

main();

