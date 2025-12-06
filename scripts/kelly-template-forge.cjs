/**
 * 🎬 KELLY TEMPLATE FORGE
 * 
 * Professional-grade template video generation system.
 * Goal: Film/Media Major quality - not settling for "good enough"
 * 
 * This script systematically generates and tests template videos
 * until we achieve professional quality.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config();

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;

// Output directories
const OUTPUT_DIR = path.join(__dirname, '..', 'template-forge');
const TEMPLATES_DIR = path.join(OUTPUT_DIR, 'templates');
const TESTS_DIR = path.join(OUTPUT_DIR, 'tests');
fs.mkdirSync(TEMPLATES_DIR, { recursive: true });
fs.mkdirSync(TESTS_DIR, { recursive: true });

// ============================================================================
// VIDEO GENERATION MODELS ON REPLICATE
// ============================================================================

const VIDEO_MODELS = {
  // Minimax Video-01 - High quality text-to-video (6s clips)
  minimax: {
    version: 'minimax/video-01:5aa835260ff7f40f4069c41185f72036accf99e29957bb4a3b3a911f3b6c1912',
    name: 'Minimax Video-01',
    type: 'text-to-video',
    maxDuration: 6,
    quality: 'high',
    inputFormat: 'minimax', // Special format for minimax
  },
  
  // Stable Video Diffusion - Image-to-video
  svd: {
    version: 'stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438',
    name: 'Stable Video Diffusion',
    type: 'image-to-video',
    maxDuration: 4,
    quality: 'medium',
  },
  
  // AnimateDiff - Motion from still image
  animatediff: {
    version: 'lucataco/animate-diff:beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f',
    name: 'AnimateDiff',
    type: 'image-to-video',
    quality: 'medium',
  },
  
  // CogVideoX - Advanced text-to-video
  cogvideo: {
    version: 'fofr/cogvideox-5b:7cdaa7ecf62c8a27db149c509c2aabe12e4c0aaa6ed0d569d922ec90f2891417',
    name: 'CogVideoX-5B',
    type: 'text-to-video',
    maxDuration: 6,
    quality: 'high',
  },
  
  // Luma Dream Machine - Cinematic quality
  luma: {
    version: 'luma/dream-machine',
    name: 'Luma Dream Machine',
    type: 'text-to-video',
    quality: 'cinematic',
  },
};

// ============================================================================
// KELLY'S CHARACTER DEFINITION (Film Production Bible)
// ============================================================================

const KELLY_CHARACTER = {
  appearance: {
    hair: 'wavy brown hair, shoulder length, natural highlights',
    face: 'warm smile, friendly expression, approachable',
    skin: 'natural skin tone, minimal makeup, healthy glow',
    outfit: 'light blue crew-neck sweater, casual but polished',
    age: 'late 20s to early 30s',
  },
  
  environments: {
    forest: 'sunlit forest path, dappled light through trees, green foliage, depth of field',
    garden: 'colorful flower garden, soft natural light, bokeh background',
    library: 'warm library interior, bookshelves, soft lamp light, cozy',
    studio: 'clean neutral background, soft even lighting, professional',
    kitchen: 'bright modern kitchen, natural window light, warm tones',
  },
  
  // Detailed prompts for each template
  templates: {
    T01_welcome_walk: {
      name: 'Welcome Walk',
      duration: '5-6s',
      action: 'walking toward camera, stopping, opening arms in welcome gesture',
      emotion: 'warm, inviting, excited to see viewer',
      camera: 'medium shot tracking to close-up',
      environment: 'forest',
      prompt: 'A warm friendly woman with wavy brown hair wearing a light blue sweater walks toward the camera on a sunlit forest path. She stops and opens her arms in a welcoming gesture with a genuine smile. Cinematic lighting, shallow depth of field, professional film quality, 4K.',
    },
    
    T02_present_explain: {
      name: 'Present & Explain',
      duration: '5-6s',
      action: 'standing, natural hand gestures while explaining, occasional nods',
      emotion: 'engaged, knowledgeable, patient',
      camera: 'medium shot, stable',
      environment: 'studio',
      prompt: 'A friendly woman with wavy brown hair in a light blue sweater stands against a soft neutral background. She gestures naturally with her hands while explaining something, nodding occasionally. Professional studio lighting, clean composition, 4K film quality.',
    },
    
    T03_curious_examine: {
      name: 'Curious Examine',
      duration: '4-5s',
      action: 'looking at something in hands, tilting head, examining closely',
      emotion: 'curious, intrigued, thoughtful',
      camera: 'close-up',
      environment: 'garden',
      prompt: 'Close-up of a curious woman with wavy brown hair examining something small in her hands. She tilts her head thoughtfully, eyes focused with genuine interest. Soft garden background with bokeh, golden hour light, cinematic quality.',
    },
    
    T04_heartfelt_share: {
      name: 'Heartfelt Share',
      duration: '5-6s',
      action: 'hand on heart, sincere expression, gentle nodding',
      emotion: 'sincere, warm, emotional',
      camera: 'close-up',
      environment: 'library',
      prompt: 'A sincere woman with wavy brown hair places her hand on her heart while speaking. Warm expression, gentle nodding, meaningful eye contact with camera. Soft warm library lighting, shallow depth of field, intimate cinematic moment.',
    },
    
    T05_excited_discovery: {
      name: 'Excited Discovery',
      duration: '4-5s',
      action: 'eyes widen, big smile, hands come up in excitement',
      emotion: 'excited, surprised, joyful',
      camera: 'medium to close',
      environment: 'forest',
      prompt: 'A woman with wavy brown hair in a light blue sweater reacts with genuine excitement. Her eyes widen, a big smile spreads across her face, hands come up in an excited gesture. Sunlit forest background, natural lighting, joyful cinematic moment.',
    },
    
    T06_thoughtful_pause: {
      name: 'Thoughtful Pause',
      duration: '4-5s',
      action: 'hand to chin, looking slightly up and away, considering',
      emotion: 'contemplative, wise, reflective',
      camera: 'close-up',
      environment: 'library',
      prompt: 'A thoughtful woman with wavy brown hair brings her hand to her chin, looking slightly upward in contemplation. Reflective expression, slight smile forming. Warm library background, soft lighting, intimate thoughtful moment.',
    },
  },
};

// ============================================================================
// QUALITY CRITERIA (Film School Standards)
// ============================================================================

const QUALITY_CRITERIA = {
  technical: [
    'Consistent lighting throughout clip',
    'No visible artifacts or glitches',
    'Smooth motion, no jitter',
    'Proper exposure and color balance',
    'Sharp focus on subject',
    'Natural skin tones',
  ],
  
  performance: [
    'Natural facial expressions',
    'Believable body movement',
    'Consistent character appearance',
    'Appropriate emotion for context',
    'Smooth gesture transitions',
  ],
  
  cinematic: [
    'Professional composition',
    'Depth of field appropriate for shot',
    'Motivated camera movement',
    'Cohesive visual style',
    'Film-like color grading',
  ],
};

// ============================================================================
// HTTP REQUEST HELPER
// ============================================================================

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

// ============================================================================
// REPLICATE API FUNCTIONS
// ============================================================================

async function listAvailableModels() {
  console.log('\n📋 Checking available video generation models on Replicate...\n');
  
  // Check which models are actually accessible
  const available = [];
  
  for (const [key, model] of Object.entries(VIDEO_MODELS)) {
    try {
      // Try to get model info
      const modelPath = model.version.includes(':') 
        ? model.version.split(':')[0] 
        : model.version;
      
      const response = await makeRequest({
        hostname: 'api.replicate.com',
        path: `/v1/models/${modelPath}`,
        method: 'GET',
        headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
      });
      
      if (response.status === 200) {
        available.push({ key, ...model, accessible: true });
        console.log(`   ✅ ${model.name} - Available`);
      } else {
        console.log(`   ❌ ${model.name} - Not accessible (${response.status})`);
      }
    } catch (e) {
      console.log(`   ⚠️  ${model.name} - Error checking: ${e.message}`);
    }
  }
  
  return available;
}

async function generateVideo(model, prompt, options = {}) {
  console.log(`\n🎬 Generating with ${model.name}...`);
  console.log(`   Prompt: "${prompt.substring(0, 80)}..."`);
  
  const version = model.version.includes(':') 
    ? model.version.split(':')[1] 
    : null;
  
  // Build input based on model type and format
  let input = {};
  
  if (model.inputFormat === 'minimax') {
    // Minimax specific format
    input = {
      prompt,
      prompt_optimizer: true, // Enable prompt optimization for better results
    };
  } else if (model.type === 'text-to-video') {
    input = {
      prompt,
      ...options,
    };
  } else if (model.type === 'image-to-video' && options.image) {
    input = {
      image: options.image,
      ...options,
    };
  }
  
  // Always use version format for Replicate API
  const requestBody = { version, input };
  
  console.log(`   Version: ${version}`);
  console.log(`   Input: ${JSON.stringify(input).substring(0, 100)}...`);
  
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify(requestBody));
  
  if (createResponse.status !== 201) {
    // Try to decode error message
    let errorMsg = JSON.stringify(createResponse.data);
    if (createResponse.data?.type === 'Buffer') {
      errorMsg = Buffer.from(createResponse.data.data).toString('utf8');
    }
    throw new Error(`Failed to create prediction: ${createResponse.status} - ${errorMsg}`);
  }
  
  const predictionId = createResponse.data.id;
  console.log(`   Prediction: ${predictionId}`);
  
  // Poll for completion (3 min for Minimax is typically enough)
  let attempts = 0;
  while (attempts < 200) { // 10 minutes max
    await new Promise(r => setTimeout(r, 3000));
    
    const statusResponse = await makeRequest({
      hostname: 'api.replicate.com',
      path: `/v1/predictions/${predictionId}`,
      method: 'GET',
      headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
    });
    
    const status = statusResponse.data.status;
    const elapsed = attempts * 3;
    process.stdout.write(`\r   Status: ${status} (${elapsed}s)...                    `);
    
    if (status === 'succeeded') {
      console.log('\n   ✅ Video generated!');
      return statusResponse.data.output;
    } else if (status === 'failed') {
      console.log('');
      throw new Error(`Generation failed: ${statusResponse.data.error}`);
    } else if (status === 'canceled') {
      throw new Error('Generation was canceled');
    }
    
    attempts++;
  }
  
  throw new Error('Timeout waiting for video generation');
}

async function downloadFile(url, filepath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(filepath);
    https.get(url, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301) {
        file.close();
        fs.unlinkSync(filepath);
        return downloadFile(res.headers.location, filepath).then(resolve).catch(reject);
      }
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(filepath); });
    }).on('error', (e) => { fs.unlinkSync(filepath); reject(e); });
  });
}

// ============================================================================
// TEMPLATE GENERATION PIPELINE
// ============================================================================

async function generateTemplate(templateKey, model) {
  const template = KELLY_CHARACTER.templates[templateKey];
  if (!template) throw new Error(`Unknown template: ${templateKey}`);
  
  console.log('\n' + '═'.repeat(70));
  console.log(`🎬 GENERATING: ${template.name}`);
  console.log('═'.repeat(70));
  console.log(`   Action: ${template.action}`);
  console.log(`   Emotion: ${template.emotion}`);
  console.log(`   Environment: ${template.environment}`);
  console.log(`   Model: ${model.name}`);
  
  const startTime = Date.now();
  
  try {
    const output = await generateVideo(model, template.prompt, {
      num_frames: 49, // ~6 seconds at 8fps
      fps: 8,
    });
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    // Handle different output formats
    let videoUrl;
    if (typeof output === 'string') {
      videoUrl = output;
    } else if (Array.isArray(output)) {
      videoUrl = output[0];
    } else if (output?.video) {
      videoUrl = output.video;
    }
    
    if (!videoUrl) {
      throw new Error(`No video URL in output: ${JSON.stringify(output)}`);
    }
    
    // Download the video
    const filename = `${templateKey}_${model.name.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}.mp4`;
    const filepath = path.join(TEMPLATES_DIR, filename);
    await downloadFile(videoUrl, filepath);
    
    console.log('\n   ✅ TEMPLATE GENERATED');
    console.log(`   Time: ${duration}s`);
    console.log(`   File: ${filepath}`);
    console.log(`   URL: ${videoUrl}`);
    
    return {
      success: true,
      template: templateKey,
      model: model.name,
      duration,
      filepath,
      videoUrl,
    };
    
  } catch (error) {
    console.log(`\n   ❌ FAILED: ${error.message}`);
    return {
      success: false,
      template: templateKey,
      model: model.name,
      error: error.message,
    };
  }
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

async function main() {
  console.log('═'.repeat(70));
  console.log('🎬 KELLY TEMPLATE FORGE');
  console.log('   Professional-Grade Template Video Generation');
  console.log('═'.repeat(70));
  
  if (!REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not configured');
    process.exit(1);
  }
  
  // Parse command line args
  const args = process.argv.slice(2);
  const listModels = args.includes('--list');
  const templateArg = args.find(a => a.startsWith('--template='));
  const modelArg = args.find(a => a.startsWith('--model='));
  const allTemplates = args.includes('--all');
  
  // List available models
  const availableModels = await listAvailableModels();
  
  if (listModels || availableModels.length === 0) {
    console.log('\n📋 Available templates:');
    for (const [key, template] of Object.entries(KELLY_CHARACTER.templates)) {
      console.log(`   ${key}: ${template.name} (${template.duration})`);
    }
    
    if (availableModels.length === 0) {
      console.log('\n⚠️  No video generation models accessible. Check API token permissions.');
    }
    return;
  }
  
  // Select model
  let selectedModel = availableModels[0]; // Default to first available
  if (modelArg) {
    const modelKey = modelArg.split('=')[1];
    selectedModel = availableModels.find(m => m.key === modelKey) || availableModels[0];
  }
  
  console.log(`\n🎯 Using model: ${selectedModel.name}`);
  
  // Generate templates
  const results = [];
  
  if (allTemplates) {
    // Generate all 6 templates
    for (const templateKey of Object.keys(KELLY_CHARACTER.templates)) {
      const result = await generateTemplate(templateKey, selectedModel);
      results.push(result);
    }
  } else if (templateArg) {
    // Generate specific template
    const templateKey = templateArg.split('=')[1];
    const result = await generateTemplate(templateKey, selectedModel);
    results.push(result);
  } else {
    // Default: generate first template as test
    const result = await generateTemplate('T02_present_explain', selectedModel);
    results.push(result);
  }
  
  // Summary
  console.log('\n' + '═'.repeat(70));
  console.log('📊 GENERATION SUMMARY');
  console.log('═'.repeat(70));
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`   Total: ${results.length}`);
  console.log(`   Success: ${successful.length}`);
  console.log(`   Failed: ${failed.length}`);
  
  if (successful.length > 0) {
    console.log('\n   ✅ Successful:');
    for (const r of successful) {
      console.log(`      - ${r.template} (${r.duration}s): ${r.videoUrl}`);
    }
  }
  
  if (failed.length > 0) {
    console.log('\n   ❌ Failed:');
    for (const r of failed) {
      console.log(`      - ${r.template}: ${r.error}`);
    }
  }
  
  // Save results
  const resultsFile = path.join(OUTPUT_DIR, `forge_results_${Date.now()}.json`);
  fs.writeFileSync(resultsFile, JSON.stringify({ results, timestamp: new Date().toISOString() }, null, 2));
  console.log(`\n   📄 Results saved: ${resultsFile}`);
  
  console.log('\n═'.repeat(70));
  console.log('🎬 Next: Test templates with V2V lipsync pipeline');
  console.log('═'.repeat(70));
}

main().catch(console.error);

