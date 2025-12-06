#!/usr/bin/env node
/**
 * High Quality Source Pipeline
 * 
 * Generate higher quality videos from the START:
 * 1. Flux Dev at megapixels "1" (1344×768) - bigger face
 * 2. SVD with higher motion and longer duration
 * 3. Wav2Lip with resize_factor 2 for better face blend
 * 
 * Run: node hq-source-pipeline.cjs --phase hook --archetype "The Explorer"
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const https = require('https');
const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const OUTPUT_DIR = path.join(__dirname, '../../template-forge/hq-source');
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Kelly LoRA on HuggingFace
const KELLY_LORA = 'CuriousKellycom/curious-kelly-lora';

// Phase to prompt mapping with high quality settings
const PHASE_PROMPTS = {
  hook: {
    prompt: "kelly, woman with long wavy brown hair and warm brown eyes, wearing a soft powder blue sweater, standing in a sunlit forest clearing, warm golden hour lighting, looking directly at camera with an excited welcoming expression, arms slightly open in greeting gesture, shallow depth of field, professional portrait photography, 8k, detailed face",
    emotion: "excited"
  },
  q1: {
    prompt: "kelly, woman with long wavy brown hair and warm brown eyes, wearing a soft powder blue sweater, sitting in a cozy library with warm lighting, books in background, curious thoughtful expression, hand near chin in thinking pose, professional portrait, 8k, detailed face",
    emotion: "curious"
  },
  q2: {
    prompt: "kelly, woman with long wavy brown hair and warm brown eyes, wearing a soft powder blue sweater, in a bright modern studio, explaining something with natural hand gestures, engaged confident expression, soft professional lighting, 8k, detailed face",
    emotion: "explaining"
  },
  q3: {
    prompt: "kelly, woman with long wavy brown hair and warm brown eyes, wearing a soft powder blue sweater, in a serene garden setting, contemplative thoughtful expression, gentle natural lighting, 8k, detailed face",
    emotion: "thoughtful"
  },
  wisdom: {
    prompt: "kelly, woman with long wavy brown hair and warm brown eyes, wearing a soft powder blue sweater, in a warm cozy setting with soft lighting, sincere heartfelt expression, hand on heart gesture, emotional connection with viewer, 8k, detailed face",
    emotion: "heartfelt"
  }
};

class ReplicateAPI {
  constructor(token) {
    this.token = token;
  }
  
  async request(method, urlPath, data = null) {
    return new Promise((resolve, reject) => {
      const options = {
        hostname: 'api.replicate.com',
        path: `/v1${urlPath}`,
        method,
        headers: {
          'Authorization': `Bearer ${this.token}`,
          'Content-Type': 'application/json',
        },
      };
      
      const req = https.request(options, (res) => {
        let body = [];
        res.on('data', chunk => body.push(chunk));
        res.on('end', () => {
          try {
            resolve(JSON.parse(Buffer.concat(body).toString()));
          } catch (e) { reject(e); }
        });
      });
      req.on('error', reject);
      if (data) req.write(JSON.stringify(data));
      req.end();
    });
  }
  
  async runAndWait(version, input, label = '') {
    process.stdout.write(`  ${label}...`);
    const prediction = await this.request('POST', '/predictions', { version, input });
    
    while (true) {
      await this.sleep(3000);
      const status = await this.request('GET', `/predictions/${prediction.id}`);
      
      if (status.status === 'succeeded') {
        console.log(` ✅ (${status.metrics?.predict_time?.toFixed(1) || '?'}s)`);
        return status.output;
      }
      if (status.status === 'failed') {
        console.log(' ❌');
        throw new Error(status.error);
      }
      if (status.status === 'canceled') throw new Error('Canceled');
      
      process.stdout.write('.');
    }
  }
  
  sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
}

async function downloadFile(url, filepath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(filepath);
    https.get(url, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        file.close();
        fs.unlinkSync(filepath);
        downloadFile(res.headers.location, filepath).then(resolve).catch(reject);
        return;
      }
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(filepath); });
    }).on('error', (e) => { 
      try { fs.unlinkSync(filepath); } catch(e) {}
      reject(e); 
    });
  });
}

async function main() {
  const args = process.argv.slice(2);
  
  const phaseIndex = args.indexOf('--phase');
  const phase = phaseIndex > -1 ? args[phaseIndex + 1] : 'hook';
  
  const archetypeIndex = args.indexOf('--archetype');
  const archetype = archetypeIndex > -1 ? args[archetypeIndex + 1] : 'The Explorer';
  
  console.log('═'.repeat(70));
  console.log('🎬 HIGH QUALITY SOURCE PIPELINE');
  console.log('   Generating from high-res source → better final quality');
  console.log('═'.repeat(70));
  console.log(`\n  Phase: ${phase}`);
  console.log(`  Archetype: ${archetype}`);
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  const phaseConfig = PHASE_PROMPTS[phase];
  
  if (!phaseConfig) {
    console.log('\n  ❌ Unknown phase:', phase);
    return;
  }
  
  const timestamp = Date.now();
  const prefix = `hq_${phase}_${archetype.replace(/\s+/g, '_')}_${timestamp}`;
  
  // Step 1: Generate HIGH-RES Kelly image with LoRA
  console.log('\n📸 Step 1: Generating high-res Kelly image (1344×768)...');
  
  const imageOutput = await api.runAndWait(
    'black-forest-labs/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
    {
      prompt: phaseConfig.prompt,
      lora_weights: KELLY_LORA,
      lora_scale: 0.85,
      aspect_ratio: "16:9",
      megapixels: "1",  // HIGH RES: 1344×768
      output_format: "png",
      num_inference_steps: 28,
      guidance_scale: 3.5
    },
    '🎨 Flux Dev + Kelly LoRA (megapixels=1)'
  );
  
  const imageUrl = Array.isArray(imageOutput) ? imageOutput[0] : imageOutput;
  const imagePath = path.join(OUTPUT_DIR, `${prefix}_image.png`);
  await downloadFile(imageUrl, imagePath);
  console.log(`     Saved: ${imagePath}`);
  
  // Step 2: Generate LONGER animation with SVD
  console.log('\n🎬 Step 2: Generating animation (25 frames, higher motion)...');
  
  const animOutput = await api.runAndWait(
    'stability-ai/stable-video-diffusion:3f0457e4619be7ac65841f7d7a9f4dfd3bb3637a0f55ca7ab09dd37dfc8f5f08',
    {
      input_image: imageUrl,
      video_length: "25_frames_with_svd_xt",
      sizing_strategy: "maintain_aspect_ratio",
      motion_bucket_id: 40,  // More motion for natural movement
      cond_aug: 0.02,
      fps: 12  // Slightly slower for smoother lipsync
    },
    '🎥 SVD-XT (25 frames, motion=40)'
  );
  
  const animUrl = animOutput;
  const animPath = path.join(OUTPUT_DIR, `${prefix}_animation.mp4`);
  await downloadFile(animUrl, animPath);
  console.log(`     Saved: ${animPath}`);
  
  // Step 3: Get audio
  console.log('\n🎙️ Step 3: Fetching audio...');
  
  const { data: audioData } = await supabase
    .from('kelly_video_assets')
    .select('public_url')
    .eq('day_number', 1)
    .eq('phase', phase)
    .eq('age_bucket', archetype)
    .eq('asset_type', 'audio')
    .single();
  
  if (!audioData) {
    console.log('     ❌ No audio found');
    return;
  }
  console.log(`     ✅ Audio: ${audioData.public_url.split('/').pop()}`);
  
  // Step 4: Apply lipsync with HIGHER quality settings
  console.log('\n👄 Step 4: Applying lipsync (resize_factor=2)...');
  
  const lipsyncOutput = await api.runAndWait(
    'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
    {
      face: animUrl,
      audio: audioData.public_url,
      fps: 25,
      smooth: true,
      resize_factor: 2  // HIGHER quality face blending
    },
    '👄 Wav2Lip (resize_factor=2)'
  );
  
  const videoPath = path.join(OUTPUT_DIR, `${prefix}_final.mp4`);
  await downloadFile(lipsyncOutput, videoPath);
  
  console.log('\n' + '═'.repeat(70));
  console.log('✅ HIGH QUALITY SOURCE PIPELINE COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\n  📁 Output files:`);
  console.log(`     Image:     ${imagePath}`);
  console.log(`     Animation: ${animPath}`);
  console.log(`     Video:     ${videoPath}`);
  console.log(`\n  🔗 Video URL: ${lipsyncOutput}`);
  console.log('\n  Compare this with the standard quality video!');
}

main().catch(console.error);

