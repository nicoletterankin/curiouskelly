#!/usr/bin/env node
/**
 * Kelly Video Factory - Systematic Calibration
 * 
 * Tests multiple parameters to find optimal settings for:
 * - Character consistency (face match)
 * - Visual quality
 * - Speed/reliability
 * 
 * Run: node systematic-calibration.cjs [--lora | --motion | --full]
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const https = require('https');
const fs = require('fs');
const path = require('path');
const config = require('./config.cjs');

const CALIBRATION_DIR = path.join(__dirname, '../../template-forge/calibration');
fs.mkdirSync(CALIBRATION_DIR, { recursive: true });

// Replicate API helpers
class ReplicateAPI {
  constructor(token) {
    this.token = token;
  }
  
  async request(method, path, data = null) {
    return new Promise((resolve, reject) => {
      const options = {
        hostname: 'api.replicate.com',
        path: `/v1${path}`,
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
          } catch (e) {
            reject(e);
          }
        });
      });
      req.on('error', reject);
      if (data) req.write(JSON.stringify(data));
      req.end();
    });
  }
  
  async getModelVersion(modelId) {
    const r = await this.request('GET', `/models/${modelId}`);
    return r.latest_version.id;
  }
  
  async runAndWait(version, input, onProgress = () => {}) {
    const prediction = await this.request('POST', '/predictions', { version, input });
    
    while (true) {
      await this.sleep(3000);
      const status = await this.request('GET', `/predictions/${prediction.id}`);
      onProgress(status.status);
      
      if (status.status === 'succeeded') return status.output;
      if (status.status === 'failed') throw new Error(status.error);
      if (status.status === 'canceled') throw new Error('Canceled');
    }
  }
  
  sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
}

// Test configurations
const LORA_SCALE_TESTS = [
  { name: 'lora_075', scale: 0.75, description: 'Lower LoRA influence' },
  { name: 'lora_080', scale: 0.80, description: 'Moderate-low LoRA' },
  { name: 'lora_085', scale: 0.85, description: 'Balanced (current)' },
  { name: 'lora_090', scale: 0.90, description: 'Moderate-high LoRA' },
  { name: 'lora_095', scale: 0.95, description: 'High LoRA influence' },
];

const MOTION_TESTS = [
  { name: 'motion_40', bucket: 40, description: 'Very subtle motion' },
  { name: 'motion_60', bucket: 60, description: 'Subtle motion' },
  { name: 'motion_80', bucket: 80, description: 'Natural motion (current)' },
  { name: 'motion_100', bucket: 100, description: 'Moderate motion' },
  { name: 'motion_127', bucket: 127, description: 'Maximum motion' },
];

const TEST_PROMPTS = {
  explain: `${config.lora.triggerWord}, woman with ${config.character.hair} and ${config.character.eyes}, wearing ${config.character.outfit}, sitting in directors chair in studio with dark background, natural hand gestures while explaining, engaged expression, professional lighting, 4K`,
  heartfelt: `${config.lora.triggerWord}, woman with ${config.character.hair} and ${config.character.eyes}, wearing ${config.character.outfit}, hand on heart, sincere warm emotional expression, soft golden lighting, close up portrait, 4K`,
};

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
    }).on('error', (e) => { fs.unlinkSync(filepath); reject(e); });
  });
}

async function runLoRACalibration(api) {
  console.log('\n' + '═'.repeat(70));
  console.log('🔬 LORA SCALE CALIBRATION');
  console.log('   Testing character consistency across different LoRA strengths');
  console.log('═'.repeat(70) + '\n');
  
  const version = await api.getModelVersion('black-forest-labs/flux-dev-lora');
  const results = [];
  const sessionDir = path.join(CALIBRATION_DIR, `lora_${Date.now()}`);
  fs.mkdirSync(sessionDir, { recursive: true });
  
  for (const test of LORA_SCALE_TESTS) {
    console.log(`\n▶ Testing ${test.name} (scale: ${test.scale})`);
    console.log(`  ${test.description}`);
    
    const startTime = Date.now();
    
    try {
      // Generate image with this LoRA scale
      const output = await api.runAndWait(version, {
        prompt: TEST_PROMPTS.explain,
        negative_prompt: config.character.negativePrompt,
        lora_weights: config.lora.weights,
        lora_scale: test.scale,
        aspect_ratio: '16:9',
        megapixels: '1',
        output_format: 'png',
        output_quality: 100,
        num_inference_steps: 28,
        guidance: 3.5,
      }, (s) => process.stdout.write(`\r  Status: ${s}...          `));
      
      const imageUrl = Array.isArray(output) ? output[0] : output;
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      
      // Download image
      const filename = `${test.name}_explain.png`;
      const filepath = path.join(sessionDir, filename);
      await downloadFile(imageUrl, filepath);
      
      console.log(`\n  ✅ Generated in ${duration}s`);
      console.log(`     Saved: ${filename}`);
      
      results.push({
        test: test.name,
        scale: test.scale,
        description: test.description,
        success: true,
        duration: parseFloat(duration),
        imageUrl,
        localPath: filepath,
      });
      
    } catch (error) {
      console.log(`\n  ❌ Failed: ${error.message}`);
      results.push({
        test: test.name,
        scale: test.scale,
        success: false,
        error: error.message,
      });
    }
    
    // Brief pause between tests
    await api.sleep(2000);
  }
  
  // Generate comparison HTML
  const htmlPath = path.join(sessionDir, 'comparison.html');
  generateComparisonHTML(results, htmlPath, 'LoRA Scale');
  
  // Save results JSON
  const jsonPath = path.join(sessionDir, 'results.json');
  fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 CALIBRATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\n  Results: ${jsonPath}`);
  console.log(`  Comparison: ${htmlPath}`);
  console.log(`  Successful: ${results.filter(r => r.success).length}/${results.length}`);
  
  return { sessionDir, results };
}

async function runMotionCalibration(api, baseImageUrl) {
  console.log('\n' + '═'.repeat(70));
  console.log('🎬 MOTION CALIBRATION');
  console.log('   Testing animation smoothness across motion bucket IDs');
  console.log('═'.repeat(70) + '\n');
  
  const version = '3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438';
  const results = [];
  const sessionDir = path.join(CALIBRATION_DIR, `motion_${Date.now()}`);
  fs.mkdirSync(sessionDir, { recursive: true });
  
  // If no base image provided, generate one first
  if (!baseImageUrl) {
    console.log('Generating base image first...');
    const imgVersion = await api.getModelVersion('black-forest-labs/flux-dev-lora');
    const output = await api.runAndWait(imgVersion, {
      prompt: TEST_PROMPTS.explain,
      lora_weights: config.lora.weights,
      lora_scale: 0.85,
      aspect_ratio: '16:9',
      megapixels: '1',
      output_format: 'png',
    }, (s) => process.stdout.write(`\r  Status: ${s}...          `));
    baseImageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`\n  ✅ Base image ready`);
  }
  
  for (const test of MOTION_TESTS) {
    console.log(`\n▶ Testing ${test.name} (bucket: ${test.bucket})`);
    console.log(`  ${test.description}`);
    
    const startTime = Date.now();
    
    try {
      const output = await api.runAndWait(version, {
        input_image: baseImageUrl,
        video_length: '14_frames_with_svd',
        fps: 8,
        motion_bucket_id: test.bucket,
        cond_aug: 0.02,
        decoding_t: 7,
      }, (s) => process.stdout.write(`\r  Status: ${s}...          `));
      
      const videoUrl = Array.isArray(output) ? output[0] : output;
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      
      // Download video
      const filename = `${test.name}.mp4`;
      const filepath = path.join(sessionDir, filename);
      await downloadFile(videoUrl, filepath);
      
      console.log(`\n  ✅ Generated in ${duration}s`);
      
      results.push({
        test: test.name,
        bucket: test.bucket,
        description: test.description,
        success: true,
        duration: parseFloat(duration),
        videoUrl,
        localPath: filepath,
      });
      
    } catch (error) {
      console.log(`\n  ❌ Failed: ${error.message}`);
      results.push({
        test: test.name,
        bucket: test.bucket,
        success: false,
        error: error.message,
      });
    }
    
    await api.sleep(2000);
  }
  
  // Generate comparison HTML
  const htmlPath = path.join(sessionDir, 'comparison.html');
  generateMotionComparisonHTML(results, htmlPath);
  
  // Save results JSON
  const jsonPath = path.join(sessionDir, 'results.json');
  fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 MOTION CALIBRATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\n  Results: ${jsonPath}`);
  console.log(`  Comparison: ${htmlPath}`);
  
  return { sessionDir, results };
}

function generateComparisonHTML(results, filepath, title) {
  const cards = results.filter(r => r.success).map(r => `
    <div class="card">
      <img src="${r.localPath.split('\\').pop()}" alt="${r.test}">
      <div class="info">
        <strong>${r.test}</strong><br>
        Scale: ${r.scale}<br>
        Time: ${r.duration}s
      </div>
    </div>
  `).join('');
  
  const html = `<!DOCTYPE html>
<html>
<head>
  <title>${title} Calibration</title>
  <style>
    body { font-family: system-ui; background: #0a0a0f; color: #eee; padding: 2rem; }
    h1 { color: #10b981; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }
    .card { background: #12121a; border-radius: 12px; overflow: hidden; }
    .card img { width: 100%; }
    .info { padding: 1rem; }
    .recommendation { background: rgba(16,185,129,0.1); border: 1px solid #10b981; padding: 1rem; border-radius: 8px; margin: 2rem 0; }
  </style>
</head>
<body>
  <h1>${title} Calibration Results</h1>
  <p>Generated: ${new Date().toISOString()}</p>
  
  <div class="recommendation">
    <strong>📋 To Review:</strong> Open each image and assess:
    <ol>
      <li>Is it Kelly? (hair, eyes, face shape)</li>
      <li>Is the sweater blue?</li>
      <li>Is the pose/expression correct?</li>
    </ol>
  </div>
  
  <div class="grid">${cards}</div>
</body>
</html>`;
  
  fs.writeFileSync(filepath, html);
}

function generateMotionComparisonHTML(results, filepath) {
  const cards = results.filter(r => r.success).map(r => `
    <div class="card">
      <video controls muted loop>
        <source src="${r.localPath.split('\\').pop()}" type="video/mp4">
      </video>
      <div class="info">
        <strong>${r.test}</strong><br>
        Bucket: ${r.bucket}<br>
        Time: ${r.duration}s
      </div>
    </div>
  `).join('');
  
  const html = `<!DOCTYPE html>
<html>
<head>
  <title>Motion Calibration</title>
  <style>
    body { font-family: system-ui; background: #0a0a0f; color: #eee; padding: 2rem; }
    h1 { color: #f59e0b; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }
    .card { background: #12121a; border-radius: 12px; overflow: hidden; }
    .card video { width: 100%; }
    .info { padding: 1rem; }
    .recommendation { background: rgba(245,158,11,0.1); border: 1px solid #f59e0b; padding: 1rem; border-radius: 8px; margin: 2rem 0; }
  </style>
</head>
<body>
  <h1>Motion Calibration Results</h1>
  <p>Generated: ${new Date().toISOString()}</p>
  
  <div class="recommendation">
    <strong>📋 To Review:</strong> Play each video and assess:
    <ol>
      <li>Is the motion natural?</li>
      <li>Is there distortion or artifacts?</li>
      <li>Does the face stay consistent?</li>
    </ol>
  </div>
  
  <div class="grid">${cards}</div>
</body>
</html>`;
  
  fs.writeFileSync(filepath, html);
}

// Main
async function main() {
  const args = process.argv.slice(2);
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  
  if (args.includes('--lora')) {
    await runLoRACalibration(api);
  } else if (args.includes('--motion')) {
    const imageUrl = args.find(a => a.startsWith('http'));
    await runMotionCalibration(api, imageUrl);
  } else if (args.includes('--full')) {
    const loraResult = await runLoRACalibration(api);
    // Use the best LoRA result for motion testing
    const bestLora = loraResult.results.find(r => r.success && r.scale === 0.85);
    if (bestLora) {
      await runMotionCalibration(api, bestLora.imageUrl);
    }
  } else {
    console.log(`
Systematic Calibration

Usage:
  node systematic-calibration.cjs --lora     Test LoRA scales (0.75-0.95)
  node systematic-calibration.cjs --motion   Test motion buckets (40-127)
  node systematic-calibration.cjs --full     Run both calibrations
`);
  }
}

main().catch(console.error);


