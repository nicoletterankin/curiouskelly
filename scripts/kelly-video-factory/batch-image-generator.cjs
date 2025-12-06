#!/usr/bin/env node
/**
 * Kelly Video Factory - Batch Image Generator
 * 
 * Generates all base Kelly images for lessons.
 * These images are reusable across all age/language variants.
 * 
 * Strategy:
 *   30 days × 5 phases = 150 unique images
 *   Each image can be animated once, then lipsynced many times
 * 
 * Run: node batch-image-generator.cjs --days 30 [--dry-run]
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const https = require('https');
const fs = require('fs');
const path = require('path');
const config = require('./config.cjs');

const OUTPUT_DIR = path.join(__dirname, '../../template-forge/production-images');
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Phase-to-template mapping
const PHASE_TEMPLATES = {
  hook: 'excited',       // Grab attention with energy
  q1: 'curious',         // First question - wonder
  q2: 'explain',         // Teaching moment - engaged
  q3: 'thoughtful',      // Deeper question - reflection
  wisdom: 'heartfelt',   // Emotional close - sincere
};

// Replicate API
class ReplicateAPI {
  constructor(token) {
    this.token = token;
    this.versionCache = null;
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
  
  async getVersion() {
    if (!this.versionCache) {
      const r = await this.request('GET', '/models/black-forest-labs/flux-dev-lora');
      this.versionCache = r.latest_version.id;
    }
    return this.versionCache;
  }
  
  async generate(input) {
    const version = await this.getVersion();
    const prediction = await this.request('POST', '/predictions', { version, input });
    
    while (true) {
      await this.sleep(3000);
      const status = await this.request('GET', `/predictions/${prediction.id}`);
      
      if (status.status === 'succeeded') return status.output;
      if (status.status === 'failed') throw new Error(status.error);
      if (status.status === 'canceled') throw new Error('Canceled');
    }
  }
  
  sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
}

function buildPrompt(template) {
  const t = config.templates[template];
  if (!t) throw new Error(`Unknown template: ${template}`);
  
  return t.prompt
    .replace('{triggerWord}', config.lora.triggerWord)
    .replace('{hair}', config.character.hair)
    .replace('{eyes}', config.character.eyes)
    .replace('{outfit}', config.character.outfit);
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
    }).on('error', (e) => { fs.unlinkSync(filepath); reject(e); });
  });
}

async function generateBatchImages(options = {}) {
  const days = options.days || 30;
  const dryRun = options.dryRun || false;
  const phases = Object.keys(PHASE_TEMPLATES);
  const totalImages = days * phases.length;
  
  console.log('═'.repeat(70));
  console.log('🎨 BATCH IMAGE GENERATOR');
  console.log('   Pre-generating Kelly images for production');
  console.log('═'.repeat(70));
  console.log(`\n  Days: ${days}`);
  console.log(`  Phases: ${phases.length} (${phases.join(', ')})`);
  console.log(`  Total images: ${totalImages}`);
  console.log(`  Estimated cost: $${(totalImages * 0.003).toFixed(2)}`);
  console.log(`  Estimated time: ${Math.ceil(totalImages * 30 / 60)} minutes`);
  
  if (dryRun) {
    console.log('\n  [DRY RUN - No images will be generated]\n');
    return;
  }
  
  console.log('\n');
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  const manifest = {
    generated: new Date().toISOString(),
    days,
    phases,
    loraScale: config.lora.scale,
    images: [],
  };
  
  let completed = 0;
  let failed = 0;
  const startTime = Date.now();
  
  for (let day = 1; day <= days; day++) {
    for (const phase of phases) {
      const template = PHASE_TEMPLATES[phase];
      const filename = `day_${String(day).padStart(3, '0')}_${phase}.png`;
      const filepath = path.join(OUTPUT_DIR, filename);
      
      // Skip if already exists
      if (fs.existsSync(filepath)) {
        console.log(`  ⏭️ [${completed + failed + 1}/${totalImages}] ${filename} (exists)`);
        manifest.images.push({
          day, phase, template, filename, 
          status: 'cached',
          localPath: filepath,
        });
        completed++;
        continue;
      }
      
      process.stdout.write(`  🎨 [${completed + failed + 1}/${totalImages}] ${filename}...`);
      
      try {
        const prompt = buildPrompt(template);
        const output = await api.generate({
          prompt,
          negative_prompt: config.character.negativePrompt,
          lora_weights: config.lora.weights,
          lora_scale: config.lora.scale,
          aspect_ratio: '16:9',
          megapixels: '1',
          output_format: 'png',
          output_quality: 100,
          num_inference_steps: 28,
          guidance: 3.5,
        });
        
        const imageUrl = Array.isArray(output) ? output[0] : output;
        await downloadFile(imageUrl, filepath);
        
        manifest.images.push({
          day, phase, template, filename,
          status: 'generated',
          localPath: filepath,
          url: imageUrl,
        });
        
        completed++;
        console.log(' ✅');
        
      } catch (error) {
        failed++;
        console.log(` ❌ ${error.message}`);
        manifest.images.push({
          day, phase, template, filename,
          status: 'failed',
          error: error.message,
        });
      }
      
      // Rate limiting pause
      await api.sleep(1000);
    }
  }
  
  const duration = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  
  // Save manifest
  const manifestPath = path.join(OUTPUT_DIR, 'manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 BATCH GENERATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\n  Completed: ${completed}`);
  console.log(`  Failed: ${failed}`);
  console.log(`  Duration: ${duration} minutes`);
  console.log(`  Manifest: ${manifestPath}`);
  console.log(`  Output: ${OUTPUT_DIR}`);
  
  return manifest;
}

// Progress tracker for resumable generation
function getProgress() {
  const manifestPath = path.join(OUTPUT_DIR, 'manifest.json');
  if (fs.existsSync(manifestPath)) {
    return JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  }
  return null;
}

// Main
async function main() {
  const args = process.argv.slice(2);
  
  const daysIndex = args.indexOf('--days');
  const days = daysIndex > -1 ? parseInt(args[daysIndex + 1]) : 30;
  
  const dryRun = args.includes('--dry-run');
  
  // Check for existing progress
  const existing = getProgress();
  if (existing && !args.includes('--force')) {
    const completed = existing.images.filter(i => i.status !== 'failed').length;
    console.log(`\n  Found existing progress: ${completed}/${existing.images.length} images`);
    console.log(`  Use --force to regenerate all\n`);
  }
  
  await generateBatchImages({ days, dryRun });
}

main().catch(console.error);

