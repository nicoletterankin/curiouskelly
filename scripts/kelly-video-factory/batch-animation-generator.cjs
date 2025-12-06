#!/usr/bin/env node
/**
 * Kelly Video Factory - Batch Animation Generator
 * 
 * Generates animations from existing Kelly images.
 * Key optimization: One animation per image, then many lipsync variants.
 * 
 * Run: node batch-animation-generator.cjs --days 5 [--dry-run]
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const https = require('https');
const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');
const config = require('./config.cjs');

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const OUTPUT_DIR = path.join(__dirname, '../../template-forge/production-animations');
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Replicate API
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
  
  async animate(imageUrl, options = {}) {
    const version = config.models.animation.svd.version;
    const videoLength = options.frames === 25 ? 'svd_xt' : 'svd';
    
    const prediction = await this.request('POST', '/predictions', {
      version,
      input: {
        input_image: imageUrl,
        video_length: videoLength === 'svd_xt' ? '25_frames_with_svd_xt' : '14_frames_with_svd',
        fps: options.fps || 8,
        motion_bucket_id: options.motion || 80,
        cond_aug: 0.02,
        decoding_t: 7,
      }
    });
    
    // Poll for completion
    while (true) {
      await this.sleep(5000);
      const status = await this.request('GET', `/predictions/${prediction.id}`);
      
      if (status.status === 'succeeded') {
        return {
          url: Array.isArray(status.output) ? status.output[0] : status.output,
          predictionId: prediction.id,
        };
      }
      if (status.status === 'failed') throw new Error(status.error);
      if (status.status === 'canceled') throw new Error('Canceled');
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
    }).on('error', (e) => { fs.unlinkSync(filepath); reject(e); });
  });
}

async function uploadToSupabase(filepath, storagePath) {
  const buffer = fs.readFileSync(filepath);
  await supabase.storage.from('kelly-templates').upload(storagePath, buffer, {
    contentType: 'video/mp4',
    upsert: true
  });
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(storagePath);
  return data.publicUrl;
}

async function generateBatchAnimations(options = {}) {
  const days = options.days || 5;
  const dryRun = options.dryRun || false;
  
  console.log('═'.repeat(70));
  console.log('🎬 BATCH ANIMATION GENERATOR');
  console.log('   Animating Kelly images for production');
  console.log('═'.repeat(70));
  
  // Get images that need animation
  const { data: images, error } = await supabase
    .from('kelly_video_assets')
    .select('*')
    .eq('asset_type', 'image')
    .lte('day_number', days)
    .order('day_number')
    .order('phase');
  
  if (error) {
    console.log('Error fetching images:', error.message);
    return;
  }
  
  // Check which already have animations
  const { data: animations } = await supabase
    .from('kelly_video_assets')
    .select('day_number, phase')
    .eq('asset_type', 'animation')
    .lte('day_number', days);
  
  const existingSet = new Set(animations?.map(a => `${a.day_number}-${a.phase}`) || []);
  const toGenerate = images.filter(img => !existingSet.has(`${img.day_number}-${img.phase}`));
  
  console.log(`\n  Total images: ${images.length}`);
  console.log(`  Already animated: ${existingSet.size}`);
  console.log(`  Need animation: ${toGenerate.length}`);
  console.log(`  Estimated cost: $${(toGenerate.length * 0.05).toFixed(2)}`);
  console.log(`  Estimated time: ${Math.ceil(toGenerate.length * 2)} minutes`);
  
  if (dryRun) {
    console.log('\n  [DRY RUN - No animations will be generated]\n');
    return;
  }
  
  if (toGenerate.length === 0) {
    console.log('\n  ✅ All images already have animations\n');
    return;
  }
  
  console.log('\n');
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  let completed = 0;
  let failed = 0;
  const startTime = Date.now();
  
  for (const img of toGenerate) {
    const filename = `day_${String(img.day_number).padStart(3, '0')}_${img.phase}.mp4`;
    const filepath = path.join(OUTPUT_DIR, filename);
    
    process.stdout.write(`  🎬 [${completed + failed + 1}/${toGenerate.length}] ${filename}...`);
    
    try {
      // Generate animation
      const result = await api.animate(img.public_url, {
        frames: 25, // Use SVD-XT for better quality
        motion: 80,
        fps: 8
      });
      
      // Download
      await downloadFile(result.url, filepath);
      
      // Upload to Supabase
      const storagePath = `production/animations/${filename}`;
      const publicUrl = await uploadToSupabase(filepath, storagePath);
      
      // Register in database
      await supabase.from('kelly_video_assets').insert({
        day_number: img.day_number,
        phase: img.phase,
        template: img.template,
        asset_type: 'animation',
        storage_bucket: 'kelly-templates',
        storage_path: storagePath,
        public_url: publicUrl,
        quality_tier: 'standard',
        resolution: img.resolution,
        duration_seconds: 3.125, // 25 frames at 8fps
        lora_scale: img.lora_scale,
        replicate_prediction_id: result.predictionId,
        generation_cost_usd: 0.05,
        status: 'generated'
      });
      
      completed++;
      console.log(' ✅');
      
    } catch (error) {
      failed++;
      console.log(` ❌ ${error.message}`);
    }
  }
  
  const duration = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 BATCH ANIMATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\n  Completed: ${completed}`);
  console.log(`  Failed: ${failed}`);
  console.log(`  Duration: ${duration} minutes`);
  console.log(`  Output: ${OUTPUT_DIR}`);
}

// Main
async function main() {
  const args = process.argv.slice(2);
  
  const daysIndex = args.indexOf('--days');
  const days = daysIndex > -1 ? parseInt(args[daysIndex + 1]) : 5;
  
  const dryRun = args.includes('--dry-run');
  
  await generateBatchAnimations({ days, dryRun });
}

main().catch(console.error);

