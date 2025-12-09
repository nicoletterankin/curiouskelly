#!/usr/bin/env node
/**
 * Generate lipsynced videos for a day
 * Takes animation + audio → final video
 * 
 * Run: node generate-day-lipsync.cjs --day 1
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

const OUTPUT_DIR = path.join(__dirname, '../../template-forge/lesson-videos');
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

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
  
  async lipsync(videoUrl, audioUrl) {
    const version = '8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef';
    
    const prediction = await this.request('POST', '/predictions', {
      version,
      input: {
        face: videoUrl,
        audio: audioUrl,
        fps: 25,
        smooth: true,
        resize_factor: 1
      }
    });
    
    // Poll for completion
    while (true) {
      await this.sleep(3000);
      const status = await this.request('GET', `/predictions/${prediction.id}`);
      
      if (status.status === 'succeeded') {
        return status.output;
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

async function main() {
  const args = process.argv.slice(2);
  const dayIndex = args.indexOf('--day');
  const day = dayIndex > -1 ? parseInt(args[dayIndex + 1]) : 1;
  const limit = args.includes('--limit') ? parseInt(args[args.indexOf('--limit') + 1]) : null;
  
  console.log('═'.repeat(70));
  console.log('👄 LIPSYNC GENERATION');
  console.log(`   Day ${day}: Animation + Audio → Final Video`);
  console.log('═'.repeat(70));
  
  // Get animations for this day
  const { data: animations } = await supabase
    .from('kelly_video_assets')
    .select('*')
    .eq('day_number', day)
    .eq('asset_type', 'animation');
  
  if (!animations || animations.length === 0) {
    console.log('\n  ❌ No animations found for this day');
    console.log('  Run: node batch-animation-generator.cjs --days ' + day);
    return;
  }
  
  // Map animations by phase
  const animMap = {};
  animations.forEach(a => { animMap[a.phase] = a.public_url; });
  console.log(`\n  Animations: ${Object.keys(animMap).length} phases`);
  
  // Get audio files for this day
  const { data: audioFiles } = await supabase
    .from('kelly_video_assets')
    .select('*')
    .eq('day_number', day)
    .eq('asset_type', 'audio');
  
  if (!audioFiles || audioFiles.length === 0) {
    console.log('\n  ❌ No audio found for this day');
    console.log('  Run: node generate-day-audio.cjs --day ' + day);
    return;
  }
  
  const toProcess = limit ? audioFiles.slice(0, limit) : audioFiles;
  console.log(`  Audio files: ${audioFiles.length}, processing ${toProcess.length}`);
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  let generated = 0;
  let failed = 0;
  let cost = 0;
  const startTime = Date.now();
  
  console.log('\n');
  
  for (const audio of toProcess) {
    const animationUrl = animMap[audio.phase];
    if (!animationUrl) {
      console.log(`  ⏭️ Skipping ${audio.age_bucket} ${audio.phase} - no animation`);
      continue;
    }
    
    const archetype = audio.age_bucket.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '');
    const filename = `day_${String(day).padStart(3, '0')}_${audio.phase}_${archetype}.mp4`;
    
    process.stdout.write(`  👄 ${filename}...`);
    
    try {
      // Apply lipsync
      const videoUrl = await api.lipsync(animationUrl, audio.public_url);
      
      // Download locally
      const localPath = path.join(OUTPUT_DIR, filename);
      await downloadFile(videoUrl, localPath);
      
      // Upload to Supabase
      const storagePath = `production/videos/${filename}`;
      const publicUrl = await uploadToSupabase(localPath, storagePath);
      
      // Register in database
      await supabase.from('kelly_video_assets').insert({
        day_number: day,
        phase: audio.phase,
        template: audio.template,
        asset_type: 'video',
        age_bucket: audio.age_bucket,
        language: audio.language,
        storage_bucket: 'kelly-templates',
        storage_path: storagePath,
        public_url: publicUrl,
        quality_tier: 'standard',
        status: 'generated',
        generation_cost_usd: 0.02
      });
      
      generated++;
      cost += 0.02;
      console.log(' ✅');
      
    } catch (err) {
      failed++;
      console.log(` ❌ ${err.message}`);
    }
  }
  
  const duration = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 LIPSYNC GENERATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\n  Generated: ${generated}`);
  console.log(`  Failed: ${failed}`);
  console.log(`  Duration: ${duration} minutes`);
  console.log(`  Cost: $${cost.toFixed(2)}`);
  console.log(`  Output: ${OUTPUT_DIR}`);
}

main().catch(console.error);


