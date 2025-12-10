#!/usr/bin/env node
/**
 * Regenerate High Quality Videos
 * 
 * Current issues with Wav2Lip:
 * - Face region is 256x256 pixels (blurry)
 * - No upscaling applied
 * 
 * Solution:
 * 1. Use Real-ESRGAN to upscale the final video to 4K
 * 2. Use higher quality base animations
 * 3. Consider alternative lipsync models (SadTalker HQ)
 * 
 * Run: node regenerate-hq.cjs --day 1 --phase hook --archetype "The Explorer"
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

const OUTPUT_DIR = path.join(__dirname, '../../template-forge/hq-videos');
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
  
  async runAndWait(version, input) {
    const prediction = await this.request('POST', '/predictions', { version, input });
    
    while (true) {
      await this.sleep(3000);
      const status = await this.request('GET', `/predictions/${prediction.id}`);
      
      if (status.status === 'succeeded') return status.output;
      if (status.status === 'failed') throw new Error(status.error);
      if (status.status === 'canceled') throw new Error('Canceled');
      
      process.stdout.write('.');
    }
  }
  
  sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
}

async function upscaleVideo(api, videoUrl) {
  console.log('\n  🔍 Upscaling with Real-ESRGAN...');
  
  // Real-ESRGAN video upscaler
  const output = await api.runAndWait(
    'lucataco/real-esrgan-video:c23768236472a5952c35606e5e88c20c7ce4fc5bcb1b97ee0ad4e0da61c24d4d',
    {
      video_path: videoUrl,
      scale: 4,
      face_enhance: true
    }
  );
  
  return output;
}

async function generateHQVideo(api, animationUrl, audioUrl) {
  console.log('\n  👄 Generating lipsync (Wav2Lip)...');
  
  // First, apply lipsync
  const lipsyncOutput = await api.runAndWait(
    'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
    {
      face: animationUrl,
      audio: audioUrl,
      fps: 25,
      smooth: true,
      resize_factor: 1
    }
  );
  
  console.log(' ✅');
  
  // Then upscale
  const hqOutput = await upscaleVideo(api, lipsyncOutput);
  console.log(' ✅');
  
  return hqOutput;
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

async function main() {
  const args = process.argv.slice(2);
  
  const dayIndex = args.indexOf('--day');
  const day = dayIndex > -1 ? parseInt(args[dayIndex + 1]) : 1;
  
  const phaseIndex = args.indexOf('--phase');
  const phase = phaseIndex > -1 ? args[phaseIndex + 1] : 'hook';
  
  const archetypeIndex = args.indexOf('--archetype');
  const archetype = archetypeIndex > -1 ? args[archetypeIndex + 1] : 'The Explorer';
  
  console.log('═'.repeat(70));
  console.log('🎬 HIGH QUALITY VIDEO REGENERATION');
  console.log('═'.repeat(70));
  console.log(`\n  Day: ${day}`);
  console.log(`  Phase: ${phase}`);
  console.log(`  Archetype: ${archetype}`);
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  
  // Get animation URL
  const { data: animData } = await supabase
    .from('kelly_video_assets')
    .select('public_url')
    .eq('day_number', day)
    .eq('phase', phase)
    .eq('asset_type', 'animation')
    .single();
  
  if (!animData) {
    console.log('\n  ❌ No animation found');
    return;
  }
  
  // Get audio URL
  const { data: audioData } = await supabase
    .from('kelly_video_assets')
    .select('public_url')
    .eq('day_number', day)
    .eq('phase', phase)
    .eq('age_bucket', archetype)
    .eq('asset_type', 'audio')
    .single();
  
  if (!audioData) {
    console.log('\n  ❌ No audio found');
    return;
  }
  
  console.log(`\n  Animation: ${animData.public_url.split('/').pop()}`);
  console.log(`  Audio: ${audioData.public_url.split('/').pop()}`);
  
  try {
    const hqVideoUrl = await generateHQVideo(api, animData.public_url, audioData.public_url);
    
    // Download
    const filename = `day_${String(day).padStart(3, '0')}_${phase}_${archetype.replace(/\s+/g, '_')}_HQ.mp4`;
    const filepath = path.join(OUTPUT_DIR, filename);
    
    console.log('\n  ⬇️ Downloading...');
    await downloadFile(hqVideoUrl, filepath);
    
    console.log('\n' + '═'.repeat(70));
    console.log('✅ HIGH QUALITY VIDEO GENERATED');
    console.log('═'.repeat(70));
    console.log(`\n  Output: ${filepath}`);
    console.log(`  URL: ${hqVideoUrl}`);
    
  } catch (error) {
    console.error('\n  ❌ Error:', error.message);
  }
}

main().catch(console.error);



