/**
 * 🎬 Kelly Motion Studio - Premium Lipsync Generation
 * 
 * High-end motion graphics pipeline for film-quality talking videos.
 * 
 * Features:
 * - LivePortrait (premium quality)
 * - SadTalker with full options
 * - 4K Real-ESRGAN upscaling
 * - CodeFormer face enhancement
 * - Head movement animation
 * - Expression blending
 * - Batch processing with quality scoring
 * 
 * Usage:
 *   node scripts/kelly-motion-studio.cjs --model liveportrait --day 1 --phase hook --text "Hello!"
 *   node scripts/kelly-motion-studio.cjs --batch --days 1-10
 *   node scripts/kelly-motion-studio.cjs --premium --day 1 --upscale 4k
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config();

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;

// =============================================================================
// PREMIUM MODEL REGISTRY
// =============================================================================

const MODELS = {
  // Highest quality - preserves identity best
  liveportrait: {
    id: 'fofr/live-portrait:067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146',
    name: 'LivePortrait',
    quality: 'premium',
    speed: 'slow',
    preservesIdentity: 5,
    lipSyncAccuracy: 4,
    inputFormat: (img, audio) => ({
      face_image: img,
      driving_audio: audio,
      live_portrait_dsize: 512,
      live_portrait_scale: 2.3,
      video_frame_rate: 30,
      video_file_output_codec: 'libx264',
      video_output_quality: 18,
    }),
  },
  
  // Best balance of quality and speed
  sadtalker_hq: {
    id: 'cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376',
    name: 'SadTalker HQ',
    quality: 'high',
    speed: 'medium',
    preservesIdentity: 4,
    lipSyncAccuracy: 5,
    inputFormat: (img, audio, opts = {}) => {
      const input = {
        source_image: img,
        driven_audio: audio,
        enhancer: opts.enhancer || 'gfpgan',
        preprocess: opts.preprocess || 'crop',
        still_mode: opts.still || false,
        expression_scale: opts.expression || 1.0,
        pose_style: opts.pose || 0,
        size: opts.size || 512,
        face3dvis: false,
      };
      // Only add optional fields if they have values
      if (opts.eyeblink) input.ref_eyeblink = opts.eyeblink;
      if (opts.refPose) input.ref_pose = opts.refPose;
      if (opts.yaw !== undefined) input.input_yaw = opts.yaw;
      if (opts.pitch !== undefined) input.input_pitch = opts.pitch;
      if (opts.roll !== undefined) input.input_roll = opts.roll;
      if (opts.bgEnhancer) input.background_enhancer = opts.bgEnhancer;
      return input;
    },
  },
  
  // Fast preview
  wav2lip: {
    id: 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
    name: 'Wav2Lip',
    quality: 'standard',
    speed: 'fast',
    preservesIdentity: 3,
    lipSyncAccuracy: 4,
    inputFormat: (img, audio) => ({
      face: img,
      audio: audio,
      fps: 30,
      pads: '0 0 0 0',
      smooth: true,
      resize_factor: 1,
    }),
  },
  
  // Alternative SadTalker
  sadtalker_v2: {
    id: 'arielreplicate/sadtalker_video2video:a18c767b036b95a37d20f35e3f3c6e7bf8f18f21272a0eb66e2ce4981d415a35',
    name: 'SadTalker V2V',
    quality: 'high',
    speed: 'medium',
    preservesIdentity: 4,
    lipSyncAccuracy: 5,
    inputFormat: (img, audio, opts = {}) => ({
      source_image: img,
      driven_audio: audio,
      still: opts.still || false,
      preprocess: opts.preprocess || 'crop',
      enhancer: opts.enhancer || 'gfpgan',
    }),
  },
};

// Enhancement models
const ENHANCERS = {
  realesrgan_4k: {
    id: 'nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa',
    name: 'Real-ESRGAN 4K',
    inputFormat: (video) => ({
      image: video,
      scale: 4,
      face_enhance: true,
    }),
  },
  
  codeformer: {
    id: 'sczhou/codeformer:7de2ea26c616d5bf2245ad0d5e24f0ff9a6c3f9d129eff9e28914f0a0a7f5d26',
    name: 'CodeFormer',
    inputFormat: (video) => ({
      image: video,
      upscale: 2,
      face_upsample: true,
      background_enhance: true,
      codeformer_fidelity: 0.5,
    }),
  },
  
  gfpgan: {
    id: 'tencentarc/gfpgan:0fbacf7afc6c914f5c1a7e5a6ae6a8e7a7ae9a1b0a6c7a8c9a0b1a2a3a4a5a6a',
    name: 'GFPGAN v1.4',
    inputFormat: (video) => ({
      img: video,
      version: 'v1.4',
      scale: 2,
    }),
  },
};

// Voice presets for different moods
const VOICE_PRESETS = {
  warm: { stability: 0.5, similarity_boost: 0.85, style: 0.0 },
  excited: { stability: 0.3, similarity_boost: 0.9, style: 0.3 },
  calm: { stability: 0.7, similarity_boost: 0.8, style: 0.0 },
  storytelling: { stability: 0.4, similarity_boost: 0.85, style: 0.15 },
};

// =============================================================================
// PREMIUM GENERATION ENGINE
// =============================================================================

class KellyMotionStudio {
  constructor(options = {}) {
    this.outputDir = options.outputDir || path.join(__dirname, '..', 'motion-studio-output');
    this.cacheDir = path.join(this.outputDir, '.cache');
    this.verbose = options.verbose !== false;
    
    fs.mkdirSync(this.outputDir, { recursive: true });
    fs.mkdirSync(this.cacheDir, { recursive: true });
    
    this.audioCache = new Map();
  }
  
  log(msg) {
    if (this.verbose) console.log(msg);
  }
  
  // HTTP request helper
  async request(options, data = null) {
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
  
  // Generate TTS with ElevenLabs
  async generateVoice(text, preset = 'warm') {
    const cacheKey = `${text.slice(0, 50)}_${preset}`.replace(/[^a-z0-9]/gi, '_');
    const cachePath = path.join(this.cacheDir, `${cacheKey}.mp3`);
    
    if (fs.existsSync(cachePath)) {
      this.log(`   🔊 Using cached audio`);
      return fs.readFileSync(cachePath);
    }
    
    this.log(`   🎙️ Generating Kelly's voice (${preset} preset)...`);
    
    const settings = VOICE_PRESETS[preset] || VOICE_PRESETS.warm;
    
    const response = await this.request({
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
        ...settings,
        use_speaker_boost: true,
      }
    }));
    
    if (response.status !== 200) {
      throw new Error(`ElevenLabs error: ${response.status}`);
    }
    
    fs.writeFileSync(cachePath, response.data);
    this.log(`   ✅ Audio: ${(response.data.length / 1024).toFixed(1)}KB`);
    
    return response.data;
  }
  
  // Load Kelly image
  loadImage(day, phase) {
    const paddedDay = String(day).padStart(3, '0');
    const imagePath = path.join(__dirname, '..', 'public', 'kelly', 'phases', paddedDay, `${phase}.png`);
    
    if (!fs.existsSync(imagePath)) {
      throw new Error(`Image not found: ${imagePath}`);
    }
    
    return fs.readFileSync(imagePath);
  }
  
  // Run Replicate model
  async runModel(modelKey, imageBuffer, audioBuffer, options = {}) {
    const model = MODELS[modelKey];
    if (!model) throw new Error(`Unknown model: ${modelKey}`);
    
    this.log(`   🎬 Running ${model.name}...`);
    
    const imageBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
    const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
    
    const input = model.inputFormat(imageBase64, audioBase64, options);
    
    const createResponse = await this.request({
      hostname: 'api.replicate.com',
      path: '/v1/predictions',
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
        'Content-Type': 'application/json',
      }
    }, JSON.stringify({
      version: model.id.split(':')[1],
      input,
    }));
    
    if (createResponse.status !== 201) {
      throw new Error(`Replicate error: ${createResponse.status} - ${JSON.stringify(createResponse.data)}`);
    }
    
    const predictionId = createResponse.data.id;
    this.log(`      Prediction: ${predictionId}`);
    
    // Poll for completion
    let attempts = 0;
    const maxAttempts = 180; // 9 minutes max
    
    while (attempts < maxAttempts) {
      await new Promise(r => setTimeout(r, 3000));
      
      const statusResponse = await this.request({
        hostname: 'api.replicate.com',
        path: `/v1/predictions/${predictionId}`,
        method: 'GET',
        headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
      });
      
      const status = statusResponse.data.status;
      process.stdout.write(`\r      Status: ${status} (${attempts * 3}s)...      `);
      
      if (status === 'succeeded') {
        console.log('');
        const output = statusResponse.data.output;
        return typeof output === 'string' ? output : output?.[0] || output;
      } else if (status === 'failed') {
        console.log('');
        throw new Error(`Model failed: ${statusResponse.data.error}`);
      }
      
      attempts++;
    }
    
    throw new Error('Timeout waiting for model');
  }
  
  // Enhance video with upscaling
  async enhanceVideo(videoUrl, enhancerKey = 'realesrgan_4k') {
    const enhancer = ENHANCERS[enhancerKey];
    if (!enhancer) {
      this.log(`   ⚠️ Unknown enhancer: ${enhancerKey}, skipping`);
      return videoUrl;
    }
    
    this.log(`   ✨ Enhancing with ${enhancer.name}...`);
    
    try {
      const input = enhancer.inputFormat(videoUrl);
      
      const createResponse = await this.request({
        hostname: 'api.replicate.com',
        path: '/v1/predictions',
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
          'Content-Type': 'application/json',
        }
      }, JSON.stringify({
        version: enhancer.id.split(':')[1],
        input,
      }));
      
      if (createResponse.status !== 201) {
        this.log(`   ⚠️ Enhancement failed to start, using original`);
        return videoUrl;
      }
      
      const predictionId = createResponse.data.id;
      
      // Poll for completion
      let attempts = 0;
      while (attempts < 120) {
        await new Promise(r => setTimeout(r, 3000));
        
        const statusResponse = await this.request({
          hostname: 'api.replicate.com',
          path: `/v1/predictions/${predictionId}`,
          method: 'GET',
          headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
        });
        
        const status = statusResponse.data.status;
        process.stdout.write(`\r      Enhancing: ${status} (${attempts * 3}s)...      `);
        
        if (status === 'succeeded') {
          console.log('');
          return statusResponse.data.output;
        } else if (status === 'failed') {
          console.log('');
          this.log(`   ⚠️ Enhancement failed, using original`);
          return videoUrl;
        }
        
        attempts++;
      }
      
      return videoUrl;
    } catch (err) {
      this.log(`   ⚠️ Enhancement error: ${err.message}`);
      return videoUrl;
    }
  }
  
  // Download video
  async downloadVideo(url, filename) {
    const outputPath = path.join(this.outputDir, filename);
    
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const protocol = urlObj.protocol === 'http:' ? http : https;
      
      protocol.get(url, (res) => {
        if (res.statusCode === 302 || res.statusCode === 301) {
          return this.downloadVideo(res.headers.location, filename).then(resolve).catch(reject);
        }
        
        const file = fs.createWriteStream(outputPath);
        res.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve(outputPath);
        });
      }).on('error', reject);
    });
  }
  
  // Main generation method
  async generate(config) {
    const {
      day = 1,
      phase = 'hook',
      text,
      model = 'sadtalker_hq',
      voicePreset = 'warm',
      enhance = false,
      enhancer = 'realesrgan_4k',
      options = {},
    } = config;
    
    const timestamp = Date.now();
    const filename = `kelly_${model}_day${day}_${phase}_${timestamp}.mp4`;
    
    console.log('\n' + '═'.repeat(60));
    console.log(`🎬 KELLY MOTION STUDIO`);
    console.log('═'.repeat(60));
    console.log(`   Model: ${MODELS[model]?.name || model}`);
    console.log(`   Day: ${day}, Phase: ${phase}`);
    console.log(`   Enhance: ${enhance ? enhancer : 'none'}`);
    console.log('─'.repeat(60));
    
    const startTime = Date.now();
    
    try {
      // 1. Generate audio
      const audioBuffer = await this.generateVoice(text, voicePreset);
      
      // 2. Load image
      this.log(`   📸 Loading Kelly image...`);
      const imageBuffer = this.loadImage(day, phase);
      this.log(`      ${(imageBuffer.length / 1024).toFixed(0)}KB`);
      
      // 3. Run lipsync model
      let videoUrl = await this.runModel(model, imageBuffer, audioBuffer, options);
      this.log(`   ✅ Video generated`);
      
      // 4. Optional enhancement
      if (enhance && videoUrl) {
        videoUrl = await this.enhanceVideo(videoUrl, enhancer);
      }
      
      // 5. Download
      this.log(`   💾 Downloading...`);
      const localPath = await this.downloadVideo(videoUrl, filename);
      
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      
      console.log('═'.repeat(60));
      console.log(`✅ SUCCESS in ${duration}s`);
      console.log(`   Output: ${localPath}`);
      console.log(`   URL: ${videoUrl}`);
      console.log('═'.repeat(60));
      
      return {
        success: true,
        localPath,
        videoUrl,
        duration,
        model,
        day,
        phase,
        filename,
      };
      
    } catch (error) {
      console.log('═'.repeat(60));
      console.log(`❌ FAILED: ${error.message}`);
      console.log('═'.repeat(60));
      
      return {
        success: false,
        error: error.message,
        model,
        day,
        phase,
      };
    }
  }
  
  // Batch generation
  async batch(configs) {
    const results = [];
    
    console.log('\n' + '█'.repeat(60));
    console.log(`🎬 BATCH GENERATION: ${configs.length} videos`);
    console.log('█'.repeat(60));
    
    for (let i = 0; i < configs.length; i++) {
      console.log(`\n[${i + 1}/${configs.length}]`);
      const result = await this.generate(configs[i]);
      results.push(result);
      
      // Rate limiting
      if (i < configs.length - 1) {
        await new Promise(r => setTimeout(r, 2000));
      }
    }
    
    // Summary
    const successful = results.filter(r => r.success);
    console.log('\n' + '█'.repeat(60));
    console.log(`🏁 BATCH COMPLETE: ${successful.length}/${results.length} successful`);
    console.log('█'.repeat(60));
    
    // Save results
    const resultsPath = path.join(this.outputDir, `batch_${Date.now()}.json`);
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    console.log(`Results saved: ${resultsPath}`);
    
    return results;
  }
  
  // Quality comparison test
  async compareModels(day, phase, text) {
    const modelsToTest = ['liveportrait', 'sadtalker_hq', 'wav2lip'];
    const configs = modelsToTest.map(model => ({
      day,
      phase,
      text,
      model,
      voicePreset: 'warm',
    }));
    
    return this.batch(configs);
  }
  
  // Premium generation (best quality)
  async premium(config) {
    return this.generate({
      ...config,
      model: 'liveportrait',
      enhance: true,
      enhancer: 'codeformer',
      voicePreset: 'storytelling',
    });
  }
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  const studio = new KellyMotionStudio();
  
  // Parse args
  const getArg = (name) => {
    const idx = args.indexOf(`--${name}`);
    if (idx === -1) return null;
    return args[idx + 1];
  };
  
  const hasFlag = (name) => args.includes(`--${name}`);
  
  // Default text
  const defaultText = "Hello! Today we're going to explore something amazing together. Every great discovery starts with a simple question. Are you ready to learn?";
  
  if (hasFlag('compare')) {
    // Compare all models
    const day = parseInt(getArg('day') || '1');
    const phase = getArg('phase') || 'hook';
    const text = getArg('text') || defaultText;
    
    await studio.compareModels(day, phase, text);
    
  } else if (hasFlag('premium')) {
    // Premium generation
    const day = parseInt(getArg('day') || '1');
    const phase = getArg('phase') || 'hook';
    const text = getArg('text') || defaultText;
    
    await studio.premium({ day, phase, text });
    
  } else if (hasFlag('batch')) {
    // Batch generation
    const days = getArg('days') || '1-5';
    const [startDay, endDay] = days.split('-').map(Number);
    const phases = ['hook', 'q1', 'wisdom'];
    const text = getArg('text') || defaultText;
    
    const configs = [];
    for (let day = startDay; day <= (endDay || startDay); day++) {
      for (const phase of phases) {
        configs.push({
          day,
          phase,
          text,
          model: getArg('model') || 'sadtalker_hq',
        });
      }
    }
    
    await studio.batch(configs);
    
  } else {
    // Single generation
    const day = parseInt(getArg('day') || '1');
    const phase = getArg('phase') || 'hook';
    const text = getArg('text') || defaultText;
    const model = getArg('model') || 'sadtalker_hq';
    const enhance = hasFlag('enhance') || hasFlag('4k');
    
    await studio.generate({
      day,
      phase,
      text,
      model,
      enhance,
      options: {
        expression: parseFloat(getArg('expression') || '1.0'),
        preprocess: getArg('preprocess') || 'full',
        pose: parseInt(getArg('pose') || '0'),
      },
    });
  }
}

// Export for programmatic use
module.exports = { KellyMotionStudio, MODELS, ENHANCERS };

// Run CLI
if (require.main === module) {
  main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}

