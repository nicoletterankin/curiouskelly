/**
 * 🎭 Kelly Character Studio
 * 
 * Full-body character animation system for Kelly.
 * As Kelly's creator, this handles:
 * - Full body animation (not just face)
 * - Gesture library
 * - Pose-to-pose transitions
 * - Scene composition
 * - Expression states
 * - Character consistency
 * 
 * Usage:
 *   node scripts/kelly-character-studio.cjs --fullbody --day 1 --phase hook
 *   node scripts/kelly-character-studio.cjs --gesture wave --day 1
 *   node scripts/kelly-character-studio.cjs --transition --from hook --to wisdom
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
// KELLY'S CHARACTER DEFINITION
// =============================================================================

const KELLY_CHARACTER = {
  name: 'Kelly',
  appearance: {
    hair: 'Brown, wavy, shoulder-length',
    eyes: 'Brown, warm',
    skin: 'Light with warm undertones',
    outfit: 'Light blue crew-neck sweater', // Her signature look
    age: 'Late 20s to early 30s',
  },
  
  personality: {
    traits: ['Curious', 'Warm', 'Intelligent', 'Patient', 'Encouraging'],
    voiceStyle: 'Conversational, like talking to a friend',
    teachingStyle: 'Socratic - asks questions, guides discovery',
  },
  
  // Lesson phase definitions
  phases: {
    hook: {
      emotion: 'excited',
      energy: 'high',
      pose: 'open',
      purpose: 'Capture attention, spark curiosity',
      gestures: ['open_arms', 'lean_forward', 'wide_eyes'],
    },
    q1: {
      emotion: 'curious',
      energy: 'medium',
      pose: 'exploring',
      purpose: 'Present the question, examine the topic',
      gestures: ['pointing', 'holding_object', 'looking_closely'],
    },
    q2: {
      emotion: 'thoughtful',
      energy: 'medium',
      pose: 'considering',
      purpose: 'Deepen the inquiry',
      gestures: ['chin_touch', 'looking_up', 'hand_gesture'],
    },
    wisdom: {
      emotion: 'heartfelt',
      energy: 'calm',
      pose: 'centered',
      purpose: 'Share the insight, connect to life',
      gestures: ['hand_on_heart', 'gentle_smile', 'open_palm'],
    },
  },
  
  // Kelly's environments
  environments: {
    forest_path: { mood: 'adventurous', lighting: 'dappled sunlight' },
    sunset_tree: { mood: 'reflective', lighting: 'golden hour' },
    library: { mood: 'scholarly', lighting: 'warm interior' },
    garden: { mood: 'nurturing', lighting: 'soft daylight' },
    kitchen: { mood: 'homey', lighting: 'warm kitchen light' },
    beach: { mood: 'expansive', lighting: 'bright ocean light' },
  },
};

// =============================================================================
// ANIMATION MODELS
// =============================================================================

const ANIMATION_MODELS = {
  // Full body talking with minimal movement
  sadtalker_full: {
    id: 'cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376',
    name: 'SadTalker Full Body',
    description: 'Preserves full body, animates face',
    inputFormat: (img, audio, opts = {}) => ({
      source_image: img,
      driven_audio: audio,
      enhancer: 'gfpgan',
      preprocess: 'full', // KEY: full body, not crop
      still_mode: opts.still || false,
      expression_scale: opts.expression || 1.0,
      pose_style: opts.pose || 0,
      size: 512,
    }),
  },
  
  // Resize mode - smaller face, more body context
  sadtalker_resize: {
    id: 'cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376',
    name: 'SadTalker Resize',
    description: 'Resizes to fit, preserves aspect ratio',
    inputFormat: (img, audio, opts = {}) => ({
      source_image: img,
      driven_audio: audio,
      enhancer: 'gfpgan',
      preprocess: 'resize', // Resize mode
      still_mode: opts.still || false,
      expression_scale: opts.expression || 1.0,
      pose_style: opts.pose || 0,
      size: 512,
    }),
  },
  
  // Wav2Lip preserves full body by default
  wav2lip_full: {
    id: 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
    name: 'Wav2Lip Full Body',
    description: 'Fast, preserves body, only modifies mouth',
    inputFormat: (img, audio) => ({
      face: img,
      audio: audio,
      fps: 30,
      smooth: true,
      resize_factor: 1,
    }),
  },
  
  // Image-to-video for subtle animation
  stable_video: {
    id: 'stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438',
    name: 'Stable Video Diffusion',
    description: 'Subtle movement, breathing, environment',
    inputFormat: (img, opts = {}) => ({
      input_image: img,
      motion_bucket_id: opts.motion || 127, // 1-255, higher = more motion
      fps: 7,
      cond_aug: 0.02,
    }),
  },
};

// =============================================================================
// KELLY'S GESTURE LIBRARY
// =============================================================================

const GESTURE_PROMPTS = {
  // Welcoming gestures
  open_arms: 'Arms open wide, palms up, welcoming gesture',
  wave: 'Hand raised, gentle wave, friendly greeting',
  
  // Teaching gestures
  pointing: 'Index finger pointing, explaining something',
  counting: 'Counting on fingers, listing items',
  showing: 'Hands presenting an object or concept',
  
  // Thinking gestures
  chin_touch: 'Hand touching chin, thoughtful expression',
  looking_up: 'Eyes looking up, considering something',
  
  // Emotional gestures
  hand_on_heart: 'Hand on heart, sincere expression',
  clapping: 'Hands clapping, celebrating',
  thumbs_up: 'Thumbs up, encouraging',
  
  // Interactive gestures
  listening: 'Head tilted, attentive listening pose',
  nodding: 'Gentle nodding, agreeing',
  shrugging: 'Shoulders raised, playful uncertainty',
};

// =============================================================================
// CHARACTER ANIMATION ENGINE
// =============================================================================

class KellyCharacterStudio {
  constructor(options = {}) {
    this.outputDir = options.outputDir || path.join(__dirname, '..', 'kelly-character-output');
    this.cacheDir = path.join(this.outputDir, '.cache');
    this.verbose = options.verbose !== false;
    
    fs.mkdirSync(this.outputDir, { recursive: true });
    fs.mkdirSync(this.cacheDir, { recursive: true });
  }
  
  log(msg) {
    if (this.verbose) console.log(msg);
  }
  
  // HTTP helper
  async request(options, data = null) {
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
  
  // Generate Kelly's voice
  async generateVoice(text, emotion = 'warm') {
    const cacheKey = `voice_${text.slice(0, 30).replace(/[^a-z0-9]/gi, '_')}_${emotion}`;
    const cachePath = path.join(this.cacheDir, `${cacheKey}.mp3`);
    
    if (fs.existsSync(cachePath)) {
      this.log(`   🔊 Using cached voice`);
      return fs.readFileSync(cachePath);
    }
    
    this.log(`   🎙️ Generating Kelly's voice (${emotion})...`);
    
    // Adjust voice settings based on emotion
    const emotionSettings = {
      excited: { stability: 0.3, similarity_boost: 0.9, style: 0.3 },
      curious: { stability: 0.5, similarity_boost: 0.85, style: 0.15 },
      thoughtful: { stability: 0.6, similarity_boost: 0.85, style: 0.1 },
      heartfelt: { stability: 0.6, similarity_boost: 0.9, style: 0.2 },
      warm: { stability: 0.5, similarity_boost: 0.85, style: 0.0 },
    };
    
    const settings = emotionSettings[emotion] || emotionSettings.warm;
    
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
      voice_settings: { ...settings, use_speaker_boost: true }
    }));
    
    if (response.status !== 200) throw new Error(`ElevenLabs error: ${response.status}`);
    
    fs.writeFileSync(cachePath, response.data);
    this.log(`   ✅ Voice: ${(response.data.length / 1024).toFixed(1)}KB`);
    return response.data;
  }
  
  // Load Kelly's image for a specific day/phase
  loadKellyImage(day, phase) {
    const paddedDay = String(day).padStart(3, '0');
    const imagePath = path.join(__dirname, '..', 'public', 'kelly', 'phases', paddedDay, `${phase}.png`);
    
    if (!fs.existsSync(imagePath)) {
      throw new Error(`Kelly image not found: ${imagePath}`);
    }
    
    this.log(`   📸 Loading Kelly: Day ${day} - ${phase}`);
    const buffer = fs.readFileSync(imagePath);
    this.log(`      ${(buffer.length / 1024).toFixed(0)}KB`);
    return buffer;
  }
  
  // Analyze Kelly's pose in an image
  analyzeKellyPose(day, phase) {
    const phaseConfig = KELLY_CHARACTER.phases[phase];
    return {
      phase,
      day,
      emotion: phaseConfig?.emotion || 'neutral',
      energy: phaseConfig?.energy || 'medium',
      suggestedGestures: phaseConfig?.gestures || [],
      purpose: phaseConfig?.purpose || '',
    };
  }
  
  // Run animation model
  async runModel(modelKey, imageBuffer, audioBuffer, options = {}) {
    const model = ANIMATION_MODELS[modelKey];
    if (!model) throw new Error(`Unknown model: ${modelKey}`);
    
    this.log(`   🎬 Running ${model.name}...`);
    this.log(`      ${model.description}`);
    
    const imageBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
    const audioBase64 = audioBuffer ? `data:audio/mpeg;base64,${audioBuffer.toString('base64')}` : null;
    
    let input;
    if (modelKey === 'stable_video') {
      input = model.inputFormat(imageBase64, options);
    } else {
      input = model.inputFormat(imageBase64, audioBase64, options);
    }
    
    const createResponse = await this.request({
      hostname: 'api.replicate.com',
      path: '/v1/predictions',
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
        'Content-Type': 'application/json',
      }
    }, JSON.stringify({ version: model.id.split(':')[1], input }));
    
    if (createResponse.status !== 201) {
      throw new Error(`Replicate error: ${createResponse.status} - ${JSON.stringify(createResponse.data)}`);
    }
    
    const predictionId = createResponse.data.id;
    this.log(`      Prediction: ${predictionId}`);
    
    // Poll for completion
    let attempts = 0;
    const maxAttempts = 180;
    
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
    
    throw new Error('Timeout');
  }
  
  // Download video
  async downloadVideo(url, filename) {
    const outputPath = path.join(this.outputDir, filename);
    
    return new Promise((resolve, reject) => {
      https.get(url, (res) => {
        if (res.statusCode === 302 || res.statusCode === 301) {
          return this.downloadVideo(res.headers.location, filename).then(resolve).catch(reject);
        }
        const file = fs.createWriteStream(outputPath);
        res.pipe(file);
        file.on('finish', () => { file.close(); resolve(outputPath); });
      }).on('error', reject);
    });
  }
  
  // ==========================================================================
  // MAIN ANIMATION METHODS
  // ==========================================================================
  
  // Generate full-body talking video
  async generateFullBody(config) {
    const {
      day = 1,
      phase = 'hook',
      text,
      model = 'sadtalker_full',
      options = {},
    } = config;
    
    const timestamp = Date.now();
    const filename = `kelly_fullbody_day${day}_${phase}_${timestamp}.mp4`;
    
    console.log('\n' + '═'.repeat(60));
    console.log('🎭 KELLY CHARACTER STUDIO - Full Body Animation');
    console.log('═'.repeat(60));
    
    const poseAnalysis = this.analyzeKellyPose(day, phase);
    console.log(`   Day: ${day}, Phase: ${phase}`);
    console.log(`   Emotion: ${poseAnalysis.emotion}`);
    console.log(`   Energy: ${poseAnalysis.energy}`);
    console.log(`   Purpose: ${poseAnalysis.purpose}`);
    console.log('─'.repeat(60));
    
    const startTime = Date.now();
    
    try {
      // Generate voice with emotion
      const audioBuffer = await this.generateVoice(text, poseAnalysis.emotion);
      
      // Load Kelly's image
      const imageBuffer = this.loadKellyImage(day, phase);
      
      // Run animation
      const videoUrl = await this.runModel(model, imageBuffer, audioBuffer, {
        expression: options.expression || 1.0,
        ...options,
      });
      
      // Download
      this.log(`   💾 Downloading...`);
      const localPath = await this.downloadVideo(videoUrl, filename);
      
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      
      console.log('═'.repeat(60));
      console.log(`✅ SUCCESS in ${duration}s`);
      console.log(`   Output: ${localPath}`);
      console.log(`   URL: ${videoUrl}`);
      console.log('═'.repeat(60));
      
      return { success: true, localPath, videoUrl, duration, filename };
      
    } catch (error) {
      console.log('═'.repeat(60));
      console.log(`❌ FAILED: ${error.message}`);
      console.log('═'.repeat(60));
      return { success: false, error: error.message };
    }
  }
  
  // Generate subtle breathing/movement video (no audio)
  async generateSubtleMotion(config) {
    const { day = 1, phase = 'hook', motion = 100 } = config;
    
    const timestamp = Date.now();
    const filename = `kelly_motion_day${day}_${phase}_${timestamp}.mp4`;
    
    console.log('\n' + '═'.repeat(60));
    console.log('🌊 KELLY CHARACTER STUDIO - Subtle Motion');
    console.log('═'.repeat(60));
    console.log(`   Day: ${day}, Phase: ${phase}`);
    console.log(`   Motion intensity: ${motion}/255`);
    console.log('─'.repeat(60));
    
    const startTime = Date.now();
    
    try {
      const imageBuffer = this.loadKellyImage(day, phase);
      
      const videoUrl = await this.runModel('stable_video', imageBuffer, null, { motion });
      
      this.log(`   💾 Downloading...`);
      const localPath = await this.downloadVideo(videoUrl, filename);
      
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      
      console.log('═'.repeat(60));
      console.log(`✅ SUCCESS in ${duration}s`);
      console.log(`   Output: ${localPath}`);
      console.log('═'.repeat(60));
      
      return { success: true, localPath, videoUrl, duration, filename };
      
    } catch (error) {
      console.log(`❌ FAILED: ${error.message}`);
      return { success: false, error: error.message };
    }
  }
  
  // Compare full body modes
  async compareFullBodyModes(day, phase, text) {
    console.log('\n' + '█'.repeat(60));
    console.log('🎭 FULL BODY MODE COMPARISON');
    console.log('█'.repeat(60));
    
    const modes = ['sadtalker_full', 'sadtalker_resize', 'wav2lip_full'];
    const results = [];
    
    for (const mode of modes) {
      const result = await this.generateFullBody({ day, phase, text, model: mode });
      results.push({ mode, ...result });
      
      if (modes.indexOf(mode) < modes.length - 1) {
        await new Promise(r => setTimeout(r, 2000));
      }
    }
    
    // Save results
    const resultsPath = path.join(this.outputDir, `fullbody_comparison_${Date.now()}.json`);
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    
    console.log('\n' + '█'.repeat(60));
    console.log(`🏁 COMPARISON COMPLETE`);
    console.log(`   Successful: ${results.filter(r => r.success).length}/${results.length}`);
    console.log('█'.repeat(60));
    
    return results;
  }
  
  // Generate all phases for a day
  async generateDaySequence(day, scripts) {
    console.log('\n' + '█'.repeat(60));
    console.log(`🎬 GENERATING DAY ${day} SEQUENCE`);
    console.log('█'.repeat(60));
    
    const phases = ['hook', 'q1', 'q2', 'wisdom'];
    const results = [];
    
    for (const phase of phases) {
      if (scripts[phase]) {
        console.log(`\n📍 Phase: ${phase}`);
        const result = await this.generateFullBody({
          day,
          phase,
          text: scripts[phase],
          model: 'wav2lip_full', // Fast for sequence
        });
        results.push({ phase, ...result });
      }
    }
    
    return results;
  }
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  const studio = new KellyCharacterStudio();
  
  const getArg = (name) => {
    const idx = args.indexOf(`--${name}`);
    return idx > -1 ? args[idx + 1] : null;
  };
  const hasFlag = (name) => args.includes(`--${name}`);
  
  const day = parseInt(getArg('day') || '1');
  const phase = getArg('phase') || 'hook';
  const text = getArg('text') || "Hello! Today we're going to explore something amazing together. Every great discovery starts with curiosity. Are you ready to learn?";
  
  if (hasFlag('compare')) {
    await studio.compareFullBodyModes(day, phase, text);
  } else if (hasFlag('motion')) {
    const motion = parseInt(getArg('motion') || '100');
    await studio.generateSubtleMotion({ day, phase, motion });
  } else if (hasFlag('fullbody')) {
    const model = getArg('model') || 'sadtalker_full';
    await studio.generateFullBody({ day, phase, text, model });
  } else {
    // Default: full body with wav2lip (fast)
    await studio.generateFullBody({ day, phase, text, model: 'wav2lip_full' });
  }
}

module.exports = { KellyCharacterStudio, KELLY_CHARACTER, GESTURE_PROMPTS };

if (require.main === module) {
  main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}



