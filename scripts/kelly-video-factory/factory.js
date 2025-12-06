/**
 * Kelly Video Factory
 * 
 * Production-grade video generation for Kelly.
 * Handles the full pipeline: Image → Animation → Audio → Lipsync → Upscale
 */

const https = require('https');
const fs = require('fs');
const path = require('path');
const config = require('./config');
const ReplicateClient = require('./replicate-client');

class KellyVideoFactory {
  constructor(options = {}) {
    this.replicateToken = options.replicateToken || process.env.REPLICATE_API_TOKEN;
    this.elevenLabsKey = options.elevenLabsKey || process.env.ELEVENLABS_API_KEY;
    this.supabaseUrl = options.supabaseUrl || process.env.PUBLIC_SUPABASE_URL;
    this.supabaseKey = options.supabaseKey || process.env.SUPABASE_SERVICE_ROLE_KEY;
    
    this.replicate = new ReplicateClient(this.replicateToken);
    this.qualityTier = options.quality || 'standard';
    this.outputDir = options.outputDir || path.join(__dirname, '../../template-forge/factory-output');
    
    fs.mkdirSync(this.outputDir, { recursive: true });
    
    // Cache for model versions
    this.modelVersionCache = {};
  }
  
  // Get quality settings for current tier
  getQualitySettings() {
    return config.quality[this.qualityTier];
  }
  
  // Build prompt from template
  buildPrompt(templateKey) {
    const template = config.templates[templateKey];
    if (!template) {
      throw new Error(`Unknown template: ${templateKey}`);
    }
    
    let prompt = template.prompt;
    prompt = prompt.replace('{triggerWord}', config.lora.triggerWord);
    prompt = prompt.replace('{hair}', config.character.hair);
    prompt = prompt.replace('{eyes}', config.character.eyes);
    prompt = prompt.replace('{outfit}', config.character.outfit);
    
    return prompt;
  }
  
  // Progress reporter
  progress(step, message, data = {}) {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    console.log(`[${timestamp}] ${step}: ${message}`);
    if (data.elapsed) {
      process.stdout.write(`\r   Status: ${data.status} (${data.elapsed}s)...                    `);
    }
  }
  
  // ============================================================
  // STEP 1: Generate Kelly Image with LoRA
  // ============================================================
  async generateImage(templateKey) {
    this.progress('IMAGE', `Generating Kelly image for "${templateKey}"...`);
    
    const prompt = this.buildPrompt(templateKey);
    const quality = this.getQualitySettings();
    
    // Get model version (cached)
    if (!this.modelVersionCache.imageGeneration) {
      this.modelVersionCache.imageGeneration = await this.replicate.getModelVersion(
        config.models.imageGeneration.id
      );
    }
    
    const input = {
      prompt,
      negative_prompt: config.character.negativePrompt,
      lora_weights: config.lora.weights,
      lora_scale: config.lora.scale,
      aspect_ratio: quality.imageSize,
      megapixels: quality.imageMegapixels,
      output_format: 'png',
      output_quality: 100,
      num_inference_steps: 28,
      guidance: 3.5,
    };
    
    const output = await this.replicate.runWithRetry(
      this.modelVersionCache.imageGeneration,
      input,
      (p) => this.progress('IMAGE', `Generating...`, p)
    );
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`\n   ✅ Image generated: ${imageUrl.substring(0, 60)}...`);
    
    return { url: imageUrl, prompt, template: templateKey };
  }
  
  // ============================================================
  // STEP 2: Animate Image
  // ============================================================
  async animateImage(imageUrl) {
    this.progress('ANIMATE', 'Animating Kelly image...');
    
    const quality = this.getQualitySettings();
    const animConfig = config.models.animation[quality.animationModel];
    
    const input = {
      input_image: imageUrl,
      video_length: animConfig.videoLength,
      fps: 8,
      motion_bucket_id: 80, // Subtle natural motion
      cond_aug: 0.02,
      decoding_t: 7,
    };
    
    const output = await this.replicate.runWithRetry(
      animConfig.version,
      input,
      (p) => this.progress('ANIMATE', `Animating...`, p)
    );
    
    const videoUrl = Array.isArray(output) ? output[0] : output;
    console.log(`\n   ✅ Animation generated: ${videoUrl.substring(0, 60)}...`);
    
    return videoUrl;
  }
  
  // ============================================================
  // STEP 3: Generate Audio with ElevenLabs
  // ============================================================
  async generateAudio(text) {
    this.progress('AUDIO', `Generating Kelly voice...`);
    
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({
        text,
        model_id: config.voice.model,
        voice_settings: config.voice.settings,
      });
      
      const options = {
        hostname: 'api.elevenlabs.io',
        path: `/v1/text-to-speech/${config.voice.id}`,
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': this.elevenLabsKey,
        },
      };
      
      const req = https.request(options, (res) => {
        const chunks = [];
        res.on('data', chunk => chunks.push(chunk));
        res.on('end', () => {
          const buffer = Buffer.concat(chunks);
          if (res.statusCode !== 200) {
            reject(new Error(`ElevenLabs error: ${res.statusCode}`));
          } else {
            console.log(`   ✅ Audio generated: ${(buffer.length / 1024).toFixed(1)}KB`);
            resolve(buffer);
          }
        });
      });
      
      req.on('error', reject);
      req.write(postData);
      req.end();
    });
  }
  
  // ============================================================
  // STEP 4: Apply Lipsync
  // ============================================================
  async applyLipsync(videoUrl, audioBuffer) {
    this.progress('LIPSYNC', 'Applying lipsync...');
    
    const quality = this.getQualitySettings();
    const lipsyncConfig = config.models.lipsync[quality.lipsyncModel];
    
    const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
    
    let input;
    if (quality.lipsyncModel === 'wav2lip') {
      input = {
        face: videoUrl,
        audio: audioBase64,
        fps: 25,
        smooth: true,
        resize_factor: 1,
      };
    } else if (quality.lipsyncModel === 'sadtalker_hq') {
      input = {
        source_image: videoUrl, // SadTalker needs image, not video
        driven_audio: audioBase64,
        ...lipsyncConfig.options,
      };
    }
    
    const output = await this.replicate.runWithRetry(
      lipsyncConfig.version,
      input,
      (p) => this.progress('LIPSYNC', `Applying...`, p)
    );
    
    const resultUrl = typeof output === 'string' ? output : output;
    console.log(`\n   ✅ Lipsync applied: ${resultUrl.substring(0, 60)}...`);
    
    return resultUrl;
  }
  
  // ============================================================
  // STEP 5: Upscale to 4K (optional)
  // ============================================================
  async upscaleTo4K(videoUrl) {
    this.progress('UPSCALE', 'Upscaling to 4K...');
    
    const upscaleConfig = config.models.upscaler;
    
    const input = {
      video_path: videoUrl,
      scale: upscaleConfig.scale,
    };
    
    const output = await this.replicate.runWithRetry(
      upscaleConfig.version,
      input,
      (p) => this.progress('UPSCALE', `Upscaling...`, p)
    );
    
    console.log(`\n   ✅ Upscaled to 4K: ${output.substring(0, 60)}...`);
    return output;
  }
  
  // ============================================================
  // DOWNLOAD HELPER
  // ============================================================
  async download(url, filename) {
    const filepath = path.join(this.outputDir, filename);
    
    return new Promise((resolve, reject) => {
      const file = fs.createWriteStream(filepath);
      
      https.get(url, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          file.close();
          fs.unlinkSync(filepath);
          return this.download(res.headers.location, filename).then(resolve).catch(reject);
        }
        
        res.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve(filepath);
        });
      }).on('error', (err) => {
        fs.unlinkSync(filepath);
        reject(err);
      });
    });
  }
  
  // ============================================================
  // FULL PIPELINE
  // ============================================================
  async generate(templateKey, scriptText, options = {}) {
    const startTime = Date.now();
    const timestamp = Date.now();
    
    console.log('═'.repeat(70));
    console.log(`🎬 KELLY VIDEO FACTORY - ${this.getQualitySettings().name} Quality`);
    console.log('═'.repeat(70));
    console.log(`Template: ${templateKey}`);
    console.log(`Script: "${scriptText.substring(0, 50)}..."`);
    console.log(`Quality: ${this.qualityTier}`);
    console.log('');
    
    const result = {
      template: templateKey,
      script: scriptText,
      quality: this.qualityTier,
      timestamp: new Date().toISOString(),
      steps: {},
    };
    
    try {
      // Step 1: Generate Image
      const imageResult = await this.generateImage(templateKey);
      result.steps.image = imageResult;
      
      // Step 2: Animate
      const animationUrl = await this.animateImage(imageResult.url);
      result.steps.animation = { url: animationUrl };
      
      // Step 3: Generate Audio
      const audioBuffer = await this.generateAudio(scriptText);
      result.steps.audio = { size: audioBuffer.length };
      
      // Step 4: Apply Lipsync
      const lipsyncUrl = await this.applyLipsync(animationUrl, audioBuffer);
      result.steps.lipsync = { url: lipsyncUrl };
      
      // Step 5: Upscale (if production quality)
      let finalUrl = lipsyncUrl;
      if (this.getQualitySettings().upscale) {
        finalUrl = await this.upscaleTo4K(lipsyncUrl);
        result.steps.upscale = { url: finalUrl };
      }
      
      // Download final video and image
      const videoFilename = `kelly_${templateKey}_${timestamp}.mp4`;
      const imageFilename = `kelly_${templateKey}_${timestamp}.png`;
      
      result.localVideo = await this.download(finalUrl, videoFilename);
      result.localImage = await this.download(imageResult.url, imageFilename);
      result.finalVideoUrl = finalUrl;
      
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      result.duration = duration;
      result.success = true;
      
      console.log('');
      console.log('═'.repeat(70));
      console.log('✅ KELLY VIDEO GENERATED SUCCESSFULLY');
      console.log('═'.repeat(70));
      console.log(`   Duration: ${duration}s`);
      console.log(`   Image: ${result.localImage}`);
      console.log(`   Video: ${result.localVideo}`);
      console.log('═'.repeat(70));
      
      // Save result metadata
      const metaFilename = `kelly_${templateKey}_${timestamp}_meta.json`;
      fs.writeFileSync(
        path.join(this.outputDir, metaFilename),
        JSON.stringify(result, null, 2)
      );
      
      return result;
      
    } catch (error) {
      result.success = false;
      result.error = error.message;
      result.duration = ((Date.now() - startTime) / 1000).toFixed(1);
      
      console.log('');
      console.log('═'.repeat(70));
      console.log(`❌ GENERATION FAILED: ${error.message}`);
      console.log('═'.repeat(70));
      
      return result;
    }
  }
}

module.exports = KellyVideoFactory;

