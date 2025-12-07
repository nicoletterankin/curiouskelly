#!/usr/bin/env node
/**
 * Kelly Expression Generator v2 - WITH LORA
 * 
 * Uses the Kelly LoRA model for FACE CONSISTENCY across all expressions.
 * This is critical for video production - Kelly must look like the SAME person.
 * 
 * Run: node generate-kelly-expressions-v2.cjs [--all | --expression 1]
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const https = require('https');
const fs = require('fs');
const path = require('path');
const config = require('./config.cjs');

// Output directory
const OUTPUT_DIR = 'C:\\iLearnStudio\\projects\\Kelly\\Ref\\Generated-v2';
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Kelly LoRA configuration - INCREASED SCALE for face consistency
const KELLY_LORA = {
  weights: config.lora.weights,
  scale: 0.95, // INCREASED from 0.85 for stronger face lock
  triggerWord: config.lora.triggerWord, // "kelly"
};

// Character spec (from config.cjs)
const CHARACTER = config.character;

// The 7 critical expressions we need
// Using the TRIGGER WORD and config-based character description for consistency
const EXPRESSIONS = {
  1: {
    name: 'teaching',
    description: 'Teaching/Explaining (HOT Reaction)',
    filename: 'kelly-chair-teaching-v2.png',
    prompt: `${KELLY_LORA.triggerWord}, ${CHARACTER.identity}, woman with ${CHARACTER.hair} and ${CHARACTER.eyes}, wearing ${CHARACTER.outfit}, ${CHARACTER.skinTone}, ${CHARACTER.age}. Patient teacher expression explaining something fascinating. One hand raised gently at chest level with palm facing up in explanatory gesture. Slight forward lean showing engagement. Eyes focused and kind with small knowing smile. Seated in black canvas director's chair with dark wooden frame. Soft professional studio lighting, clean white background, 16:9 aspect ratio, medium shot showing gesture, ${CHARACTER.style}`,
  },
  2: {
    name: 'celebration-clasped',
    description: 'Celebration Hands Clasped (NOT Reaction)',
    filename: 'kelly-chair-celebration-clasped-v2.png',
    prompt: `${KELLY_LORA.triggerWord}, ${CHARACTER.identity}, woman with ${CHARACTER.hair} and ${CHARACTER.eyes}, wearing ${CHARACTER.outfit}, ${CHARACTER.skinTone}, ${CHARACTER.age}. Genuinely excited happy expression. Bright enthusiastic smile showing teeth, eyes lit up with joy. Hands clasped together at chest level in celebration. Warm encouraging energy. Seated in black canvas director's chair with dark wooden frame. Soft professional studio lighting, clean white background, 16:9 aspect ratio, medium shot, ${CHARACTER.style}`,
  },
  3: {
    name: 'celebration-clap',
    description: 'Celebration Mid-Clap (NOT Reaction Variant)',
    filename: 'kelly-chair-celebration-clap-v2.png',
    prompt: `${KELLY_LORA.triggerWord}, ${CHARACTER.identity}, woman with ${CHARACTER.hair} and ${CHARACTER.eyes}, wearing ${CHARACTER.outfit}, ${CHARACTER.skinTone}, ${CHARACTER.age}. Excited happy expression mid-clap, hands coming together at chest level. Bright genuine smile showing teeth, sparkling eyes. Warm encouraging demeanor. Seated in black canvas director's chair with dark wooden frame. Soft professional studio lighting, clean white background, 16:9 aspect ratio, medium shot, ${CHARACTER.style}`,
  },
  4: {
    name: 'celebration-fist',
    description: 'Celebration Subtle Fist (NOT Reaction Variant)',
    filename: 'kelly-chair-celebration-fist-v2.png',
    prompt: `${KELLY_LORA.triggerWord}, ${CHARACTER.identity}, woman with ${CHARACTER.hair} and ${CHARACTER.eyes}, wearing ${CHARACTER.outfit}, ${CHARACTER.skinTone}, ${CHARACTER.age}. Triumphant controlled expression with subtle fist pump, one hand raised with gentle fist near shoulder. Bright proud smile. Seated in black canvas director's chair with dark wooden frame. Soft professional studio lighting, clean white background, 16:9 aspect ratio, medium shot, ${CHARACTER.style}`,
  },
  5: {
    name: 'wisdom',
    description: 'Wisdom Delivery (Final Phase)',
    filename: 'kelly-chair-wisdom-v2.png',
    prompt: `${KELLY_LORA.triggerWord}, ${CHARACTER.identity}, woman with ${CHARACTER.hair} and ${CHARACTER.eyes}, wearing ${CHARACTER.outfit}, ${CHARACTER.skinTone}, ${CHARACTER.age}. Serene calm wise expression, direct gentle gaze at camera. Peaceful composed demeanor, quiet confidence. Hands resting naturally, still. More contemplative than a smile but warm. Soft ethereal lighting. Seated in black canvas director's chair with dark wooden frame. Clean white background, 16:9 aspect ratio, medium close-up on face, ${CHARACTER.style}`,
  },
  6: {
    name: 'question-tilt',
    description: 'Question Head Tilt (Question Phases)',
    filename: 'kelly-chair-question-tilt-v2.png',
    prompt: `${KELLY_LORA.triggerWord}, ${CHARACTER.identity}, woman with ${CHARACTER.hair} and ${CHARACTER.eyes}, wearing ${CHARACTER.outfit}, ${CHARACTER.skinTone}, ${CHARACTER.age}. Genuinely curious inviting expression. Head tilted slightly to one side. Raised eyebrows showing interest. Friendly inquisitive smile, bright engaged eyes. Seated in black canvas director's chair with dark wooden frame. Soft professional studio lighting, clean white background, 16:9 aspect ratio, medium shot, ${CHARACTER.style}`,
  },
  7: {
    name: 'question-chin',
    description: 'Question Chin Touch (Question Variant)',
    filename: 'kelly-chair-question-chin-v2.png',
    prompt: `${KELLY_LORA.triggerWord}, ${CHARACTER.identity}, woman with ${CHARACTER.hair} and ${CHARACTER.eyes}, wearing ${CHARACTER.outfit}, ${CHARACTER.skinTone}, ${CHARACTER.age}. Contemplative inquiring expression, one finger gently touching chin. Slight knowing smile, eyes looking up thoughtfully. Seated in black canvas director's chair with dark wooden frame. Soft professional studio lighting, clean white background, 16:9 aspect ratio, medium shot, ${CHARACTER.style}`,
  },
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
            const json = JSON.parse(Buffer.concat(body).toString());
            if (res.statusCode >= 400) {
              reject(new Error(`API Error ${res.statusCode}: ${JSON.stringify(json)}`));
            } else {
              resolve(json);
            }
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
  
  async getVersion(owner, model) {
    const cacheKey = `${owner}/${model}`;
    if (!this.versionCache) this.versionCache = {};
    if (this.versionCache[cacheKey]) return this.versionCache[cacheKey];
    
    const response = await this.request('GET', `/models/${owner}/${model}`);
    this.versionCache[cacheKey] = response.latest_version.id;
    return this.versionCache[cacheKey];
  }
  
  async generateWithLoRA(prompt, loraWeights, loraScale) {
    // Using Flux Dev LoRA model for Kelly face consistency
    const owner = 'black-forest-labs';
    const model = 'flux-dev-lora';
    
    console.log(`  📡 Getting Flux Dev LoRA version...`);
    const version = await this.getVersion(owner, model);
    console.log(`  📡 Version: ${version.substring(0, 12)}...`);
    console.log(`  📡 LoRA: ${loraWeights} @ scale ${loraScale}`);
    console.log(`  📡 Creating prediction...`);
    
    const prediction = await this.request('POST', '/predictions', {
      version,
      input: {
        prompt,
        hf_lora: loraWeights,
        lora_scale: loraScale,
        aspect_ratio: '16:9',
        megapixels: '1', // Flux Dev LoRA supports 1 or 0.25
        output_format: 'png',
        output_quality: 100,
        num_inference_steps: 35,  // Higher quality
        guidance: 4.0,
      },
    });
    
    console.log(`  ⏳ Prediction ID: ${prediction.id}`);
    console.log(`  ⏳ Waiting for generation (this may take longer with LoRA)...`);
    
    // Poll for completion
    let attempts = 0;
    const maxAttempts = 120; // 6 minutes max (LoRA can be slower)
    
    while (attempts < maxAttempts) {
      await this.sleep(3000);
      attempts++;
      
      const status = await this.request('GET', `/predictions/${prediction.id}`);
      
      process.stdout.write(`\r  ⏳ Status: ${status.status} (${attempts * 3}s)     `);
      
      if (status.status === 'succeeded') {
        console.log(' ✅');
        return status.output;
      } else if (status.status === 'failed') {
        console.log(' ❌');
        throw new Error(`Generation failed: ${status.error}`);
      } else if (status.status === 'canceled') {
        throw new Error('Generation was canceled');
      }
    }
    
    throw new Error('Generation timed out');
  }
  
  sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
}

async function downloadFile(url, filepath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(filepath);
    
    const download = (downloadUrl) => {
      https.get(downloadUrl, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          file.close();
          download(res.headers.location);
          return;
        }
        res.pipe(file);
        file.on('finish', () => { 
          file.close(); 
          resolve(filepath); 
        });
      }).on('error', (e) => { 
        fs.unlinkSync(filepath); 
        reject(e); 
      });
    };
    
    download(url);
  });
}

async function generateExpression(api, expressionId) {
  const expression = EXPRESSIONS[expressionId];
  if (!expression) {
    throw new Error(`Unknown expression: ${expressionId}`);
  }
  
  const filepath = path.join(OUTPUT_DIR, expression.filename);
  
  // Check if already exists
  if (fs.existsSync(filepath)) {
    console.log(`  ⏭️  ${expression.filename} already exists`);
    return { status: 'cached', filepath };
  }
  
  console.log(`\n${'─'.repeat(70)}`);
  console.log(`🎨 Expression ${expressionId}: ${expression.description}`);
  console.log('─'.repeat(70));
  
  console.log(`  📝 Prompt: ${expression.prompt.substring(0, 80)}...`);
  
  try {
    const output = await api.generateWithLoRA(
      expression.prompt,
      KELLY_LORA.weights,
      KELLY_LORA.scale
    );
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    console.log(`  💾 Downloading to ${expression.filename}...`);
    await downloadFile(imageUrl, filepath);
    
    console.log(`  ✅ Saved: ${filepath}`);
    
    return { status: 'generated', filepath, url: imageUrl };
  } catch (error) {
    console.log(`  ❌ Failed: ${error.message}`);
    return { status: 'failed', error: error.message };
  }
}

async function main() {
  const args = process.argv.slice(2);
  
  console.log('═'.repeat(70));
  console.log('🎬 KELLY EXPRESSION GENERATOR v2 - WITH LORA');
  console.log('   Face-consistent generation using Kelly LoRA');
  console.log('═'.repeat(70));
  
  // Check for API key
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('\n❌ REPLICATE_API_TOKEN not found in environment!');
    process.exit(1);
  }
  
  console.log(`\n✅ Replicate API key found`);
  console.log(`📁 Output directory: ${OUTPUT_DIR}`);
  console.log(`\n🎭 Kelly LoRA Configuration:`);
  console.log(`   Weights: ${KELLY_LORA.weights}`);
  console.log(`   Scale: ${KELLY_LORA.scale}`);
  console.log(`   Trigger: "${KELLY_LORA.triggerWord}"`);
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  
  // Parse arguments
  let expressionsToGenerate = [];
  
  if (args.includes('--all') || args.length === 0) {
    expressionsToGenerate = Object.keys(EXPRESSIONS).map(Number);
    console.log(`\n🎯 Generating ALL 7 expressions with LoRA face consistency`);
  } else if (args.includes('--expression')) {
    const idx = args.indexOf('--expression');
    const id = parseInt(args[idx + 1]);
    if (!EXPRESSIONS[id]) {
      console.error(`\n❌ Invalid expression ID: ${id}`);
      process.exit(1);
    }
    expressionsToGenerate = [id];
    console.log(`\n🎯 Generating expression ${id}: ${EXPRESSIONS[id].description}`);
  } else if (args.includes('--priority')) {
    expressionsToGenerate = [1, 2, 5];
    console.log(`\n🎯 Generating PRIORITY expressions (1, 2, 5)`);
  }
  
  // Show what we'll generate
  console.log(`\n📋 Queue:`);
  for (const id of expressionsToGenerate) {
    console.log(`   ${id}. ${EXPRESSIONS[id].description}`);
  }
  
  // Estimate (LoRA is slower)
  const estimatedTime = expressionsToGenerate.length * 45; // ~45 sec each with LoRA
  const estimatedCost = expressionsToGenerate.length * 0.003; // ~$0.003 each for Flux Dev
  console.log(`\n⏱️  Estimated time: ${estimatedTime} seconds`);
  console.log(`💰 Estimated cost: $${estimatedCost.toFixed(3)} (Flux Dev is cheaper!)`);
  
  // Generate!
  const startTime = Date.now();
  const results = [];
  
  for (const id of expressionsToGenerate) {
    const result = await generateExpression(api, id);
    results.push({ id, ...result });
    
    // Small delay between requests
    await api.sleep(1000);
  }
  
  // Summary
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  const succeeded = results.filter(r => r.status === 'generated').length;
  const cached = results.filter(r => r.status === 'cached').length;
  const failed = results.filter(r => r.status === 'failed').length;
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 GENERATION COMPLETE (WITH LORA)');
  console.log('═'.repeat(70));
  console.log(`\n  Generated: ${succeeded}`);
  console.log(`  Cached:    ${cached}`);
  console.log(`  Failed:    ${failed}`);
  console.log(`  Duration:  ${duration}s`);
  console.log(`\n  Output:    ${OUTPUT_DIR}`);
  
  // List generated files
  console.log(`\n📁 Files:`);
  for (const result of results) {
    const exp = EXPRESSIONS[result.id];
    const status = result.status === 'generated' ? '✅' : 
                   result.status === 'cached' ? '📦' : '❌';
    console.log(`   ${status} ${exp.filename}`);
  }
  
  console.log('\n🎬 Kelly expressions ready for video pipeline!');
  console.log('   These should have CONSISTENT FACE across all expressions.\n');
}

// Show help
if (process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log(`
Kelly Expression Generator v2 - WITH LORA

This version uses the Kelly LoRA model for FACE CONSISTENCY.
All expressions will feature the SAME Kelly face.

Usage:
  node generate-kelly-expressions-v2.cjs              # Generate all 7 expressions
  node generate-kelly-expressions-v2.cjs --all        # Generate all 7 expressions
  node generate-kelly-expressions-v2.cjs --priority   # Generate priority (1, 2, 5)
  node generate-kelly-expressions-v2.cjs --expression 1  # Generate specific expression

LoRA Config:
  Weights: ${KELLY_LORA.weights}
  Scale:   ${KELLY_LORA.scale}
  Trigger: "${KELLY_LORA.triggerWord}"

Output: ${OUTPUT_DIR}
`);
  process.exit(0);
}

main().catch(console.error);

