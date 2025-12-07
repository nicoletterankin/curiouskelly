#!/usr/bin/env node
/**
 * Kelly Expression Generator
 * 
 * Generates the 7 critical missing Kelly expressions using Replicate's Flux model.
 * Uses existing LoRA and configuration for consistency.
 * 
 * Run: node generate-kelly-expressions.cjs [--all | --expression 1]
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const https = require('https');
const fs = require('fs');
const path = require('path');

// Output directory
const OUTPUT_DIR = 'C:\\iLearnStudio\\projects\\Kelly\\Ref\\Generated';
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Kelly's canonical look - matches reference images exactly
const KELLY_BASE = {
  identity: 'kelly, photorealistic 3D render, young woman in her late twenties',
  hair: 'long wavy brown hair with golden honey highlights flowing past shoulders',
  eyes: 'warm brown eyes',
  skin: 'fair-medium skin tone with warm undertones',
  sweater: 'light blue periwinkle ribbed crew-neck sweater',
  chair: 'seated in black canvas director\'s chair with dark wooden frame',
  lighting: 'soft professional studio lighting, clean white background',
  style: 'Character Creator or iClone style, film-grade photorealistic quality, ultra-high detail',
};

// The 7 critical expressions we need
const EXPRESSIONS = {
  1: {
    name: 'teaching',
    description: 'Teaching/Explaining (HOT Reaction)',
    filename: 'kelly-chair-teaching-v1.png',
    expression: 'Patient teacher expression explaining something fascinating with genuine enthusiasm. One hand raised gently at chest level with palm facing slightly up in explanatory gesture. Slight forward lean showing engagement. Eyes focused and kind with a small knowing smile as if saying "let me explain why this is so interesting." Calm composed demeanor, educational and nurturing energy. Head tilted slightly showing attentiveness.',
    framing: 'Medium shot showing head and upper torso with gesture visible',
  },
  2: {
    name: 'celebration-clasped',
    description: 'Celebration Hands Clasped (NOT Reaction)',
    filename: 'kelly-chair-celebration-clasped-v1.png',
    expression: 'Genuinely excited and happy expression but not over-the-top. Bright enthusiastic smile showing teeth, eyes lit up with joy. Hands clasped together at chest level in a celebratory gesture. Body language showing "yes, you figured it out!" energy. Warm encouraging energy, proud teacher moment. Slight lean forward with positive energy.',
    framing: 'Medium shot showing upper body and gesture',
  },
  3: {
    name: 'celebration-clap',
    description: 'Celebration Mid-Clap (NOT Reaction Variant)',
    filename: 'kelly-chair-celebration-clap-v1.png',
    expression: 'Excited and happy expression, caught mid-clap gesture with hands together at chest level. Bright genuine smile showing teeth, eyes sparkling with enthusiasm. Body language conveying "wonderful!" energy. Warm encouraging demeanor. MUST show the light blue sweater clearly.',
    framing: 'Medium shot showing hands clapping and blue sweater',
  },
  4: {
    name: 'celebration-fist',
    description: 'Celebration Subtle Fist (NOT Reaction Variant)',
    filename: 'kelly-chair-celebration-fist-v1.png',
    expression: 'Triumphant but controlled expression with subtle fist pump gesture, one hand raised with gentle fist near shoulder level. Bright proud smile, eyes conveying "you did it!" energy. Encouraging teacher celebrating student success.',
    framing: 'Medium shot',
  },
  5: {
    name: 'wisdom',
    description: 'Wisdom Delivery (Final Phase)',
    filename: 'kelly-chair-wisdom-v1.png',
    expression: 'Serene calm wise expression with direct gentle gaze straight at camera. Peaceful composed demeanor with slight maturity in expression. Hands resting naturally and still, no gestures. Embodying profound truth and quiet confidence. "I have something important to share with you" energy. More serious and contemplative than a welcome smile but still warm and approachable. Slightly softer, almost ethereal lighting quality.',
    framing: 'Medium close-up shot emphasizing face and eyes',
  },
  6: {
    name: 'question-tilt',
    description: 'Question Head Tilt (Question Phases)',
    filename: 'kelly-chair-question-tilt-v1.png',
    expression: 'Genuinely curious and inviting expression. Head tilted slightly to one side about 10 degrees. Raised eyebrows showing interest. Friendly inquisitive smile. Eyes bright and engaged as if asking "what do you think?" Open welcoming body language. One hand may be raised slightly in questioning gesture.',
    framing: 'Medium shot, slightly angled',
  },
  7: {
    name: 'question-chin',
    description: 'Question Chin Touch (Question Variant)',
    filename: 'kelly-chair-question-chin-v1.png',
    expression: 'Contemplative inquiring expression with one finger gently touching chin in a thinking gesture. Slight knowing smile as if posing an interesting puzzle. Eyes looking slightly up and to the side showing thought. "Hmm, interesting question..." energy. Relaxed but engaged posture.',
    framing: 'Medium shot showing thinking gesture',
  },
};

function buildPrompt(expression) {
  return `${KELLY_BASE.identity}, ${KELLY_BASE.hair}, ${KELLY_BASE.eyes}, ${KELLY_BASE.skin}. She wears a ${KELLY_BASE.sweater}. ${expression.expression} ${KELLY_BASE.chair}. ${KELLY_BASE.lighting}. 16:9 aspect ratio. ${KELLY_BASE.style}. ${expression.framing}.`;
}

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
  
  async generate(prompt, options = {}) {
    // Using Flux 1.1 Pro for highest quality
    const owner = 'black-forest-labs';
    const model = 'flux-1.1-pro';
    
    console.log(`  📡 Getting model version...`);
    const version = await this.getVersion(owner, model);
    console.log(`  📡 Version: ${version.substring(0, 12)}...`);
    console.log(`  📡 Creating prediction...`);
    
    const prediction = await this.request('POST', '/predictions', {
      version,
      input: {
        prompt,
        aspect_ratio: '16:9',
        output_format: 'png',
        output_quality: 100,
        safety_tolerance: 2,
        prompt_upsampling: true,
      },
    });
    
    console.log(`  ⏳ Prediction ID: ${prediction.id}`);
    console.log(`  ⏳ Waiting for generation...`);
    
    // Poll for completion
    let attempts = 0;
    const maxAttempts = 60; // 3 minutes max
    
    while (attempts < maxAttempts) {
      await this.sleep(3000);
      attempts++;
      
      const status = await this.request('GET', `/predictions/${prediction.id}`);
      
      process.stdout.write(`\r  ⏳ Status: ${status.status} (${attempts * 3}s)`);
      
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
  
  const prompt = buildPrompt(expression);
  console.log(`  📝 Prompt: ${prompt.substring(0, 100)}...`);
  
  try {
    const output = await api.generate(prompt);
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
  console.log('🎬 KELLY EXPRESSION GENERATOR');
  console.log('   Chief Video Officer - Automated Asset Generation');
  console.log('═'.repeat(70));
  
  // Check for API key
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('\n❌ REPLICATE_API_TOKEN not found in environment!');
    console.error('   Set it in .env file or export REPLICATE_API_TOKEN=...\n');
    process.exit(1);
  }
  
  console.log(`\n✅ Replicate API key found`);
  console.log(`📁 Output directory: ${OUTPUT_DIR}`);
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  
  // Parse arguments
  let expressionsToGenerate = [];
  
  if (args.includes('--all') || args.length === 0) {
    expressionsToGenerate = Object.keys(EXPRESSIONS).map(Number);
    console.log(`\n🎯 Generating ALL 7 expressions`);
  } else if (args.includes('--expression')) {
    const idx = args.indexOf('--expression');
    const id = parseInt(args[idx + 1]);
    if (!EXPRESSIONS[id]) {
      console.error(`\n❌ Invalid expression ID: ${id}`);
      console.error(`   Valid IDs: 1-7`);
      process.exit(1);
    }
    expressionsToGenerate = [id];
    console.log(`\n🎯 Generating expression ${id}: ${EXPRESSIONS[id].description}`);
  } else if (args.includes('--priority')) {
    expressionsToGenerate = [1, 2, 5]; // Teaching, Celebration, Wisdom
    console.log(`\n🎯 Generating PRIORITY expressions (1, 2, 5)`);
  }
  
  // Show what we'll generate
  console.log(`\n📋 Queue:`);
  for (const id of expressionsToGenerate) {
    console.log(`   ${id}. ${EXPRESSIONS[id].description}`);
  }
  
  // Estimate
  const estimatedTime = expressionsToGenerate.length * 30; // ~30 sec each
  const estimatedCost = expressionsToGenerate.length * 0.04; // ~$0.04 each for Flux Pro
  console.log(`\n⏱️  Estimated time: ${estimatedTime} seconds`);
  console.log(`💰 Estimated cost: $${estimatedCost.toFixed(2)}`);
  
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
  console.log('📊 GENERATION COMPLETE');
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
  
  console.log('\n🎬 Kelly expressions ready for video pipeline!\n');
}

// Show help
if (process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log(`
Kelly Expression Generator

Usage:
  node generate-kelly-expressions.cjs              # Generate all 7 expressions
  node generate-kelly-expressions.cjs --all        # Generate all 7 expressions
  node generate-kelly-expressions.cjs --priority   # Generate priority (1, 2, 5)
  node generate-kelly-expressions.cjs --expression 1  # Generate specific expression

Expressions:
  1. Teaching/Explaining    - For HOT reactions (correct answer)
  2. Celebration Clasped    - For NOT reactions (learning moment)
  3. Celebration Clap       - NOT reaction variant
  4. Celebration Fist       - NOT reaction variant
  5. Wisdom Delivery        - Final phase (profound moment)
  6. Question Head Tilt     - Question phases
  7. Question Chin Touch    - Question variant

Output: C:\\iLearnStudio\\projects\\Kelly\\Ref\\Generated\\
`);
  process.exit(0);
}

main().catch(console.error);

