#!/usr/bin/env node
/**
 * Kelly Video Factory - Enhanced Prompt Tester
 * 
 * Generates test images with the new enhanced prompts to validate quality
 * before running batch generation.
 * 
 * Run: node test-enhanced-prompts.cjs [template]
 * 
 * Templates: excited, curious, explain, thoughtful, heartfelt, celebrating, encouraging, listening
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const https = require('https');
const fs = require('fs');
const path = require('path');
const config = require('./config.cjs');

const TEST_DIR = path.join(__dirname, '../../template-forge/prompt-tests');
fs.mkdirSync(TEST_DIR, { recursive: true });

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
      process.stdout.write('.');
      
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
    .replace(/\{triggerWord\}/g, config.lora.triggerWord)
    .replace(/\{hair\}/g, config.character.hair)
    .replace(/\{eyes\}/g, config.character.eyes)
    .replace(/\{outfit\}/g, config.character.outfit)
    .replace(/\{identity\}/g, config.character.identity || '')
    .replace(/\{style\}/g, config.character.style || '');
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

async function testTemplate(templateName) {
  console.log('═'.repeat(70));
  console.log('🧪 ENHANCED PROMPT TESTER');
  console.log('   Testing new prompt quality before batch generation');
  console.log('═'.repeat(70));
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('\n❌ REPLICATE_API_TOKEN not set in environment');
    process.exit(1);
  }
  
  const template = config.templates[templateName];
  if (!template) {
    console.log(`\n❌ Unknown template: ${templateName}`);
    console.log('\nAvailable templates:');
    Object.keys(config.templates).forEach(t => {
      console.log(`  - ${t}: ${config.templates[t].emotion}`);
    });
    process.exit(1);
  }
  
  const prompt = buildPrompt(templateName);
  
  console.log(`\n📋 Template: ${templateName}`);
  console.log(`   Emotion: ${template.emotion}`);
  console.log(`   Environment: ${template.environment}`);
  console.log(`   Framing: ${template.framing || 'default'}`);
  console.log(`\n📝 Full Prompt:\n   ${prompt}`);
  console.log(`\n⛔ Negative Prompt:\n   ${config.character.negativePrompt}`);
  
  console.log(`\n🎨 Generating image`);
  process.stdout.write('   Progress: ');
  
  const api = new ReplicateAPI(process.env.REPLICATE_API_TOKEN);
  const startTime = Date.now();
  
  try {
    const output = await api.generate({
      prompt,
      negative_prompt: config.character.negativePrompt,
      lora_weights: config.lora.weights,
      lora_scale: config.lora.scale,
      aspect_ratio: '16:9',
      megapixels: '1',
      output_format: 'png',
      output_quality: 100,
      num_inference_steps: 35,
      guidance: 4.0,
    });
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    const filename = `test_${templateName}_${Date.now()}.png`;
    const filepath = path.join(TEST_DIR, filename);
    
    await downloadFile(imageUrl, filepath);
    
    console.log(` ✅ (${duration}s)`);
    console.log(`\n📁 Saved to: ${filepath}`);
    console.log(`🔗 URL: ${imageUrl}`);
    
    // Save metadata
    const meta = {
      template: templateName,
      prompt,
      negativePrompt: config.character.negativePrompt,
      loraScale: config.lora.scale,
      duration,
      url: imageUrl,
      filepath,
      timestamp: new Date().toISOString(),
    };
    
    fs.writeFileSync(
      path.join(TEST_DIR, `test_${templateName}_${Date.now()}.json`),
      JSON.stringify(meta, null, 2)
    );
    
    // Create comparison HTML
    createComparisonPage();
    
    console.log('\n═'.repeat(70));
    console.log('✅ TEST COMPLETE');
    console.log('═'.repeat(70));
    console.log(`\n👀 Review the image and compare with previous versions.`);
    console.log(`   Open: ${path.join(TEST_DIR, 'comparison.html')}`);
    
  } catch (error) {
    console.log(` ❌ ${error.message}`);
    process.exit(1);
  }
}

function createComparisonPage() {
  const files = fs.readdirSync(TEST_DIR)
    .filter(f => f.endsWith('.png'))
    .sort()
    .reverse();
  
  const cards = files.map(f => {
    const template = f.replace(/^test_/, '').replace(/_\d+\.png$/, '');
    return `
      <div class="card">
        <img src="${f}" alt="${template}">
        <div class="info">
          <strong>${template}</strong><br>
          <small>${f}</small>
        </div>
      </div>
    `;
  }).join('');
  
  const html = `<!DOCTYPE html>
<html>
<head>
  <title>Enhanced Prompt Test Results</title>
  <style>
    body { font-family: system-ui; background: #0a0a0f; color: #eee; padding: 2rem; }
    h1 { color: #10b981; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 1.5rem; }
    .card { background: #12121a; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); }
    .card img { width: 100%; }
    .info { padding: 1rem; }
    .checklist { background: rgba(16,185,129,0.1); border: 1px solid #10b981; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .checklist h3 { margin-top: 0; color: #10b981; }
    .checklist li { margin: 0.5rem 0; }
  </style>
</head>
<body>
  <h1>🧪 Enhanced Prompt Test Results</h1>
  <p>Generated: ${new Date().toISOString()}</p>
  
  <div class="checklist">
    <h3>✅ Quality Checklist</h3>
    <ol>
      <li><strong>Face Consistency:</strong> Does Kelly look like Kelly? Check hair, eyes, face shape.</li>
      <li><strong>Sweater Color:</strong> Is it powder blue? (Not pink, teal, or other)</li>
      <li><strong>Expression:</strong> Does it match the template emotion?</li>
      <li><strong>Composition:</strong> Is the framing professional? Good lighting?</li>
      <li><strong>Environment:</strong> Does the background enhance the mood?</li>
      <li><strong>Hands:</strong> Are they anatomically correct? (Common AI issue)</li>
    </ol>
  </div>
  
  <div class="grid">${cards}</div>
</body>
</html>`;
  
  fs.writeFileSync(path.join(TEST_DIR, 'comparison.html'), html);
}

// Generate ALL templates for comparison
async function testAll() {
  const templates = ['excited', 'curious', 'explain', 'thoughtful', 'heartfelt'];
  
  console.log('═'.repeat(70));
  console.log('🧪 TESTING ALL CORE TEMPLATES');
  console.log('═'.repeat(70));
  
  for (const template of templates) {
    await testTemplate(template);
    console.log('\n⏳ Waiting 5 seconds before next generation...\n');
    await new Promise(r => setTimeout(r, 5000));
  }
  
  console.log('\n\n═'.repeat(70));
  console.log('🎉 ALL TESTS COMPLETE');
  console.log('═'.repeat(70));
}

// Main
async function main() {
  const args = process.argv.slice(2);
  const templateArg = args[0];
  
  if (!templateArg) {
    console.log(`
Enhanced Prompt Tester
Usage:
  node test-enhanced-prompts.cjs <template>    Test specific template
  node test-enhanced-prompts.cjs --all         Test all 5 core templates

Templates:
  excited     - Hook phase (grab attention)
  curious     - Q1 phase (invite exploration)
  explain     - Q2 phase (teaching moment)
  thoughtful  - Q3 phase (deeper reflection)
  heartfelt   - Wisdom phase (emotional landing)
  
  celebrating - Success feedback
  encouraging - Redirect feedback
  listening   - Waiting for response
  welcome     - Opening greeting
`);
    return;
  }
  
  if (templateArg === '--all') {
    await testAll();
  } else {
    await testTemplate(templateArg);
  }
}

main().catch(console.error);


