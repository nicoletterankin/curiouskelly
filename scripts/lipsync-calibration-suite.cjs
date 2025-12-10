/**
 * 🔬 Kelly Lipsync Calibration Suite
 * 
 * Comprehensive testing of all lipsync models and settings
 * to find the optimal configuration for 4K film quality.
 * 
 * Tests:
 * - Multiple models (SadTalker, LivePortrait, Wav2Lip)
 * - Different image types (hook, q1, wisdom)
 * - Various text lengths (short, medium, long)
 * - Expression scales
 * - Enhancers (GFPGAN, etc.)
 * 
 * Usage:
 *   node scripts/lipsync-calibration-suite.cjs
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

// Load env vars
require('dotenv').config();

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;

// =============================================================================
// TEST CONFIGURATIONS
// =============================================================================

const LIPSYNC_MODELS = {
  sadtalker: {
    id: 'cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376',
    name: 'SadTalker (cjwbw)',
    inputFormat: 'sadtalker',
  },
  sadtalker_v2: {
    id: 'lucataco/sadtalker:85f79f4a1d369fc190998c3dbbf6e67a8b6bee9fcbae33ff6be3261aaaefd85e', 
    name: 'SadTalker v2 (Lucataco)',
    inputFormat: 'sadtalker',
  },
  wav2lip: {
    id: 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
    name: 'Wav2Lip',
    inputFormat: 'wav2lip',
  },
};

const TEST_CONFIGS = [
  // Model comparison - same image, same text
  { name: 'sadtalker_baseline', model: 'sadtalker', day: 1, phase: 'hook', textType: 'short', enhancer: 'gfpgan', expressionScale: 1.0 },
  { name: 'sadtalker_v2_baseline', model: 'sadtalker_v2', day: 1, phase: 'hook', textType: 'short', enhancer: 'gfpgan', expressionScale: 1.0 },
  { name: 'wav2lip_baseline', model: 'wav2lip', day: 1, phase: 'hook', textType: 'short' },
  
  // Expression scale tests
  { name: 'sadtalker_exp_0.8', model: 'sadtalker', day: 1, phase: 'hook', textType: 'short', enhancer: 'gfpgan', expressionScale: 0.8 },
  { name: 'sadtalker_exp_1.2', model: 'sadtalker', day: 1, phase: 'hook', textType: 'short', enhancer: 'gfpgan', expressionScale: 1.2 },
  { name: 'sadtalker_exp_1.5', model: 'sadtalker', day: 1, phase: 'hook', textType: 'short', enhancer: 'gfpgan', expressionScale: 1.5 },
  
  // Different phases (pose variations)
  { name: 'sadtalker_q1', model: 'sadtalker', day: 1, phase: 'q1', textType: 'short', enhancer: 'gfpgan', expressionScale: 1.0 },
  { name: 'sadtalker_wisdom', model: 'sadtalker', day: 1, phase: 'wisdom', textType: 'short', enhancer: 'gfpgan', expressionScale: 1.0 },
  
  // Different days (different Kelly images)
  { name: 'sadtalker_day5', model: 'sadtalker', day: 5, phase: 'hook', textType: 'short', enhancer: 'gfpgan', expressionScale: 1.0 },
  { name: 'sadtalker_day10', model: 'sadtalker', day: 10, phase: 'hook', textType: 'short', enhancer: 'gfpgan', expressionScale: 1.0 },
  
  // Text length tests
  { name: 'sadtalker_long_text', model: 'sadtalker', day: 1, phase: 'hook', textType: 'long', enhancer: 'gfpgan', expressionScale: 1.0 },
  
  // No enhancer test
  { name: 'sadtalker_no_enhance', model: 'sadtalker', day: 1, phase: 'hook', textType: 'short', enhancer: null, expressionScale: 1.0 },
];

const TEST_TEXTS = {
  short: "Hello! Today we're going to explore something amazing together. Are you ready?",
  medium: "Welcome to your daily lesson! Today we're diving into a fascinating topic that connects to everything around us. Let me show you something incredible that will change how you see the world.",
  long: "Good morning, curious learner! Today's lesson is one of my favorites. We're going to explore a concept that seems simple at first, but the more you think about it, the more amazing it becomes. I've been so excited to share this with you. Let's discover something wonderful together, and remember - every question you ask makes you smarter!",
};

// =============================================================================
// HELPERS
// =============================================================================

function makeRequest(options, data = null) {
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

async function generateTTS(text) {
  const response = await makeRequest({
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
    voice_settings: { stability: 0.5, similarity_boost: 0.85, style: 0.0, use_speaker_boost: true }
  }));
  
  if (response.status !== 200) {
    throw new Error(`ElevenLabs error: ${response.status}`);
  }
  return response.data;
}

function loadImage(day, phase) {
  const paddedDay = String(day).padStart(3, '0');
  const imagePath = path.join(__dirname, '..', 'public', 'kelly', 'phases', paddedDay, `${phase}.png`);
  if (!fs.existsSync(imagePath)) {
    throw new Error(`Image not found: ${imagePath}`);
  }
  return fs.readFileSync(imagePath);
}

async function runReplicateModel(modelConfig, imageBuffer, audioBuffer, config) {
  const imageBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
  const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  let input;
  if (modelConfig.inputFormat === 'wav2lip') {
    input = { face: imageBase64, audio: audioBase64 };
  } else {
    input = {
      source_image: imageBase64,
      driven_audio: audioBase64,
      preprocess: 'crop',
      still_mode: false,
      expression_scale: config.expressionScale || 1.0,
    };
    if (config.enhancer) {
      input.enhancer = config.enhancer;
    }
  }
  
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify({ version: modelConfig.id.split(':')[1], input }));
  
  if (createResponse.status !== 201) {
    throw new Error(`Replicate create error: ${createResponse.status} - ${JSON.stringify(createResponse.data)}`);
  }
  
  const predictionId = createResponse.data.id;
  let attempts = 0;
  const maxAttempts = 120;
  
  while (attempts < maxAttempts) {
    await new Promise(r => setTimeout(r, 3000));
    
    const statusResponse = await makeRequest({
      hostname: 'api.replicate.com',
      path: `/v1/predictions/${predictionId}`,
      method: 'GET',
      headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
    });
    
    const status = statusResponse.data.status;
    process.stdout.write(`\r      Status: ${status} (${attempts * 3}s)...      `);
    
    if (status === 'succeeded') {
      return { url: statusResponse.data.output, metrics: statusResponse.data.metrics };
    } else if (status === 'failed') {
      throw new Error(`Failed: ${statusResponse.data.error}`);
    }
    attempts++;
  }
  throw new Error('Timeout');
}

async function downloadVideo(url, outputPath) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const protocol = urlObj.protocol === 'http:' ? http : https;
    protocol.get(url, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301) {
        return downloadVideo(res.headers.location, outputPath).then(resolve).catch(reject);
      }
      const file = fs.createWriteStream(outputPath);
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(outputPath); });
    }).on('error', reject);
  });
}

// =============================================================================
// MAIN TEST RUNNER
// =============================================================================

async function runTest(config, outputDir, audioCache) {
  const testStart = Date.now();
  const result = {
    name: config.name,
    model: config.model,
    day: config.day,
    phase: config.phase,
    textType: config.textType,
    enhancer: config.enhancer,
    expressionScale: config.expressionScale,
    status: 'pending',
    videoUrl: null,
    localPath: null,
    duration: null,
    error: null,
  };
  
  try {
    console.log(`\n  🧪 Test: ${config.name}`);
    console.log(`     Model: ${LIPSYNC_MODELS[config.model].name}`);
    console.log(`     Image: Day ${config.day} - ${config.phase}`);
    
    // Get or generate audio
    const text = TEST_TEXTS[config.textType];
    let audioBuffer;
    if (audioCache[config.textType]) {
      audioBuffer = audioCache[config.textType];
      console.log(`     Audio: Using cached ${config.textType}`);
    } else {
      console.log(`     Audio: Generating ${config.textType}...`);
      audioBuffer = await generateTTS(text);
      audioCache[config.textType] = audioBuffer;
    }
    
    // Load image
    const imageBuffer = loadImage(config.day, config.phase);
    console.log(`     Image loaded: ${(imageBuffer.length / 1024).toFixed(0)}KB`);
    
    // Run model
    console.log(`     Running ${config.model}...`);
    const modelResult = await runReplicateModel(
      LIPSYNC_MODELS[config.model],
      imageBuffer,
      audioBuffer,
      config
    );
    
    result.videoUrl = typeof modelResult.url === 'string' ? modelResult.url : modelResult.url?.[0];
    
    // Download video
    const filename = `${config.name}_${Date.now()}.mp4`;
    const localPath = path.join(outputDir, filename);
    await downloadVideo(result.videoUrl, localPath);
    result.localPath = localPath;
    result.localFilename = filename;
    
    result.status = 'success';
    result.duration = ((Date.now() - testStart) / 1000).toFixed(1);
    console.log(`\n     ✅ Success in ${result.duration}s`);
    
  } catch (error) {
    result.status = 'failed';
    result.error = error.message;
    result.duration = ((Date.now() - testStart) / 1000).toFixed(1);
    console.log(`\n     ❌ Failed: ${error.message}`);
  }
  
  return result;
}

async function main() {
  console.log('═'.repeat(70));
  console.log('🔬 KELLY LIPSYNC CALIBRATION SUITE');
  console.log('═'.repeat(70));
  console.log(`Total tests: ${TEST_CONFIGS.length}`);
  console.log(`Models: ${Object.keys(LIPSYNC_MODELS).join(', ')}`);
  console.log('═'.repeat(70));
  
  // Check env
  if (!ELEVENLABS_API_KEY) throw new Error('ELEVENLABS_API_KEY not set');
  if (!REPLICATE_API_TOKEN) throw new Error('REPLICATE_API_TOKEN not set');
  
  // Create output directory
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const outputDir = path.join(__dirname, '..', 'calibration-results', timestamp);
  fs.mkdirSync(outputDir, { recursive: true });
  console.log(`\nOutput: ${outputDir}`);
  
  // Run tests
  const results = [];
  const audioCache = {};
  const suiteStart = Date.now();
  
  for (let i = 0; i < TEST_CONFIGS.length; i++) {
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`📊 Test ${i + 1}/${TEST_CONFIGS.length}`);
    
    const result = await runTest(TEST_CONFIGS[i], outputDir, audioCache);
    results.push(result);
    
    // Save intermediate results
    const resultsPath = path.join(outputDir, 'results.json');
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  }
  
  // Generate HTML report
  const totalTime = ((Date.now() - suiteStart) / 1000 / 60).toFixed(1);
  const successful = results.filter(r => r.status === 'success').length;
  
  const html = generateHTMLReport(results, timestamp, totalTime);
  const htmlPath = path.join(outputDir, 'index.html');
  fs.writeFileSync(htmlPath, html);
  
  // Copy to public folder for deployment
  const publicPath = path.join(__dirname, '..', 'public', 'lipsync');
  fs.mkdirSync(publicPath, { recursive: true });
  fs.writeFileSync(path.join(publicPath, 'index.html'), html);
  fs.writeFileSync(path.join(publicPath, 'results.json'), JSON.stringify(results, null, 2));
  
  // Copy videos to public
  for (const result of results) {
    if (result.localPath && fs.existsSync(result.localPath)) {
      fs.copyFileSync(result.localPath, path.join(publicPath, result.localFilename));
    }
  }
  
  console.log('\n' + '═'.repeat(70));
  console.log('🏁 CALIBRATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`Total time: ${totalTime} minutes`);
  console.log(`Successful: ${successful}/${results.length}`);
  console.log(`Results: ${outputDir}`);
  console.log(`Public page: ${publicPath}`);
  console.log('═'.repeat(70));
}

function generateHTMLReport(results, timestamp, totalTime) {
  const successful = results.filter(r => r.status === 'success');
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kelly Lipsync Calibration - ${timestamp}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'SF Pro Display', -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 2rem;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #f39c12, #e74c3c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #8892b0; margin-bottom: 2rem; }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .stat {
            background: rgba(255,255,255,0.05);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }
        .stat-value { font-size: 2rem; font-weight: bold; color: #64ffda; }
        .stat-label { font-size: 0.85rem; color: #8892b0; margin-top: 0.5rem; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1.5rem;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card.failed { opacity: 0.5; }
        .card video {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            background: #000;
        }
        .card-body { padding: 1rem; }
        .card-title {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #64ffda;
        }
        .card-meta {
            font-size: 0.8rem;
            color: #8892b0;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.25rem;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .badge-success { background: rgba(46, 204, 113, 0.2); color: #2ecc71; }
        .badge-failed { background: rgba(231, 76, 60, 0.2); color: #e74c3c; }
        .controls {
            margin-bottom: 2rem;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }
        select, button {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
            font-size: 1rem;
            cursor: pointer;
        }
        .explanation {
            background: rgba(100, 255, 218, 0.1);
            border: 1px solid rgba(100, 255, 218, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
        }
        .explanation h3 { color: #64ffda; margin-bottom: 1rem; }
        .explanation p { color: #ccd6f6; line-height: 1.6; margin-bottom: 0.5rem; }
        @media (max-width: 768px) {
            .stats { grid-template-columns: repeat(2, 1fr); }
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Kelly Lipsync Calibration</h1>
        <p class="subtitle">Comparing models, settings, and images for optimal quality • ${timestamp}</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">${results.length}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat">
                <div class="stat-value">${successful.length}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat">
                <div class="stat-value">${Object.keys(LIPSYNC_MODELS).length}</div>
                <div class="stat-label">Models Tested</div>
            </div>
            <div class="stat">
                <div class="stat-value">${totalTime}m</div>
                <div class="stat-label">Total Time</div>
            </div>
        </div>
        
        <div class="controls">
            <select id="filterModel" onchange="filterResults()">
                <option value="all">All Models</option>
                ${Object.entries(LIPSYNC_MODELS).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join('')}
            </select>
            <select id="filterStatus" onchange="filterResults()">
                <option value="all">All Status</option>
                <option value="success">Success Only</option>
                <option value="failed">Failed Only</option>
            </select>
            <button onclick="playAll()">▶️ Play All</button>
            <button onclick="pauseAll()">⏸️ Pause All</button>
        </div>
        
        <div class="grid" id="resultsGrid">
            ${results.map((r, i) => `
                <div class="card ${r.status}" data-model="${r.model}" data-status="${r.status}">
                    ${r.status === 'success' ? `
                        <video controls loop muted>
                            <source src="${r.localFilename}" type="video/mp4">
                        </video>
                    ` : `
                        <div style="aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#1a1a2e;">
                            <span style="color:#e74c3c;">❌ Failed</span>
                        </div>
                    `}
                    <div class="card-body">
                        <div class="card-title">${r.name}</div>
                        <span class="badge ${r.status === 'success' ? 'badge-success' : 'badge-failed'}">
                            ${r.status}
                        </span>
                        <div class="card-meta">
                            <div>Model: ${r.model}</div>
                            <div>Day: ${r.day}</div>
                            <div>Phase: ${r.phase}</div>
                            <div>Text: ${r.textType}</div>
                            <div>Enhancer: ${r.enhancer || 'none'}</div>
                            <div>Exp: ${r.expressionScale || 'N/A'}</div>
                            <div>Time: ${r.duration}s</div>
                            ${r.error ? `<div style="grid-column:span 2;color:#e74c3c;">Error: ${r.error}</div>` : ''}
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
        
        <div class="explanation">
            <h3>🎯 How to Use This Page</h3>
            <p><strong>Compare models:</strong> Watch each video and note lip-sync accuracy, naturalness, and quality.</p>
            <p><strong>Expression scale:</strong> Higher values = more exaggerated mouth movements. 1.0 is default.</p>
            <p><strong>Enhancer:</strong> GFPGAN improves face quality but may introduce artifacts.</p>
            <p><strong>Best practices:</strong> Look for smooth transitions, accurate timing, and minimal distortion.</p>
            <p><strong>Report issues:</strong> Note any test that looks particularly good or bad for calibration.</p>
        </div>
    </div>
    
    <script>
        function filterResults() {
            const model = document.getElementById('filterModel').value;
            const status = document.getElementById('filterStatus').value;
            document.querySelectorAll('.card').forEach(card => {
                const matchModel = model === 'all' || card.dataset.model === model;
                const matchStatus = status === 'all' || card.dataset.status === status;
                card.style.display = matchModel && matchStatus ? 'block' : 'none';
            });
        }
        function playAll() {
            document.querySelectorAll('video').forEach(v => v.play());
        }
        function pauseAll() {
            document.querySelectorAll('video').forEach(v => v.pause());
        }
    </script>
</body>
</html>`;
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});



