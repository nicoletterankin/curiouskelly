/**
 * 🎭 Kelly Head Pose & Expression Test
 * 
 * Tests different head positions and expressions for natural talking.
 * High-end motion graphics calibration.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config();

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;

const SADTALKER_MODEL = 'cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376';

// Head pose tests
const HEAD_POSE_TESTS = [
  { name: 'neutral', pose: 0, yaw: 0, pitch: 0, roll: 0 },
  { name: 'slight_left', pose: 0, yaw: -10, pitch: 0, roll: 0 },
  { name: 'slight_right', pose: 0, yaw: 10, pitch: 0, roll: 0 },
  { name: 'nod_down', pose: 0, yaw: 0, pitch: 10, roll: 0 },
  { name: 'tilt_left', pose: 0, yaw: 0, pitch: 0, roll: -10 },
  { name: 'engaged', pose: 0, yaw: -5, pitch: 5, roll: 0 },
  { name: 'pose_style_10', pose: 10, yaw: 0, pitch: 0, roll: 0 },
  { name: 'pose_style_20', pose: 20, yaw: 0, pitch: 0, roll: 0 },
  { name: 'pose_style_30', pose: 30, yaw: 0, pitch: 0, roll: 0 },
];

// Expression scale tests
const EXPRESSION_TESTS = [
  { name: 'subtle_0.6', expression: 0.6 },
  { name: 'normal_1.0', expression: 1.0 },
  { name: 'expressive_1.3', expression: 1.3 },
  { name: 'dramatic_1.6', expression: 1.6 },
];

const outputDir = path.join(__dirname, '..', 'motion-studio-output', 'head-pose-tests');
fs.mkdirSync(outputDir, { recursive: true });

function makeRequest(options, data = null) {
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

async function generateAudio(text) {
  const cachePath = path.join(outputDir, 'audio_cache.mp3');
  if (fs.existsSync(cachePath)) return fs.readFileSync(cachePath);
  
  console.log('🎙️ Generating audio...');
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
    voice_settings: { stability: 0.5, similarity_boost: 0.85, use_speaker_boost: true }
  }));
  
  if (response.status !== 200) throw new Error(`ElevenLabs error: ${response.status}`);
  fs.writeFileSync(cachePath, response.data);
  return response.data;
}

function loadImage(day, phase) {
  const paddedDay = String(day).padStart(3, '0');
  const imagePath = path.join(__dirname, '..', 'public', 'kelly', 'phases', paddedDay, `${phase}.png`);
  return fs.readFileSync(imagePath);
}

async function runSadTalker(imageBuffer, audioBuffer, options) {
  const imageBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
  const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  const input = {
    source_image: imageBase64,
    driven_audio: audioBase64,
    enhancer: 'gfpgan',
    preprocess: 'crop',
    still_mode: false,
    expression_scale: options.expression || 1.0,
    pose_style: options.pose || 0,
    size: 512,
    face3dvis: false,
  };
  
  if (options.yaw !== undefined && options.yaw !== 0) input.input_yaw = [options.yaw];
  if (options.pitch !== undefined && options.pitch !== 0) input.input_pitch = [options.pitch];
  if (options.roll !== undefined && options.roll !== 0) input.input_roll = [options.roll];
  
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify({ version: SADTALKER_MODEL.split(':')[1], input }));
  
  if (createResponse.status !== 201) {
    throw new Error(`Replicate error: ${createResponse.status} - ${JSON.stringify(createResponse.data)}`);
  }
  
  const predictionId = createResponse.data.id;
  
  // Poll for completion
  let attempts = 0;
  while (attempts < 120) {
    await new Promise(r => setTimeout(r, 3000));
    const statusResponse = await makeRequest({
      hostname: 'api.replicate.com',
      path: `/v1/predictions/${predictionId}`,
      method: 'GET',
      headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
    });
    
    const status = statusResponse.data.status;
    process.stdout.write(`\r   Status: ${status} (${attempts * 3}s)      `);
    
    if (status === 'succeeded') {
      console.log('');
      return statusResponse.data.output;
    } else if (status === 'failed') {
      throw new Error(`Failed: ${statusResponse.data.error}`);
    }
    attempts++;
  }
  throw new Error('Timeout');
}

async function downloadVideo(url, filename) {
  const outputPath = path.join(outputDir, filename);
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301) {
        return downloadVideo(res.headers.location, filename).then(resolve).catch(reject);
      }
      const file = fs.createWriteStream(outputPath);
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(outputPath); });
    }).on('error', reject);
  });
}

async function runTest(testConfig, imageBuffer, audioBuffer, index, total) {
  console.log(`\n[${index + 1}/${total}] 🎭 ${testConfig.name}`);
  
  try {
    const startTime = Date.now();
    const videoUrl = await runSadTalker(imageBuffer, audioBuffer, testConfig);
    const filename = `${testConfig.name}_${Date.now()}.mp4`;
    const localPath = await downloadVideo(videoUrl, filename);
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    console.log(`   ✅ Success in ${duration}s`);
    
    return {
      name: testConfig.name,
      success: true,
      videoUrl,
      localPath,
      filename,
      duration,
      config: testConfig,
    };
  } catch (error) {
    console.log(`   ❌ Failed: ${error.message}`);
    return {
      name: testConfig.name,
      success: false,
      error: error.message,
      config: testConfig,
    };
  }
}

async function main() {
  console.log('═'.repeat(60));
  console.log('🎭 KELLY HEAD POSE & EXPRESSION CALIBRATION');
  console.log('═'.repeat(60));
  
  const text = "Hello! Let me show you something amazing today. Every great discovery starts with curiosity. Are you ready?";
  
  // Generate shared audio
  const audioBuffer = await generateAudio(text);
  console.log(`Audio: ${(audioBuffer.length / 1024).toFixed(1)}KB`);
  
  // Load image
  const imageBuffer = loadImage(1, 'hook');
  console.log(`Image: ${(imageBuffer.length / 1024).toFixed(0)}KB`);
  
  // Combine tests
  const allTests = [
    ...HEAD_POSE_TESTS,
    ...EXPRESSION_TESTS.map(t => ({ ...t, pose: 0, yaw: 0, pitch: 0, roll: 0 })),
  ];
  
  console.log(`\nRunning ${allTests.length} tests...`);
  
  const results = [];
  for (let i = 0; i < allTests.length; i++) {
    const result = await runTest(allTests[i], imageBuffer, audioBuffer, i, allTests.length);
    results.push(result);
    
    // Rate limiting
    if (i < allTests.length - 1) {
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  
  // Save results
  const resultsPath = path.join(outputDir, 'results.json');
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  
  // Generate HTML gallery
  const successful = results.filter(r => r.success);
  const html = `<!DOCTYPE html>
<html>
<head>
  <title>Kelly Head Pose Test Results</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui; background: #111; color: #fff; padding: 2rem; }
    h1 { font-size: 2rem; margin-bottom: 1rem; background: linear-gradient(90deg, #f39c12, #e74c3c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
    .card { background: #222; border-radius: 12px; overflow: hidden; }
    .card video { width: 100%; aspect-ratio: 1; object-fit: cover; }
    .card-body { padding: 0.75rem; }
    .card-title { font-weight: 600; margin-bottom: 0.25rem; }
    .card-meta { font-size: 0.75rem; color: #888; }
    button { padding: 0.5rem 1rem; background: #333; color: #fff; border: none; border-radius: 6px; cursor: pointer; margin-right: 0.5rem; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <h1>🎭 Kelly Head Pose Tests</h1>
  <p style="color:#888;margin-bottom:1rem;">${successful.length}/${results.length} successful</p>
  <button onclick="document.querySelectorAll('video').forEach(v=>v.play())">▶️ Play All</button>
  <button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">⏸️ Pause All</button>
  <div class="grid">
    ${successful.map(r => `
      <div class="card">
        <video controls loop muted playsinline><source src="${r.videoUrl}" type="video/mp4"></video>
        <div class="card-body">
          <div class="card-title">${r.name}</div>
          <div class="card-meta">
            ${r.config.pose !== undefined ? `pose:${r.config.pose} ` : ''}
            ${r.config.yaw !== undefined && r.config.yaw !== 0 ? `yaw:${r.config.yaw} ` : ''}
            ${r.config.pitch !== undefined && r.config.pitch !== 0 ? `pitch:${r.config.pitch} ` : ''}
            ${r.config.roll !== undefined && r.config.roll !== 0 ? `roll:${r.config.roll} ` : ''}
            ${r.config.expression !== undefined ? `exp:${r.config.expression} ` : ''}
            | ${r.duration}s
          </div>
        </div>
      </div>
    `).join('')}
  </div>
</body>
</html>`;
  
  fs.writeFileSync(path.join(outputDir, 'index.html'), html);
  
  console.log('\n' + '═'.repeat(60));
  console.log(`🏁 COMPLETE: ${successful.length}/${results.length} successful`);
  console.log(`Results: ${outputDir}`);
  console.log('═'.repeat(60));
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});

