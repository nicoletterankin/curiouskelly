/**
 * 🎬 Test Template with V2V Lipsync
 * 
 * Takes a generated template video and applies new audio/lipsync.
 * This proves we can voice-over any template.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config();

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;

const WAV2LIP_MODEL = 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef';

const outputDir = path.join(__dirname, '..', 'template-forge', 'v2v-tests');
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
  console.log('🎙️ Generating Kelly voice audio...');
  console.log(`   Text: "${text.substring(0, 60)}..."`);
  
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
  console.log(`   ✅ Audio: ${(response.data.length / 1024).toFixed(1)}KB`);
  return response.data;
}

async function applyV2VLipsync(videoUrl, audioBuffer) {
  console.log('\n🎬 Applying V2V lipsync to template...');
  
  const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify({
    version: WAV2LIP_MODEL.split(':')[1],
    input: {
      face: videoUrl,  // VIDEO input
      audio: audioBase64,
      fps: 25,
      smooth: true,
      resize_factor: 1,
    },
  }));
  
  if (createResponse.status !== 201) {
    throw new Error(`Replicate error: ${createResponse.status} - ${JSON.stringify(createResponse.data)}`);
  }
  
  const predictionId = createResponse.data.id;
  console.log(`   Prediction: ${predictionId}`);
  
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
    process.stdout.write(`\r   Status: ${status} (${attempts * 3}s)...      `);
    
    if (status === 'succeeded') {
      console.log('\n   ✅ V2V lipsync complete!');
      return statusResponse.data.output;
    } else if (status === 'failed') {
      console.log('');
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

async function main() {
  console.log('═'.repeat(70));
  console.log('🎬 TEMPLATE V2V LIPSYNC TEST');
  console.log('═'.repeat(70));
  
  // The template we just generated
  const templateVideoUrl = 'https://replicate.delivery/xezq/7VSe9KNTeZt2V0g4Hs0E6GUOleq8GzZp2Sw0r63TmonLLwgrA/tmpbhg9lvto.mp4';
  
  // Kelly's script to voice-over
  const script = `Hello curious learner! Today we're going to discover something amazing. 
Did you know that your brain has about 86 billion neurons? That's more stars than in many galaxies! 
Each one of these tiny cells is working right now, helping you learn and grow.`;
  
  console.log('\n📼 Template: Present & Explain (Minimax)');
  console.log('📝 Script: Brain neurons fact');
  console.log('');
  
  const startTime = Date.now();
  
  try {
    const audioBuffer = await generateAudio(script);
    const outputVideoUrl = await applyV2VLipsync(templateVideoUrl, audioBuffer);
    
    const filename = `template_v2v_${Date.now()}.mp4`;
    const localPath = await downloadVideo(outputVideoUrl, filename);
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    console.log('');
    console.log('═'.repeat(70));
    console.log('✅ TEMPLATE V2V LIPSYNC SUCCESS!');
    console.log('═'.repeat(70));
    console.log(`   Time: ${duration}s`);
    console.log(`   Output: ${localPath}`);
    console.log(`   URL: ${outputVideoUrl}`);
    console.log('');
    console.log('🎯 This proves: We can voice-over ANY template video!');
    console.log('═'.repeat(70));
    
  } catch (error) {
    console.log('');
    console.log(`❌ FAILED: ${error.message}`);
  }
}

main();


