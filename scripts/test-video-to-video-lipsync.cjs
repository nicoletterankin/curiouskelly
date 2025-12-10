/**
 * 🎬 Video-to-Video Lipsync Test
 * 
 * Critical proof-of-concept:
 * Can we take an existing video of Kelly moving and replace the audio?
 * 
 * If this works, we can:
 * 1. Create template videos of Kelly doing actions
 * 2. Voice-over with any script
 * 3. Full puppet control!
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config();

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;

// Wav2Lip accepts VIDEO as input (not just image!)
const WAV2LIP_MODEL = 'devxpy/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef';

const outputDir = path.join(__dirname, '..', 'video-to-video-test');
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
  console.log('🎙️ Generating new audio with ElevenLabs...');
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

async function runWav2LipOnVideo(videoUrl, audioBuffer) {
  console.log('\n🎬 Running Wav2Lip on VIDEO input...');
  console.log(`   Video: ${videoUrl.substring(0, 60)}...`);
  
  const audioBase64 = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  const input = {
    face: videoUrl,  // KEY: This is a VIDEO URL, not an image!
    audio: audioBase64,
    fps: 30,
    smooth: true,
    resize_factor: 1,
  };
  
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
    input,
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
      console.log('\n   ✅ Video generated!');
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
  console.log('🎬 VIDEO-TO-VIDEO LIPSYNC TEST');
  console.log('═'.repeat(70));
  console.log('');
  console.log('This test will:');
  console.log('1. Take an EXISTING video of Kelly');
  console.log('2. Generate NEW audio with different text');
  console.log('3. Apply lipsync to replace the speech');
  console.log('4. See if body movement is PRESERVED');
  console.log('');
  console.log('═'.repeat(70));
  
  // Use one of our previously generated full-body Kelly videos
  // This video has Kelly with open arms, talking
  const sourceVideoUrl = 'https://replicate.delivery/yhqm/MebBdrcgayy0byEbv70Qq3gCQiBvkMFkfGEz7pDQfMXpJvgrA/result_voice.mp4';
  
  // NEW text - completely different from original
  const newText = "Wow! Did you know that butterflies taste with their feet? Nature is full of surprises like this. Let's discover more together!";
  
  console.log('📼 Source video: Kelly full body hook (open arms)');
  console.log('📝 New script: Butterfly fact');
  console.log('');
  
  const startTime = Date.now();
  
  try {
    // Generate new audio
    const audioBuffer = await generateAudio(newText);
    
    // Run Wav2Lip with VIDEO input
    const outputVideoUrl = await runWav2LipOnVideo(sourceVideoUrl, audioBuffer);
    
    // Download result
    const filename = `v2v_test_${Date.now()}.mp4`;
    const localPath = await downloadVideo(outputVideoUrl, filename);
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    console.log('');
    console.log('═'.repeat(70));
    console.log('✅ VIDEO-TO-VIDEO LIPSYNC SUCCESS!');
    console.log('═'.repeat(70));
    console.log(`   Time: ${duration}s`);
    console.log(`   Output: ${localPath}`);
    console.log(`   URL: ${outputVideoUrl}`);
    console.log('');
    console.log('🎯 NEXT STEP: Watch the video and verify:');
    console.log('   - Does Kelly say the NEW text (butterfly fact)?');
    console.log('   - Are her arms/body movements PRESERVED from original?');
    console.log('   - Is the lipsync accurate?');
    console.log('═'.repeat(70));
    
    return { success: true, outputVideoUrl, localPath };
    
  } catch (error) {
    console.log('');
    console.log('═'.repeat(70));
    console.log(`❌ FAILED: ${error.message}`);
    console.log('═'.repeat(70));
    return { success: false, error: error.message };
  }
}

main();



