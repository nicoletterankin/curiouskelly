/**
 * 🎬 Kelly S2V (Subject-to-Video) Test
 * 
 * Uses Minimax's S2V-01 model with Kelly's actual photo
 * as subject_reference to generate character-consistent videos.
 * 
 * THIS IS THE KEY: Same face across all videos!
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

require('dotenv').config();

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;

// Minimax with S2V-01 for character consistency
const MINIMAX_VERSION = '5aa835260ff7f40f4069c41185f72036accf99e29957bb4a3b3a911f3b6c1912';

// CANONICAL Kelly reference from Best Character Reference (LOCKED)
const KELLY_CANONICAL_REFERENCE = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/reference/kelly_primary_face.jpeg';

// Character spec from production factory
const KELLY_SPEC = {
  eyes: 'brown eyes',
  hair: 'long wavy brown hair',
  outfit: 'light blue sweater',
  minResolution: '1344x768',
};

const outputDir = path.join(__dirname, '..', 'template-forge', 'kelly-s2v');
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

async function generateS2V(subjectReferenceUrl, prompt) {
  console.log('\n🎬 Generating with S2V-01 (Subject Reference)...');
  console.log(`   Reference: ${subjectReferenceUrl.split('/').pop()}`);
  console.log(`   Prompt: "${prompt.substring(0, 70)}..."`);
  
  const input = {
    prompt,
    prompt_optimizer: true,
    subject_reference: subjectReferenceUrl,  // THIS IS THE KEY
  };
  
  console.log(`   Input: ${JSON.stringify(input).substring(0, 100)}...`);
  
  const createResponse = await makeRequest({
    hostname: 'api.replicate.com',
    path: '/v1/predictions',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${REPLICATE_API_TOKEN}`,
      'Content-Type': 'application/json',
    }
  }, JSON.stringify({
    version: MINIMAX_VERSION,
    input,
  }));
  
  if (createResponse.status !== 201) {
    let errorMsg = JSON.stringify(createResponse.data);
    if (createResponse.data?.type === 'Buffer') {
      errorMsg = Buffer.from(createResponse.data.data).toString('utf8');
    }
    throw new Error(`Failed to create: ${createResponse.status} - ${errorMsg}`);
  }
  
  const predictionId = createResponse.data.id;
  console.log(`   Prediction: ${predictionId}`);
  
  // Poll for completion
  let attempts = 0;
  while (attempts < 200) {
    await new Promise(r => setTimeout(r, 3000));
    
    const statusResponse = await makeRequest({
      hostname: 'api.replicate.com',
      path: `/v1/predictions/${predictionId}`,
      method: 'GET',
      headers: { 'Authorization': `Bearer ${REPLICATE_API_TOKEN}` }
    });
    
    const status = statusResponse.data.status;
    process.stdout.write(`\r   Status: ${status} (${attempts * 3}s)...                    `);
    
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
  console.log('🎬 KELLY S2V (Subject-to-Video) TEST');
  console.log('   Using Kelly\'s actual photo for character consistency');
  console.log('═'.repeat(70));
  
  // Test configurations using CANONICAL reference and Kelly's character spec
  const tests = [
    {
      name: 'kelly_welcome_s2v',
      reference: KELLY_CANONICAL_REFERENCE,
      prompt: `A woman with ${KELLY_SPEC.hair} and ${KELLY_SPEC.eyes} wearing a ${KELLY_SPEC.outfit} walks toward the camera on a sunlit forest path, opens her arms in a warm welcoming gesture with a genuine smile, natural graceful movement, cinematic film quality, 4K`,
    },
    {
      name: 'kelly_explain_s2v',
      reference: KELLY_CANONICAL_REFERENCE,
      prompt: `A woman with ${KELLY_SPEC.hair} and ${KELLY_SPEC.eyes} wearing a ${KELLY_SPEC.outfit} speaks and explains with natural hand gestures, engaged knowledgeable expression, standing against a soft neutral background, professional studio lighting, film quality`,
    },
    {
      name: 'kelly_heartfelt_s2v',
      reference: KELLY_CANONICAL_REFERENCE,
      prompt: `A woman with ${KELLY_SPEC.hair} and ${KELLY_SPEC.eyes} wearing a ${KELLY_SPEC.outfit} places her hand on her heart with a sincere warm expression, speaking emotionally, warm golden hour lighting, intimate cinematic moment`,
    },
  ];
  
  const args = process.argv.slice(2);
  const testIndex = args[0] ? parseInt(args[0]) : 0;
  const test = tests[testIndex];
  
  if (!test) {
    console.log('\nAvailable tests:');
    tests.forEach((t, i) => console.log(`  ${i}: ${t.name}`));
    return;
  }
  
  console.log(`\n🎯 Test: ${test.name}`);
  
  const startTime = Date.now();
  
  try {
    const output = await generateS2V(test.reference, test.prompt);
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    const filename = `${test.name}_${Date.now()}.mp4`;
    const localPath = await downloadVideo(output, filename);
    
    console.log('\n' + '═'.repeat(70));
    console.log('✅ S2V GENERATION SUCCESS!');
    console.log('═'.repeat(70));
    console.log(`   Time: ${duration}s`);
    console.log(`   File: ${localPath}`);
    console.log(`   URL: ${output}`);
    console.log('');
    console.log('🎯 CHECK: Does this video show KELLY (from the reference photo)?');
    console.log('   If yes, we have character-consistent templates!');
    console.log('═'.repeat(70));
    
    // Save result
    const result = {
      test: test.name,
      reference: test.reference,
      prompt: test.prompt,
      duration,
      output,
      localPath,
      timestamp: new Date().toISOString(),
    };
    
    fs.writeFileSync(
      path.join(outputDir, `${test.name}_result.json`),
      JSON.stringify(result, null, 2)
    );
    
  } catch (error) {
    console.log('\n' + '═'.repeat(70));
    console.log(`❌ FAILED: ${error.message}`);
    console.log('═'.repeat(70));
  }
}

main();

