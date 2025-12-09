/**
 * Test script for ElevenLabs Omnihuman 1.5 lip-sync video generation
 * 
 * This script:
 * 1. Reads a Kelly static image
 * 2. Generates TTS audio using ElevenLabs
 * 3. Calls the Omnihuman API to create a lip-synced video
 * 4. Saves the video locally
 * 
 * Usage:
 *   npx ts-node scripts/test-omnihuman.ts
 * 
 * Prerequisites:
 *   - ELEVENLABS_API_KEY environment variable set
 */

import * as dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });
dotenv.config();

import * as fs from 'fs';
import * as path from 'path';

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

// Test text for Kelly to speak
const TEST_TEXT = "Hello! I'm Kelly, and I'm so excited to learn with you today! Let's discover something amazing together.";

// Output directory
const OUTPUT_DIR = path.join(process.cwd(), 'test-output');

/**
 * Generate TTS audio using ElevenLabs
 */
async function generateTTSAudio(text: string): Promise<Buffer> {
  console.log('📢 Generating TTS audio...');
  console.log(`   Text: "${text.substring(0, 50)}..."`);
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY!,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.0,
          use_speaker_boost: true,
        },
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`TTS API error ${response.status}: ${errorText}`);
  }

  const audioBuffer = await response.arrayBuffer();
  console.log(`   ✅ Audio generated: ${audioBuffer.byteLength} bytes`);
  
  return Buffer.from(audioBuffer);
}

/**
 * Test different potential ElevenLabs video API endpoints
 */
async function testVideoEndpoints(imageBuffer: Buffer, audioBuffer: Buffer): Promise<{endpoint: string, result: any}[]> {
  const results: {endpoint: string, result: any}[] = [];
  
  // List of potential endpoints to test
  const potentialEndpoints = [
    'https://api.elevenlabs.io/v1/image-to-video',
    'https://api.elevenlabs.io/v1/image-to-video/omnihuman',
    'https://api.elevenlabs.io/v1/omnihuman/generate',
    'https://api.elevenlabs.io/v1/text-to-video',
    'https://api.elevenlabs.io/v1/avatar/generate',
  ];

  for (const endpoint of potentialEndpoints) {
    console.log(`\n🔍 Testing endpoint: ${endpoint}`);
    
    try {
      // Try with FormData (multipart)
      const formData = new FormData();
      formData.append('source_image', new Blob([imageBuffer], { type: 'image/png' }), 'kelly.png');
      formData.append('audio', new Blob([audioBuffer], { type: 'audio/mpeg' }), 'speech.mp3');
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'xi-api-key': ELEVENLABS_API_KEY!,
        },
        body: formData,
      });

      const contentType = response.headers.get('content-type');
      let result;
      
      if (contentType?.includes('application/json')) {
        result = await response.json();
      } else if (contentType?.includes('video')) {
        result = { type: 'video', size: (await response.arrayBuffer()).byteLength };
      } else {
        result = await response.text();
      }

      console.log(`   Status: ${response.status}`);
      console.log(`   Response:`, typeof result === 'object' ? JSON.stringify(result, null, 2) : result.substring(0, 200));
      
      results.push({ endpoint, result: { status: response.status, body: result } });
      
      // If we got a 200, this endpoint works!
      if (response.ok) {
        console.log(`   ✅ ENDPOINT WORKS!`);
        return results;
      }
    } catch (error) {
      console.log(`   ❌ Error:`, error instanceof Error ? error.message : error);
      results.push({ endpoint, result: { error: String(error) } });
    }
  }
  
  return results;
}

/**
 * Try the Python SDK style endpoint (based on search results)
 */
async function tryImageToVideo(imageBuffer: Buffer, audioBuffer: Buffer): Promise<Buffer | null> {
  console.log('\n🎬 Attempting image-to-video generation...');
  
  // First, let's check what endpoints are available
  console.log('\n📋 Checking available API endpoints...');
  
  try {
    const docsResponse = await fetch('https://api.elevenlabs.io/v1', {
      headers: { 'xi-api-key': ELEVENLABS_API_KEY! }
    });
    console.log(`   API root response: ${docsResponse.status}`);
  } catch (e) {
    console.log('   Could not reach API root');
  }

  // Try the most likely endpoint based on search results
  const endpoints = [
    {
      url: 'https://api.elevenlabs.io/v1/image-to-video',
      method: 'formdata'
    },
    {
      url: 'https://api.elevenlabs.io/v1/image-to-video/generate',
      method: 'formdata'
    }
  ];

  for (const { url, method } of endpoints) {
    console.log(`\n🔄 Trying: ${url}`);
    
    try {
      const formData = new FormData();
      formData.append('image', new Blob([imageBuffer], { type: 'image/png' }), 'kelly.png');
      formData.append('audio', new Blob([audioBuffer], { type: 'audio/mpeg' }), 'speech.mp3');
      // Optional parameters that might be needed
      formData.append('crop_to_face', 'false');

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'xi-api-key': ELEVENLABS_API_KEY!,
        },
        body: formData,
      });

      console.log(`   Status: ${response.status}`);
      console.log(`   Headers:`, Object.fromEntries(response.headers.entries()));

      if (response.ok) {
        const contentType = response.headers.get('content-type');
        
        if (contentType?.includes('video') || contentType?.includes('octet-stream')) {
          // Direct video response
          const videoBuffer = Buffer.from(await response.arrayBuffer());
          console.log(`   ✅ Got video directly: ${videoBuffer.length} bytes`);
          return videoBuffer;
        }
        
        // JSON response - might contain video URL or generation ID
        const result = await response.json();
        console.log(`   Response:`, JSON.stringify(result, null, 2));
        
        // If there's a video URL, download it
        if (result.video_url || result.output_url || result.url) {
          const videoUrl = result.video_url || result.output_url || result.url;
          console.log(`   📥 Downloading video from: ${videoUrl}`);
          
          const videoResponse = await fetch(videoUrl);
          if (videoResponse.ok) {
            const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
            console.log(`   ✅ Downloaded video: ${videoBuffer.length} bytes`);
            return videoBuffer;
          }
        }
        
        // If async, poll for completion
        if (result.generation_id || result.id || result.request_id) {
          const id = result.generation_id || result.id || result.request_id;
          console.log(`   ⏳ Async generation started: ${id}`);
          return await pollForVideo(id);
        }
      } else {
        const errorText = await response.text();
        console.log(`   ❌ Error: ${errorText.substring(0, 500)}`);
      }
    } catch (error) {
      console.log(`   ❌ Request failed:`, error instanceof Error ? error.message : error);
    }
  }

  return null;
}

/**
 * Poll for async video generation completion
 */
async function pollForVideo(generationId: string, maxAttempts = 60): Promise<Buffer | null> {
  console.log(`\n⏳ Polling for video completion...`);
  
  const statusEndpoints = [
    `https://api.elevenlabs.io/v1/image-to-video/${generationId}`,
    `https://api.elevenlabs.io/v1/image-to-video/status/${generationId}`,
    `https://api.elevenlabs.io/v1/generations/${generationId}`,
  ];

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    console.log(`   Attempt ${attempt + 1}/${maxAttempts}...`);
    
    for (const statusUrl of statusEndpoints) {
      try {
        const response = await fetch(statusUrl, {
          headers: { 'xi-api-key': ELEVENLABS_API_KEY! }
        });

        if (!response.ok) continue;

        const result = await response.json();
        console.log(`   Status: ${result.status}`);

        if (result.status === 'completed' || result.status === 'done' || result.status === 'succeeded') {
          const videoUrl = result.video_url || result.output_url || result.url;
          if (videoUrl) {
            const videoResponse = await fetch(videoUrl);
            if (videoResponse.ok) {
              return Buffer.from(await videoResponse.arrayBuffer());
            }
          }
        }

        if (result.status === 'failed' || result.status === 'error') {
          console.log(`   ❌ Generation failed:`, result.error || result.message);
          return null;
        }

        // Found valid endpoint, break inner loop
        break;
      } catch (e) {
        // Try next endpoint
      }
    }

    // Wait before next poll
    await new Promise(r => setTimeout(r, 5000));
  }

  console.log('   ❌ Polling timeout');
  return null;
}

/**
 * Main test function
 */
async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🎬 ELEVENLABS OMNIHUMAN 1.5 TEST');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  // Check API key
  if (!ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY environment variable not set');
    console.log('   Set it in .env.local or export ELEVENLABS_API_KEY=your_key');
    process.exit(1);
  }
  console.log('✅ API key found:', ELEVENLABS_API_KEY.substring(0, 10) + '...');

  // Create output directory
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
  console.log(`📁 Output directory: ${OUTPUT_DIR}`);

  // Step 1: Read Kelly image
  console.log('\n📷 Loading Kelly image...');
  const imagePath = path.join(process.cwd(), 'public/kelly/poses/kelly_welcome.png');
  
  if (!fs.existsSync(imagePath)) {
    console.error(`❌ Image not found: ${imagePath}`);
    console.log('   Available poses:');
    const posesDir = path.join(process.cwd(), 'public/kelly/poses');
    if (fs.existsSync(posesDir)) {
      fs.readdirSync(posesDir).forEach(f => console.log(`     - ${f}`));
    }
    process.exit(1);
  }
  
  const imageBuffer = fs.readFileSync(imagePath);
  console.log(`   ✅ Image loaded: ${imageBuffer.length} bytes`);

  // Step 2: Generate TTS audio
  const audioBuffer = await generateTTSAudio(TEST_TEXT);
  
  // Save audio for reference
  const audioPath = path.join(OUTPUT_DIR, 'test-audio.mp3');
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   💾 Audio saved: ${audioPath}`);

  // Step 3: Try image-to-video generation
  const videoBuffer = await tryImageToVideo(imageBuffer, audioBuffer);
  
  if (videoBuffer) {
    // Save video
    const videoPath = path.join(OUTPUT_DIR, 'kelly-talking.mp4');
    fs.writeFileSync(videoPath, videoBuffer);
    console.log(`\n✅ SUCCESS! Video saved: ${videoPath}`);
    console.log(`   File size: ${videoBuffer.length} bytes`);
  } else {
    console.log('\n⚠️  Could not generate video with standard endpoints.');
    console.log('   The image-to-video API might:');
    console.log('   1. Require beta access / feature flag');
    console.log('   2. Have a different endpoint structure');
    console.log('   3. Not be available on your plan');
    console.log('');
    console.log('   Let me check what API endpoints are available...');
    
    // List available API endpoints
    await checkAvailableEndpoints();
  }
}

/**
 * Check what ElevenLabs API endpoints are available
 */
async function checkAvailableEndpoints() {
  console.log('\n📋 Checking available ElevenLabs API features...');
  
  const testEndpoints = [
    { url: 'https://api.elevenlabs.io/v1/user', name: 'User Info' },
    { url: 'https://api.elevenlabs.io/v1/voices', name: 'Voices' },
    { url: 'https://api.elevenlabs.io/v1/models', name: 'Models' },
    { url: 'https://api.elevenlabs.io/v1/history', name: 'History' },
    { url: 'https://api.elevenlabs.io/v1/convai/agents', name: 'Conversational AI Agents' },
  ];

  for (const { url, name } of testEndpoints) {
    try {
      const response = await fetch(url, {
        headers: { 'xi-api-key': ELEVENLABS_API_KEY! }
      });
      console.log(`   ${name}: ${response.status === 200 ? '✅' : '❌'} (${response.status})`);
      
      // If it's user info, show subscription details
      if (url.includes('/user') && response.ok) {
        const userData = await response.json();
        console.log(`      Subscription: ${userData.subscription?.tier || 'Unknown'}`);
        console.log(`      Character count: ${userData.subscription?.character_count || 0}`);
        console.log(`      Character limit: ${userData.subscription?.character_limit || 0}`);
      }
    } catch (e) {
      console.log(`   ${name}: ❌ Error`);
    }
  }

  // Try to find video-related endpoints
  console.log('\n🔍 Searching for video-related endpoints...');
  const videoEndpoints = [
    'https://api.elevenlabs.io/v1/image-to-video',
    'https://api.elevenlabs.io/v1/video',
    'https://api.elevenlabs.io/v1/avatar',
    'https://api.elevenlabs.io/v1/talking-head',
    'https://api.elevenlabs.io/v1/lipsync',
    'https://api.elevenlabs.io/v1/omnihuman',
  ];

  for (const url of videoEndpoints) {
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'xi-api-key': ELEVENLABS_API_KEY! }
      });
      const status = response.status;
      const statusText = status === 404 ? 'Not found' : 
                         status === 405 ? 'Method not allowed (POST only?)' :
                         status === 401 ? 'Unauthorized' :
                         status === 403 ? 'Forbidden (need higher tier?)' :
                         status === 200 ? 'Available!' : `Status ${status}`;
      console.log(`   ${url.replace('https://api.elevenlabs.io/v1/', '')}: ${statusText}`);
    } catch (e) {
      console.log(`   ${url}: Error`);
    }
  }
}

// Run
main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});



