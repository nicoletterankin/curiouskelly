/**
 * Direct ElevenLabs TTS Test
 * Tests the API key directly without going through the server
 * 
 * Run: node test-tts-direct.mjs
 */

import { config } from 'dotenv';
config();

const API_KEY = process.env.ELEVENLABS_API_KEY;
const VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

console.log('\n🔊 ElevenLabs Direct TTS Test\n');
console.log('═'.repeat(50));

// Check API Key
if (!API_KEY) {
  console.log('❌ ELEVENLABS_API_KEY not found in .env');
  console.log('\nMake sure you have a .env file with:');
  console.log('ELEVENLABS_API_KEY=sk_your_key_here');
  process.exit(1);
}

console.log('✅ API Key found:', API_KEY.substring(0, 15) + '...');
console.log('✅ Voice ID:', VOICE_ID);

async function testTTS() {
  const testText = "Hello! I'm Kelly, your curious learning companion. Let's explore something amazing today!";
  
  console.log('\n📤 Sending TTS Request...');
  console.log('   Text:', testText.substring(0, 50) + '...');
  console.log('   Endpoint:', `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`);
  
  const startTime = Date.now();
  
  try {
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`,
      {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': API_KEY,
        },
        body: JSON.stringify({
          text: testText,
          model_id: 'eleven_monolingual_v1',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.0,
            use_speaker_boost: true,
          },
        }),
      }
    );

    const duration = Date.now() - startTime;
    
    console.log('\n📥 Response received in', duration, 'ms');
    console.log('   Status:', response.status, response.statusText);
    
    // Log response headers
    console.log('\n   Headers:');
    response.headers.forEach((value, key) => {
      if (!key.startsWith('x-') || key.includes('character') || key.includes('remaining')) {
        console.log(`     ${key}: ${value}`);
      }
    });
    
    if (response.ok) {
      const audioBuffer = await response.arrayBuffer();
      console.log('\n✅ SUCCESS!');
      console.log('   Audio size:', audioBuffer.byteLength, 'bytes');
      console.log('   Audio duration estimate:', Math.round(audioBuffer.byteLength / 16000), 'seconds');
      
      // Save audio for verification
      const fs = await import('fs');
      fs.writeFileSync('test-output.mp3', Buffer.from(audioBuffer));
      console.log('   Saved to: test-output.mp3');
      
      // Check remaining quota
      const remaining = response.headers.get('x-ratelimit-remaining-characters');
      if (remaining) {
        console.log('\n   📊 Remaining characters:', remaining);
      }
    } else {
      const errorText = await response.text();
      console.log('\n❌ FAILED!');
      console.log('   Error:', errorText);
      
      // Parse error for more info
      try {
        const errorJson = JSON.parse(errorText);
        console.log('\n   Parsed error:');
        console.log('     Status:', errorJson.detail?.status || 'unknown');
        console.log('     Message:', errorJson.detail?.message || errorJson.error || 'unknown');
      } catch (e) {
        // Not JSON
      }
      
      // Suggestions based on status code
      console.log('\n   💡 Suggestions:');
      if (response.status === 401) {
        console.log('     - API key is invalid or expired');
        console.log('     - Get a new key from https://elevenlabs.io/app/settings/api-keys');
      } else if (response.status === 403) {
        console.log('     - API key does not have permission');
        console.log('     - Check your ElevenLabs subscription');
      } else if (response.status === 429) {
        console.log('     - Rate limit exceeded');
        console.log('     - Wait a bit or upgrade your plan');
      } else if (response.status === 400) {
        console.log('     - Invalid request parameters');
        console.log('     - Check voice_id is valid');
      }
    }
  } catch (error) {
    console.log('\n❌ Network Error:', error.message);
    console.log('\n   Possible causes:');
    console.log('     - No internet connection');
    console.log('     - Firewall blocking request');
    console.log('     - DNS resolution failure');
  }
}

// Also test voice list to verify API key is valid
async function testVoiceList() {
  console.log('\n\n🎤 Testing Voice List API (verifies API key)...');
  
  try {
    const response = await fetch('https://api.elevenlabs.io/v1/voices', {
      headers: {
        'xi-api-key': API_KEY,
      },
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ API Key is valid!');
      console.log('   Available voices:', data.voices?.length || 0);
      
      // Find Kelly's voice
      const kellyVoice = data.voices?.find(v => v.voice_id === VOICE_ID);
      if (kellyVoice) {
        console.log(`   ✅ Kelly's voice found: "${kellyVoice.name}"`);
      } else {
        console.log(`   ⚠️  Voice ${VOICE_ID} not found in your voice list`);
        console.log('   Available voice IDs:');
        data.voices?.slice(0, 5).forEach(v => {
          console.log(`     - ${v.voice_id}: ${v.name}`);
        });
      }
    } else {
      const error = await response.text();
      console.log('❌ API Key verification failed:', response.status);
      console.log('   Error:', error);
    }
  } catch (error) {
    console.log('❌ Voice list check failed:', error.message);
  }
}

// Run tests
async function main() {
  await testVoiceList();
  await testTTS();
  
  console.log('\n' + '═'.repeat(50));
  console.log('Test complete!\n');
}

main().catch(console.error);


