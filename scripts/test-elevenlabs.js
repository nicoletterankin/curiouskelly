/**
 * ElevenLabs TTS Connection Test Script
 * Tests connectivity to ElevenLabs API and generates a test audio clip
 * 
 * Run from project root: node scripts/test-elevenlabs.js
 */

import { createRequire } from 'module';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { writeFileSync, mkdirSync, existsSync } from 'fs';

// Setup paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(resolve(__dirname, '../daily-lesson-marketing/package.json'));

const dotenv = require('dotenv');

// Load environment variables from project root .env
const envPath = resolve(__dirname, '..', '.env');
dotenv.config({ path: envPath });

const ELEVENLABS_API_URL = 'https://api.elevenlabs.io/v1';

async function testElevenLabsConnection() {
  console.log('🔄 Testing ElevenLabs connection...\n');

  // Check for required environment variables
  const apiKey = process.env.ELEVENLABS_API_KEY;
  const voiceId = process.env.ELEVENLABS_KELLY_VOICE_ID;

  if (!apiKey) {
    console.error('❌ Error: ELEVENLABS_API_KEY is not set in .env');
    process.exit(1);
  }

  if (!voiceId) {
    console.error('❌ Error: ELEVENLABS_KELLY_VOICE_ID is not set in .env');
    process.exit(1);
  }

  console.log(`🔑 API Key: ${apiKey.substring(0, 10)}...[redacted]`);
  console.log(`🎤 Voice ID: ${voiceId}`);
  console.log('');

  const testText = "Hello! I'm Kelly, your curious learning companion!";
  console.log(`📝 Generating audio for: "${testText}"\n`);

  try {
    // Call ElevenLabs Text-to-Speech API
    const response = await fetch(`${ELEVENLABS_API_URL}/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': apiKey
      },
      body: JSON.stringify({
        text: testText,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.0,
          use_speaker_boost: true
        }
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errorJson = JSON.parse(errorText);
        errorMessage = errorJson.detail?.message || errorJson.detail || errorMessage;
      } catch {
        errorMessage = errorText || errorMessage;
      }
      throw new Error(errorMessage);
    }

    // Get audio buffer
    const audioBuffer = await response.arrayBuffer();
    
    // Ensure output directory exists
    const outputDir = resolve(__dirname, '..', 'test-output');
    if (!existsSync(outputDir)) {
      mkdirSync(outputDir, { recursive: true });
      console.log(`📁 Created output directory: test-output/`);
    }

    // Save audio file
    const outputPath = resolve(outputDir, 'kelly-test.mp3');
    writeFileSync(outputPath, Buffer.from(audioBuffer));
    
    const fileSizeKB = (audioBuffer.byteLength / 1024).toFixed(1);
    console.log(`✅ ElevenLabs working! Audio saved to test-output/kelly-test.mp3`);
    console.log(`📊 File size: ${fileSizeKB} KB`);
    console.log('\n🎉 Connection test passed successfully!');

  } catch (error) {
    console.error('❌ Connection failed:', error.message);
    
    if (error.message.includes('401') || error.message.includes('Unauthorized')) {
      console.log('\n💡 Hint: Check that ELEVENLABS_API_KEY is valid.');
    } else if (error.message.includes('voice_not_found') || error.message.includes('404')) {
      console.log('\n💡 Hint: Check that ELEVENLABS_KELLY_VOICE_ID is correct.');
      console.log('   You can find voice IDs at: https://elevenlabs.io/voice-library');
    } else if (error.message.includes('quota') || error.message.includes('limit')) {
      console.log('\n💡 Hint: You may have exceeded your ElevenLabs quota.');
    }
    
    process.exit(1);
  }
}

testElevenLabsConnection();













