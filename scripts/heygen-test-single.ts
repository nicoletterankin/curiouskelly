#!/usr/bin/env npx tsx
/**
 * Test a single archetype to debug API errors
 */
import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const archetype = process.argv[2] || 'rebel';

async function test() {
  const library = JSON.parse(fs.readFileSync('generated-images/kelly-motion-library.json', 'utf-8'));
  
  if (!library[archetype]) {
    console.log('Unknown archetype:', archetype);
    return;
  }
  
  const avatarId = library[archetype].A;
  console.log(`\n🔍 Testing ${archetype} Motion A: ${avatarId}\n`);

  try {
    const response = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        video_inputs: [{
          character: { type: 'talking_photo', talking_photo_id: avatarId },
          voice: { type: 'text', input_text: 'This is a test.', voice_id: '0015ce4f932b405b9fc3a5e2f5e92c46', speed: 1.0 },
          background: { type: 'color', value: '#1a1a2e' },
        }],
        dimension: { width: 1920, height: 1080 },
      }),
    });

    console.log('HTTP Status:', response.status);
    console.log('Headers:', Object.fromEntries(response.headers.entries()));
    
    const text = await response.text();
    console.log('\nResponse Body:');
    console.log(text);
    
    try {
      const json = JSON.parse(text);
      console.log('\nParsed JSON:');
      console.log(JSON.stringify(json, null, 2));
    } catch (e) {
      console.log('(Not valid JSON)');
    }
  } catch (error) {
    console.error('Fetch error:', error);
  }
}

test();
