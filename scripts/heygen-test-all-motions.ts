#!/usr/bin/env npx tsx
import 'dotenv/config';
import * as fs from 'fs';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const archetype = process.argv[2] || 'architect';

async function test() {
  const library = JSON.parse(fs.readFileSync('generated-images/kelly-motion-library.json', 'utf-8'));
  
  console.log(`\n🔍 Testing all ${archetype} motion IDs:\n`);
  
  for (const [motion, id] of Object.entries(library[archetype] as Record<string, string>)) {
    console.log(`Motion ${motion}: ${id}`);
    
    try {
      const response = await fetch('https://api.heygen.com/v2/video/generate', {
        method: 'POST',
        headers: {
          'X-Api-Key': HEYGEN_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_inputs: [{
            character: { type: 'talking_photo', talking_photo_id: id },
            voice: { type: 'text', input_text: 'Quick test.', voice_id: '0015ce4f932b405b9fc3a5e2f5e92c46', speed: 1.0 },
            background: { type: 'color', value: '#1a1a2e' },
          }],
          dimension: { width: 1920, height: 1080 },
        }),
      });
      
      const data = await response.json();
      
      if (response.ok && data.data?.video_id) {
        console.log(`  ✅ OK - video_id: ${data.data.video_id}\n`);
      } else {
        console.log(`  ❌ Error (${response.status}): ${JSON.stringify(data)}\n`);
      }
    } catch (err) {
      console.log(`  ❌ Network error: ${err}\n`);
    }
    
    // Small delay between requests
    await new Promise(r => setTimeout(r, 1000));
  }
}

test();
