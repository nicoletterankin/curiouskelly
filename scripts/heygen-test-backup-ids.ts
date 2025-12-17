#!/usr/bin/env npx tsx
import 'dotenv/config';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

const testIds: Record<string, string> = {
  'architect_C_backup': 'dbfa11bf4568497fa7cfccc083951cfe',
  'diplomat_A': '0c110d20a92d47e68340baef8b2816b3',
  'survivor_B_backup': 'e85a5db4609c44dd898ea2876dbc1cfb',
  'survivor_C_backup': 'a0676076ed6549a1be3ce7a1c8d03353',
};

async function test() {
  console.log('\n🔍 Testing backup avatar IDs:\n');
  
  for (const [name, id] of Object.entries(testIds)) {
    try {
      const response = await fetch('https://api.heygen.com/v2/video/generate', {
        method: 'POST',
        headers: { 'X-Api-Key': HEYGEN_API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_inputs: [{
            character: { type: 'talking_photo', talking_photo_id: id },
            voice: { type: 'text', input_text: 'Test.', voice_id: '0015ce4f932b405b9fc3a5e2f5e92c46', speed: 1.0 },
            background: { type: 'color', value: '#1a1a2e' },
          }],
          dimension: { width: 1920, height: 1080 },
        }),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        console.log(`✅ ${name}: ${id}`);
        console.log(`   video_id: ${data.data?.video_id}\n`);
      } else {
        console.log(`❌ ${name}: ${id}`);
        console.log(`   Error: ${data.error?.message || JSON.stringify(data)}\n`);
      }
    } catch (err) {
      console.log(`❌ ${name}: Network error\n`);
    }
    
    await new Promise(r => setTimeout(r, 1000));
  }
}

test();
