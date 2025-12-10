/**
 * Test generating a SINGLE Kelly frame with mouth position
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const KELLY_LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

async function main() {
  console.log('🎨 Testing single Kelly frame generation...\n');
  
  if (!REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not set');
    process.exit(1);
  }

  const replicate = new Replicate({ auth: REPLICATE_API_TOKEN });
  
  const prompt = "curious_kelly, mouth slightly open speaking, warm smile, professional portrait photo, looking at camera, warm studio lighting, soft background, 4k, high quality, photorealistic";
  
  console.log('📝 Prompt:', prompt);
  console.log('🔗 LoRA:', KELLY_LORA_URL);
  console.log('\n⏳ Generating...\n');

  try {
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: prompt,
          hf_lora: KELLY_LORA_URL,
          lora_scale: 0.9,
          num_outputs: 1,
          aspect_ratio: "1:1",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28,
          seed: 12345,
        }
      }
    );

    console.log('📦 Raw output:', JSON.stringify(output, null, 2));

    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    if (typeof imageUrl === 'string' && imageUrl.startsWith('http')) {
      console.log('\n✅ Got image URL:', imageUrl);
      
      // Download it
      const response = await fetch(imageUrl);
      const buffer = Buffer.from(await response.arrayBuffer());
      
      const outputDir = path.join(process.cwd(), 'test-output', 'frame-gen');
      if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
      
      const framePath = path.join(outputDir, 'test-frame.png');
      fs.writeFileSync(framePath, buffer);
      
      console.log(`💾 Saved: ${framePath} (${(buffer.length/1024).toFixed(1)} KB)`);
      
      // Open it
      const { execSync } = require('child_process');
      execSync(`start "" "${framePath}"`, { stdio: 'ignore' });
    } else {
      console.log('❌ Unexpected output format');
    }
  } catch (error: any) {
    console.error('\n❌ ERROR:', error.message);
    console.error('Full error:', JSON.stringify(error, null, 2));
  }
}

main();




