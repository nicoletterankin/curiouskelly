import Replicate from 'replicate';
import fs from 'fs';
import https from 'https';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const replicate = new Replicate();

const KELLY_LORA = 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors';

async function downloadImage(url, outputPath) {
  return new Promise((resolve, reject) => {
    const dir = path.dirname(outputPath);
    fs.mkdirSync(dir, { recursive: true });
    
    const file = fs.createWriteStream(outputPath);
    
    const handleResponse = (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        https.get(response.headers.location, handleResponse).on('error', reject);
      } else {
        response.pipe(file);
        file.on('finish', () => { file.close(); resolve(); });
        file.on('error', reject);
      }
    };
    
    https.get(url, handleResponse).on('error', reject);
  });
}

async function collectStream(stream) {
  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return chunks.join('');
}

async function generate() {
  console.log('🎨 KELLY VISUAL GENERATOR - Day 8 Test');
  console.log('=====================================\n');
  console.log('📍 Topic: What Makes a Real Friend');
  
  const prompt = `kelly, photorealistic woman named Kelly, late 20s, brown wavy shoulder-length hair with caramel highlights, hazel-brown almond-shaped eyes, wearing soft powder blue cashmere sweater, in cozy living room with warm lighting and comfortable atmosphere, welcoming open stance with arms slightly open in invitation, warm genuine smile showing excitement, looking directly at viewer with curiosity and enthusiasm, full body visible, natural confident posture, mood: warm connected trusting, cinematic photography, natural lighting, 8K, shallow depth of field`;
  
  console.log('\n🚀 Generating with Kelly LoRA (this takes 30-60 seconds)...');
  
  try {
    // Use prediction API for more control
    const prediction = await replicate.predictions.create({
      model: 'lucataco/flux-dev-lora',
      version: 'a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
      input: {
        prompt: prompt,
        hf_lora: KELLY_LORA,
        lora_scale: 0.85,
        num_outputs: 1,
        aspect_ratio: '16:9',
        output_format: 'png',
        guidance_scale: 3.5,
        num_inference_steps: 28
      }
    });
    
    console.log('⏳ Prediction started:', prediction.id);
    
    // Wait for completion
    let result = prediction;
    while (result.status !== 'succeeded' && result.status !== 'failed') {
      await new Promise(r => setTimeout(r, 2000));
      result = await replicate.predictions.get(prediction.id);
      console.log('   Status:', result.status);
    }
    
    if (result.status === 'failed') {
      console.error('❌ Generation failed:', result.error);
      return;
    }
    
    const imageUrl = result.output?.[0];
    
    if (imageUrl && typeof imageUrl === 'string') {
      console.log('✅ Image URL:', imageUrl);
      
      const outputPath = path.join(process.cwd(), 'public/kelly/phases/008/hook.png');
      console.log('\n💾 Downloading to:', outputPath);
      
      await downloadImage(imageUrl, outputPath);
      console.log('✅ Saved successfully!');
      
      const stats = fs.statSync(outputPath);
      console.log(`📊 File size: ${(stats.size / 1024).toFixed(1)} KB`);
      console.log(`💰 Cost: ~$0.04`);
    } else {
      console.log('⚠️ No image URL in output');
      console.log('Output:', JSON.stringify(result.output, null, 2));
    }
    
    console.log('\n🎉 TEST COMPLETE!');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.error(error.stack);
  }
}

generate();
