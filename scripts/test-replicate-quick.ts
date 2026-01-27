import Replicate from 'replicate';
import * as dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });

async function test() {
  console.log('Token prefix:', process.env.REPLICATE_API_TOKEN?.substring(0, 8));
  
  const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
  console.log('Testing Replicate API with Kelly LoRA...\n');
  
  try {
    const output = await replicate.run(
      'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
      {
        input: {
          prompt: 'kelly, professional female teacher, close-up portrait, warm friendly smile, brown wavy hair, soft blue sweater, white studio background, professional headshot, corporate photography',
          hf_lora: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
          lora_scale: 0.85,
          num_outputs: 1,
          aspect_ratio: '1:1',
          output_format: 'png',
          guidance_scale: 3.5,
          num_inference_steps: 28,
          disable_safety_checker: true,
        },
      }
    );
    
    console.log('✅ Success!');
    console.log('Output type:', typeof output);
    console.log('Is array:', Array.isArray(output));
    console.log('Output:', output);
    
  } catch (err: any) {
    console.error('❌ Error:', err.message);
    if (err.response) {
      console.error('Response:', await err.response.text());
    }
  }
}

test();
