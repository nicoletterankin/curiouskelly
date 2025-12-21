import 'dotenv/config';
import Replicate from 'replicate';

async function testReplicate() {
  console.log('Starting Replicate test...');
  console.log('Token prefix:', process.env.REPLICATE_API_TOKEN?.substring(0, 6));
  
  const replicate = new Replicate({ 
    auth: process.env.REPLICATE_API_TOKEN 
  });
  
  console.log('Calling Replicate...');
  
  try {
    const output = await replicate.run(
      'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d' as `${string}/${string}:${string}`,
      {
        input: {
          prompt: 'kelly, woman with brown hair, blue sweater, smiling warmly, studio background',
          hf_lora: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
          lora_scale: 0.85,
          num_outputs: 1,
          aspect_ratio: '16:9',
        }
      }
    );
    
    console.log('Output received!');
    console.log('Type:', typeof output);
    console.log('Is Array:', Array.isArray(output));
    console.log('Raw:', JSON.stringify(output, null, 2));
    
    if (Array.isArray(output) && output.length > 0) {
      const first = output[0];
      console.log('\nFirst item:');
      console.log('  Type:', typeof first);
      console.log('  Constructor:', first?.constructor?.name);
      console.log('  String():', String(first).substring(0, 100));
    }
    
  } catch (error: any) {
    console.error('Error:', error.message);
    console.error('Full:', error);
  }
}

testReplicate();


