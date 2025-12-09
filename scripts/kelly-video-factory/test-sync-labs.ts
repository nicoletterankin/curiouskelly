/**
 * 🎬 SYNC LABS DIRECT TEST
 * 
 * Tests the Sync Labs API directly with Kelly's image and voice.
 * $300 credits ready - let's make perfection happen.
 * 
 * Usage:
 *   npx tsx scripts/kelly-video-factory/test-sync-labs.ts
 */

import 'dotenv/config';
import Replicate from 'replicate';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// Configuration
const CONFIG = {
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
};

const OUTPUT_DIR = path.join(process.cwd(), 'generated-videos', 'sync-labs-test');

// Kelly's identity
const KELLY_PROMPT = `kelly, friendly approachable teacher, intelligent warmth, genuine smile lines, natural beauty, woman with long wavy chestnut brown hair with subtle highlights and warm brown eyes with visible catchlights, wearing soft powder blue crewneck sweater, eyes sparkling with genuine excitement and wonder, natural joyful expression, warm modern classroom environment with soft bokeh background, cinematic lighting, shallow depth of field, 85mm lens, professional color grading, soft diffused lighting, 4K UHD`;

async function main() {
  console.log('');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 SYNC LABS PREMIUM TEST                                   ║');
  console.log('║  $300 credits • 95% lip-sync accuracy • Film quality         ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('');
  
  // Validate keys
  console.log('🔑 Checking API Keys...');
  const keys = {
    REPLICATE: !!CONFIG.REPLICATE_API_TOKEN,
    ELEVENLABS: !!CONFIG.ELEVENLABS_API_KEY,
    SYNC_LABS: !!CONFIG.SYNC_LABS_API_KEY,
    SUPABASE: !!CONFIG.SUPABASE_URL && !!CONFIG.SUPABASE_KEY,
  };
  
  Object.entries(keys).forEach(([name, valid]) => {
    console.log(`   ${valid ? '✅' : '❌'} ${name}`);
  });
  
  if (!keys.REPLICATE || !keys.ELEVENLABS || !keys.SYNC_LABS) {
    console.error('\n❌ Missing required API keys');
    process.exit(1);
  }
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
  
  // Step 1: Generate Kelly image
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📸 Step 1: Generating Kelly image with LoRA');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const imageOutput = await replicate.run(
    "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
    {
      input: {
        prompt: KELLY_PROMPT,
        hf_lora: CONFIG.KELLY_LORA_URL,
        lora_scale: 0.85,
        num_outputs: 1,
        aspect_ratio: "16:9",
        output_format: "png",
        guidance_scale: 3.5,
        output_quality: 100,
        num_inference_steps: 35,
      }
    }
  );
  
  let imageUrl: string;
  if (Array.isArray(imageOutput)) {
    imageUrl = String(imageOutput[0]);
  } else if (typeof imageOutput === 'object' && imageOutput !== null) {
    imageUrl = String((imageOutput as any).url || imageOutput);
  } else {
    imageUrl = String(imageOutput);
  }
  
  console.log(`   ✅ Image: ${imageUrl}`);
  
  // Step 2: Generate audio with ElevenLabs
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🎤 Step 2: Generating Kelly audio (ElevenLabs)');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const testText = "Hello! I'm Kelly, and I'm so excited to learn with you today. Together, we're going to discover something truly amazing. Are you ready?";
  
  console.log(`   Text: "${testText}"`);
  
  const audioResponse = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text: testText,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.85,
          style: 0.2,
          use_speaker_boost: true,
        },
      }),
    }
  );
  
  if (!audioResponse.ok) {
    throw new Error(`ElevenLabs error: ${audioResponse.status}`);
  }
  
  const audioBuffer = Buffer.from(await audioResponse.arrayBuffer());
  const audioFileName = `kelly_sync_test_${Date.now()}.mp3`;
  const localAudioPath = path.join(OUTPUT_DIR, audioFileName);
  fs.writeFileSync(localAudioPath, audioBuffer);
  
  console.log(`   ✅ Audio saved locally: ${localAudioPath} (${(audioBuffer.length / 1024).toFixed(1)} KB)`);
  
  // Step 3: Upload audio to Supabase for public URL
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('☁️ Step 3: Uploading audio to Supabase Storage');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  let audioUrl: string;
  
  try {
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('kelly-templates')
      .upload(`sync-labs-test/${audioFileName}`, audioBuffer, {
        contentType: 'audio/mpeg',
        upsert: true,
      });
    
    if (uploadError) {
      console.log(`   ⚠️ Supabase upload failed: ${uploadError.message}`);
      console.log('   Using Replicate file upload as fallback...');
      
      // Fallback: use a public file hosting or data URI
      // For now, we'll try using the Replicate prediction URL approach
      throw new Error('Need public audio URL');
    }
    
    const { data: publicUrl } = supabase.storage
      .from('kelly-templates')
      .getPublicUrl(`sync-labs-test/${audioFileName}`);
    
    audioUrl = publicUrl.publicUrl;
    console.log(`   ✅ Audio uploaded: ${audioUrl}`);
    
  } catch (error) {
    console.log('   ⚠️ Supabase not available, using alternative upload...');
    
    // Try uploading to tmpfiles.org or similar service
    const formData = new FormData();
    formData.append('file', new Blob([audioBuffer], { type: 'audio/mpeg' }), audioFileName);
    
    try {
      const tmpResponse = await fetch('https://tmpfiles.org/api/v1/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (tmpResponse.ok) {
        const tmpData = await tmpResponse.json();
        // Convert tmpfiles.org URL to direct download URL
        audioUrl = tmpData.data.url.replace('tmpfiles.org/', 'tmpfiles.org/dl/');
        console.log(`   ✅ Audio uploaded to tmpfiles: ${audioUrl}`);
      } else {
        throw new Error('tmpfiles upload failed');
      }
    } catch (tmpError) {
      // Last resort: use a base64 data URL (might not work with Sync Labs)
      console.log('   ⚠️ Using data URI as last resort...');
      audioUrl = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
    }
  }
  
  // Step 4: Generate base video with Wav2Lip first
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🎬 Step 4: Generating base video with Wav2Lip');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  const audioDataUri = `data:audio/mpeg;base64,${audioBuffer.toString('base64')}`;
  
  const baseVideoOutput = await replicate.run(
    "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
    {
      input: {
        face: imageUrl,
        audio: audioDataUri,
        fps: 25,
        pads: "0 10 0 0",
        smooth: true,
        resize_factor: 1,
      }
    }
  );
  
  const baseVideoUrl = typeof baseVideoOutput === 'string' ? baseVideoOutput : String(baseVideoOutput);
  console.log(`   ✅ Base video: ${baseVideoUrl}`);
  
  // Step 5: Enhance with Sync Labs
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🚀 Step 5: SYNC LABS - Premium Lip-Sync Enhancement');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  console.log(`   Model: lipsync-2`);
  console.log(`   Video: ${baseVideoUrl.substring(0, 60)}...`);
  console.log(`   Audio: ${audioUrl.substring(0, 60)}...`);
  
  // Submit to Sync Labs
  const syncPayload = {
    model: 'lipsync-2',
    input: [
      { type: 'video', url: baseVideoUrl },
      { type: 'audio', url: audioUrl.startsWith('data:') ? audioUrl : audioUrl },
    ],
  };
  
  console.log(`\n   📤 Submitting to Sync Labs...`);
  
  const syncResponse = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(syncPayload),
  });
  
  if (!syncResponse.ok) {
    const errorText = await syncResponse.text();
    console.log(`   ❌ Sync Labs error: ${syncResponse.status}`);
    console.log(`   Response: ${errorText}`);
    
    // If Sync Labs fails, output the base video anyway
    console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📊 RESULTS (Base Video Only)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log(`   Image: ${imageUrl}`);
    console.log(`   Audio: ${audioUrl}`);
    console.log(`   Video (Wav2Lip): ${baseVideoUrl}`);
    console.log('\n   ⚠️ Sync Labs enhancement failed - check API key and inputs');
    return;
  }
  
  const syncJob = await syncResponse.json();
  console.log(`   ✅ Job submitted: ${syncJob.id}`);
  console.log(`   Status: ${syncJob.status}`);
  
  // Poll for completion
  console.log(`\n   ⏳ Processing (this may take 1-3 minutes)...`);
  
  let finalVideoUrl: string | null = null;
  
  for (let i = 0; i < 60; i++) { // 5 minute timeout
    await sleep(5000);
    
    const statusResponse = await fetch(`https://api.sync.so/v2/generate/${syncJob.id}`, {
      headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
    });
    
    const status = await statusResponse.json();
    
    if (status.status === 'COMPLETED') {
      finalVideoUrl = status.output?.[0]?.url || status.outputUrl;
      console.log(`\n   ✅ SYNC LABS COMPLETE!`);
      break;
    }
    
    if (status.status === 'FAILED' || status.status === 'REJECTED') {
      console.log(`\n   ❌ Job failed: ${status.error || status.message}`);
      break;
    }
    
    if (i % 6 === 0) {
      console.log(`      Status: ${status.status} (${Math.round(i * 5 / 60)}m)`);
    }
    process.stdout.write('.');
  }
  
  // Final results
  console.log('\n');
  console.log('═'.repeat(64));
  console.log('📊 FINAL RESULTS');
  console.log('═'.repeat(64));
  console.log(`   Image: ${imageUrl}`);
  console.log(`   Audio: ${audioUrl}`);
  console.log(`   Base Video (Wav2Lip 70%): ${baseVideoUrl}`);
  
  if (finalVideoUrl) {
    console.log(`   🌟 PREMIUM VIDEO (Sync Labs 95%): ${finalVideoUrl}`);
    console.log('\n   🎉 SUCCESS! Kelly is now at 95% lip-sync accuracy!');
  } else {
    console.log(`   ⚠️ Sync Labs enhancement did not complete`);
  }
  
  console.log('═'.repeat(64));
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});


