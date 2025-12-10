#!/usr/bin/env npx tsx
/**
 * 🚀 PHOTOREALISTIC KELLY VIDEO TEST
 * 
 * Uses the REAL Kelly photo (not iClone, not AI-generated)
 * and tests multiple state-of-the-art talking head APIs
 * 
 * Goal: Find which one makes Kelly look HUMAN and PERFECT
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

const CONFIG = {
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  KELLY_VOICE_ID: 'wAdymQH5YucAkXwmrdL0',
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  FAL_KEY: process.env.FAL_KEY!,
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  
  // THE REAL KELLY - photorealistic reference
  KELLY_PHOTO: 'C:\\iLearnStudio\\projects\\Kelly\\Ref\\Best Character Reference\\head and shoulders without chair.png',
  
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'photorealistic-test'),
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// Short test script
const TEST_SCRIPT = "Hey! Ever notice how New Year's Day makes you feel like anything is possible? That's not just a feeling—it's science.";

async function generateAudio(): Promise<string> {
  console.log('\n🎤 Generating audio with ElevenLabs...');
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: TEST_SCRIPT,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.85,
          style: 0.3,
          use_speaker_boost: true,
        },
      }),
    }
  );
  
  if (!response.ok) throw new Error(`ElevenLabs: ${response.status}`);
  
  const audioPath = path.join(CONFIG.OUTPUT_DIR, 'kelly_test.mp3');
  const audioBuffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(audioPath, audioBuffer);
  console.log(`   ✅ Audio: ${audioPath}`);
  
  return audioPath;
}

async function uploadToSupabase(filePath: string, remotePath: string): Promise<string> {
  const fileBuffer = fs.readFileSync(filePath);
  await supabase.storage.from('kelly-templates').upload(remotePath, fileBuffer, { upsert: true });
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
  return data.publicUrl;
}

// =============================================================================
// TEST 1: Hedra Character-2 (best for natural expressions)
// =============================================================================
async function testHedra(imageUrl: string, audioUrl: string): Promise<string | null> {
  console.log('\n🎬 TEST 1: HEDRA Character-2...');
  
  try {
    const { fal } = await import('@fal-ai/client');
    fal.config({ credentials: CONFIG.FAL_KEY });
    
    const result = await fal.subscribe('fal-ai/hedra/character-2', {
      input: {
        audio_url: audioUrl,
        character_image_url: imageUrl,
        aspect_ratio: '16:9',
      },
      logs: true,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') process.stdout.write('.');
      },
    });
    
    const videoUrl = (result as any)?.video?.url;
    if (videoUrl) {
      console.log(`\n   ✅ Hedra SUCCESS: ${videoUrl}`);
      return videoUrl;
    }
  } catch (e: any) {
    console.log(`\n   ❌ Hedra failed: ${e.message}`);
  }
  return null;
}

// =============================================================================
// TEST 2: LivePortrait (realistic micro-expressions)
// =============================================================================
async function testLivePortrait(imageUrl: string, audioUrl: string): Promise<string | null> {
  console.log('\n🎬 TEST 2: LivePortrait...');
  
  try {
    const { fal } = await import('@fal-ai/client');
    fal.config({ credentials: CONFIG.FAL_KEY });
    
    const result = await fal.subscribe('fal-ai/liveportrait', {
      input: {
        image_url: imageUrl,
        video_url: audioUrl, // Some models use this for audio
      },
      logs: true,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') process.stdout.write('.');
      },
    });
    
    const videoUrl = (result as any)?.video?.url;
    if (videoUrl) {
      console.log(`\n   ✅ LivePortrait SUCCESS: ${videoUrl}`);
      return videoUrl;
    }
  } catch (e: any) {
    console.log(`\n   ❌ LivePortrait failed: ${e.message}`);
  }
  return null;
}

// =============================================================================
// TEST 3: SyncLabs lipsync-2 with static image (premium quality)
// =============================================================================
async function testSyncLabsImage(imageUrl: string, audioUrl: string): Promise<string | null> {
  console.log('\n🎬 TEST 3: Sync Labs lipsync-2 (image input)...');
  
  try {
    const createResponse = await fetch('https://api.sync.so/v2/generate', {
      method: 'POST',
      headers: {
        'x-api-key': CONFIG.SYNC_LABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'lipsync-2',
        input: [
          { type: 'image', url: imageUrl },
          { type: 'audio', url: audioUrl },
        ],
        options: { output_format: 'mp4' },
      }),
    });
    
    if (!createResponse.ok) throw new Error(await createResponse.text());
    
    const { id: jobId } = await createResponse.json();
    console.log(`   Job: ${jobId}`);
    
    // Poll for completion
    for (let i = 0; i < 60; i++) {
      await new Promise(r => setTimeout(r, 5000));
      const statusResponse = await fetch(`https://api.sync.so/v2/generate/${jobId}`, {
        headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
      });
      const result = await statusResponse.json();
      process.stdout.write(result.status === 'COMPLETED' ? '✅' : '.');
      
      if (result.status === 'COMPLETED') {
        console.log(`\n   ✅ SyncLabs SUCCESS: ${result.outputUrl}`);
        return result.outputUrl;
      }
      if (result.status === 'FAILED') throw new Error(result.error);
    }
  } catch (e: any) {
    console.log(`\n   ❌ SyncLabs failed: ${e.message}`);
  }
  return null;
}

// =============================================================================
// TEST 4: Hallo2 (audio-driven portrait animation)
// =============================================================================
async function testHallo2(imageUrl: string, audioUrl: string): Promise<string | null> {
  console.log('\n🎬 TEST 4: Hallo2...');
  
  try {
    const { fal } = await import('@fal-ai/client');
    fal.config({ credentials: CONFIG.FAL_KEY });
    
    const result = await fal.subscribe('fal-ai/hallo2', {
      input: {
        source_image: imageUrl,
        driving_audio: audioUrl,
        pose_weight: 1.0,
        face_weight: 1.0,
        lip_weight: 1.5,
        face_expand_ratio: 1.2,
      },
      logs: true,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') process.stdout.write('.');
      },
    });
    
    const videoUrl = (result as any)?.video?.url;
    if (videoUrl) {
      console.log(`\n   ✅ Hallo2 SUCCESS: ${videoUrl}`);
      return videoUrl;
    }
  } catch (e: any) {
    console.log(`\n   ❌ Hallo2 failed: ${e.message}`);
  }
  return null;
}

// =============================================================================
// TEST 5: MuseTalk (latest, very natural)
// =============================================================================
async function testMuseTalk(imageUrl: string, audioUrl: string): Promise<string | null> {
  console.log('\n🎬 TEST 5: MuseTalk...');
  
  try {
    const { fal } = await import('@fal-ai/client');
    fal.config({ credentials: CONFIG.FAL_KEY });
    
    const result = await fal.subscribe('fal-ai/musetalk', {
      input: {
        source_image: imageUrl,
        audio_input: audioUrl,
      },
      logs: true,
      onQueueUpdate: (update: any) => {
        if (update.status === 'IN_PROGRESS') process.stdout.write('.');
      },
    });
    
    const videoUrl = (result as any)?.video?.url;
    if (videoUrl) {
      console.log(`\n   ✅ MuseTalk SUCCESS: ${videoUrl}`);
      return videoUrl;
    }
  } catch (e: any) {
    console.log(`\n   ❌ MuseTalk failed: ${e.message}`);
  }
  return null;
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🚀 PHOTOREALISTIC KELLY VIDEO COMPARISON                  ║');
  console.log('║  Testing 5 different APIs to find THE BEST                 ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  
  // Generate audio
  const audioPath = await generateAudio();
  
  // Upload assets to Supabase
  console.log('\n☁️ Uploading to Supabase...');
  const timestamp = Date.now();
  const imageUrl = await uploadToSupabase(CONFIG.KELLY_PHOTO, `photorealistic-test/kelly_${timestamp}.png`);
  const audioUrl = await uploadToSupabase(audioPath, `photorealistic-test/audio_${timestamp}.mp3`);
  console.log(`   Image: ${imageUrl}`);
  console.log(`   Audio: ${audioUrl}`);
  
  // Run all tests
  const results: { name: string; url: string | null }[] = [];
  
  results.push({ name: 'Hedra Character-2', url: await testHedra(imageUrl, audioUrl) });
  results.push({ name: 'LivePortrait', url: await testLivePortrait(imageUrl, audioUrl) });
  results.push({ name: 'SyncLabs lipsync-2', url: await testSyncLabsImage(imageUrl, audioUrl) });
  results.push({ name: 'Hallo2', url: await testHallo2(imageUrl, audioUrl) });
  results.push({ name: 'MuseTalk', url: await testMuseTalk(imageUrl, audioUrl) });
  
  // Summary
  console.log('\n\n' + '═'.repeat(60));
  console.log('📊 RESULTS SUMMARY');
  console.log('═'.repeat(60));
  
  const successful = results.filter(r => r.url);
  const failed = results.filter(r => !r.url);
  
  if (successful.length > 0) {
    console.log('\n✅ SUCCESSFUL:');
    successful.forEach(r => {
      console.log(`   ${r.name}: ${r.url}`);
    });
  }
  
  if (failed.length > 0) {
    console.log('\n❌ FAILED:');
    failed.forEach(r => console.log(`   ${r.name}`));
  }
  
  if (successful.length > 0) {
    console.log('\n' + '═'.repeat(60));
    console.log('🎬 WATCH THESE VIDEOS AND TELL ME WHICH LOOKS BEST!');
    console.log('═'.repeat(60));
  } else {
    console.log('\n⚠️ All tests failed. Check API keys and quotas.');
  }
}

main().catch(console.error);

