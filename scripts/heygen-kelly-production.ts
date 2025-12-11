#!/usr/bin/env npx tsx
/**
 * 🚀 HEYGEN KELLY PRODUCTION PIPELINE
 * 
 * 13 Kelly archetypes mapped by head accessory.
 * Generates Day 1 videos for all archetypes.
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY!,
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  ELEVENLABS_KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'heygen-production'),
  BUCKET: 'kelly-videos',
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// 13 KELLY ARCHETYPES - MAPPED BY HEAD ACCESSORY
// Using the first 13 processed avatar IDs from your HeyGen account
// =============================================================================

const KELLY_ARCHETYPES: Record<string, { id: string; accessory: string }> = {
  "Base":       { id: "433ad96bf5d647d9964cecf784d008f6", accessory: "Animated base" },
  "Neutral":    { id: "7bb18cddacd44333813cc90ffa44f766", accessory: "None" },
  "Survivor":   { id: "a2b31ed0b5f84b0fa02d15d411735d3a", accessory: "Olive bandana" },
  "Mystic":     { id: "45e5ef8b651846e0b62b7477e552e87b", accessory: "White beanie" },
  "Rebel":      { id: "aa8b5eb1d711468a9a6e2085a4f8469c", accessory: "Red headband" },
  "MacGyver":   { id: "06b78109ad22489ea2165ebbf180f77b", accessory: "Aviator goggles" },
  "Architect":  { id: "9ffd06bd986a4e3086612921f3ac87ea", accessory: "Thin glasses" },
  "Consultant": { id: "e614671b193c40f99772f7de5d1c51f7", accessory: "Purple bindi" },
  "Empath":     { id: "b9032c922c6e4e35b58a98abd499d060", accessory: "Praying pose" },
  "Scientist":  { id: "3f44bd33bfd1494d916d2746808a1a39", accessory: "Round glasses" },
  "Explorer":   { id: "d4eccf6a8d4c427b9313208d640db407", accessory: "Goggles" },
  "Strategist": { id: "4227be1001a3431db2cb4c59f9c25287", accessory: "Sunglasses up" },
  "Provider":   { id: "d1d731dcdd5d4bb9af1c020a907671dc", accessory: "Dog tags" },
  "Storyteller":{ id: "4f28f8a7e7d44eab99f2cdd0d1530d5f", accessory: "Headphones" },
};

// =============================================================================
// AUDIO GENERATION
// =============================================================================

async function generateAudio(script: string): Promise<string> {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.ELEVENLABS_KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: script,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    }
  );

  if (!response.ok) throw new Error(`ElevenLabs: ${response.status}`);
  
  const buffer = Buffer.from(await response.arrayBuffer());
  const fileName = `audio_${Date.now()}.mp3`;
  
  await supabase.storage.from('kelly-templates').upload(
    `heygen/audio/${fileName}`, buffer,
    { contentType: 'audio/mpeg', upsert: true }
  );
  
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(`heygen/audio/${fileName}`);
  return data.publicUrl;
}

// =============================================================================
// VIDEO GENERATION
// =============================================================================

async function generateVideo(avatarId: string, audioUrl: string): Promise<string> {
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': CONFIG.HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: avatarId,
        },
        voice: {
          type: 'audio',
          audio_url: audioUrl,
        },
      }],
      dimension: { width: 1920, height: 1080 },
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HeyGen: ${error}`);
  }

  const result = await response.json();
  return result.data.video_id;
}

async function waitForVideo(videoId: string): Promise<string> {
  while (true) {
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
      { headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY } }
    );
    const result = await response.json();
    
    if (result.data.status === 'completed') return result.data.video_url;
    if (result.data.status === 'failed') throw new Error(`Failed: ${result.data.error}`);
    
    process.stdout.write('.');
    await new Promise(r => setTimeout(r, 10000));
  }
}

async function downloadAndUpload(videoUrl: string, name: string): Promise<string> {
  const response = await fetch(videoUrl);
  const buffer = Buffer.from(await response.arrayBuffer());
  
  // Save locally
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(CONFIG.OUTPUT_DIR, `${name}.mp4`), buffer);
  
  // Upload to Supabase
  const remotePath = `production/day_001/${name}.mp4`;
  await supabase.storage.from(CONFIG.BUCKET).upload(remotePath, buffer, {
    contentType: 'video/mp4', upsert: true
  });
  
  const { data } = supabase.storage.from(CONFIG.BUCKET).getPublicUrl(remotePath);
  return data.publicUrl;
}

// =============================================================================
// MAIN PRODUCTION RUN
// =============================================================================

async function generateForArchetype(archetype: string, script: string) {
  const config = KELLY_ARCHETYPES[archetype];
  if (!config) throw new Error(`Unknown archetype: ${archetype}`);

  console.log(`\n🎬 ${archetype} (${config.accessory})`);
  console.log(`   Script: "${script.substring(0, 50)}..."`);

  // 1. Audio
  console.log('   🎤 Generating audio...');
  const audioUrl = await generateAudio(script);

  // 2. Video
  console.log('   🎬 Generating video...');
  const videoId = await generateVideo(config.id, audioUrl);
  console.log(`   📹 Video ID: ${videoId}`);

  // 3. Wait
  console.log('   ⏳ Processing');
  const videoUrl = await waitForVideo(videoId);
  console.log(' ✅');

  // 4. Download & Upload
  console.log('   ☁️ Uploading...');
  const outputName = `${archetype.toLowerCase()}_hook`;
  const supabaseUrl = await downloadAndUpload(videoUrl, outputName);
  console.log(`   ✅ ${supabaseUrl}`);

  return supabaseUrl;
}

async function runDay1() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🚀 KELLY PRODUCTION - DAY 1 - 13 ARCHETYPES               ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // Test script for Day 1 Hook
  const hookScript = `Hey there, curious mind! I'm Kelly, and I'm SO excited to learn with you today! 
Did you know that everything around you is made of tiny building blocks called atoms? 
They're so small you can't see them, but they're everywhere - in the air you breathe, 
the water you drink, even in YOU! Pretty amazing, right? Let's discover more!`;

  const results: Record<string, string> = {};

  for (const archetype of Object.keys(KELLY_ARCHETYPES)) {
    try {
      const url = await generateForArchetype(archetype, hookScript);
      results[archetype] = url;
    } catch (error: any) {
      console.error(`   ❌ ${archetype}: ${error.message}`);
      results[archetype] = 'FAILED';
    }
  }

  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log('📋 RESULTS');
  console.log('═'.repeat(60));
  
  Object.entries(results).forEach(([arch, url]) => {
    const status = url.startsWith('http') ? '✅' : '❌';
    console.log(`${status} ${arch}`);
  });

  fs.writeFileSync(
    path.join(CONFIG.OUTPUT_DIR, 'day1_results.json'),
    JSON.stringify(results, null, 2)
  );

  console.log('\n🎯 Day 1 complete!');
}

// Quick test with ONE archetype
async function testOne() {
  console.log('🧪 Testing with Neutral archetype...\n');
  
  const script = "Hey! I'm Kelly. Just a quick test to make sure everything works!";
  
  try {
    await generateForArchetype('Neutral', script);
    console.log('\n✅ TEST PASSED!');
  } catch (error: any) {
    console.error(`\n❌ TEST FAILED: ${error.message}`);
  }
}

// Run
const args = process.argv.slice(2);
if (args.includes('--day1')) {
  runDay1().catch(console.error);
} else {
  testOne().catch(console.error);
}

