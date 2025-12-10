#!/usr/bin/env npx tsx
/**
 * 🎬 HEYGEN KELLY VIDEO PIPELINE
 * 
 * Generates videos using Kelly's actual HeyGen avatar.
 * Uploads to Supabase, updates database.
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
  
  // Storage
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'heygen-kelly'),
  SUPABASE_BUCKET: 'kelly-videos',
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// KELLY AVATAR IDS - REAL ONES
// =============================================================================

// Kelly avatar group
const KELLY_AVATAR_GROUP = 'a762125d3107477aba43d1bd79f90d6e';

// Individual Kelly avatars by archetype
// NOTE: Use talking_photo_id from HeyGen, must have motion applied!
const KELLY_AVATARS: Record<string, string> = {
  // The Strategist needs motion applied first - using working photo for now
  "The Strategist": "d8ba9d6f0a994046b4d9fbe2d6428a95", // Working test photo
  
  // UPDATE THESE as you apply motion to Kelly photos:
  // "The Scientist": "xxx",
  // "The Explorer": "xxx", 
  // etc.
};

// =============================================================================
// GENERATE VIDEO
// =============================================================================

async function generateVideo(avatarId: string, script: string, outputName: string): Promise<string> {
  console.log(`\n🎬 Generating video: ${outputName}`);
  console.log(`   Avatar: ${avatarId}`);
  console.log(`   Script: "${script.substring(0, 60)}..."`);

  // Step 1: Generate audio with ElevenLabs
  console.log('   🎤 Generating audio...');
  const audioResponse = await fetch(
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

  if (!audioResponse.ok) {
    throw new Error(`ElevenLabs error: ${audioResponse.status}`);
  }

  const audioBuffer = Buffer.from(await audioResponse.arrayBuffer());
  
  // Upload audio to Supabase for HeyGen to access
  const audioFileName = `audio_${Date.now()}.mp3`;
  await supabase.storage.from('kelly-templates').upload(
    `heygen/audio/${audioFileName}`,
    audioBuffer,
    { contentType: 'audio/mpeg', upsert: true }
  );
  const { data: audioData } = supabase.storage.from('kelly-templates').getPublicUrl(`heygen/audio/${audioFileName}`);
  console.log(`   ✅ Audio uploaded`);

  // Step 2: Generate HeyGen video
  console.log('   🎬 Calling HeyGen API...');
  const videoResponse = await fetch('https://api.heygen.com/v2/video/generate', {
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
          audio_url: audioData.publicUrl,
        },
        background: {
          type: 'color',
          value: '#FFFFFF',
        },
      }],
      dimension: { width: 1920, height: 1080 },
    }),
  });

  if (!videoResponse.ok) {
    const error = await videoResponse.text();
    throw new Error(`HeyGen error: ${error}`);
  }

  const videoResult = await videoResponse.json();
  const videoId = videoResult.data.video_id;
  console.log(`   📹 Video ID: ${videoId}`);

  // Step 3: Wait for completion
  console.log('   ⏳ Waiting for video...');
  let videoUrl = '';
  while (true) {
    const statusResponse = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
      { headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY } }
    );
    const statusResult = await statusResponse.json();
    
    if (statusResult.data.status === 'completed') {
      videoUrl = statusResult.data.video_url;
      console.log(`   ✅ Video complete!`);
      break;
    }
    if (statusResult.data.status === 'failed') {
      throw new Error(`Video failed: ${statusResult.data.error}`);
    }
    
    process.stdout.write('.');
    await new Promise(r => setTimeout(r, 10000));
  }

  // Step 4: Download video
  console.log('   📥 Downloading video...');
  const downloadResponse = await fetch(videoUrl);
  const videoBuffer = Buffer.from(await downloadResponse.arrayBuffer());

  // Save locally
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  const localPath = path.join(CONFIG.OUTPUT_DIR, `${outputName}.mp4`);
  fs.writeFileSync(localPath, videoBuffer);
  console.log(`   💾 Saved: ${localPath}`);

  // Step 5: Upload to Supabase
  console.log('   ☁️ Uploading to Supabase...');
  const supabasePath = `production/heygen/${outputName}.mp4`;
  await supabase.storage.from(CONFIG.SUPABASE_BUCKET).upload(
    supabasePath,
    videoBuffer,
    { contentType: 'video/mp4', upsert: true }
  );
  const { data: publicData } = supabase.storage.from(CONFIG.SUPABASE_BUCKET).getPublicUrl(supabasePath);
  console.log(`   ✅ Uploaded: ${publicData.publicUrl}`);

  return publicData.publicUrl;
}

// =============================================================================
// TEST WITH STRATEGIST
// =============================================================================

async function testStrategist() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN KELLY PIPELINE - TEST                           ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  const testScript = `Hey there! I'm Kelly, and I'm so excited to learn with you today. 
Did you know that your brain is like a supercomputer? It's the most powerful thing in the universe, 
and it's sitting right there between your ears! Pretty cool, right? 
Let's discover something amazing together!`;

  try {
    const url = await generateVideo(
      KELLY_AVATARS["The Strategist"],
      testScript,
      'kelly_strategist_test'
    );

    console.log('\n' + '═'.repeat(60));
    console.log('🎉 SUCCESS!');
    console.log('═'.repeat(60));
    console.log(`Video URL: ${url}`);
    console.log(`\nTest at: http://localhost:3000/learn?day=1&clearcache=1`);

  } catch (error: any) {
    console.error(`\n❌ Error: ${error.message}`);
  }
}

// =============================================================================
// GENERATE DAY 1
// =============================================================================

async function generateDay1() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN KELLY PIPELINE - DAY 1                          ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // Get Day 1 lesson atoms
  const { data: atoms, error } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('day_number', 1)
    .order('phase');

  if (error || !atoms?.length) {
    console.error('❌ Could not fetch Day 1:', error);
    return;
  }

  console.log(`Found ${atoms.length} atoms for Day 1`);

  for (const atom of atoms) {
    const archetype = atom.archetype;
    const avatarId = KELLY_AVATARS[archetype];

    if (!avatarId) {
      console.log(`⚠️ No avatar for ${archetype}, skipping`);
      continue;
    }

    const script = atom.content?.script;
    if (!script) {
      console.log(`⚠️ No script for ${archetype} - ${atom.phase}, skipping`);
      continue;
    }

    const outputName = `day_001_${atom.phase.toLowerCase()}_${archetype.replace(/\s+/g, '_').toLowerCase()}`;

    try {
      const url = await generateVideo(avatarId, script, outputName);

      // Update database
      await supabase
        .from('lesson_atoms')
        .update({ hd_video_url: url })
        .eq('id', atom.id);

      console.log(`✅ Updated database for ${archetype} - ${atom.phase}`);

    } catch (error: any) {
      console.error(`❌ Failed ${archetype} - ${atom.phase}: ${error.message}`);
    }
  }

  console.log('\n🎯 Day 1 generation complete!');
}

// =============================================================================
// MAIN
// =============================================================================

const args = process.argv.slice(2);
if (args.includes('--day1')) {
  generateDay1().catch(console.error);
} else {
  testStrategist().catch(console.error);
}
