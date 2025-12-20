#!/usr/bin/env npx tsx
/**
 * 🚀 HEYGEN DAY 1 FULL PRODUCTION
 * 
 * Generates ALL phases (Hook, Fact1, Fact2, Fact3, Wisdom) for ALL archetypes.
 * Pulls scripts from Supabase database.
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
// KELLY ARCHETYPES - 14 AVATARS
// =============================================================================

const KELLY_AVATARS: Record<string, string> = {
  "Base":           "433ad96bf5d647d9964cecf784d008f6",
  "Neutral":        "7bb18cddacd44333813cc90ffa44f766",
  "The Survivor":   "a2b31ed0b5f84b0fa02d15d411735d3a",
  "The Mystic":     "45e5ef8b651846e0b62b7477e552e87b",
  "The Rebel":      "aa8b5eb1d711468a9a6e2085a4f8469c",
  "The MacGyver":   "06b78109ad22489ea2165ebbf180f77b",
  "The Architect":  "9ffd06bd986a4e3086612921f3ac87ea",
  "Consultant":     "e614671b193c40f99772f7de5d1c51f7",
  "The Empath":     "b9032c922c6e4e35b58a98abd499d060",
  "The Scientist":  "3f44bd33bfd1494d916d2746808a1a39",
  "The Explorer":   "d4eccf6a8d4c427b9313208d640db407",
  "The Strategist": "4227be1001a3431db2cb4c59f9c25287",
  "Provider":       "d1d731dcdd5d4bb9af1c020a907671dc",
  "The Storyteller":"4f28f8a7e7d44eab99f2cdd0d1530d5f",
};

// =============================================================================
// CORE FUNCTIONS
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

async function downloadAndUpload(videoUrl: string, day: number, phase: string, archetype: string): Promise<string> {
  const response = await fetch(videoUrl);
  const buffer = Buffer.from(await response.arrayBuffer());
  
  const safeName = archetype.replace(/\s+/g, '_').replace(/^The_/, '').toLowerCase();
  const fileName = `day_${String(day).padStart(3, '0')}_${phase.toLowerCase()}_${safeName}.mp4`;
  
  // Save locally
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(CONFIG.OUTPUT_DIR, fileName), buffer);
  
  // Upload to Supabase
  const remotePath = `production/day_${String(day).padStart(3, '0')}/${fileName}`;
  await supabase.storage.from(CONFIG.BUCKET).upload(remotePath, buffer, {
    contentType: 'video/mp4', upsert: true
  });
  
  const { data } = supabase.storage.from(CONFIG.BUCKET).getPublicUrl(remotePath);
  return data.publicUrl;
}

// =============================================================================
// MAIN PRODUCTION
// =============================================================================

async function runDay1Production() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🚀 DAY 1 FULL PRODUCTION - ALL PHASES                     ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // Get Day 1 lesson
  const { data: lessons } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', 1)
    .single();

  if (!lessons) {
    console.error('❌ Day 1 lesson not found');
    return;
  }

  // Get all atoms for Day 1
  const { data: atoms } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lessons.id);

  if (!atoms?.length) {
    console.error('❌ No atoms found for Day 1');
    return;
  }

  console.log(`📚 Found ${atoms.length} atoms for Day 1\n`);

  const phases = ['Fact1', 'Fact2', 'Fact3', 'Wisdom'];
  const results: any[] = [];

  for (const phase of phases) {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`📌 PHASE: ${phase}`);
    console.log('═'.repeat(60));

    const phaseAtoms = atoms.filter(a => a.phase === phase);
    
    for (const atom of phaseAtoms) {
      const archetype = atom.archetype;
      const script = atom.content?.script;
      
      if (!script) {
        console.log(`⚠️ No script for ${archetype} - ${phase}`);
        continue;
      }

      // Find avatar ID
      const avatarId = KELLY_AVATARS[archetype];
      if (!avatarId) {
        console.log(`⚠️ No avatar for ${archetype}`);
        continue;
      }

      console.log(`\n🎬 ${archetype} - ${phase}`);
      console.log(`   Script: "${script.substring(0, 50)}..."`);

      try {
        // 1. Audio
        console.log('   🎤 Audio...');
        const audioUrl = await generateAudio(script);

        // 2. Video
        console.log('   🎬 Video...');
        const videoId = await generateVideo(avatarId, audioUrl);
        console.log(`   📹 ID: ${videoId}`);

        // 3. Wait
        process.stdout.write('   ⏳ ');
        const videoUrl = await waitForVideo(videoId);
        console.log(' ✅');

        // 4. Upload
        console.log('   ☁️ Upload...');
        const supabaseUrl = await downloadAndUpload(videoUrl, 1, phase, archetype);

        // 5. Update DB
        await supabase
          .from('lesson_atoms')
          .update({ hd_video_url: supabaseUrl })
          .eq('id', atom.id);

        console.log(`   ✅ ${supabaseUrl}`);
        results.push({ archetype, phase, status: 'success', url: supabaseUrl });

      } catch (error: any) {
        console.error(`   ❌ ${error.message}`);
        results.push({ archetype, phase, status: 'failed', error: error.message });
      }
    }
  }

  // Summary
  console.log('\n\n' + '═'.repeat(60));
  console.log('📋 FINAL RESULTS');
  console.log('═'.repeat(60));
  
  const success = results.filter(r => r.status === 'success').length;
  const failed = results.filter(r => r.status === 'failed').length;
  
  console.log(`✅ Success: ${success}`);
  console.log(`❌ Failed: ${failed}`);

  fs.writeFileSync(
    path.join(CONFIG.OUTPUT_DIR, 'day1_full_results.json'),
    JSON.stringify(results, null, 2)
  );

  console.log('\n🎯 Day 1 production complete!');
}

runDay1Production().catch(console.error);

















