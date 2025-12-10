#!/usr/bin/env npx tsx
/**
 * 🎬 KELLY iCLONE + SYNC LABS PIPELINE
 * 
 * Uses YOUR actual iClone-rendered Kelly video as the source,
 * then applies Sync Labs for perfect lip-sync.
 * 
 * This produces NATURAL Kelly, not AI-generated robotic Kelly.
 * 
 * Usage:
 *   npx tsx scripts/kelly-iclone-lipsync.ts --script "Your script here"
 *   npx tsx scripts/kelly-iclone-lipsync.ts --phase hook --day 1
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

const CONFIG = {
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  SYNC_LABS_API_KEY: process.env.SYNC_LABS_API_KEY!,
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  
  // YOUR ACTUAL iClone render
  ICLONE_SOURCE_VIDEO: 'C:\\iLearnStudio\\projects\\Kelly\\Video\\kv1.mp4',
  
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'iclone-lipsync'),
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// Day 1 conversational scripts (already updated in database)
const DAY1_SCRIPTS = {
  Hook: "Hey! Ever notice how New Year's Day or even just Monday morning makes you feel like anything is possible? That's not just a feeling—it's science. Your brain literally resets at these moments. Today, let's explore how you can use this to your advantage.",
  Fact1: "Here's what's wild: researchers found that people who start a goal on a 'fresh start' day—like New Year's or a birthday—are way more likely to stick with it. It's called the Fresh Start Effect. Your brain treats these moments as a clean slate, like yesterday's failures don't count anymore.",
  Fact2: "But here's the thing—you don't have to wait for January 1st. You can create your own fresh starts. The first day of the month, a Monday, even just tomorrow morning. The key is making it feel significant to YOU.",
  Fact3: "And get this—when you start fresh, your brain's reward system lights up. You literally feel more optimistic. It's like your mind gives you permission to be a different person than you were yesterday.",
  Wisdom: "So here's what I want you to take away: you have the power to create a fresh start whenever you need one. Tomorrow could be day one. Next week could be your reset. The calendar doesn't decide—you do.",
};

async function generateAudio(script: string, outputPath: string): Promise<string> {
  console.log('\n🎤 Generating audio with ElevenLabs...');
  console.log(`   Script: "${script.substring(0, 50)}..."`);
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: script,
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
  
  if (!response.ok) {
    throw new Error(`ElevenLabs error: ${response.status}`);
  }
  
  const audioBuffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(outputPath, audioBuffer);
  console.log(`   ✅ Audio saved: ${outputPath} (${(audioBuffer.length / 1024).toFixed(1)} KB)`);
  
  return outputPath;
}

async function uploadToSupabase(filePath: string, bucket: string, remotePath: string): Promise<string> {
  console.log(`\n☁️ Uploading to Supabase...`);
  
  const fileBuffer = fs.readFileSync(filePath);
  const { error } = await supabase.storage
    .from(bucket)
    .upload(remotePath, fileBuffer, { upsert: true });
  
  if (error) throw error;
  
  const { data } = supabase.storage.from(bucket).getPublicUrl(remotePath);
  console.log(`   ✅ Uploaded: ${data.publicUrl}`);
  
  return data.publicUrl;
}

async function applySyncLabsLipSync(
  videoUrl: string, 
  audioUrl: string,
  outputPath: string
): Promise<string> {
  console.log('\n🎬 Applying Sync Labs lip-sync to iClone video...');
  console.log(`   Video: ${videoUrl.substring(0, 60)}...`);
  console.log(`   Audio: ${audioUrl.substring(0, 60)}...`);
  
  // Create job
  const createResponse = await fetch('https://api.sync.so/v2/generate', {
    method: 'POST',
    headers: {
      'x-api-key': CONFIG.SYNC_LABS_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'lipsync-2',
      input: [
        { type: 'video', url: videoUrl },
        { type: 'audio', url: audioUrl },
      ],
      options: {
        output_format: 'mp4',
      },
    }),
  });
  
  if (!createResponse.ok) {
    const err = await createResponse.text();
    throw new Error(`Sync Labs error: ${err}`);
  }
  
  const { id: jobId } = await createResponse.json();
  console.log(`   ✅ Job created: ${jobId}`);
  
  // Poll for completion
  let result: any = null;
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 5000));
    
    const statusResponse = await fetch(`https://api.sync.so/v2/generate/${jobId}`, {
      headers: { 'x-api-key': CONFIG.SYNC_LABS_API_KEY },
    });
    
    result = await statusResponse.json();
    process.stdout.write(result.status === 'COMPLETED' ? '✅' : '.');
    
    if (result.status === 'COMPLETED') {
      console.log(`\n   ✅ Sync Labs complete!`);
      break;
    }
    
    if (result.status === 'FAILED') {
      throw new Error(`Sync Labs failed: ${result.error || 'Unknown error'}`);
    }
  }
  
  if (!result?.outputUrl) {
    throw new Error('No output URL from Sync Labs');
  }
  
  // Download the result
  const videoResponse = await fetch(result.outputUrl);
  const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
  fs.writeFileSync(outputPath, videoBuffer);
  
  console.log(`   ✅ Video saved: ${outputPath}`);
  return outputPath;
}

async function generatePhaseVideo(phase: keyof typeof DAY1_SCRIPTS): Promise<string> {
  console.log('\n' + '═'.repeat(60));
  console.log(`🎬 GENERATING: Day 1 - ${phase}`);
  console.log('═'.repeat(60));
  
  const script = DAY1_SCRIPTS[phase];
  const timestamp = Date.now();
  
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  
  // Step 1: Generate audio
  const audioPath = path.join(CONFIG.OUTPUT_DIR, `day1_${phase.toLowerCase()}_${timestamp}.mp3`);
  await generateAudio(script, audioPath);
  
  // Step 2: Upload audio to Supabase
  const audioUrl = await uploadToSupabase(
    audioPath,
    'kelly-templates',
    `iclone-lipsync/day1_${phase.toLowerCase()}_${timestamp}.mp3`
  );
  
  // Step 3: Upload iClone source video to Supabase (if not already there)
  const icloneVideoUrl = await uploadToSupabase(
    CONFIG.ICLONE_SOURCE_VIDEO,
    'kelly-templates',
    'iclone-source/kv1.mp4'
  );
  
  // Step 4: Apply Sync Labs lip-sync
  const outputPath = path.join(CONFIG.OUTPUT_DIR, `day1_${phase.toLowerCase()}_final_${timestamp}.mp4`);
  await applySyncLabsLipSync(icloneVideoUrl, audioUrl, outputPath);
  
  console.log('\n' + '═'.repeat(60));
  console.log(`✅ COMPLETE: ${outputPath}`);
  console.log('═'.repeat(60));
  
  return outputPath;
}

async function main() {
  const args = process.argv.slice(2);
  
  // Parse arguments
  let phase: keyof typeof DAY1_SCRIPTS = 'Hook';
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--phase' && args[i + 1]) {
      phase = args[i + 1] as keyof typeof DAY1_SCRIPTS;
    }
  }
  
  console.log('╔' + '═'.repeat(60) + '╗');
  console.log('║  🎬 KELLY iCLONE + SYNC LABS PIPELINE                      ║');
  console.log('║  Using YOUR actual iClone render for natural results      ║');
  console.log('╚' + '═'.repeat(60) + '╝');
  
  // Verify iClone source exists
  if (!fs.existsSync(CONFIG.ICLONE_SOURCE_VIDEO)) {
    console.error(`❌ iClone source not found: ${CONFIG.ICLONE_SOURCE_VIDEO}`);
    console.error('   Please render Kelly in iClone first.');
    process.exit(1);
  }
  
  console.log(`✅ iClone source found: ${CONFIG.ICLONE_SOURCE_VIDEO}`);
  
  await generatePhaseVideo(phase);
}

main().catch(console.error);

