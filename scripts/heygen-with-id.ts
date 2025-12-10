#!/usr/bin/env npx tsx
/**
 * 🎬 HEYGEN KELLY PIPELINE
 * 
 * Usage: npx tsx scripts/heygen-with-id.ts --id=YOUR_TALKING_PHOTO_ID --phase=Hook
 * 
 * To get the talking_photo_id:
 * 1. Go to app.heygen.com
 * 2. Create a Photo Avatar with Kelly's image
 * 3. The ID will appear in the URL or API response
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY!,
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  KELLY_VOICE_ID: 'wAdymQH5YucAkXwmrdL0',
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'heygen'),
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

const DAY1_SCRIPTS: Record<string, string> = {
  Hook: "Hey! Ever notice how New Year's Day or even just Monday morning makes you feel like anything is possible? That's not just a feeling—it's science. Your brain literally resets at these moments. Today, let's explore how you can use this to your advantage.",
  Fact1: "Here's what's wild: researchers found that people who start a goal on a 'fresh start' day—like New Year's or a birthday—are way more likely to stick with it. It's called the Fresh Start Effect. Your brain treats these moments as a clean slate, like yesterday's failures don't count anymore.",
  Fact2: "But here's the thing—you don't have to wait for January 1st. You can create your own fresh starts. The first day of the month, a Monday, even just tomorrow morning. The key is making it feel significant to YOU.",
  Fact3: "And get this—when you start fresh, your brain's reward system lights up. You literally feel more optimistic. It's like your mind gives you permission to be a different person than you were yesterday.",
  Wisdom: "So here's what I want you to take away: you have the power to create a fresh start whenever you need one. Tomorrow could be day one. Next week could be your reset. The calendar doesn't decide—you do.",
};

async function generateAudio(script: string, phase: string): Promise<string> {
  console.log(`   🎤 Generating audio...`);
  
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
        voice_settings: { stability: 0.5, similarity_boost: 0.85, style: 0.3, use_speaker_boost: true },
      }),
    }
  );
  
  if (!response.ok) throw new Error(`ElevenLabs: ${response.status}`);
  
  const audioPath = path.join(CONFIG.OUTPUT_DIR, `day1_${phase.toLowerCase()}.mp3`);
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(audioPath, Buffer.from(await response.arrayBuffer()));
  
  // Upload to Supabase
  const remotePath = `heygen/audio_${phase.toLowerCase()}_${Date.now()}.mp3`;
  await supabase.storage.from('kelly-templates').upload(remotePath, fs.readFileSync(audioPath), { upsert: true });
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
  console.log(`   ✅ Audio: ${data.publicUrl}`);
  return data.publicUrl;
}

async function createVideo(talkingPhotoId: string, audioUrl: string): Promise<string> {
  console.log(`   🎬 Creating HeyGen video...`);
  
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
          talking_photo_id: talkingPhotoId,
        },
        voice: {
          type: 'audio',
          audio_url: audioUrl,
        },
      }],
      dimension: { width: 1920, height: 1080 },
    }),
  });
  
  const result = await response.json();
  if (!response.ok) throw new Error(`HeyGen: ${JSON.stringify(result)}`);
  
  console.log(`   ✅ Job started: ${result.data.video_id}`);
  return result.data.video_id;
}

async function waitForVideo(videoId: string): Promise<string> {
  console.log(`   ⏳ Waiting...`);
  
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 10000));
    
    const response = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY },
    });
    const result = await response.json();
    
    process.stdout.write(result.data?.status === 'completed' ? '✅' : '.');
    
    if (result.data?.status === 'completed') {
      console.log(`\n   ✅ Done: ${result.data.video_url}`);
      return result.data.video_url;
    }
    if (result.data?.status === 'failed') throw new Error(result.data.error);
  }
  throw new Error('Timeout');
}

async function main() {
  const args = process.argv.slice(2);
  const idArg = args.find(a => a.startsWith('--id='))?.split('=')[1];
  const phaseArg = args.find(a => a.startsWith('--phase='))?.split('=')[1] || 'Hook';
  
  if (!idArg) {
    console.log('Usage: npx tsx scripts/heygen-with-id.ts --id=YOUR_TALKING_PHOTO_ID --phase=Hook');
    console.log('\nTo get the ID:');
    console.log('1. Go to app.heygen.com');
    console.log('2. Create a Photo Avatar with Kelly image');
    console.log('3. Copy the talking_photo_id');
    process.exit(1);
  }
  
  console.log(`\n🎬 Generating Day 1 ${phaseArg} with HeyGen...`);
  
  const script = DAY1_SCRIPTS[phaseArg];
  if (!script) throw new Error(`Unknown phase: ${phaseArg}`);
  
  const audioUrl = await generateAudio(script, phaseArg);
  const videoId = await createVideo(idArg, audioUrl);
  const videoUrl = await waitForVideo(videoId);
  
  // Download and upload to our Supabase
  console.log('   💾 Saving to Supabase...');
  const videoResponse = await fetch(videoUrl);
  const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
  
  const remotePath = `production/videos/heygen/day_001_${phaseArg}_heygen.mp4`;
  await supabase.storage.from('kelly-videos').upload(remotePath, videoBuffer, { upsert: true });
  const { data } = supabase.storage.from('kelly-videos').getPublicUrl(remotePath);
  
  // Update database
  const { data: lesson } = await supabase.from('core_lessons').select('id').eq('day_number', 1).single();
  if (lesson) {
    await supabase.from('lesson_atoms')
      .update({ hd_video_url: data.publicUrl })
      .eq('core_lesson_id', lesson.id)
      .eq('phase', phaseArg)
      .eq('archetype', 'The Scientist');
  }
  
  console.log('\n✅ COMPLETE!');
  console.log(`   Video: ${data.publicUrl}`);
  console.log('   Test: http://localhost:3000/learn?day=1&clearcache=1');
}

main().catch(e => {
  console.error('❌ Error:', e.message);
  process.exit(1);
});

