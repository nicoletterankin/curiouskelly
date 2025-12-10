#!/usr/bin/env npx tsx
/**
 * 🎬 HEYGEN VIDEO GENERATION - Using Existing Talking Photo
 * 
 * Uses the talking_photo_id found in the account: 759063ab989242f8910f8013747c8f40
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
  // HeyGen's stock Kelly avatar (Blue Shirt, Front view)
  KELLY_AVATAR_ID: 'Kelly_Blue_Shirt_Front',
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'heygen'),
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// Day 1 Hook script (conversational version)
const HOOK_SCRIPT = "Hey! Ever notice how New Year's Day or even just Monday morning makes you feel like anything is possible? That's not just a feeling—it's science. Your brain literally resets at these moments. Today, let's explore how you can use this to your advantage.";

async function generateAudio(): Promise<string> {
  console.log('🎤 Generating audio with ElevenLabs...');
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: HOOK_SCRIPT,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { 
          stability: 0.5, 
          similarity_boost: 0.85, 
          style: 0.3, 
          use_speaker_boost: true 
        },
      }),
    }
  );
  
  if (!response.ok) {
    throw new Error(`ElevenLabs error: ${response.status}`);
  }
  
  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const audioPath = path.join(CONFIG.OUTPUT_DIR, `day1_hook_audio.mp3`);
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(audioPath, audioBuffer);
  console.log('   ✅ Audio saved locally');
  
  // Upload to Supabase
  const remotePath = `heygen/day1_hook_audio_${Date.now()}.mp3`;
  await supabase.storage.from('kelly-templates').upload(remotePath, audioBuffer, { 
    upsert: true,
    contentType: 'audio/mpeg'
  });
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
  console.log('   ✅ Uploaded to Supabase:', data.publicUrl);
  return data.publicUrl;
}

async function createHeyGenVideo(audioUrl: string): Promise<string> {
  console.log('🎬 Creating HeyGen video...');
  console.log(`   Using Kelly Avatar ID: ${CONFIG.KELLY_AVATAR_ID}`);
  console.log(`   Using audio: ${audioUrl}`);
  
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': CONFIG.HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'avatar',
          avatar_id: CONFIG.KELLY_AVATAR_ID,
          avatar_style: 'normal',
        },
        voice: {
          type: 'audio',
          audio_url: audioUrl,
        },
      }],
      dimension: { 
        width: 1920, 
        height: 1080 
      },
    }),
  });
  
  const result = await response.json();
  console.log('   Response:', JSON.stringify(result, null, 2));
  
  if (result.error) {
    throw new Error(`HeyGen error: ${result.error.message}`);
  }
  
  return result.data.video_id;
}

async function waitForVideo(videoId: string): Promise<string> {
  console.log(`⏳ Waiting for video ${videoId}...`);
  
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 10000)); // 10 second intervals
    
    const response = await fetch(
      `https://api.heygen.com/v1/video_status.get?video_id=${videoId}`,
      { headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY } }
    );
    const result = await response.json();
    
    const status = result.data?.status;
    process.stdout.write(status === 'completed' ? '\n' : '.');
    
    if (status === 'completed') {
      console.log('✅ Video complete!');
      return result.data.video_url;
    }
    if (status === 'failed') {
      throw new Error(`Video failed: ${result.data.error}`);
    }
  }
  throw new Error('Timeout waiting for video');
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN VIDEO GENERATION                                ║');
  console.log('║  Using existing talking photo from your account            ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  
  try {
    // 1. Generate audio
    const audioUrl = await generateAudio();
    
    // 2. Create video
    const videoId = await createHeyGenVideo(audioUrl);
    
    // 3. Wait for completion
    const videoUrl = await waitForVideo(videoId);
    
    // 4. Download and save
    console.log('💾 Downloading video...');
    const videoResponse = await fetch(videoUrl);
    const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
    const videoPath = path.join(CONFIG.OUTPUT_DIR, `day1_hook_heygen.mp4`);
    fs.writeFileSync(videoPath, videoBuffer);
    console.log(`   Saved: ${videoPath}`);
    
    // 5. Upload to Supabase
    console.log('☁️ Uploading to Supabase...');
    const remotePath = 'production/videos/heygen/day_001_Hook_heygen.mp4';
    await supabase.storage.from('kelly-videos').upload(remotePath, videoBuffer, { upsert: true });
    const { data } = supabase.storage.from('kelly-videos').getPublicUrl(remotePath);
    
    // 6. Update database
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('id')
      .eq('day_number', 1)
      .single();
    
    if (lesson) {
      await supabase
        .from('lesson_atoms')
        .update({ hd_video_url: data.publicUrl })
        .eq('core_lesson_id', lesson.id)
        .eq('phase', 'Hook')
        .eq('archetype', 'The Scientist');
      console.log('   ✅ Database updated!');
    }
    
    console.log('\n' + '═'.repeat(60));
    console.log('🎉 SUCCESS!');
    console.log('═'.repeat(60));
    console.log(`Video URL: ${data.publicUrl}`);
    console.log('\n🔗 Test now: http://localhost:3000/learn?day=1&clearcache=1');
    
  } catch (error: any) {
    console.error('\n❌ ERROR:', error.message);
    process.exit(1);
  }
}

main();

