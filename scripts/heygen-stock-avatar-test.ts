#!/usr/bin/env npx tsx
/**
 * 🎬 HEYGEN VIDEO - Using Stock Avatar (to prove pipeline works)
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

const HOOK_SCRIPT = "Hey! Ever notice how New Year's Day or even just Monday morning makes you feel like anything is possible? That's not just a feeling—it's science. Your brain literally resets at these moments. Today, let's explore how you can use this to your advantage.";

async function findSuitableAvatar(): Promise<{ avatar_id: string; avatar_name: string }> {
  console.log('🔍 Finding a suitable female avatar...');
  
  const response = await fetch('https://api.heygen.com/v2/avatars', {
    headers: { 'X-Api-Key': CONFIG.HEYGEN_API_KEY }
  });
  const result = await response.json();
  
  if (!result.data?.avatars) {
    throw new Error('No avatars found');
  }
  
  // Find a female avatar, preferring ones with "teacher" or similar names
  const femaleAvatars = result.data.avatars.filter((a: any) => 
    a.gender === 'female' || 
    a.avatar_name?.toLowerCase().includes('female') ||
    a.avatar_name?.toLowerCase().includes('woman')
  );
  
  console.log(`   Found ${femaleAvatars.length} potential female avatars`);
  
  // Show first 5 options
  console.log('\n   Sample avatars:');
  femaleAvatars.slice(0, 5).forEach((a: any) => {
    console.log(`      ${a.avatar_id}: ${a.avatar_name} (type: ${a.avatar_type})`);
  });
  
  // Use the first suitable one
  const avatar = femaleAvatars[0];
  if (!avatar) {
    throw new Error('No suitable avatar found');
  }
  
  console.log(`\n   Using: ${avatar.avatar_name} (${avatar.avatar_id})`);
  return { avatar_id: avatar.avatar_id, avatar_name: avatar.avatar_name };
}

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
        text: HOOK_SCRIPT,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.85, style: 0.3, use_speaker_boost: true },
      }),
    }
  );
  
  if (!response.ok) throw new Error(`ElevenLabs error: ${response.status}`);
  
  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const audioPath = path.join(CONFIG.OUTPUT_DIR, `day1_hook_audio.mp3`);
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(audioPath, audioBuffer);
  
  // Upload to Supabase
  const remotePath = `heygen/day1_hook_audio_${Date.now()}.mp3`;
  await supabase.storage.from('kelly-templates').upload(remotePath, audioBuffer, { upsert: true, contentType: 'audio/mpeg' });
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
  console.log('   ✅ Audio ready:', data.publicUrl);
  return data.publicUrl;
}

async function createVideo(avatarId: string, audioUrl: string): Promise<string> {
  console.log('\n🎬 Creating HeyGen video...');
  
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
          avatar_id: avatarId,
          avatar_style: 'normal',
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
  console.log('   Response:', JSON.stringify(result, null, 2));
  
  if (result.error) throw new Error(`HeyGen error: ${result.error.message}`);
  
  return result.data.video_id;
}

async function waitForVideo(videoId: string): Promise<string> {
  console.log(`\n⏳ Waiting for video ${videoId}...`);
  
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 10000));
    
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
    if (status === 'failed') throw new Error(`Video failed: ${result.data.error}`);
  }
  throw new Error('Timeout');
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN TEST - Stock Avatar with Kelly Voice            ║');
  console.log('║  Testing if the pipeline works before custom avatar        ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  
  try {
    const avatar = await findSuitableAvatar();
    const audioUrl = await generateAudio();
    const videoId = await createVideo(avatar.avatar_id, audioUrl);
    const videoUrl = await waitForVideo(videoId);
    
    console.log('\n💾 Downloading video...');
    const videoResponse = await fetch(videoUrl);
    const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
    const videoPath = path.join(CONFIG.OUTPUT_DIR, `stock_avatar_test.mp4`);
    fs.writeFileSync(videoPath, videoBuffer);
    
    console.log('\n' + '═'.repeat(60));
    console.log('✅ SUCCESS! Stock avatar video generated.');
    console.log(`   Local: ${videoPath}`);
    console.log('═'.repeat(60));
    console.log('\nThis proves the HeyGen pipeline works!');
    console.log('Next step: Create Kelly as a Photo Avatar in HeyGen web UI.');
    
  } catch (error: any) {
    console.error('\n❌ ERROR:', error.message);
    process.exit(1);
  }
}

main();

