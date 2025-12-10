#!/usr/bin/env npx tsx
/**
 * Test different HeyGen avatar API approaches
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY!;
const ELEVENLABS_KELLY_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Avatar IDs from user
const STRATEGIST_AVATAR = '0cbc4757846646399e020d418e0dff56';
const KELLY_GROUP = 'a762125d3107477aba43d1bd79f90d6e';

async function generateAudioUrl(text: string): Promise<string> {
  console.log('🎤 Generating audio...');
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_KELLY_VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    }
  );

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const fileName = `test_${Date.now()}.mp3`;
  
  await supabase.storage.from('kelly-templates').upload(
    `heygen/audio/${fileName}`,
    audioBuffer,
    { contentType: 'audio/mpeg', upsert: true }
  );
  
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(`heygen/audio/${fileName}`);
  console.log(`✅ Audio URL: ${data.publicUrl}`);
  return data.publicUrl;
}

async function tryV1Generate(avatarId: string, audioUrl: string) {
  console.log('\n📹 Trying v1/video.generate...');
  
  const response = await fetch('https://api.heygen.com/v1/video.generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      avatar_id: avatarId,
      audio_url: audioUrl,
      dimension: { width: 1920, height: 1080 },
    }),
  });

  const result = await response.json();
  console.log('v1 result:', JSON.stringify(result, null, 2));
  return result;
}

async function tryV2Generate(avatarId: string, audioUrl: string) {
  console.log('\n📹 Trying v2/video/generate with avatar_id...');
  
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'avatar',
          avatar_id: avatarId,
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
  console.log('v2 avatar result:', JSON.stringify(result, null, 2));
  return result;
}

async function tryV2TalkingPhoto(talkingPhotoId: string, audioUrl: string) {
  console.log('\n📹 Trying v2/video/generate with talking_photo...');
  
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
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
  console.log('v2 talking_photo result:', JSON.stringify(result, null, 2));
  return result;
}

async function tryV2WithStyle(avatarId: string, audioUrl: string) {
  console.log('\n📹 Trying v2/video/generate with avatar_style...');
  
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
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
  console.log('v2 avatar_style result:', JSON.stringify(result, null, 2));
  return result;
}

async function getAvatarDetails(avatarId: string) {
  console.log(`\n🔍 Getting avatar details for ${avatarId}...`);
  
  const response = await fetch(`https://api.heygen.com/v2/avatars/${avatarId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY },
  });
  
  const result = await response.json();
  console.log('Avatar details:', JSON.stringify(result, null, 2));
  return result;
}

async function listAvatarLooks(groupId: string) {
  console.log(`\n🔍 Listing looks for group ${groupId}...`);
  
  const response = await fetch(`https://api.heygen.com/v2/avatars?avatar_group_id=${groupId}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY },
  });
  
  const result = await response.json();
  console.log('Avatar looks:', JSON.stringify(result, null, 2));
  return result;
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🔬 HEYGEN API TESTING                                     ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // Get avatar details first
  await getAvatarDetails(STRATEGIST_AVATAR);
  await listAvatarLooks(KELLY_GROUP);

  // Generate audio
  const audioUrl = await generateAudioUrl("Hey! I'm Kelly and I'm testing this video generation.");

  // Try different approaches
  await tryV1Generate(STRATEGIST_AVATAR, audioUrl);
  await tryV2Generate(STRATEGIST_AVATAR, audioUrl);
  await tryV2TalkingPhoto(STRATEGIST_AVATAR, audioUrl);
  await tryV2WithStyle(STRATEGIST_AVATAR, audioUrl);
}

main().catch(console.error);

