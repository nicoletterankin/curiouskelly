#!/usr/bin/env npx tsx
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const supabase = createClient(process.env.PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!);

// Try with the second talking photo ID
const TALKING_PHOTO_ID = 'd8ba9d6f0a994046b4d9fbe2d6428a95'; 

async function getAudioUrl() {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0'}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': process.env.ELEVENLABS_API_KEY!,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: "Hey! I'm Kelly testing video generation.",
        model_id: 'eleven_multilingual_v2',
      }),
    }
  );
  const buffer = Buffer.from(await response.arrayBuffer());
  const name = `test_${Date.now()}.mp3`;
  await supabase.storage.from('kelly-templates').upload(`heygen/audio/${name}`, buffer, { contentType: 'audio/mpeg', upsert: true });
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(`heygen/audio/${name}`);
  return data.publicUrl;
}

async function test() {
  console.log('🎤 Getting audio...');
  const audioUrl = await getAudioUrl();
  console.log('Audio:', audioUrl);

  // Try v1/video.generate format
  console.log('\n--- Testing v1/video.generate ---');
  const v1Response = await fetch('https://api.heygen.com/v1/video.generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      background: '#ffffff',
      clips: [{
        avatar_id: TALKING_PHOTO_ID,
        input_audio: audioUrl,
      }],
      ratio: '16:9',
      test: true, // test mode - faster, lower quality
    }),
  });
  console.log('v1 status:', v1Response.status);
  console.log('v1 response:', await v1Response.text());

  // Try v2 with the second photo
  console.log('\n--- Testing v2 with different photo ---');
  const v2Response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: TALKING_PHOTO_ID,
        },
        voice: {
          type: 'audio',
          audio_url: audioUrl,
        },
      }],
      test: true,
    }),
  });
  console.log('v2 status:', v2Response.status);
  const v2Result = await v2Response.json();
  console.log('v2 response:', JSON.stringify(v2Result, null, 2));

  if (v2Result.data?.video_id) {
    console.log('\n✅ Video started! ID:', v2Result.data.video_id);
    console.log('Run: npx tsx scripts/heygen-check-video.ts', v2Result.data.video_id);
  }
}

test().catch(console.error);

