#!/usr/bin/env npx tsx
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const supabase = createClient(process.env.PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!);

// Avatar IDs
const STRATEGIST = '0cbc4757846646399e020d418e0dff56';

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
        text: "Hey! Testing Kelly video.",
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
  const audioUrl = await getAudioUrl();
  console.log('Audio:', audioUrl);

  // Try the exact format HeyGen wants for talking photos
  const payload = {
    video_inputs: [{
      character: {
        type: 'talking_photo',
        talking_photo_id: STRATEGIST,
      },
      voice: {
        type: 'audio',
        audio_url: audioUrl,
      },
    }],
    dimension: { width: 1920, height: 1080 },
  };

  console.log('\nPayload:', JSON.stringify(payload, null, 2));

  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const text = await response.text();
  console.log('\nResponse status:', response.status);
  console.log('Response:', text);

  // Also try listing talking photos to see the structure
  console.log('\n--- Listing talking photos ---');
  const listResp = await fetch('https://api.heygen.com/v1/talking_photo.list', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY },
  });
  const listData = await listResp.json();
  console.log('First 3 photos:', JSON.stringify(listData.data?.slice(0, 3), null, 2));
}

test().catch(console.error);

