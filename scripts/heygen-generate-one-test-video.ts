#!/usr/bin/env npx tsx
/**
 * Generate EXACTLY ONE HeyGen test video (no Supabase writes).
 *
 * - Picks adult-scientist (scientist_adult) from DEFAULT_AVATAR_IDS in avatar-registry.html
 *   or falls back to the first available ID in that registry.
 * - Uses HeyGen TTS (voice.type='text') with a short script.
 * - Calls v2/video/generate then polls until completed.
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

if (!HEYGEN_API_KEY) {
  console.error('❌ Missing HEYGEN_API_KEY in environment');
  process.exit(1);
}

const TEST_SCRIPT = "Hello! I'm Kelly, your learning companion. Welcome to Curious Kelly where every day is a chance to discover something amazing.";

type PickedAvatar = { key: string; avatar_id: string };

function readAvatarRegistryHtml(): string {
  const p = path.join(process.cwd(), 'public', 'admin', 'avatar-registry.html');
  return fs.readFileSync(p, 'utf8');
}

function pickTestAvatarFromRegistry(html: string): PickedAvatar {
  // Prefer adult-scientist (scientist_adult)
  const preferred = /\bscientist_adult\s*:\s*'([^']+)'/i.exec(html);
  if (preferred?.[1] && preferred[1].length >= 10) {
    return { key: 'scientist_adult', avatar_id: preferred[1] };
  }

  // Otherwise pick the first available in DEFAULT_AVATAR_IDS block
  const block = /const\s+DEFAULT_AVATAR_IDS\s*=\s*\{([\s\S]*?)\}\s*;/m.exec(html);
  if (block?.[1]) {
    const first = /\b([a-z_]+)\s*:\s*'([^']+)'/i.exec(block[1]);
    if (first?.[1] && first?.[2] && first[2].length >= 10) {
      return { key: first[1], avatar_id: first[2] };
    }
  }

  throw new Error(
    `Could not find any DEFAULT_AVATAR_IDS in public/admin/avatar-registry.html (contains DEFAULT_AVATAR_IDS: ${html.includes('DEFAULT_AVATAR_IDS')})`
  );
}

async function heygenGenerateOne(params: { avatarId: string; script: string }): Promise<string> {
  // HeyGen requires a voice_id for text-to-speech. Fetch and pick one.
  const voice = await (async () => {
    const r = await fetch('https://api.heygen.com/v2/voices', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY!, Accept: 'application/json' },
    });
    const txt = await r.text();
    let j: any;
    try {
      j = JSON.parse(txt);
    } catch {
      throw new Error(`HeyGen v2/voices returned non-JSON (status ${r.status}): ${txt.slice(0, 200)}`);
    }
    if (!r.ok) {
      throw new Error(`HeyGen v2/voices failed (status ${r.status}): ${JSON.stringify(j).slice(0, 500)}`);
    }

    const voices: any[] = j?.data?.voices || [];
    if (!Array.isArray(voices) || voices.length === 0) {
      throw new Error(`HeyGen v2/voices returned no voices: ${JSON.stringify(j).slice(0, 500)}`);
    }

    // Prefer an English female voice, else just take first.
    const preferred =
      voices.find(v => String(v?.language).toLowerCase().includes('english') && String(v?.gender).toLowerCase() === 'female') ||
      voices[0];

    return {
      voice_id: preferred.voice_id as string,
      name: preferred.name as string | undefined,
      language: preferred.language as string | undefined,
      gender: preferred.gender as string | undefined,
    };
  })();

  console.log(`✅ Using HeyGen voice_id: ${voice.voice_id}${voice.name ? ` (${voice.name})` : ''}`);

  const payload = {
    video_inputs: [
      {
        character: {
          type: 'talking_photo',
          talking_photo_id: params.avatarId,
        },
        voice: {
          type: 'text',
          // HeyGen's API has varied schema expectations across accounts/versions.
          // Provide both the "flattened" and "nested text" forms to satisfy validation.
          voice_id: voice.voice_id,
          input_text: params.script,
          text: {
            voice_id: voice.voice_id,
            input_text: params.script,
          },
        },
      },
    ],
    dimension: { width: 1920, height: 1080 },
    test: false,
  };

  const r = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY!,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const txt = await r.text();
  let j: any;
  try {
    j = JSON.parse(txt);
  } catch {
    throw new Error(`HeyGen returned non-JSON (status ${r.status}): ${txt.slice(0, 200)}`);
  }

  if (!r.ok) {
    throw new Error(`HeyGen generate failed (status ${r.status}): ${JSON.stringify(j)}`);
  }

  const videoId = j?.data?.video_id;
  if (!videoId) {
    throw new Error(`HeyGen generate response missing data.video_id: ${JSON.stringify(j)}`);
  }

  return videoId;
}

async function heygenStatus(videoId: string): Promise<{ status: string; video_url?: string; error?: any }> {
  const r = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY! },
  });

  const txt = await r.text();
  const j: any = JSON.parse(txt);

  return {
    status: j?.data?.status,
    video_url: j?.data?.video_url,
    error: j?.data?.error,
  };
}

async function main() {
  const startedAt = Date.now();

  // 1) Pick test avatar
  const html = readAvatarRegistryHtml();
  const picked = pickTestAvatarFromRegistry(html);
  console.log(`✅ Picked test avatar key: ${picked.key}`);
  console.log(`✅ Using avatar_id: ${picked.avatar_id}`);

  // 2) Pick test script
  console.log(`\n✅ Test script (${TEST_SCRIPT.length} chars):`);
  console.log(TEST_SCRIPT);

  // 3) Generate video
  console.log('\n🎬 Calling HeyGen v2/video/generate...');
  const videoId = await heygenGenerateOne({ avatarId: picked.avatar_id, script: TEST_SCRIPT });
  console.log(`✅ video_id: ${videoId}`);

  // Poll until completed (log each status check)
  console.log('\n⏳ Polling status until completed...');
  const pollStartedAt = Date.now();

  const maxMs = 15 * 60 * 1000; // 15 minutes
  const intervalMs = 10 * 1000;
  let attempts = 0;

  while (true) {
    attempts += 1;
    const now = Date.now();
    const elapsed = now - pollStartedAt;

    const st = await heygenStatus(videoId);
    console.log(`[${attempts}] +${Math.round(elapsed / 1000)}s status=${st.status}`);

    if (st.status === 'completed' && st.video_url) {
      const totalMs = Date.now() - startedAt;
      console.log('\n✅ COMPLETED');
      console.log(`video_url: ${st.video_url}`);
      console.log(`time_taken_seconds: ${Math.round(totalMs / 1000)}`);
      return;
    }

    if (st.status === 'failed') {
      const totalMs = Date.now() - startedAt;
      console.log('\n❌ FAILED');
      console.log(`time_taken_seconds: ${Math.round(totalMs / 1000)}`);
      console.log('error:', typeof st.error === 'string' ? st.error : JSON.stringify(st.error));
      process.exit(1);
    }

    if (Date.now() - pollStartedAt > maxMs) {
      const totalMs = Date.now() - startedAt;
      console.log('\n❌ TIMEOUT');
      console.log(`time_taken_seconds: ${Math.round(totalMs / 1000)}`);
      process.exit(1);
    }

    await new Promise(r => setTimeout(r, intervalMs));
  }
}

main().catch(err => {
  console.error('❌ ERROR:', err?.message || String(err));
  process.exit(1);
});


















