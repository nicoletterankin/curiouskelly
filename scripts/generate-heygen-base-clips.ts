#!/usr/bin/env npx tsx
/**
 * Generate one short "base clip" per avatar (12 personas × 5 age buckets).
 *
 * Reads avatar IDs from `public/admin/avatar-registry.html` (DEFAULT_AVATAR_IDS).
 * Generates a single shared audio file (ElevenLabs) and reuses it for every avatar
 * so motion differences come from the avatar/persona, not the audio.
 *
 * Default output:
 * - local/heygen-base-clips/<key>.mp4   (key = scientist_adult, rebel_kid, ...)
 *
 * Usage:
 *   npx tsx scripts/generate-heygen-base-clips.ts --dry-run
 *   npx tsx scripts/generate-heygen-base-clips.ts
 *   npx tsx scripts/generate-heygen-base-clips.ts --only kid
 *   npx tsx scripts/generate-heygen-base-clips.ts --only adult --include scientist_adult
 *
 * Required env:
 * - HEYGEN_API_KEY
 * - ELEVENLABS_API_KEY
 * - ELEVENLABS_KELLY_VOICE_ID (optional; repo default used if absent)
 * - PUBLIC_SUPABASE_URL (or SUPABASE_URL) + SUPABASE_SERVICE_ROLE_KEY (audio upload)
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const DEFAULT_ELEVEN_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

type Options = {
  dryRun: boolean;
  includeScientistAdult: boolean;
  only: string | null; // 'kid'|'teen'|'adult'|'elder'|'super_elder' OR persona key substring
  outDir: string;
  concurrency: number;
};

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    dryRun: false,
    includeScientistAdult: false,
    only: null,
    outDir: path.join(process.cwd(), 'local', 'heygen-base-clips'),
    concurrency: 2,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--dry-run') opts.dryRun = true;
    else if (a === '--include' && argv[i + 1] === 'scientist_adult') {
      opts.includeScientistAdult = true;
      i++;
    } else if (a === '--include-scientist-adult') opts.includeScientistAdult = true;
    else if (a === '--only') opts.only = argv[++i] || null;
    else if (a === '--out-dir') opts.outDir = argv[++i] || opts.outDir;
    else if (a === '--concurrency') opts.concurrency = Math.max(1, parseInt(argv[++i] || '2', 10) || 2);
    else if (a === '--help' || a === '-h') {
      console.log(`
Generate one base clip per HeyGen avatar.

Usage:
  npx tsx scripts/generate-heygen-base-clips.ts [options]

Options:
  --dry-run                     Print what would be generated
  --only <kid|teen|adult|elder|super_elder|substring>
                                Filter which keys to generate
  --include-scientist-adult     Also generate scientist_adult (default: skipped)
  --out-dir <dir>               Output directory (default: ./local/heygen-base-clips)
  --concurrency <n>             Concurrent HeyGen jobs (default: 2)
`);
      process.exit(0);
    }
  }
  return opts;
}

function readAvatarRegistryHtml(): string {
  const p = path.join(process.cwd(), 'public', 'admin', 'avatar-registry.html');
  if (!fs.existsSync(p)) throw new Error(`Missing file: ${p}`);
  return fs.readFileSync(p, 'utf-8');
}

function parseDefaultAvatarIdsFromHtml(html: string): Record<string, string> {
  const m = html.match(/const\\s+DEFAULT_AVATAR_IDS\\s*=\\s*\\{([\\s\\S]*?)\\};/);
  if (!m) throw new Error('Could not find DEFAULT_AVATAR_IDS in avatar-registry.html');
  const body = m[1];

  // Matches: key: 'value' OR key: "value"
  const re = /([a-zA-Z0-9_]+)\\s*:\\s*['"]([a-f0-9]{32})['"]\\s*,?/g;
  const out: Record<string, string> = {};
  for (const match of body.matchAll(re)) {
    out[match[1]] = match[2];
  }

  if (Object.keys(out).length < 10) {
    throw new Error(`Parsed too few avatar IDs (${Object.keys(out).length}). Parser likely failed.`);
  }
  return out;
}

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

async function runConcurrent<T>(items: T[], fn: (item: T, idx: number) => Promise<void>, limit: number) {
  const executing = new Set<Promise<void>>();
  let i = 0;
  for (const item of items) {
    const idx = i++;
    const p = Promise.resolve().then(() => fn(item, idx));
    executing.add(p);
    const clean = () => executing.delete(p);
    p.then(clean).catch(clean);
    if (executing.size >= limit) await Promise.race(executing);
  }
  await Promise.all(executing);
}

async function elevenlabsTtsMp3(text: string): Promise<Buffer> {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) throw new Error('Missing ELEVENLABS_API_KEY');

  const res = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${DEFAULT_ELEVEN_VOICE_ID}?output_format=mp3_44100_192`,
    {
      method: 'POST',
      headers: { 'xi-api-key': apiKey, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    }
  );
  if (!res.ok) throw new Error(`ElevenLabs error: ${res.status} ${await res.text()}`);
  return Buffer.from(await res.arrayBuffer());
}

async function uploadAudioToSupabase(mp3: Buffer): Promise<string> {
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!supabaseUrl || !serviceKey) throw new Error('Missing PUBLIC_SUPABASE_URL/SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');

  const supabase = createClient(supabaseUrl, serviceKey);
  const fileName = `heygen/base-clip-audio/base_clip_${Date.now()}.mp3`;

  const { error } = await supabase.storage.from('kelly-templates').upload(fileName, mp3, {
    contentType: 'audio/mpeg',
    upsert: true,
  });
  if (error) throw error;

  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(fileName);
  return data.publicUrl;
}

async function heygenGenerateVideo(talkingPhotoId: string, audioUrl: string): Promise<string> {
  const apiKey = process.env.HEYGEN_API_KEY;
  if (!apiKey) throw new Error('Missing HEYGEN_API_KEY');

  const res = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: { 'X-Api-Key': apiKey, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_inputs: [
        {
          character: { type: 'talking_photo', talking_photo_id: talkingPhotoId },
          voice: { type: 'audio', audio_url: audioUrl },
          background: { type: 'color', value: '#FFFFFF' },
        },
      ],
      dimension: { width: 1920, height: 1080 },
    }),
  });

  if (!res.ok) throw new Error(`HeyGen generate error: ${res.status} ${await res.text()}`);
  const j: any = await res.json();
  const id = j?.data?.video_id;
  if (!id) throw new Error(`HeyGen response missing video_id: ${JSON.stringify(j).slice(0, 500)}`);
  return id;
}

async function heygenWaitForVideoUrl(videoId: string): Promise<string> {
  const apiKey = process.env.HEYGEN_API_KEY;
  if (!apiKey) throw new Error('Missing HEYGEN_API_KEY');

  while (true) {
    const res = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
      headers: { 'X-Api-Key': apiKey },
    });
    const j: any = await res.json();
    const status = j?.data?.status;
    if (status === 'completed') return j.data.video_url;
    if (status === 'failed') throw new Error(`HeyGen failed: ${j?.data?.error || 'unknown error'}`);
    await sleep(10_000);
  }
}

async function downloadToFile(url: string, filePath: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, buf);
}

function filterKeys(all: Record<string, string>, opts: Options): [string, string][] {
  let entries = Object.entries(all);

  if (!opts.includeScientistAdult) {
    entries = entries.filter(([k]) => k !== 'scientist_adult');
  }

  if (opts.only) {
    const needle = opts.only.toLowerCase();
    entries = entries.filter(([k]) => k.toLowerCase().includes(needle));
  }

  // Stable sort by key for reproducibility
  entries.sort((a, b) => a[0].localeCompare(b[0]));
  return entries;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  const html = readAvatarRegistryHtml();
  const ids = parseDefaultAvatarIdsFromHtml(html);
  const targets = filterKeys(ids, opts);

  console.log('Base clip generation');
  console.log('Targets:', targets.length);
  console.log('Output:', opts.outDir);
  console.log('Concurrency:', opts.concurrency);
  if (opts.dryRun) {
    for (const [k, v] of targets.slice(0, 25)) console.log('-', k, v);
    if (targets.length > 25) console.log('... and', targets.length - 25, 'more');
    return;
  }

  const scriptText =
    "Hello! I'm Kelly. Welcome to learning. I'm excited to explore something new with you today. Let's begin.";

  console.log('Generating shared audio (ElevenLabs) ...');
  const mp3 = await elevenlabsTtsMp3(scriptText);
  console.log('Uploading shared audio to Supabase ...');
  const audioUrl = await uploadAudioToSupabase(mp3);
  console.log('Audio URL ready.');

  let ok = 0;
  let failed = 0;

  await runConcurrent(
    targets,
    async ([key, avatarId], idx) => {
      const outPath = path.join(opts.outDir, `${key}.mp4`);
      if (fs.existsSync(outPath)) {
        console.log(`[SKIP] ${key} (exists)`);
        ok++;
        return;
      }

      try {
        console.log(`[${idx + 1}/${targets.length}] start ${key}`);
        const videoId = await heygenGenerateVideo(avatarId, audioUrl);
        const url = await heygenWaitForVideoUrl(videoId);
        await downloadToFile(url, outPath);
        ok++;
        console.log(`[OK] ${key} -> ${path.relative(process.cwd(), outPath)}`);
      } catch (e: any) {
        failed++;
        console.log(`[FAIL] ${key}: ${e?.message || e}`);
      }
    },
    opts.concurrency
  );

  console.log('Done.');
  console.log('OK:', ok);
  console.log('Failed:', failed);
}

main().catch((e) => {
  console.error('Fatal:', e?.message || e);
  process.exit(1);
});


