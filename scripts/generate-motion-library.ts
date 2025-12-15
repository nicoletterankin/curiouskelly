#!/usr/bin/env npx tsx
/**
 * KELLY MOTION LIBRARY GENERATOR (420 base clips)
 *
 * Generates 7 phase-specific HeyGen videos per avatar (12 personas × 5 age buckets).
 * Stores job state + results in Supabase table `kelly_motion_library`.
 *
 * Usage:
 *   npx tsx scripts/generate-motion-library.ts --export-registry
 *   npx tsx scripts/generate-motion-library.ts --batch=proof
 *   npx tsx scripts/generate-motion-library.ts --batch=adults
 *   npx tsx scripts/generate-motion-library.ts --batch=all
 *
 * Required env (for generation runs):
 * - HEYGEN_API_KEY
 * - SUPABASE_URL or PUBLIC_SUPABASE_URL
 * - SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY
 *
 * Optional env:
 * - HEYGEN_VOICE_ID (otherwise we pick one from /v2/voices)
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

const REGISTRY_JSON_PATH = path.join(process.cwd(), 'public', 'data', 'avatar-registry.json');
const REGISTRY_HTML_PATH = path.join(process.cwd(), 'public', 'admin', 'avatar-registry.html');

const HEYGEN_GENERATE_URL = 'https://api.heygen.com/v2/video/generate';

// 7 phases with specific motion energy
const PHASES = {
  hook: {
    name: 'Hook',
    duration: '15-20s',
    energy: 'engaging, welcoming',
    script:
      "Welcome! I'm so glad you're here today. We're going to explore something truly fascinating together. This is going to be a journey worth taking. Are you ready to begin?",
  },
  cliff: {
    name: 'Cliff',
    duration: '20-25s',
    energy: 'contemplative, presenting choice',
    script:
      'Now I want you to consider something important. There are two paths we could take from here. Both are valid, both are meaningful. Think about which one resonates with you more deeply. Take your time.',
  },
  fact1: {
    name: 'Fact1',
    duration: '20-25s',
    energy: 'teaching, first gesture set',
    script:
      "Let me share the first important idea with you. This is something that might surprise you at first. Pay attention to how it makes you feel. I think you'll find this really valuable.",
  },
  fact2: {
    name: 'Fact2',
    duration: '20-25s',
    energy: 'teaching, second gesture set',
    script:
      "Building on that, here's another key insight. Notice how this connects to what we just discussed. Can you see the pattern forming? This is where it starts to get really interesting.",
  },
  fact3: {
    name: 'Fact3',
    duration: '20-25s',
    energy: 'teaching, emphatic third gesture set',
    script:
      "And here's where it all comes together. This is the piece that ties everything else into place. If you remember nothing else from today, remember this. This is the heart of what I want you to carry forward.",
  },
  wisdom: {
    name: 'Wisdom',
    duration: '15-20s',
    energy: 'reflective, deeper',
    script:
      "So what's the deeper meaning here? When you step back and look at everything we've explored, there's a profound truth waiting for you. Let it sink in. This wisdom is yours now.",
  },
  outro: {
    name: 'Outro',
    duration: '15-20s',
    energy: 'warm, celebratory',
    script:
      "You did it! Another day of growth complete. I'm genuinely proud of you for showing up today. Tomorrow we'll continue this journey together. Until then, carry what you've learned with you.",
  },
} as const;

type PhaseKey = keyof typeof PHASES;

const PERSONAS = [
  'scientist',
  'explorer',
  'rebel',
  'architect',
  'diplomat',
  'empath',
  'macgyver',
  'mystic',
  'provider',
  'storyteller',
  'strategist',
  'survivor',
] as const;

type PersonaKey = (typeof PERSONAS)[number];

const AGE_BUCKETS = ['kid', 'teen', 'adult', 'elder', 'super_elder'] as const;

type AgeBucketKey = (typeof AGE_BUCKETS)[number];

type Options = {
  batch: string;
  delayMs: number;
  pollMs: number;
  timeoutMs: number;
  exportRegistry: boolean;
  dryRun: boolean;
};

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    batch: 'proof',
    delayMs: 5000,
    pollMs: 10_000,
    timeoutMs: 25 * 60 * 1000,
    exportRegistry: false,
    dryRun: false,
  };

  for (const a of argv) {
    if (a === '--export-registry') opts.exportRegistry = true;
    else if (a === '--dry-run') opts.dryRun = true;
    else if (a.startsWith('--batch=')) opts.batch = a.split('=')[1] || opts.batch;
    else if (a.startsWith('--delay-ms=')) opts.delayMs = Math.max(0, parseInt(a.split('=')[1] || '5000', 10) || opts.delayMs);
    else if (a.startsWith('--poll-ms=')) opts.pollMs = Math.max(1000, parseInt(a.split('=')[1] || '10000', 10) || opts.pollMs);
    else if (a.startsWith('--timeout-ms=')) opts.timeoutMs = Math.max(30_000, parseInt(a.split('=')[1] || String(opts.timeoutMs), 10) || opts.timeoutMs);
    else if (a === '--help' || a === '-h') {
      console.log(`
Kelly Motion Library Generator

Usage:
  npx tsx scripts/generate-motion-library.ts --export-registry
  npx tsx scripts/generate-motion-library.ts --batch=proof
  npx tsx scripts/generate-motion-library.ts --batch=adults
  npx tsx scripts/generate-motion-library.ts --batch=all

Options:
  --export-registry        Create public/data/avatar-registry.json from public/admin/avatar-registry.html
  --batch=<name>           proof | adults | teens | kids | elders | super_elders | all
  --dry-run                Print targets only (no API calls)
  --delay-ms=<n>           Delay between clip generations (default 5000)
  --poll-ms=<n>            Poll interval (default 10000)
  --timeout-ms=<n>         Max time per clip (default 1500000)
`);
      process.exit(0);
    }
  }

  return opts;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function parseDefaultAvatarIdsFromHtml(html: string): Record<string, string> {
  const m = html.match(/const\s+DEFAULT_AVATAR_IDS\s*=\s*\{([\s\S]*?)\}\s*;/m);
  if (!m) throw new Error('Could not find DEFAULT_AVATAR_IDS in public/admin/avatar-registry.html');

  const body = m[1];
  const re = /([a-zA-Z0-9_]+)\s*:\s*['"]([a-f0-9]{32})['"]\s*,?/g;

  const out: Record<string, string> = {};
  for (const match of body.matchAll(re)) {
    out[match[1]] = match[2];
  }

  if (Object.keys(out).length !== 60) {
    console.log(`⚠️ Parsed ${Object.keys(out).length} avatar IDs (expected 60).`);
  }

  return out;
}

async function loadAvatarRegistry(params: { writeIfMissing: boolean }): Promise<Record<string, string>> {
  // 1) Prefer JSON file
  if (fs.existsSync(REGISTRY_JSON_PATH)) {
    const raw = fs.readFileSync(REGISTRY_JSON_PATH, 'utf8');
    const json = JSON.parse(raw);
    if (json && typeof json === 'object' && !Array.isArray(json)) {
      // ensure string->string
      const out: Record<string, string> = {};
      for (const [k, v] of Object.entries(json as any)) {
        if (typeof v === 'string' && v.length >= 10) out[k] = v;
      }
      return out;
    }
  }

  // 2) Fallback: parse from HTML
  if (!fs.existsSync(REGISTRY_HTML_PATH)) {
    throw new Error(`Missing avatar registry sources. Expected one of:\n- ${REGISTRY_JSON_PATH}\n- ${REGISTRY_HTML_PATH}`);
  }

  const html = fs.readFileSync(REGISTRY_HTML_PATH, 'utf8');
  const out = parseDefaultAvatarIdsFromHtml(html);

  if (params.writeIfMissing) {
    fs.mkdirSync(path.dirname(REGISTRY_JSON_PATH), { recursive: true });
    fs.writeFileSync(REGISTRY_JSON_PATH, JSON.stringify(out, null, 2));
  }

  return out;
}

function requireEnvForGeneration() {
  const missing: string[] = [];
  if (!HEYGEN_API_KEY) missing.push('HEYGEN_API_KEY');
  if (!SUPABASE_URL) missing.push('SUPABASE_URL (or PUBLIC_SUPABASE_URL)');
  if (!SUPABASE_SERVICE_KEY) missing.push('SUPABASE_SERVICE_KEY (or SUPABASE_SERVICE_ROLE_KEY)');

  if (missing.length) {
    console.error('❌ Missing required env vars for generation:');
    for (const m of missing) console.error('  -', m);
    process.exit(1);
  }
}

let cachedHeygenVoiceId: string | null = null;

async function getHeygenVoiceId(): Promise<string> {
  if (cachedHeygenVoiceId) return cachedHeygenVoiceId;

  const fromEnv = process.env.HEYGEN_VOICE_ID;
  if (fromEnv && fromEnv.length > 5) {
    cachedHeygenVoiceId = fromEnv;
    return cachedHeygenVoiceId;
  }

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
  if (!r.ok) throw new Error(`HeyGen v2/voices failed (status ${r.status}): ${JSON.stringify(j).slice(0, 500)}`);

  const voices: any[] = j?.data?.voices || [];
  if (!Array.isArray(voices) || voices.length === 0) {
    throw new Error(`HeyGen v2/voices returned no voices: ${JSON.stringify(j).slice(0, 500)}`);
  }

  const preferred =
    voices.find((v) => String(v?.language).toLowerCase().includes('english') && String(v?.gender).toLowerCase() === 'female') ||
    voices[0];

  cachedHeygenVoiceId = preferred.voice_id;
  return cachedHeygenVoiceId;
}

async function heygenGenerateVideo(params: { talkingPhotoId: string; script: string }): Promise<string> {
  const voiceId = await getHeygenVoiceId();

  const payload: any = {
    video_inputs: [
      {
        character: { type: 'talking_photo', talking_photo_id: params.talkingPhotoId },
        voice: {
          type: 'text',
          voice_id: voiceId,
          input_text: params.script,
          text: { voice_id: voiceId, input_text: params.script },
        },
      },
    ],
    dimension: { width: 1280, height: 720 },
  };

  const r = await fetch(HEYGEN_GENERATE_URL, {
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
    throw new Error(`HeyGen generate returned non-JSON (status ${r.status}): ${txt.slice(0, 200)}`);
  }

  if (!r.ok) throw new Error(`HeyGen generate failed (status ${r.status}): ${JSON.stringify(j).slice(0, 800)}`);

  const videoId = j?.data?.video_id;
  if (!videoId) throw new Error(`HeyGen generate response missing data.video_id: ${JSON.stringify(j).slice(0, 800)}`);

  return videoId;
}

async function heygenStatus(videoId: string): Promise<{ status: string; videoUrl?: string; duration?: number; error?: any }> {
  // Try v2 first
  {
    const r = await fetch(`https://api.heygen.com/v2/video_status.get?video_id=${encodeURIComponent(videoId)}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY! },
    });
    const txt = await r.text();
    try {
      const j: any = JSON.parse(txt);
      if (j?.data?.status) {
        return { status: j.data.status, videoUrl: j.data.video_url, duration: j.data.duration, error: j.data.error };
      }
    } catch {
      // ignore
    }
  }

  // Fallback to v1
  {
    const r = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY! },
    });
    const j: any = await r.json();
    return { status: j?.data?.status, videoUrl: j?.data?.video_url, duration: j?.data?.duration, error: j?.data?.error };
  }
}

async function pollForCompletion(videoId: string, opts: Options): Promise<{ videoUrl: string; duration?: number }> {
  const startedAt = Date.now();
  let attempt = 0;

  while (true) {
    attempt += 1;
    const elapsed = Date.now() - startedAt;
    if (elapsed > opts.timeoutMs) {
      throw new Error(`Timeout waiting for HeyGen video (${videoId}) after ${Math.round(elapsed / 1000)}s`);
    }

    const st = await heygenStatus(videoId);
    const status = st.status;

    console.log(`  ⏳ Status: ${status} (+${Math.round(elapsed / 1000)}s, attempt ${attempt})`);

    if (status === 'completed' && st.videoUrl) {
      return { videoUrl: st.videoUrl, duration: st.duration };
    }

    if (status === 'failed') {
      throw new Error(`Video generation failed: ${typeof st.error === 'string' ? st.error : JSON.stringify(st.error)}`);
    }

    await sleep(opts.pollMs);
  }
}

type ClipResult =
  | { outcome: 'completed'; videoUrl: string }
  | { outcome: 'skipped'; videoUrl: string }
  | { outcome: 'failed'; error: string };

function isMissingTableError(e: any): boolean {
  const msg = String(e?.message || '');
  return msg.includes('relation') && msg.includes('does not exist') && msg.includes('kelly_motion_library');
}

async function generatePhaseClip(params: {
  supabase: ReturnType<typeof createClient>;
  avatarKey: string;
  persona: string;
  ageBucket: string;
  talkingPhotoId: string;
  phase: PhaseKey;
  opts: Options;
}): Promise<ClipResult> {
  const phaseConfig = PHASES[params.phase];
  console.log(`\n🎬 Generating: ${params.avatarKey} - ${params.phase} (${phaseConfig.energy})`);

  try {
    // Check if already exists
    const existing = await (async () => {
      const { data, error } = await params.supabase
        .from('kelly_motion_library')
        .select('video_url,status,video_id')
        .eq('avatar_key', params.avatarKey)
        .eq('phase', params.phase)
        .maybeSingle();

      if (error) throw error;
      return data as any;
    })();

    if (existing?.status === 'completed' && existing?.video_url) {
      console.log('  ⏭️ Already completed, skipping');
      return { outcome: 'skipped', videoUrl: existing.video_url };
    }

    // Resume if a previous run started a HeyGen job
    if (existing?.status === 'generating' && existing?.video_id) {
      console.log(`  🔁 Resuming existing job: ${existing.video_id}`);
      const done = await pollForCompletion(existing.video_id, params.opts);

      await params.supabase
        .from('kelly_motion_library')
        .update({
          video_url: done.videoUrl,
          status: 'completed',
          duration: done.duration ?? null,
          completed_at: new Date().toISOString(),
        })
        .eq('avatar_key', params.avatarKey)
        .eq('phase', params.phase);

      console.log(`  ✅ Complete: ${done.videoUrl.substring(0, 80)}...`);
      return { outcome: 'completed', videoUrl: done.videoUrl };
    }

    // Upsert as generating
    {
      const { error } = await params.supabase
        .from('kelly_motion_library')
        .upsert(
          {
            avatar_key: params.avatarKey,
            persona: params.persona,
            age_bucket: params.ageBucket,
            phase: params.phase,
            status: 'generating',
          },
          { onConflict: 'avatar_key,phase' }
        );
      if (error) throw error;
    }

    // Generate video (retry a couple times for transient errors)
    const maxAttempts = 3;
    let videoId: string | null = null;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        videoId = await heygenGenerateVideo({ talkingPhotoId: params.talkingPhotoId, script: phaseConfig.script });
        break;
      } catch (e: any) {
        const msg = e?.message || String(e);
        console.log(`  ⚠️ HeyGen generate attempt ${attempt}/${maxAttempts} failed: ${msg}`);
        if (attempt === maxAttempts) throw e;
        await sleep(1500 * attempt);
      }
    }

    if (!videoId) throw new Error('HeyGen generate did not return a video_id');

    console.log(`  📹 Video ID: ${videoId}`);

    // Persist video_id
    {
      const { error } = await params.supabase
        .from('kelly_motion_library')
        .update({ video_id: videoId })
        .eq('avatar_key', params.avatarKey)
        .eq('phase', params.phase);
      if (error) throw error;
    }

    // Poll for completion
    const done = await pollForCompletion(videoId, params.opts);

    // Update as completed
    {
      const { error } = await params.supabase
        .from('kelly_motion_library')
        .update({
          video_url: done.videoUrl,
          status: 'completed',
          duration: done.duration ?? null,
          completed_at: new Date().toISOString(),
        })
        .eq('avatar_key', params.avatarKey)
        .eq('phase', params.phase);
      if (error) throw error;
    }

    console.log(`  ✅ Complete: ${done.videoUrl.substring(0, 80)}...`);
    return { outcome: 'completed', videoUrl: done.videoUrl };
  } catch (error: any) {
    if (isMissingTableError(error)) {
      throw new Error(
        `Supabase table kelly_motion_library does not exist. Create it first in Supabase SQL Editor (see mission doc / prompt). Original error: ${error.message}`
      );
    }

    const msg = error?.message || String(error);
    console.error(`  ❌ Failed: ${msg}`);

    try {
      await params.supabase
        .from('kelly_motion_library')
        .update({ status: 'failed' })
        .eq('avatar_key', params.avatarKey)
        .eq('phase', params.phase);
    } catch {
      // ignore
    }

    return { outcome: 'failed', error: msg };
  }
}

function phasesList(): PhaseKey[] {
  return Object.keys(PHASES) as PhaseKey[];
}

async function generateBatch(params: {
  supabase: ReturnType<typeof createClient>;
  avatarRegistry: Record<string, string>;
  personas: readonly PersonaKey[];
  ageBuckets: readonly AgeBucketKey[];
  phases: readonly PhaseKey[];
  opts: Options;
}): Promise<void> {
  const results = { total: 0, completed: 0, skipped: 0, failed: 0 };

  for (const persona of params.personas) {
    for (const ageBucket of params.ageBuckets) {
      const avatarKey = `${persona}_${ageBucket}`;
      const talkingPhotoId = params.avatarRegistry[avatarKey];

      if (!talkingPhotoId) {
        console.log(`⚠️ Missing avatar ID for ${avatarKey}, skipping`);
        continue;
      }

      for (const phase of params.phases) {
        results.total++;

        if (params.opts.dryRun) {
          console.log(`[DRY] would generate ${avatarKey} - ${phase}`);
          continue;
        }

        const r = await generatePhaseClip({
          supabase: params.supabase,
          avatarKey,
          persona,
          ageBucket,
          talkingPhotoId,
          phase,
          opts: params.opts,
        });

        if (r.outcome === 'completed') results.completed++;
        if (r.outcome === 'skipped') results.skipped++;
        if (r.outcome === 'failed') results.failed++;

        await sleep(params.opts.delayMs);
      }
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log('BATCH COMPLETE');
  console.log('='.repeat(60));
  console.log(`Total attempted: ${results.total}`);
  console.log(`Completed:      ${results.completed}`);
  console.log(`Skipped:        ${results.skipped}`);
  console.log(`Failed:         ${results.failed}`);
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  console.log('🚀 KELLY MOTION LIBRARY GENERATOR');
  console.log('='.repeat(60));
  console.log(`Batch: ${opts.batch}${opts.dryRun ? ' (dry-run)' : ''}`);
  console.log(`Phases: ${phasesList().join(', ')}`);
  console.log(`Delay: ${opts.delayMs}ms | Poll: ${opts.pollMs}ms | Timeout: ${opts.timeoutMs}ms`);
  console.log('='.repeat(60));

  // Registry export mode (no env required)
  if (opts.exportRegistry) {
    const registry = await loadAvatarRegistry({ writeIfMissing: true });
    console.log(`✅ Registry ready at: ${path.relative(process.cwd(), REGISTRY_JSON_PATH)}`);
    console.log(`✅ Avatar IDs: ${Object.keys(registry).length}`);
    return;
  }

  // Load registry (write JSON if missing so future runs are stable)
  const avatarRegistry = await loadAvatarRegistry({ writeIfMissing: true });

  // Batches
  const phases = phasesList();

  let personas: readonly PersonaKey[] = PERSONAS;
  let ages: readonly AgeBucketKey[] = AGE_BUCKETS;

  switch (opts.batch) {
    case 'proof':
      personas = ['scientist'];
      ages = ['adult'];
      break;
    case 'adults':
      personas = PERSONAS;
      ages = ['adult'];
      break;
    case 'teens':
      personas = PERSONAS;
      ages = ['teen'];
      break;
    case 'kids':
      personas = PERSONAS;
      ages = ['kid'];
      break;
    case 'elders':
      personas = PERSONAS;
      ages = ['elder'];
      break;
    case 'super_elders':
      personas = PERSONAS;
      ages = ['super_elder'];
      break;
    case 'all':
      personas = PERSONAS;
      ages = AGE_BUCKETS;
      break;
    default:
      console.log('Usage: npx tsx scripts/generate-motion-library.ts --batch=<batch>');
      console.log('Batches: proof, adults, teens, kids, elders, super_elders, all');
      process.exit(1);
  }

  console.log(`Targets: ${personas.length} personas × ${ages.length} ages × ${phases.length} phases`);

  if (!opts.dryRun) {
    requireEnvForGeneration();
  }

  const supabase = createClient(SUPABASE_URL!, SUPABASE_SERVICE_KEY!);

  await generateBatch({
    supabase,
    avatarRegistry,
    personas,
    ageBuckets: ages,
    phases,
    opts,
  });
}

main().catch((err) => {
  console.error('❌ Fatal error:', err?.message || String(err));
  process.exit(1);
});
