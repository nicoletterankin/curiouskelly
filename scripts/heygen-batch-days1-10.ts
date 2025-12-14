#!/usr/bin/env npx tsx
/**
 * PROMPT 3: PRODUCTION RUN - BATCH 2 (Days 11-50)
 *
 * - Loads days 11-50 from Supabase (core_lessons + The Scientist/Hook atom script)
 * - Skips lessons that already have a completed video_url recorded
 * - Uses adult-scientist avatar for all lessons (scientist_adult from avatar-registry.html)
 * - Rate limits submissions: 5 seconds between HeyGen generate calls (global)
 * - Max concurrency: 3 (generation+poll+upload tasks)
 * - Updates Supabase:
 *   - lesson_video_generation_status: video_url, status='completed', completed_at
 *   - lesson_atoms.hd_video_url for the targeted atom (best-effort)
 * - Writes results to pipeline-results-batch2.json
 *
 * Run:
 *   npx tsx scripts/heygen-batch-days1-10.ts
 */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY =
  process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;

const VIDEO_BUCKET = process.env.KELLY_VIDEO_BUCKET || 'kelly-videos';

if (!HEYGEN_API_KEY) {
  console.error('❌ Missing HEYGEN_API_KEY');
  process.exit(1);
}
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error('❌ Missing Supabase env vars. Required one of:');
  console.error('   - SUPABASE_URL or PUBLIC_SUPABASE_URL');
  console.error('   - SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

function assertSupabaseEnvLooksReal() {
  // Guardrail: a common failure mode is leaving template placeholders in .env
  let host = '';
  try {
    host = new URL(SUPABASE_URL!).host;
  } catch {
    // We'll still fail later, but this message is clearer.
    throw new Error('SUPABASE_URL is not a valid URL (expected https://<project>.supabase.co)');
  }

  const keyLen = String(SUPABASE_SERVICE_KEY || '').length;
  const looksPlaceholder =
    host === 'your_project.supabase.co' ||
    host.includes('your_project') ||
    host.includes('your-project') ||
    SUPABASE_URL!.includes('YOUR_') ||
    keyLen < 80;

  if (looksPlaceholder) {
    throw new Error(
      [
        'Supabase env vars look like placeholders (not real credentials).',
        `- SUPABASE host: ${host}`,
        `- SUPABASE service key length: ${keyLen} (expected a long service-role JWT)`,
        '',
        'Fix: update your local `.env` with real values:',
        '- PUBLIC_SUPABASE_URL (or SUPABASE_URL)',
        '- SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SERVICE_KEY)',
        '',
        'References in this repo:',
        '- GET_SUPABASE_KEY_INSTRUCTIONS.md',
        '- docs/backend/SUPABASE_MCP_SETUP.md',
      ].join('\n')
    );
  }
}

try {
  assertSupabaseEnvLooksReal();
} catch (e: any) {
  console.error(`❌ ${e?.message || String(e)}`);
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

type BatchResultStatus = 'completed' | 'skipped' | 'failed';

type LessonRunResult = {
  day: number;
  core_lesson_id?: string;
  marketing_headline?: string | null;
  archetype: 'The Scientist';
  phase: 'Hook';
  video_type: 'main';
  avatar_key: 'scientist_adult';
  avatar_id: string;
  atom_id?: string;
  script_chars?: number;
  heygen_video_id?: string;
  heygen_video_url?: string;
  final_video_url?: string;
  status: BatchResultStatus;
  started_at?: string;
  completed_at?: string;
  error?: string;
};

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function safeJsonStringify(value: unknown) {
  return JSON.stringify(value, (_k, v) => (typeof v === 'bigint' ? v.toString() : v), 2);
}

function readAvatarRegistryHtml(): string {
  const p = path.join(process.cwd(), 'public', 'admin', 'avatar-registry.html');
  return fs.readFileSync(p, 'utf8');
}

function pickAdultScientistAvatarId(html: string): string {
  const preferred = /\bscientist_adult\s*:\s*'([^']+)'/i.exec(html);
  if (preferred?.[1] && preferred[1].length >= 10) return preferred[1];
  throw new Error(
    `Could not find scientist_adult in public/admin/avatar-registry.html (contains DEFAULT_AVATAR_IDS: ${html.includes('DEFAULT_AVATAR_IDS')})`
  );
}

async function heygenGenerateVideo(params: { avatarId: string; script: string }): Promise<{ videoId: string }> {
  // Use HeyGen text-to-speech to avoid pulling in ElevenLabs for this batch.
  // (Test video already worked with voice.type='text'.)
  const voiceId = await (async () => {
    const r = await fetch('https://api.heygen.com/v2/voices', {
      headers: { 'X-Api-Key': HEYGEN_API_KEY!, Accept: 'application/json' },
    });
    const txt = await r.text();
    let j: any;
    try {
      j = JSON.parse(txt);
    } catch {
      throw new Error(`HeyGen v2/voices returned non-JSON (status ${r.status}): ${txt.slice(0, 250)}`);
    }
    if (!r.ok) throw new Error(`HeyGen v2/voices failed (status ${r.status}): ${txt.slice(0, 500)}`);
    const voices: any[] = j?.data?.voices || [];
    if (!Array.isArray(voices) || voices.length === 0) throw new Error('HeyGen v2/voices returned no voices');
    const preferred =
      voices.find(v => String(v?.language).toLowerCase().includes('english') && String(v?.gender).toLowerCase() === 'female') ||
      voices[0];
    const id = preferred?.voice_id;
    if (!id) throw new Error('HeyGen voice missing voice_id');
    return id as string;
  })();

  const payload = {
    video_inputs: [
      {
        character: {
          type: 'talking_photo',
          talking_photo_id: params.avatarId,
        },
        voice: {
          type: 'text',
          voice_id: voiceId,
          input_text: params.script,
          text: {
            voice_id: voiceId,
            input_text: params.script,
          },
        },
        background: { type: 'color', value: '#FFFFFF' },
      },
    ],
    dimension: { width: 1080, height: 1920 },
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
    throw new Error(`HeyGen generate returned non-JSON (status ${r.status}): ${txt.slice(0, 250)}`);
  }

  if (!r.ok) throw new Error(`HeyGen generate failed (status ${r.status}): ${txt.slice(0, 800)}`);
  const videoId = j?.data?.video_id;
  if (!videoId) throw new Error(`HeyGen generate response missing data.video_id: ${txt.slice(0, 800)}`);
  return { videoId };
}

async function heygenStatus(videoId: string): Promise<{ status: string; video_url?: string; duration?: number; error?: any }> {
  // Prefer v2; fallback to v1
  {
    const r = await fetch(`https://api.heygen.com/v2/video_status.get?video_id=${encodeURIComponent(videoId)}`, {
      headers: { 'X-Api-Key': HEYGEN_API_KEY! },
    });
    const txt = await r.text();
    try {
      const j: any = JSON.parse(txt);
      if (j?.data?.status) {
        return { status: j.data.status, video_url: j.data.video_url, duration: j.data.duration, error: j.data.error };
      }
    } catch {
      // ignore
    }
  }

  const r = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${encodeURIComponent(videoId)}`, {
    headers: { 'X-Api-Key': HEYGEN_API_KEY! },
  });
  const txt = await r.text();
  const j: any = JSON.parse(txt);
  return { status: j?.data?.status, video_url: j?.data?.video_url, duration: j?.data?.duration, error: j?.data?.error };
}

async function downloadToBuffer(url: string): Promise<Buffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Video download failed: ${r.status} ${r.statusText}`);
  return Buffer.from(await r.arrayBuffer());
}

async function uploadVideoAndGetPublicUrl(params: { day: number; videoBuffer: Buffer; ext: 'mp4' }) {
  const dayStr = String(params.day).padStart(3, '0');
  const hash = crypto.createHash('sha1').update(params.videoBuffer).digest('hex').slice(0, 10);
  const storagePath = `heygen/day_${dayStr}/scientist_adult/main/${Date.now()}_${hash}.${params.ext}`;

  const { error: upErr } = await supabase.storage
    .from(VIDEO_BUCKET)
    .upload(storagePath, params.videoBuffer, { upsert: true, contentType: 'video/mp4' });

  if (upErr) throw new Error(`Supabase upload failed: ${upErr.message}`);
  const { data } = supabase.storage.from(VIDEO_BUCKET).getPublicUrl(storagePath);
  if (!data?.publicUrl) throw new Error('Supabase getPublicUrl failed');
  return { publicUrl: data.publicUrl, storagePath };
}

async function upsertGenerationStatus(params: {
  coreLessonId: string;
  status: 'pending' | 'generating' | 'completed' | 'failed';
  videoUrl?: string | null;
  errorMessage?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
}) {
  const payload: any = {
    core_lesson_id: params.coreLessonId,
    archetype: 'The Scientist',
    phase: 'Hook',
    video_type: 'main',
    status: params.status,
    video_url: params.videoUrl ?? null,
    error_message: params.errorMessage ?? null,
    started_at: params.startedAt ?? null,
    completed_at: params.completedAt ?? null,
  };

  const { error } = await supabase
    .from('lesson_video_generation_status')
    .upsert(payload, { onConflict: 'core_lesson_id,archetype,phase,video_type' as any });

  if (error) throw new Error(`Supabase upsert lesson_video_generation_status failed: ${error.message}`);
}

async function tryUpdateAtomHdUrl(atomId: string, url: string) {
  // This column exists in some deployments but not all; ignore errors.
  const { error } = await supabase.from('lesson_atoms').update({ hd_video_url: url } as any).eq('id', atomId);
  if (error) {
    // best-effort only; do not throw
  }
}

async function loadDayInputs(day: number): Promise<{
  coreLessonId: string;
  marketingHeadline: string | null;
  atomId: string;
  script: string;
  alreadyHasVideoUrl: boolean;
}> {
  const { data: lesson, error: lessonErr } = await supabase
    .from('core_lessons')
    .select('id, marketing_headline')
    .eq('day_number', day)
    .single();
  if (lessonErr || !lesson?.id) throw new Error(`Supabase core_lessons fetch failed for day ${day}: ${lessonErr?.message || 'not found'}`);

  const { data: atom, error: atomErr } = await supabase
    .from('lesson_atoms')
    .select('id, content')
    .eq('core_lesson_id', lesson.id)
    .eq('archetype', 'The Scientist')
    .eq('phase', 'Hook')
    .single();

  if (atomErr || !atom?.id) throw new Error(`Supabase lesson_atoms fetch failed (The Scientist/Hook) for day ${day}: ${atomErr?.message || 'not found'}`);

  const script = (atom as any)?.content?.script || (atom as any)?.content?.text;
  if (!script || typeof script !== 'string') throw new Error(`Day ${day} atom content missing script/text`);

  // Check whether there is already a completed URL in lesson_video_generation_status
  const { data: statusRow, error: stErr } = await supabase
    .from('lesson_video_generation_status')
    .select('video_url, status')
    .eq('core_lesson_id', lesson.id)
    .eq('archetype', 'The Scientist')
    .eq('phase', 'Hook')
    .eq('video_type', 'main')
    .maybeSingle();

  if (stErr) {
    // If table isn't present, fail loudly — prompt requires Supabase updates.
    throw new Error(`Supabase lesson_video_generation_status read failed: ${stErr.message}`);
  }

  const alreadyHasVideoUrl = Boolean(statusRow?.video_url) && String(statusRow?.status || '').toLowerCase() === 'completed';

  return {
    coreLessonId: lesson.id,
    marketingHeadline: (lesson as any).marketing_headline ?? null,
    atomId: atom.id,
    script,
    alreadyHasVideoUrl,
  };
}

class SubmissionRateLimiter {
  private nextAllowedMs = Date.now();
  constructor(private minIntervalMs: number) {}
  async waitTurn() {
    const now = Date.now();
    const wait = Math.max(0, this.nextAllowedMs - now);
    if (wait > 0) await sleep(wait);
    // Reserve next slot *after* we pass the gate
    this.nextAllowedMs = Date.now() + this.minIntervalMs;
  }
}

async function runConcurrent<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = [];
  const executing = new Set<Promise<void>>();
  let idx = 0;

  const enqueue = async () => {
    while (idx < items.length) {
      const current = items[idx++];
      const p = (async () => {
        const r = await fn(current);
        results.push(r);
      })();
      executing.add(p);
      const cleanup = () => executing.delete(p);
      p.then(cleanup).catch(cleanup);
      if (executing.size >= limit) return;
    }
  };

  await enqueue();
  while (executing.size) {
    await Promise.race(executing);
    await enqueue();
  }
  return results;
}

async function processDay(day: number, avatarId: string, limiter: SubmissionRateLimiter): Promise<LessonRunResult> {
  const base: LessonRunResult = {
    day,
    archetype: 'The Scientist',
    phase: 'Hook',
    video_type: 'main',
    avatar_key: 'scientist_adult',
    avatar_id: avatarId,
    status: 'failed',
  };

  try {
    const inputs = await loadDayInputs(day);
    base.core_lesson_id = inputs.coreLessonId;
    base.marketing_headline = inputs.marketingHeadline;
    base.atom_id = inputs.atomId;
    base.script_chars = inputs.script.length;

    if (inputs.alreadyHasVideoUrl) {
      base.status = 'skipped';
      console.log(`Day ${day}: ⏭️ Skipped (already had completed video_url)`);
      return base;
    }

    const startedAt = new Date().toISOString();
    base.started_at = startedAt;

    // Mark generating
    await upsertGenerationStatus({
      coreLessonId: inputs.coreLessonId,
      status: 'generating',
      videoUrl: null,
      errorMessage: null,
      startedAt,
      completedAt: null,
    });

    // Global submission spacing
    await limiter.waitTurn();

    // Generate
    const { videoId } = await heygenGenerateVideo({ avatarId, script: inputs.script });
    base.heygen_video_id = videoId;

    // Poll until complete
    const pollStarted = Date.now();
    const maxMs = 25 * 60 * 1000; // 25 minutes safety
    const intervalMs = 10 * 1000;

    while (true) {
      const st = await heygenStatus(videoId);
      if (st.status === 'completed' && st.video_url) {
        base.heygen_video_url = st.video_url;
        break;
      }
      if (st.status === 'failed') {
        throw new Error(`HeyGen failed: ${typeof st.error === 'string' ? st.error : JSON.stringify(st.error)}`);
      }
      if (Date.now() - pollStarted > maxMs) throw new Error(`Timeout waiting for HeyGen completion (>${Math.round(maxMs / 60000)}m)`);
      await sleep(intervalMs);
    }

    // Download -> Upload to Supabase -> store stable URL
    const buf = await downloadToBuffer(base.heygen_video_url!);
    const { publicUrl } = await uploadVideoAndGetPublicUrl({ day, videoBuffer: buf, ext: 'mp4' });
    base.final_video_url = publicUrl;

    const completedAt = new Date().toISOString();
    base.completed_at = completedAt;

    await upsertGenerationStatus({
      coreLessonId: inputs.coreLessonId,
      status: 'completed',
      videoUrl: publicUrl,
      errorMessage: null,
      startedAt,
      completedAt,
    });

    // Best-effort convenience: write hd_video_url onto the atom too
    await tryUpdateAtomHdUrl(inputs.atomId, publicUrl);

    base.status = 'completed';
    const elapsedS = Math.round((Date.now() - pollStarted) / 1000);
    console.log(`Day ${day}: ✅ ${publicUrl} (+${elapsedS}s after queued)`);
    return base;
  } catch (e: any) {
    const msg = e?.message || String(e);
    base.status = 'failed';
    base.error = msg;

    try {
      if (base.core_lesson_id) {
        await upsertGenerationStatus({
          coreLessonId: base.core_lesson_id,
          status: 'failed',
          videoUrl: null,
          errorMessage: msg,
          startedAt: base.started_at ?? null,
          completedAt: new Date().toISOString(),
        });
      }
    } catch {
      // ignore secondary failure; primary error is more important
    }

    console.log(`Day ${day}: ❌ ${msg}`);
    return base;
  }
}

async function main() {
  console.log('========================================');
  console.log('PROMPT 3: PRODUCTION RUN - BATCH 2');
  console.log('Generating HeyGen videos for Days 11-50');
  console.log('========================================');

  // Avatar selection: adult-scientist
  const avatarHtml = readAvatarRegistryHtml();
  const avatarId = pickAdultScientistAvatarId(avatarHtml);
  console.log(`Avatar: scientist_adult -> avatar_id=${avatarId}`);

  const fromDay = 11;
  const toDay = 50;
  const days = Array.from({ length: toDay - fromDay + 1 }, (_, i) => i + fromDay);
  const limiter = new SubmissionRateLimiter(5000);
  const startedAt = new Date().toISOString();

  const results = await runConcurrent(days, 3, async day => processDay(day, avatarId, limiter));

  // Stable sort by day for reporting
  results.sort((a, b) => a.day - b.day);

  const completed = results.filter(r => r.status === 'completed');
  const skipped = results.filter(r => r.status === 'skipped');
  const failed = results.filter(r => r.status === 'failed');

  const report = {
    batch: 'batch2',
    range: { fromDay, toDay },
    avatar: { key: 'scientist_adult', avatar_id: avatarId },
    started_at: startedAt,
    finished_at: new Date().toISOString(),
    summary: {
      completed: completed.length,
      skipped: skipped.length,
      failed: failed.length,
    },
    results,
    failed_lessons: failed.map(f => ({ day: f.day, error: f.error })),
  };

  const outPath = path.join(process.cwd(), 'pipeline-results-batch2.json');
  fs.writeFileSync(outPath, safeJsonStringify(report));

  console.log(`\nBATCH COMPLETE: Days ${fromDay}-${toDay}`);
  console.log(`✅ Completed: ${completed.length}`);
  console.log(`⏭️ Skipped (had video): ${skipped.length}`);
  console.log(`❌ Failed: ${failed.length}`);
  if (failed.length) {
    console.log('\nFailed lessons:');
    for (const f of failed) console.log(`- Day ${f.day}: ${f.error}`);
  }
  console.log(`\nSaved results to: ${outPath}`);
}

main().catch(err => {
  console.error('❌ Fatal error:', err?.message || String(err));
  process.exit(1);
});

