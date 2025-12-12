#!/usr/bin/env npx tsx
/**
 * CURIOUS KELLY - HEYGEN VIDEO GENERATION (DAY N)
 *
 * Pipeline:
 * lesson_atoms.content.script -> ElevenLabs TTS -> upload audio -> HeyGen lip-sync video
 * -> (records generation metadata into kelly_video_assets with schema fallbacks)
 *
 * IMPORTANT:
 * - This script will REFUSE to generate if talking_photo_ids are missing/placeholder.
 *
 * Usage:
 *   npx tsx scripts/generate-day-videos-heygen.ts --day=1 --age=adult --lang=en
 *   npx tsx scripts/generate-day-videos-heygen.ts --day=1 --age=adult --lang=en --archetype="The Explorer"
 *   npx tsx scripts/generate-day-videos-heygen.ts --day=1 --age=adult --lang=en --dry-run
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY =
  process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

const AUDIO_BUCKET = process.env.KELLY_AUDIO_BUCKET || 'kelly-templates';

if (!HEYGEN_API_KEY) {
  console.error('❌ Missing HEYGEN_API_KEY');
  process.exit(1);
}
if (!ELEVENLABS_API_KEY) {
  console.error('❌ Missing ELEVENLABS_API_KEY');
  process.exit(1);
}
if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error('❌ Missing Supabase env vars. Required one of:');
  console.error('   - SUPABASE_URL or PUBLIC_SUPABASE_URL');
  console.error('   - SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

const ARCHETYPE_TO_PERSONA: Record<string, string> = {
  'The Architect': 'architect',
  'The Diplomat': 'diplomat',
  'The Empath': 'empath',
  'The Explorer': 'explorer',
  'The MacGyver': 'macgyver',
  'The Mystic': 'mystic',
  'The Provider': 'provider',
  'The Rebel': 'rebel',
  'The Scientist': 'scientist',
  'The Storyteller': 'storyteller',
  'The Strategist': 'strategist',
  'The Survivor': 'survivor',
};

const PHASES = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'] as const;

const PHASE_TO_ASSET_PHASE: Record<string, string> = {
  Hook: 'hook',
  Fact1: 'q1',
  Fact2: 'q2',
  Fact3: 'q3',
  Wisdom: 'wisdom',
};

type AgeVariant = 'kid' | 'teen' | 'adult' | 'elder' | 'super_elder' | 'mature';

function getArg(name: string, fallback?: string) {
  const arg = process.argv.slice(2).find(a => a.startsWith(`--${name}=`));
  return arg ? arg.split('=')[1] : fallback;
}

function hasFlag(name: string) {
  return process.argv.slice(2).includes(`--${name}`);
}

function readTalkingPhotoMap(age: AgeVariant): Record<string, string> {
  const idMapPath = path.join(
    process.cwd(),
    'generated-images',
    'kelly-archetypes-head-only',
    'age',
    age,
    'heygen_talking_photo_ids.json'
  );

  if (!fs.existsSync(idMapPath)) {
    throw new Error(`Talking photo ID map not found: ${idMapPath}`);
  }

  const raw = JSON.parse(fs.readFileSync(idMapPath, 'utf-8'));
  if (!raw || typeof raw !== 'object') {
    throw new Error(`Talking photo ID map invalid JSON: ${idMapPath}`);
  }

  return raw;
}

function assertTalkingPhotoMapComplete(map: Record<string, string>, age: string) {
  const required = Object.values(ARCHETYPE_TO_PERSONA);
  const missingKeys = required.filter(p => !(p in map));
  const missingIds = required.filter(p => !map[p] || map[p] === 'PASTE_ID_HERE');

  if (missingKeys.length || missingIds.length) {
    const parts: string[] = [];
    if (missingKeys.length) parts.push(`missing keys: ${missingKeys.join(', ')}`);
    if (missingIds.length) parts.push(`missing IDs: ${missingIds.join(', ')}`);
    throw new Error(`HeyGen talking_photo_ids not ready for age=${age}: ${parts.join(' | ')}`);
  }
}

function normalizeAgeBucketForDb(age: AgeVariant): string {
  // Try canonical first; if DB uses constrained set, we fall back on insert.
  return age;
}

function normalizePhaseForDb(phase: string): string {
  return PHASE_TO_ASSET_PHASE[phase] || phase.toLowerCase();
}

function sanitizeForPath(s: string) {
  return s.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_\-]/g, '');
}

async function generateElevenLabsAudio(script: string): Promise<Buffer> {
  const resp = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}?output_format=mp3_44100_192`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY!,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: script,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    }
  );

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`ElevenLabs error: ${resp.status} ${resp.statusText} - ${txt.slice(0, 400)}`);
  }

  return Buffer.from(await resp.arrayBuffer());
}

async function uploadAudioAndGetUrl(audio: Buffer, storagePath: string): Promise<string> {
  const { error: upErr } = await supabase.storage
    .from(AUDIO_BUCKET)
    .upload(storagePath, audio, { upsert: true, contentType: 'audio/mpeg' });

  if (upErr) throw new Error(`Supabase audio upload failed: ${upErr.message}`);

  const { data } = supabase.storage.from(AUDIO_BUCKET).getPublicUrl(storagePath);
  if (!data?.publicUrl) throw new Error('Supabase audio getPublicUrl failed');
  return data.publicUrl;
}

async function heygenGenerateVideo(talkingPhotoId: string, audioUrl: string) {
  const resp = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY!,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [
        {
          character: { type: 'talking_photo', talking_photo_id: talkingPhotoId },
          voice: { type: 'audio', audio_url: audioUrl },
          background: { type: 'color', value: '#FFFFFF' },
        },
      ],
      dimension: { width: 1080, height: 1920 },
      super_resolution: true,
    }),
  });

  const text = await resp.text();
  let json: any;
  try {
    json = JSON.parse(text);
  } catch {
    json = null;
  }

  if (!resp.ok) {
    throw new Error(`HeyGen generate failed: ${resp.status} ${resp.statusText} - ${text.slice(0, 400)}`);
  }

  const videoId = json?.data?.video_id;
  if (!videoId) throw new Error(`HeyGen response missing video_id: ${text.slice(0, 400)}`);
  return { videoId };
}

async function upsertKellyVideoAsset(params: {
  day: number;
  phase: string;
  ageBucket: string;
  language: string;
  archetype: string;
  persona: string;
  script: string;
  audioUrl: string;
  heygenVideoId: string;
  sourceImagePath: string;
}) {
  const now = new Date().toISOString();

  // Attempt A: "day_number" + "public_url" schema (used by several scripts in this repo)
  {
    const payload: any = {
      day_number: params.day,
      phase: params.phase,
      template: params.persona,
      asset_type: 'video',
      age_bucket: params.ageBucket,
      language: params.language,
      storage_bucket: AUDIO_BUCKET,
      storage_path: `heygen/${params.heygenVideoId}`,
      public_url: '',
      status: 'generating',
      generation_prompt: params.script,
      created_at: now,
      updated_at: now,
    };

    const { error } = await supabase
      .from('kelly_video_assets')
      .upsert(payload, { onConflict: 'day_number,phase,template,asset_type,age_bucket,language' as any });

    if (!error) return;

    // If constraints/columns don't match, fall through to Attempt B.
  }

  // Attempt B: migration 004 schema (lesson_day + video_public_url)
  {
    const payload: any = {
      lesson_day: params.day,
      phase: params.phase === 'hook' ? 'welcome' : params.phase,
      age_bucket: params.ageBucket,
      language: params.language,
      archetype: params.archetype,
      source_image_path: params.sourceImagePath,
      source_audio_url: params.audioUrl,
      script_text: params.script,
      video_storage_path: `heygen/${params.heygenVideoId}`,
      status: 'generating',
      generation_started_at: now,
      updated_at: now,
    };

    // Try insert/upsert using the unique constraint from migration 004.
    const { error } = await supabase
      .from('kelly_video_assets')
      .upsert(payload, { onConflict: 'lesson_day,phase,age_bucket,language' as any });

    if (!error) return;

    // Final fallback: try again with constrained age buckets
    const ageFallback: Record<string, string> = {
      kid: 'child',
      teen: 'teen',
      adult: 'adult',
      elder: 'elder',
      super_elder: 'elder',
      mature: 'adult',
    };
    payload.age_bucket = ageFallback[params.ageBucket] || params.ageBucket;

    const { error: error2 } = await supabase
      .from('kelly_video_assets')
      .upsert(payload, { onConflict: 'lesson_day,phase,age_bucket,language' as any });

    if (error2) throw new Error(`Failed to write kelly_video_assets: ${error2.message}`);
  }
}

async function main() {
  const day = parseInt(getArg('day', '1')!, 10);
  const age = (getArg('age', 'adult') as AgeVariant) || 'adult';
  const lang = getArg('lang', 'en')!;
  const targetArchetype = getArg('archetype');
  const dryRun = hasFlag('dry-run') || hasFlag('dryRun');

  console.log('========================================');
  console.log(`HEYGEN GENERATION (QUEUING) - Day ${day}`);
  console.log(`Age: ${age} | Lang: ${lang} | DryRun: ${dryRun}`);
  console.log('========================================');

  const talkingPhotoMap = readTalkingPhotoMap(age);
  assertTalkingPhotoMapComplete(talkingPhotoMap, age);

  // Get lesson id
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', day)
    .single();

  if (lessonError || !lesson) {
    throw new Error(`No core_lesson found for day ${day}: ${lessonError?.message || 'not found'}`);
  }

  let query = supabase
    .from('lesson_atoms')
    .select('id, phase, archetype, content')
    .eq('core_lesson_id', lesson.id);

  if (targetArchetype) {
    query = query.eq('archetype', targetArchetype);
  }

  const { data: atoms, error } = await query;
  if (error || !atoms?.length) throw new Error(`No lesson_atoms found: ${error?.message || 'empty'}`);

  const atomsToProcess = atoms
    .filter(a => PHASES.includes(a.phase as any))
    .sort((a, b) => (PHASES.indexOf(a.phase as any) - PHASES.indexOf(b.phase as any)));

  console.log(`\nFound ${atomsToProcess.length} atoms to queue`);

  for (const atom of atomsToProcess) {
    const persona = ARCHETYPE_TO_PERSONA[atom.archetype] || 'diplomat';
    const talkingPhotoId = talkingPhotoMap[persona];

    const script = (atom as any).content?.script || (atom as any).content?.text || '';
    if (!script) {
      console.log(`⚠️ Skipping ${atom.phase}/${atom.archetype}: no script`);
      continue;
    }

    const assetPhase = normalizePhaseForDb(atom.phase);

    const sourceImagePath =
      age === 'adult'
        ? `heygen/archetypes-head-only/kelly_${persona}_head.png`
        : `heygen/archetypes-head-only/age/${age}/kelly_${persona}_head.png`;

    console.log(`\n[${atom.phase}] ${atom.archetype} -> persona=${persona} (age=${age})`);

    if (dryRun) {
      console.log('  DRY RUN: would generate audio + queue HeyGen video');
      continue;
    }

    // 1) Audio
    console.log('  🎤 Generating ElevenLabs audio...');
    const audioBuf = await generateElevenLabsAudio(script);

    const dayStr = String(day).padStart(3, '0');
    const ts = Date.now();
    const audioStoragePath = `heygen/audio/day_${dayStr}/${age}/${persona}/${sanitizeForPath(atom.phase)}_${ts}.mp3`;
    const audioUrl = await uploadAudioAndGetUrl(audioBuf, audioStoragePath);

    console.log(`  ✅ Audio URL: ${audioUrl}`);

    // 2) HeyGen queue
    console.log('  🚀 Queuing HeyGen video...');
    const { videoId } = await heygenGenerateVideo(talkingPhotoId, audioUrl);
    console.log(`  ⏳ HeyGen video_id: ${videoId}`);

    // 3) Record to DB (schema tolerant)
    console.log('  💾 Recording generation in kelly_video_assets...');
    await upsertKellyVideoAsset({
      day,
      phase: assetPhase,
      ageBucket: normalizeAgeBucketForDb(age),
      language: lang,
      archetype: atom.archetype,
      persona,
      script,
      audioUrl,
      heygenVideoId: videoId,
      sourceImagePath,
    });

    // Rate limit safety
    await new Promise(r => setTimeout(r, 1500));
  }

  console.log('\n✅ HeyGen jobs queued (or dry-run complete)');
}

main().catch(err => {
  console.error('❌ Fatal error:', err);
  process.exit(1);
});
