#!/usr/bin/env npx tsx
/**
 * CURIOUS KELLY - GENERATE DAY AUDIO (ELEVENLABS) 
 *
 * Generates precomputed MP3 audio from lesson_atoms.content.script using ElevenLabs,
 * uploads to Supabase Storage, and registers rows in kelly_video_assets.
 *
 * This enables launch-mode: static persona head + audio now, video later.
 *
 * Usage:
 *   npx tsx scripts/generate-day-audio-elevenlabs.ts --day=1 --age=adult --lang=en
 *   npx tsx scripts/generate-day-audio-elevenlabs.ts --day=1 --age=adult --lang=en --all
 *   npx tsx scripts/generate-day-audio-elevenlabs.ts --day=1 --age=adult --lang=en --dry-run
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY =
  process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

const AUDIO_BUCKET = process.env.KELLY_AUDIO_BUCKET || 'kelly-templates';

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

const PHASE_TO_DB_PHASE: Record<string, string> = {
  Hook: 'hook',
  Fact1: 'q1',
  Fact2: 'q2',
  Fact3: 'q3',
  Wisdom: 'wisdom',
};

type AgeVariant = 'kid' | 'teen' | 'adult' | 'elder' | 'super_elder';

function getArg(name: string, fallback?: string) {
  const arg = process.argv.slice(2).find(a => a.startsWith(`--${name}=`));
  return arg ? arg.split('=')[1] : fallback;
}

function hasFlag(name: string) {
  return process.argv.slice(2).includes(`--${name}`);
}

function safeSlug(s: string) {
  return s.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_\-]/g, '');
}

async function elevenlabsTtsMp3(text: string): Promise<Buffer> {
  const resp = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}?output_format=mp3_44100_192`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY!,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
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

async function uploadAudio(storagePath: string, audio: Buffer): Promise<string> {
  const { error: upErr } = await supabase.storage
    .from(AUDIO_BUCKET)
    .upload(storagePath, audio, { upsert: true, contentType: 'audio/mpeg' });

  if (upErr) throw new Error(`Supabase upload failed: ${upErr.message}`);

  const { data } = supabase.storage.from(AUDIO_BUCKET).getPublicUrl(storagePath);
  if (!data?.publicUrl) throw new Error('Supabase getPublicUrl failed');
  return data.publicUrl;
}

async function upsertAudioAsset(params: {
  dayNumber: number;
  dbPhase: string;
  persona: string;
  ageVariant: AgeVariant;
  language: string;
  publicUrl: string;
  storagePath: string;
  scriptText: string;
}) {
  const now = new Date().toISOString();

  const payload: any = {
    day_number: params.dayNumber,
    phase: params.dbPhase,
    template: params.persona,
    asset_type: 'audio',
    age_bucket: params.ageVariant,
    language: params.language,
    storage_bucket: AUDIO_BUCKET,
    storage_path: params.storagePath,
    public_url: params.publicUrl,
    status: 'validated',
    generation_prompt: params.scriptText,
    created_at: now,
    updated_at: now,
  };

  const { error } = await supabase
    .from('kelly_video_assets')
    .upsert(payload, { onConflict: 'day_number,phase,template,asset_type,age_bucket,language' as any });

  if (error) {
    throw new Error(`DB upsert failed: ${error.message}`);
  }
}

async function main() {
  const dayNumber = parseInt(getArg('day', '1')!, 10);
  const ageVariant = (getArg('age', 'adult') as AgeVariant) || 'adult';
  const language = getArg('lang', 'en')!;
  const dryRun = hasFlag('dry-run') || hasFlag('dryRun');
  const all = hasFlag('all');

  const defaultArchetypes = ['The Scientist', 'The Explorer', 'The Rebel'];
  const archetypes = all ? Object.keys(ARCHETYPE_TO_PERSONA) : defaultArchetypes;

  console.log('==========================================');
  console.log('ELEVENLABS AUDIO GENERATION');
  console.log(`Day: ${dayNumber} | Age: ${ageVariant} | Lang: ${language}`);
  console.log(`Archetypes: ${archetypes.join(', ')}`);
  console.log(`Mode: ${dryRun ? 'DRY RUN' : 'LIVE'}`);
  console.log('==========================================\n');

  // Resolve lesson
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();

  if (lessonError || !lesson) {
    throw new Error(`core_lessons lookup failed for day ${dayNumber}: ${lessonError?.message || 'not found'}`);
  }

  // Fetch atoms for the lesson
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('id, phase, archetype, content')
    .eq('core_lesson_id', lesson.id);

  if (atomsError || !atoms?.length) {
    throw new Error(`lesson_atoms query failed: ${atomsError?.message || 'no atoms'}`);
  }

  const targets = atoms
    .filter(a => archetypes.includes(a.archetype))
    .filter(a => a.phase in PHASE_TO_DB_PHASE);

  console.log(`Found ${targets.length} atoms matching archetypes+phases\n`);

  for (const atom of targets) {
    const persona = ARCHETYPE_TO_PERSONA[atom.archetype] || 'scientist';
    const dbPhase = PHASE_TO_DB_PHASE[atom.phase] || 'hook';
    const scriptText = (atom as any).content?.script || (atom as any).content?.text || '';

    if (!scriptText) {
      console.log(`⚠️ Skipping ${atom.archetype}/${atom.phase}: no content.script`);
      continue;
    }

    const dayStr = String(dayNumber).padStart(3, '0');
    const storagePath = `heygen/audio/day_${dayStr}/${ageVariant}/${persona}/${safeSlug(dbPhase)}_${language}.mp3`;

    console.log(`🎤 ${atom.archetype} | ${atom.phase} -> ${dbPhase} | persona=${persona}`);
    console.log(`    path: ${storagePath}`);

    if (dryRun) continue;

    const audio = await elevenlabsTtsMp3(scriptText);
    const url = await uploadAudio(storagePath, audio);
    await upsertAudioAsset({
      dayNumber,
      dbPhase,
      persona,
      ageVariant,
      language,
      publicUrl: url,
      storagePath,
      scriptText,
    });

    console.log(`    ✅ uploaded + registered: ${url}`);

    // polite pacing
    await new Promise(r => setTimeout(r, 750));
  }

  console.log('\n✅ Done');
}

main().catch(err => {
  console.error('❌ Fatal error:', err);
  process.exit(1);
});
