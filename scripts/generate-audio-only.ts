#!/usr/bin/env npx tsx
/**
 * 🎙️ AUDIO-ONLY GENERATION SCRIPT
 * 
 * Generates ElevenLabs audio for all 7 phases without HeyGen video generation.
 * Uploads to Supabase storage for later SadTalker video creation.
 * 
 * Usage:
 *   npx tsx scripts/generate-audio-only.ts --day=355
 *   npx tsx scripts/generate-audio-only.ts --days=355,356
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY!;
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Phase mapping from lesson_atoms to video phases
const PHASE_MAP: Record<string, string> = {
  'Hook': 'hook',
  'Cliff': 'cliff',
  'Fact1': 'q1',
  'Fact2': 'q2', 
  'Fact3': 'q3',
  'Wisdom': 'wisdom',
  'Outro': 'outro',
};

const TARGET_ARCHETYPE = 'The Explorer';

function parseArgs(): { days: number[] } {
  const args = process.argv.slice(2);
  let days: number[] = [];
  
  for (const arg of args) {
    if (arg.startsWith('--day=')) {
      days = [parseInt(arg.split('=')[1], 10)];
    } else if (arg.startsWith('--days=')) {
      days = arg.split('=')[1].split(',').map(d => parseInt(d.trim(), 10));
    }
  }
  
  return { days };
}

async function generateElevenLabsAudio(script: string): Promise<Buffer> {
  if (!ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY not set');
  }
  
  const resp = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}?output_format=mp3_44100_192`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY,
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
    const error = await resp.text();
    throw new Error(`ElevenLabs API error: ${error}`);
  }
  
  return Buffer.from(await resp.arrayBuffer());
}

async function uploadAudio(buffer: Buffer, day: number, phase: string): Promise<string> {
  const timestamp = Date.now();
  const fileName = `heygen/audio/day_${day}_${phase}_${timestamp}.mp3`;
  
  const { error } = await supabase.storage
    .from('kelly-templates')
    .upload(fileName, buffer, { contentType: 'audio/mpeg', upsert: true });
  
  if (error) throw new Error(`Audio upload failed: ${error.message}`);
  
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(fileName);
  return data.publicUrl;
}

async function getAtoms(day: number): Promise<Array<{phase: string; script: string}>> {
  // Get atoms directly with join - picks the core_lesson that actually has atoms
  const { data: atoms, error: atomErr } = await supabase
    .from('lesson_atoms')
    .select(`
      phase, 
      content,
      core_lessons!inner(day_number)
    `)
    .eq('core_lessons.day_number', day)
    .eq('archetype', TARGET_ARCHETYPE);
  
  if (atomErr) {
    throw new Error(`Error fetching atoms for day ${day}: ${atomErr.message}`);
  }
  
  if (!atoms || atoms.length === 0) {
    console.log(`  ⚠️  No atoms found for day ${day} with archetype "${TARGET_ARCHETYPE}"`);
    return [];
  }
  
  return atoms
    .filter(a => a.content?.script && PHASE_MAP[a.phase])
    .map(a => ({
      phase: PHASE_MAP[a.phase],
      script: a.content.script
    }));
}

async function processDay(day: number): Promise<{success: number; failed: number}> {
  console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  console.log(`📅 DAY ${day}`);
  console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  
  const atoms = await getAtoms(day);
  console.log(`Found ${atoms.length} phases with scripts`);
  
  let success = 0;
  let failed = 0;
  
  for (const atom of atoms) {
    console.log(`\n🎤 ${atom.phase}: ${atom.script.substring(0, 50)}...`);
    
    try {
      console.log('   🔊 Generating audio...');
      const audioBuffer = await generateElevenLabsAudio(atom.script);
      
      console.log('   📤 Uploading...');
      const audioUrl = await uploadAudio(audioBuffer, day, atom.phase);
      
      console.log(`   ✅ Done: ${audioUrl.substring(audioUrl.lastIndexOf('/') + 1)}`);
      success++;
      
      // Rate limit delay
      await new Promise(r => setTimeout(r, 500));
    } catch (error) {
      console.log(`   ❌ Failed: ${(error as Error).message}`);
      failed++;
    }
  }
  
  return { success, failed };
}

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║           🎙️ AUDIO-ONLY GENERATION                           ║
╚══════════════════════════════════════════════════════════════╝
`);

  if (!ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY not set');
    process.exit(1);
  }

  const { days } = parseArgs();
  
  if (days.length === 0) {
    console.log('Usage: npx tsx scripts/generate-audio-only.ts --days=355,356');
    process.exit(1);
  }

  console.log(`Target Days: ${days.join(', ')}`);
  console.log(`Archetype: ${TARGET_ARCHETYPE}`);
  console.log(`Voice ID: ${ELEVENLABS_VOICE_ID}`);

  const results: Record<number, {success: number; failed: number}> = {};

  for (const day of days) {
    results[day] = await processDay(day);
  }

  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                        📊 SUMMARY                            ║
╚══════════════════════════════════════════════════════════════╝
`);

  let totalSuccess = 0;
  let totalFailed = 0;

  for (const [day, result] of Object.entries(results)) {
    const icon = result.failed === 0 ? '✅' : '🟡';
    console.log(`  ${icon} Day ${day}: ${result.success}/7 audio files`);
    totalSuccess += result.success;
    totalFailed += result.failed;
  }

  console.log(`
Total: ${totalSuccess} success, ${totalFailed} failed

✨ Next steps:
   1. Generate videos: npx tsx scripts/generate-sadtalker.ts --days=${days.join(',')}
   2. Sync database: npx tsx scripts/sync-bucket-to-database.ts
   3. Verify: npx tsx scripts/verify-day-ready.ts --days=${days.join(',')}
`);
}

main().catch(console.error);
