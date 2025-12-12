#!/usr/bin/env npx tsx
/**
 * CURIOUS KELLY - DAY N HEYGEN AUDIT
 *
 * Audits:
 * - Supabase lesson_atoms for a given day
 * - Local HeyGen talking_photo_id maps for age variants
 * - Existing rows in kelly_video_assets (schema-tolerant)
 *
 * Usage:
 *   npx tsx scripts/audit-day1-for-heygen.ts --day=1
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY =
  process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error('❌ Missing Supabase env vars. Required one of:');
  console.error('   - SUPABASE_URL or PUBLIC_SUPABASE_URL');
  console.error('   - SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

const CANONICAL_PERSONAS = [
  'architect',
  'diplomat',
  'empath',
  'explorer',
  'macgyver',
  'mystic',
  'provider',
  'rebel',
  'scientist',
  'storyteller',
  'strategist',
  'survivor',
] as const;

const CANONICAL_AGE_VARIANTS = ['kid', 'teen', 'adult', 'elder', 'super_elder'] as const;

function getArg(name: string, fallback?: string) {
  const arg = process.argv.slice(2).find(a => a.startsWith(`--${name}=`));
  return arg ? arg.split('=')[1] : fallback;
}

function readJsonIfExists(p: string): any | null {
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf-8'));
  } catch {
    return null;
  }
}

function summarizeCounts<T extends string>(items: T[]) {
  const out: Record<string, number> = {};
  for (const i of items) out[i] = (out[i] || 0) + 1;
  return out;
}

async function countKellyVideoAssets(day: number) {
  // Try day_number schema
  {
    const { count, error } = await supabase
      .from('kelly_video_assets')
      .select('*', { count: 'exact', head: true })
      // @ts-expect-error - schema varies
      .eq('day_number', day);

    if (!error) return { dayColumn: 'day_number' as const, count: count || 0 };
  }

  // Fallback: lesson_day schema
  {
    const { count, error } = await supabase
      .from('kelly_video_assets')
      .select('*', { count: 'exact', head: true })
      // @ts-expect-error - schema varies
      .eq('lesson_day', day);

    if (!error) return { dayColumn: 'lesson_day' as const, count: count || 0 };
  }

  return { dayColumn: null as const, count: 0 };
}

async function main() {
  const day = parseInt(getArg('day', '1')!, 10);

  console.log('=== CURIOUS KELLY - HEYGEN AUDIT ===');
  console.log(`Day: ${day}`);

  // 1) Resolve core lesson
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id, day_number')
    .eq('day_number', day)
    .single();

  if (lessonError || !lesson) {
    console.error('❌ core_lessons lookup failed:', lessonError?.message || 'not found');
    process.exit(1);
  }

  // 2) Lesson atoms
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('id, archetype, phase, content')
    .eq('core_lesson_id', lesson.id);

  if (atomsError) {
    console.error('❌ lesson_atoms query failed:', atomsError.message);
    process.exit(1);
  }

  console.log(`\n[Supabase] lesson_atoms for Day ${day}: ${atoms?.length || 0}`);

  const byArchetype: Record<string, number> = {};
  const byPhase: Record<string, number> = {};

  for (const a of atoms || []) {
    byArchetype[a.archetype] = (byArchetype[a.archetype] || 0) + 1;
    byPhase[a.phase] = (byPhase[a.phase] || 0) + 1;
  }

  console.log('\nAtoms by archetype:');
  for (const [k, v] of Object.entries(byArchetype).sort((a, b) => a[0].localeCompare(b[0]))) {
    console.log(`  ${k}: ${v}`);
  }

  console.log('\nAtoms by phase:');
  for (const [k, v] of Object.entries(byPhase).sort((a, b) => a[0].localeCompare(b[0]))) {
    console.log(`  ${k}: ${v}`);
  }

  // 3) HeyGen talking photo IDs
  console.log('\n[HeyGen] Talking Photo ID maps (local files)');

  for (const age of CANONICAL_AGE_VARIANTS) {
    const idMapPath = path.join(
      process.cwd(),
      'generated-images',
      'kelly-archetypes-head-only',
      'age',
      age,
      'heygen_talking_photo_ids.json'
    );

    const map = readJsonIfExists(idMapPath);
    if (!map) {
      console.log(`  ${age}: ❌ missing or invalid JSON at ${idMapPath}`);
      continue;
    }

    const keys = Object.keys(map);
    const missingKeys = CANONICAL_PERSONAS.filter(p => !(p in map));
    const placeholderKeys = CANONICAL_PERSONAS.filter(p => map[p] === 'PASTE_ID_HERE' || !map[p]);

    console.log(
      `  ${age}: keys=${keys.length} | missingKeys=${missingKeys.length} | missingIds=${placeholderKeys.length}`
    );

    if (missingKeys.length) console.log(`    missing keys: ${missingKeys.join(', ')}`);
    if (placeholderKeys.length) console.log(`    missing IDs: ${placeholderKeys.join(', ')}`);
  }

  // 4) Existing kelly_video_assets rows
  const kva = await countKellyVideoAssets(day);
  console.log(`\n[kelly_video_assets] Day ${day} rows: ${kva.count} (day column: ${kva.dayColumn || 'unknown'})`);

  if (kva.dayColumn) {
    const { data: rows } = await supabase
      .from('kelly_video_assets')
      .select('*')
      // @ts-expect-error - schema varies
      .eq(kva.dayColumn, day)
      .limit(500);

    if (rows?.length) {
      const statuses = rows
        .map((r: any) => r.status)
        .filter(Boolean)
        .map(String);

      console.log('  status counts:', summarizeCounts(statuses));
    }
  }

  // 5) Sample scripts
  console.log('\n=== SAMPLE SCRIPTS (first found per phase) ===');
  const phases = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'];
  for (const phase of phases) {
    const atom = (atoms || []).find(a => a.phase === phase);
    if (!atom) {
      console.log(`\n${phase}: (no atom found)`);
      continue;
    }

    const script =
      (atom as any).content?.script || (atom as any).content?.text || JSON.stringify((atom as any).content || {});

    console.log(`\n${phase} (${atom.archetype}):`);
    console.log(`"${String(script).slice(0, 220)}${String(script).length > 220 ? '...' : ''}"`);
  }

  console.log('\n✅ Audit complete');
}

main().catch(err => {
  console.error('❌ Fatal error:', err);
  process.exit(1);
});
