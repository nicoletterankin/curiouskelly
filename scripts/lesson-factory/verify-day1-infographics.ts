#!/usr/bin/env npx tsx
/**
 * ✅ VERIFY DAY INFOGRAPHICS IN SUPABASE
 *
 * Checks that:
 *   - Exactly 5 infographic image assets exist in lesson_assets for the given day
 *   - Each phase (hook, fact1, fact2, fact3, wisdom) has one asset
 *   - Each asset's storage_path exists in storage.objects (lesson-visuals bucket)
 *   - lesson_atoms.visual_url is populated for atoms per phase
 *
 * Usage:
 *   # Default: Day 1
 *   npx tsx scripts/lesson-factory/verify-day1-infographics.ts
 *
 *   # Explicit day
 *   npx tsx scripts/lesson-factory/verify-day1-infographics.ts --day 7
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const DEFAULT_DAY = 1;
const BUCKET_NAME = 'lesson-visuals';

const PHASES = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'] as const;
type PhaseName = (typeof PHASES)[number];

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

if (!CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_KEY) {
  // eslint-disable-next-line no-console
  console.error('❌ Missing Supabase credentials (PUBLIC_SUPABASE_URL / SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)');
  process.exit(1);
}

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

async function getCoreLessonId(dayNumber: number): Promise<string> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();

  if (error || !data) {
    throw new Error(`Core lesson not found for day ${dayNumber}: ${error?.message ?? 'no data'}`);
  }

  return data.id as string;
}

async function verifyLessonAssets(coreLessonId: string) {
  const { data, error } = await supabase
    .from('lesson_assets')
    .select('id, asset_type, asset_subtype, variant_name, storage_path, public_url')
    .eq('lesson_id', coreLessonId)
    .eq('asset_type', 'image')
    .eq('asset_subtype', 'infographic');

  if (error) {
    throw new Error(`Failed to query lesson_assets: ${error.message}`);
  }

  const assets = data ?? [];
  console.log(`\n🧾 lesson_assets: found ${assets.length} infographic image rows for Day ${DAY_NUMBER}`);

  const expectedCount = PHASES.length;
  const variantMap = new Map<string, typeof assets[number]>();
  for (const asset of assets) {
    if (asset.variant_name) {
      variantMap.set(String(asset.variant_name), asset);
    }
  }

  let pass = assets.length === expectedCount;
  if (!pass) {
    console.log(`   ⚠️ Expected ${expectedCount} rows, got ${assets.length}`);
  }

  const missingPhases: PhaseName[] = [];

  for (const phase of PHASES) {
    const key = phase.toLowerCase();
    const asset = variantMap.get(key);
    if (!asset) {
      missingPhases.push(phase);
    }
  }

  if (missingPhases.length > 0) {
    pass = false;
    console.log(`   ❌ Missing lesson_assets for phases: ${missingPhases.join(', ')}`);
  } else {
    console.log('   ✅ All 5 phases present in lesson_assets');
  }

  return { pass, assets };
}

async function verifyStorageObjects(assets: { storage_path: string | null }[]) {
  const paths = assets
    .map(a => a.storage_path)
    .filter((p): p is string => typeof p === 'string' && p.length > 0);

  if (paths.length === 0) {
    console.log('\n📦 Storage: no storage_path values to verify');
    return { pass: false, missing: [] as string[] };
  }

  const { data, error } = await supabase
    .from('storage.objects')
    .select('name')
    .eq('bucket_id', BUCKET_NAME)
    .in('name', paths);

  if (error) {
    throw new Error(`Failed to query storage.objects: ${error.message}`);
  }

  const existingNames = new Set<string>((data ?? []).map(row => row.name as string));
  const missing: string[] = [];

  for (const p of paths) {
    if (!existingNames.has(p)) {
      missing.push(p);
    }
  }

  if (missing.length === 0) {
    console.log('\n📦 Storage: ✅ all infographic storage_path files exist in bucket lesson-visuals');
    return { pass: true, missing };
  }

  console.log('\n📦 Storage: ❌ missing the following paths in lesson-visuals:');
  for (const m of missing) {
    console.log(`   - ${m}`);
  }

  return { pass: false, missing };
}

async function verifyLessonAtoms(coreLessonId: string) {
  const { data, error } = await supabase
    .from('lesson_atoms')
    .select('id, phase, visual_url')
    .eq('core_lesson_id', coreLessonId);

  if (error) {
    throw new Error(`Failed to query lesson_atoms: ${error.message}`);
  }

  const atoms = data ?? [];
  const { data: lessonRow } = await supabase
    .from('core_lessons')
    .select('day_number')
    .eq('id', coreLessonId)
    .single();

  const dayLabel = lessonRow?.day_number ?? 'unknown';

  console.log(`\n🧬 lesson_atoms: ${atoms.length} atoms for Day ${dayLabel}`);

  let pass = true;
  for (const phase of PHASES) {
    const phaseAtoms = atoms.filter(a => a.phase === phase);
    const withVisual = phaseAtoms.filter(a => a.visual_url && String(a.visual_url).length > 0);

    if (phaseAtoms.length === 0) {
      console.log(`   ⚠️ No atoms found for phase ${phase}`);
      pass = false;
      continue;
    }

    if (withVisual.length === 0) {
      console.log(`   ❌ Phase ${phase}: ${phaseAtoms.length} atoms, NONE have visual_url set`);
      pass = false;
    } else {
      console.log(
        `   ✅ Phase ${phase}: ${withVisual.length}/${phaseAtoms.length} atoms have visual_url populated`
      );
    }
  }

  return { pass };
}

async function main() {
  console.log('═'.repeat(72));
  console.log('✅ VERIFY DAY INFOGRAPHICS IN SUPABASE');
  console.log('═'.repeat(72));

  // CLI args
  const args = process.argv.slice(2);
  let dayNumber = DEFAULT_DAY;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--day') {
      const value = args[i + 1];
      if (!value) {
        console.error('❌ Missing value for --day');
        process.exit(1);
      }
      dayNumber = Number.parseInt(value, 10);
      i++;
    }
  }

  console.log(`📅 Day: ${dayNumber}`);
  console.log(`🪣 Bucket: ${BUCKET_NAME}`);

  const coreLessonId = await getCoreLessonId(dayNumber);
  console.log(`\n✅ core_lessons.id for Day ${dayNumber}: ${coreLessonId}`);

  const assetResult = await verifyLessonAssets(coreLessonId);
  const storageResult = await verifyStorageObjects(assetResult.assets);
  const atomsResult = await verifyLessonAtoms(coreLessonId);

  const allPass = assetResult.pass && storageResult.pass && atomsResult.pass;

  console.log('\n' + '-'.repeat(72));
  if (allPass) {
    console.log('🎉 OVERALL STATUS: ✅ Day 1 infographics are fully wired for production');
  } else {
    console.log('⚠️ OVERALL STATUS: Issues detected with Day 1 infographics');
  }
  console.log('-'.repeat(72));

  if (!allPass) {
    process.exit(1);
  }
}

// eslint-disable-next-line unicorn/prefer-top-level-await
main().catch(err => {
  // eslint-disable-next-line no-console
  console.error('❌ Unexpected error in verify-day1-infographics:', err);
  process.exit(1);
});


