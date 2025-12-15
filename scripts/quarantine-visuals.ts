#!/usr/bin/env npx tsx
/**
 * 🚨 QUARANTINE VISUALS
 * 
 * Clears lesson_atoms.visual_url for a day range.
 * Use this when visuals are not production quality.
 * 
 * Usage:
 *   npx tsx scripts/quarantine-visuals.ts --range=1-50
 *   npx tsx scripts/quarantine-visuals.ts --all
 *   npx tsx scripts/quarantine-visuals.ts --dry-run --range=1-50
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_URL || !SERVICE_KEY) {
  console.error('❌ Missing PUBLIC_SUPABASE_URL/SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_KEY);

function parseRange(range: string): { start: number; end: number } {
  const [a, b] = range.split('-').map((x) => Number(x));
  if (!a || !b || a < 1 || b < a) throw new Error(`Invalid range: ${range}`);
  return { start: a, end: b };
}

async function getLessonIds(start: number, end: number): Promise<string[]> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('id')
    .gte('day_number', start)
    .lte('day_number', end);

  if (error) throw error;
  return (data || []).map((d: any) => d.id);
}

async function countAtoms(lessonIds: string[]) {
  const { count, error } = await supabase
    .from('lesson_atoms')
    .select('id', { count: 'exact', head: true })
    .in('core_lesson_id', lessonIds);

  if (error) throw error;
  return count || 0;
}

async function clearVisualUrls(lessonIds: string[]) {
  const { error } = await supabase
    .from('lesson_atoms')
    .update({ visual_url: null })
    .in('core_lesson_id', lessonIds);

  if (error) throw error;
}

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const all = args.includes('--all');
  const rangeArg = args.find((a) => a.startsWith('--range='));

  let start = 1;
  let end = 365;

  if (!all) {
    if (!rangeArg) {
      console.log('Usage: --range=1-50 or --all [--dry-run]');
      process.exit(1);
    }
    ({ start, end } = parseRange(rangeArg.split('=')[1]));
  }

  console.log(`\n🚨 QUARANTINE VISUALS ${dryRun ? '(DRY RUN)' : ''}`);
  console.log(`Days: ${start}-${end}`);

  const lessonIds = await getLessonIds(start, end);
  console.log(`Core lessons found: ${lessonIds.length}`);

  if (lessonIds.length === 0) {
    console.log('Nothing to do.');
    return;
  }

  const atomCount = await countAtoms(lessonIds);
  console.log(`Lesson atoms to update: ${atomCount}`);

  if (dryRun) {
    console.log('\n✅ Dry run complete. Run without --dry-run to clear visual_url.');
    return;
  }

  await clearVisualUrls(lessonIds);
  console.log('\n✅ Cleared lesson_atoms.visual_url for selected days.');
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
