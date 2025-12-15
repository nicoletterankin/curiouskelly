#!/usr/bin/env npx tsx
/**
 * ✅ LINK APPROVED INFOGRAPHICS (EXISTING)
 *
 * Links *existing* approved infographics from Supabase Storage into lesson_atoms.visual_url.
 *
 * Source bucket/prefix (current approved set):
 *   lesson-visuals/day-XXX/infographics/{hook|fact1|fact2|fact3|wisdom}.png
 *
 * Usage:
 *   npx tsx scripts/link-approved-infographics.ts --day=1 --allow-db-mutation
 *   npx tsx scripts/link-approved-infographics.ts --range=1-30 --dry-run
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_URL || !SERVICE_KEY) {
  console.error('❌ Missing PUBLIC_SUPABASE_URL/SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const sb = createClient(SUPABASE_URL, SERVICE_KEY);

const BUCKET = 'lesson-visuals';

type Phase = 'Hook' | 'Fact1' | 'Fact2' | 'Fact3' | 'Wisdom';

const PHASE_TO_FILE: Record<Phase, string> = {
  Hook: 'hook.png',
  Fact1: 'fact1.png',
  Fact2: 'fact2.png',
  Fact3: 'fact3.png',
  Wisdom: 'wisdom.png',
};

function pad3(n: number) {
  return String(n).padStart(3, '0');
}

function parseRange(range: string): { start: number; end: number } {
  const [a, b] = range.split('-').map((x) => Number(x));
  if (!a || !b || a < 1 || b < a) throw new Error(`Invalid range: ${range}`);
  return { start: a, end: b };
}

async function getCoreLessonId(day: number): Promise<{ id: string; topic: string } | null> {
  const { data, error } = await sb
    .from('core_lessons')
    .select('id, topic')
    .eq('day_number', day)
    .maybeSingle();
  if (error) throw error;
  if (!data?.id) return null;
  return { id: data.id, topic: data.topic };
}

async function listInfographicFiles(day: number) {
  const prefix = `day-${pad3(day)}/infographics`;
  const { data, error } = await sb.storage.from(BUCKET).list(prefix, { limit: 50 });
  if (error) throw error;
  const names = new Set((data || []).map((f) => f.name));
  return { prefix, names };
}

function publicUrl(prefix: string, filename: string) {
  const path = `${prefix}/${filename}`;
  return sb.storage.from(BUCKET).getPublicUrl(path).data.publicUrl;
}

async function updatePhaseUrls(coreLessonId: string, phase: Phase, url: string) {
  const { error } = await sb
    .from('lesson_atoms')
    .update({ visual_url: url })
    .eq('core_lesson_id', coreLessonId)
    .eq('phase', phase);
  if (error) throw error;
}

async function countAtoms(coreLessonId: string) {
  const { count, error } = await sb
    .from('lesson_atoms')
    .select('id', { head: true, count: 'exact' })
    .eq('core_lesson_id', coreLessonId);
  if (error) throw error;
  return count || 0;
}

async function processDay(day: number, opts: { dryRun: boolean }) {
  const core = await getCoreLessonId(day);
  if (!core) {
    console.log(`Day ${day}: no core lesson`);
    return;
  }

  const { prefix, names } = await listInfographicFiles(day);
  const required = Object.values(PHASE_TO_FILE);
  const missing = required.filter((f) => !names.has(f));

  if (missing.length > 0) {
    console.log(`Day ${day} (${core.topic}): infographics incomplete (missing: ${missing.join(', ')})`);
    return;
  }

  const atoms = await countAtoms(core.id);
  console.log(`\nDay ${day} (${core.topic}) • atoms=${atoms} • prefix=${prefix}`);

  const phases: Phase[] = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'];
  for (const phase of phases) {
    const file = PHASE_TO_FILE[phase];
    const url = publicUrl(prefix, file);
    if (opts.dryRun) {
      console.log(`  [DRY] ${phase} -> ${url}`);
    } else {
      await updatePhaseUrls(core.id, phase, url);
      console.log(`  ✅ ${phase} linked`);
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const allowDbMutation = args.includes('--allow-db-mutation') || process.env.CK_ALLOW_DB_MUTATIONS === '1';

  const dayArg = args.find((a) => a.startsWith('--day='));
  const rangeArg = args.find((a) => a.startsWith('--range='));

  if (!dryRun && !allowDbMutation) {
    console.error('\n❌ Refusing to mutate DB (safety lock).');
    console.error('   Re-run with --allow-db-mutation or set CK_ALLOW_DB_MUTATIONS=1\n');
    process.exit(1);
  }

  let start = 1;
  let end = 365;

  if (dayArg) {
    start = Number(dayArg.split('=')[1]);
    end = start;
  } else if (rangeArg) {
    ({ start, end } = parseRange(rangeArg.split('=')[1]));
  } else {
    console.log('Usage: --day=1 or --range=1-30 [--dry-run] [--allow-db-mutation]');
    process.exit(1);
  }

  console.log(`\n🔎 Linking approved infographics (${dryRun ? 'DRY RUN' : 'APPLY'}) Days ${start}-${end}`);

  for (let d = start; d <= end; d++) {
    try {
      await processDay(d, { dryRun });
    } catch (e: any) {
      console.error(`Day ${d} failed: ${e?.message || String(e)}`);
    }
  }
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
