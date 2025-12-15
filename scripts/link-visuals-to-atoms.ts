#!/usr/bin/env npx tsx
/**
 * 🔗 LINK VISUALS TO ATOMS
 * 
 * Updates lesson_atoms.visual_url with URLs from kelly_video_assets.
 * This makes infographics appear in the UI when users click the 📊 button.
 * 
 * Usage:
 *   npx tsx scripts/link-visuals-to-atoms.ts --range=1-50
 *   npx tsx scripts/link-visuals-to-atoms.ts --day=1
 *   npx tsx scripts/link-visuals-to-atoms.ts --all
 *   npx tsx scripts/link-visuals-to-atoms.ts --dry-run
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Phase name mapping
const PHASE_MAP: Record<string, string> = {
  'hook': 'Hook',
  'q1': 'Fact1',
  'q2': 'Fact2',
  'q3': 'Fact3',
  'wisdom': 'Wisdom',
};

const REVERSE_PHASE_MAP: Record<string, string> = {
  'Hook': 'hook',
  'Fact1': 'q1',
  'Fact2': 'q2',
  'Fact3': 'q3',
  'Wisdom': 'wisdom',
};

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

async function getLessonsInRange(start: number, end: number) {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic')
    .gte('day_number', start)
    .lte('day_number', end)
    .order('day_number');
  
  if (error) throw error;
  return data || [];
}

async function getAssetsForDay(dayNumber: number) {
  const { data, error } = await supabase
    .from('kelly_video_assets')
    .select('phase, asset_type, public_url')
    .eq('day_number', dayNumber)
    .eq('asset_type', 'image');
  
  if (error) throw error;
  return data || [];
}

async function getAtomsForLesson(lessonId: string) {
  const { data, error } = await supabase
    .from('lesson_atoms')
    .select('id, phase, archetype, visual_url')
    .eq('core_lesson_id', lessonId);
  
  if (error) throw error;
  return data || [];
}

async function updateAtomVisualUrl(atomId: string, visualUrl: string) {
  const { error } = await supabase
    .from('lesson_atoms')
    .update({ visual_url: visualUrl })
    .eq('id', atomId);
  
  return !error;
}

async function processDay(dayNumber: number, dryRun: boolean) {
  console.log(`\n📚 Day ${dayNumber}`);
  
  // Get lesson
  const lessons = await getLessonsInRange(dayNumber, dayNumber);
  if (lessons.length === 0) {
    console.log('  ⚠️ No lesson found');
    return { linked: 0, skipped: 0 };
  }
  
  const lesson = lessons[0];
  console.log(`  Topic: ${lesson.topic}`);
  
  // Get assets for this day
  const assets = await getAssetsForDay(dayNumber);
  if (assets.length === 0) {
    console.log('  ⚠️ No image assets found in kelly_video_assets');
    return { linked: 0, skipped: 0 };
  }
  
  console.log(`  Found ${assets.length} image assets`);
  
  // Build asset map (phase -> URL)
  const assetMap: Record<string, string> = {};
  assets.forEach(asset => {
    const atomPhase = PHASE_MAP[asset.phase];
    if (atomPhase) {
      assetMap[atomPhase] = asset.public_url;
    }
  });
  
  // Get atoms for this lesson
  const atoms = await getAtomsForLesson(lesson.id);
  console.log(`  Found ${atoms.length} atoms`);
  
  let linked = 0;
  let skipped = 0;
  
  // Update each atom with matching visual URL
  for (const atom of atoms) {
    const visualUrl = assetMap[atom.phase];
    
    if (!visualUrl) {
      console.log(`    ⏭️ ${atom.archetype} ${atom.phase}: No matching asset`);
      continue;
    }
    
    if (atom.visual_url === visualUrl) {
      console.log(`    ⏭️ ${atom.archetype} ${atom.phase}: Already linked`);
      skipped++;
      continue;
    }
    
    if (dryRun) {
      console.log(`    🔗 ${atom.archetype} ${atom.phase}: Would link to ${visualUrl.substring(visualUrl.length - 30)}`);
      linked++;
    } else {
      const success = await updateAtomVisualUrl(atom.id, visualUrl);
      if (success) {
        console.log(`    ✅ ${atom.archetype} ${atom.phase}: Linked`);
        linked++;
      } else {
        console.log(`    ❌ ${atom.archetype} ${atom.phase}: Update failed`);
      }
    }
  }
  
  return { linked, skipped };
}

async function processRange(start: number, end: number, dryRun: boolean) {
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  🔗 LINK VISUALS TO ATOMS`);
  console.log(`  Days ${start} to ${end} ${dryRun ? '(DRY RUN)' : ''}`);
  console.log(`${'█'.repeat(60)}`);
  
  const totals = { linked: 0, skipped: 0, days: 0 };
  
  for (let day = start; day <= end; day++) {
    const result = await processDay(day, dryRun);
    totals.linked += result.linked;
    totals.skipped += result.skipped;
    totals.days++;
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📊 RESULTS`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`📅 Days processed: ${totals.days}`);
  console.log(`✅ Atoms linked: ${totals.linked}`);
  console.log(`⏭️ Already linked: ${totals.skipped}`);
  
  if (dryRun) {
    console.log(`\n💡 Run without --dry-run to actually update database`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  
  const dayArg = args.find(a => a.startsWith('--day='));
  const rangeArg = args.find(a => a.startsWith('--range='));
  const allArg = args.includes('--all');
  
  if (dayArg) {
    const day = parseInt(dayArg.split('=')[1]);
    await processRange(day, day, dryRun);
  } else if (rangeArg) {
    const [start, end] = rangeArg.split('=')[1].split('-').map(Number);
    await processRange(start, end, dryRun);
  } else if (allArg) {
    await processRange(1, 365, dryRun);
  } else {
    console.log(`
🔗 LINK VISUALS TO ATOMS

This script connects existing images in kelly_video_assets
to lesson_atoms.visual_url so they appear in the UI.

Usage:
  npx tsx scripts/link-visuals-to-atoms.ts --all          # All 365 days
  npx tsx scripts/link-visuals-to-atoms.ts --range=1-50   # Day range
  npx tsx scripts/link-visuals-to-atoms.ts --day=1        # Single day
  npx tsx scripts/link-visuals-to-atoms.ts --dry-run      # Preview

Environment Required:
  SUPABASE_URL (or PUBLIC_SUPABASE_URL)
  SUPABASE_SERVICE_ROLE_KEY
    `);
  }
}

main().catch(console.error);
