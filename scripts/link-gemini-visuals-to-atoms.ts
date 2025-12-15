#!/usr/bin/env npx tsx
/**
 * 🔗 LINK GEMINI VISUALS TO ATOMS
 * 
 * The generate-lesson-visuals.ts script uploads to lesson-visuals bucket
 * but doesn't link to lesson_atoms. This script does that linking.
 * 
 * Gemini generates: thumbnail, illustration, infographic-1, infographic-2, etc.
 * We map these to lesson phases:
 *   - infographic-1 → Hook
 *   - infographic-2 → Fact1
 *   - infographic-3 → Fact2
 *   - (or use illustration for all phases as fallback)
 * 
 * Usage:
 *   npx tsx scripts/link-gemini-visuals-to-atoms.ts --range=6-10
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

const STORAGE_BASE = `${SUPABASE_URL}/storage/v1/object/public/lesson-visuals`;

// Map infographic index to phase
const INFOGRAPHIC_TO_PHASE: Record<number, string> = {
  1: 'Hook',
  2: 'Fact1',
  3: 'Fact2',
  4: 'Fact3',
  5: 'Wisdom',
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

async function checkIfFileExists(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

async function processDay(dayNumber: number, dryRun: boolean) {
  console.log(`\n📚 Day ${dayNumber}`);
  
  const lessons = await getLessonsInRange(dayNumber, dayNumber);
  if (lessons.length === 0) {
    console.log('  ⚠️ No lesson found');
    return { linked: 0, skipped: 0 };
  }
  
  const lesson = lessons[0];
  console.log(`  Topic: ${lesson.topic}`);
  
  const paddedDay = String(dayNumber).padStart(3, '0');
  
  // Check what Gemini visuals exist
  const baseUrl = `${STORAGE_BASE}/day-${paddedDay}`;
  const illustration = `${baseUrl}/illustration.png`;
  const infographics: string[] = [];
  
  // Check for infographic-1, infographic-2, etc.
  for (let i = 1; i <= 5; i++) {
    const url = `${baseUrl}/infographic-${i}.png`;
    const exists = await checkIfFileExists(url);
    if (exists) {
      infographics.push(url);
      console.log(`  ✅ Found: infographic-${i}.png`);
    }
  }
  
  // Check illustration
  const illustrationExists = await checkIfFileExists(illustration);
  if (illustrationExists) {
    console.log(`  ✅ Found: illustration.png`);
  }
  
  if (infographics.length === 0 && !illustrationExists) {
    console.log('  ⚠️ No Gemini visuals found for this day');
    return { linked: 0, skipped: 0 };
  }
  
  // Get atoms for this lesson
  const atoms = await getAtomsForLesson(lesson.id);
  console.log(`  Found ${atoms.length} atoms`);
  
  let linked = 0;
  let skipped = 0;
  
  // Strategy: Map infographics to phases
  // If we have 2 infographics, use them for Hook and Fact1
  // If we have 3+, map to Hook, Fact1, Fact2, etc.
  // Use illustration as fallback for remaining phases
  
  for (const atom of atoms) {
    let visualUrl = '';
    
    // Map phase to infographic
    if (atom.phase === 'Hook' && infographics[0]) {
      visualUrl = infographics[0];
    } else if (atom.phase === 'Fact1' && infographics[1]) {
      visualUrl = infographics[1];
    } else if (atom.phase === 'Fact2' && infographics[2]) {
      visualUrl = infographics[2];
    } else if (atom.phase === 'Fact3' && infographics[3]) {
      visualUrl = infographics[3];
    } else if (atom.phase === 'Wisdom' && infographics[4]) {
      visualUrl = infographics[4];
    } else if (illustrationExists) {
      // Fallback: use illustration for all phases
      visualUrl = illustration;
    }
    
    if (!visualUrl) {
      continue;
    }
    
    if (atom.visual_url === visualUrl) {
      skipped++;
      continue;
    }
    
    if (dryRun) {
      console.log(`    🔗 ${atom.archetype} ${atom.phase}: Would link`);
      linked++;
    } else {
      const success = await updateAtomVisualUrl(atom.id, visualUrl);
      if (success) {
        linked++;
      }
    }
  }
  
  if (!dryRun && linked > 0) {
    console.log(`  ✅ Linked ${linked} atoms`);
  }
  
  return { linked, skipped };
}

async function processRange(start: number, end: number, dryRun: boolean) {
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  🔗 LINK GEMINI VISUALS TO ATOMS`);
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
🔗 LINK GEMINI VISUALS TO ATOMS

Links Gemini-generated visuals from lesson-visuals bucket to lesson_atoms.

Usage:
  npx tsx scripts/link-gemini-visuals-to-atoms.ts --range=6-50
  npx tsx scripts/link-gemini-visuals-to-atoms.ts --day=6
  npx tsx scripts/link-gemini-visuals-to-atoms.ts --all
  npx tsx scripts/link-gemini-visuals-to-atoms.ts --dry-run
    `);
  }
}

main().catch(console.error);
