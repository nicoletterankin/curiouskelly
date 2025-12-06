/**
 * UPDATE ATOM VISUAL URLs
 * 
 * Links generated phase images to lesson_atoms in the database.
 * 
 * Strategy:
 * - Each lesson has 60 atoms (12 archetypes × 5 phases)
 * - All archetypes for the same phase share the same image
 * - URLs are relative paths to public/kelly/phases/{day}/{phase}.png
 * 
 * Usage:
 *   npx ts-node scripts/update-atom-visual-urls.ts --preview
 *   npx ts-node scripts/update-atom-visual-urls.ts --update
 *   npx ts-node scripts/update-atom-visual-urls.ts --update --range=1-50
 */

import * as dotenv from 'dotenv';
dotenv.config();

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const supabase = createClient(
  process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_KEY!
);

// Phase mapping from atom phases to image filenames
const PHASE_TO_IMAGE: Record<string, string> = {
  'Hook': 'hook.png',
  'Fact1': 'q1.png',
  'Fact2': 'q2.png',
  'Fact3': 'q3.png',
  'Wisdom': 'wisdom.png'
};

// Base URL for images (can be changed to CDN URL later)
const BASE_URL = '/kelly/phases';

interface AtomUpdate {
  atom_id: string;
  day_number: number;
  phase: string;
  archetype: string;
  visual_url: string;
}

async function getAtomsNeedingUrls(startDay?: number, endDay?: number): Promise<any[]> {
  let query = supabase
    .from('lesson_atoms')
    .select(`
      id,
      phase,
      archetype,
      core_lesson_id,
      core_lessons!inner(day_number, topic)
    `)
    .is('visual_url', null);
  
  if (startDay !== undefined && endDay !== undefined) {
    query = query
      .gte('core_lessons.day_number', startDay)
      .lte('core_lessons.day_number', endDay);
  }
  
  const { data, error } = await query.order('core_lessons(day_number)');
  
  if (error) {
    console.error('Query error:', error);
    throw error;
  }
  
  return data || [];
}

async function checkImageExists(dayNumber: number, phase: string): Promise<boolean> {
  const paddedDay = String(dayNumber).padStart(3, '0');
  const imageName = PHASE_TO_IMAGE[phase];
  
  if (!imageName) return false;
  
  const imagePath = path.join(process.cwd(), 'public', 'kelly', 'phases', paddedDay, imageName);
  return fs.existsSync(imagePath);
}

async function generateUrlForAtom(dayNumber: number, phase: string): Promise<string | null> {
  const imageName = PHASE_TO_IMAGE[phase];
  if (!imageName) return null;
  
  const paddedDay = String(dayNumber).padStart(3, '0');
  return `${BASE_URL}/${paddedDay}/${imageName}`;
}

async function previewUpdates(atoms: any[]): Promise<void> {
  console.log('\n📊 PREVIEW OF ATOM VISUAL URL UPDATES\n');
  
  // Group by day
  const byDay = new Map<number, any[]>();
  for (const atom of atoms) {
    const day = atom.core_lessons.day_number;
    if (!byDay.has(day)) byDay.set(day, []);
    byDay.get(day)!.push(atom);
  }
  
  let totalWithImages = 0;
  let totalMissingImages = 0;
  
  // Check first 10 days
  const days = Array.from(byDay.keys()).sort((a, b) => a - b).slice(0, 10);
  
  for (const day of days) {
    const dayAtoms = byDay.get(day)!;
    console.log(`\nDay ${day}: ${dayAtoms[0].core_lessons.topic}`);
    
    const phases = [...new Set(dayAtoms.map(a => a.phase))];
    for (const phase of phases) {
      const hasImage = await checkImageExists(day, phase);
      const url = await generateUrlForAtom(day, phase);
      const atomCount = dayAtoms.filter(a => a.phase === phase).length;
      
      if (hasImage) {
        console.log(`  ✅ ${phase}: ${url} (${atomCount} atoms)`);
        totalWithImages += atomCount;
      } else {
        console.log(`  ❌ ${phase}: Missing image (${atomCount} atoms)`);
        totalMissingImages += atomCount;
      }
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log(`Total atoms needing URLs: ${atoms.length}`);
  console.log(`Days with atoms needing URLs: ${byDay.size}`);
  console.log(`Sample - Atoms with available images: ${totalWithImages}`);
  console.log(`Sample - Atoms missing images: ${totalMissingImages}`);
}

async function updateAtomUrls(atoms: any[]): Promise<{ updated: number; skipped: number; errors: number }> {
  const results = { updated: 0, skipped: 0, errors: 0 };
  
  // Group by day for batch processing
  const byDay = new Map<number, any[]>();
  for (const atom of atoms) {
    const day = atom.core_lessons.day_number;
    if (!byDay.has(day)) byDay.set(day, []);
    byDay.get(day)!.push(atom);
  }
  
  const days = Array.from(byDay.keys()).sort((a, b) => a - b);
  
  for (const day of days) {
    const dayAtoms = byDay.get(day)!;
    const paddedDay = String(day).padStart(3, '0');
    
    console.log(`\nProcessing Day ${day}: ${dayAtoms[0].core_lessons.topic}`);
    
    // Group atoms by phase
    const byPhase = new Map<string, any[]>();
    for (const atom of dayAtoms) {
      if (!byPhase.has(atom.phase)) byPhase.set(atom.phase, []);
      byPhase.get(atom.phase)!.push(atom);
    }
    
    for (const [phase, phaseAtoms] of byPhase) {
      const hasImage = await checkImageExists(day, phase);
      
      if (!hasImage) {
        console.log(`  ⏭️ ${phase}: No image available`);
        results.skipped += phaseAtoms.length;
        continue;
      }
      
      const url = await generateUrlForAtom(day, phase);
      
      if (!url) {
        console.log(`  ⏭️ ${phase}: Unknown phase mapping`);
        results.skipped += phaseAtoms.length;
        continue;
      }
      
      // Update all atoms for this phase
      const atomIds = phaseAtoms.map(a => a.id);
      
      const { error } = await supabase
        .from('lesson_atoms')
        .update({ visual_url: url })
        .in('id', atomIds);
      
      if (error) {
        console.error(`  ❌ ${phase}: Error - ${error.message}`);
        results.errors += phaseAtoms.length;
      } else {
        console.log(`  ✅ ${phase}: Updated ${phaseAtoms.length} atoms → ${url}`);
        results.updated += phaseAtoms.length;
      }
    }
  }
  
  return results;
}

async function main() {
  const args = process.argv.slice(2);
  
  console.log('\n🔗 ATOM VISUAL URL UPDATER');
  console.log('   Linking phase images to lesson atoms\n');
  
  // Check environment
  if (!process.env.SUPABASE_URL && !process.env.NEXT_PUBLIC_SUPABASE_URL) {
    console.error('❌ SUPABASE_URL not set');
    process.exit(1);
  }
  if (!process.env.SUPABASE_SERVICE_KEY) {
    console.error('❌ SUPABASE_SERVICE_KEY not set');
    process.exit(1);
  }
  
  // Parse range argument
  const rangeArg = args.find(a => a.startsWith('--range='));
  let startDay: number | undefined;
  let endDay: number | undefined;
  
  if (rangeArg) {
    const [start, end] = rangeArg.split('=')[1].split('-').map(Number);
    startDay = start;
    endDay = end;
    console.log(`📅 Processing days ${startDay} to ${endDay}`);
  }
  
  // Get atoms needing URLs
  console.log('🔍 Finding atoms without visual URLs...');
  const atoms = await getAtomsNeedingUrls(startDay, endDay);
  console.log(`Found ${atoms.length} atoms needing URLs`);
  
  if (atoms.length === 0) {
    console.log('✅ All atoms already have visual URLs!');
    return;
  }
  
  if (args.includes('--preview')) {
    await previewUpdates(atoms);
  } else if (args.includes('--update')) {
    const results = await updateAtomUrls(atoms);
    
    console.log('\n' + '═'.repeat(60));
    console.log('📊 FINAL RESULTS');
    console.log('═'.repeat(60));
    console.log(`✅ Updated: ${results.updated} atoms`);
    console.log(`⏭️ Skipped: ${results.skipped} atoms (no image available)`);
    console.log(`❌ Errors: ${results.errors} atoms`);
  } else {
    console.log(`
Usage:
  npx ts-node scripts/update-atom-visual-urls.ts --preview           # Preview changes
  npx ts-node scripts/update-atom-visual-urls.ts --update            # Update all
  npx ts-node scripts/update-atom-visual-urls.ts --update --range=1-50  # Update range
    `);
  }
}

main().catch(console.error);

