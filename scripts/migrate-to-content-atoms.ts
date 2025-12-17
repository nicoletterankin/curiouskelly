/**
 * Migrate Static Lesson Files to content_atoms Table
 * 
 * This script reads all day-XXX-complete.js files and populates
 * the content_atoms table in Supabase.
 * 
 * Usage:
 *   npx tsx scripts/migrate-to-content-atoms.ts --all
 *   npx tsx scripts/migrate-to-content-atoms.ts --day 17
 *   npx tsx scripts/migrate-to-content-atoms.ts --range 1-50
 */

import * as fs from 'fs';
import * as path from 'path';
import * as vm from 'vm';
import { createClient, SupabaseClient } from '@supabase/supabase-js';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  console.log('Set these environment variables:');
  console.log('  export PUBLIC_SUPABASE_URL=https://xxx.supabase.co');
  console.log('  export SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const DATA_DIR = path.join(process.cwd(), 'public', 'data');

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

interface AtomContent {
  script?: string;
  options?: Array<{
    text: string;
    letter: string;
    quality: string;
    response: string;
  }>;
  kellyPose?: string;
  kellyEmotion?: string;
  visual_cue?: string;
  factNumber?: number;
  factTitle?: string;
  cliffPrompt?: string;
}

interface LessonAtom {
  id: string;
  phase: string;
  archetype?: string;
  content: AtomContent;
}

interface LessonPack {
  meta: {
    day_number: number;
    version?: string;
  };
  lesson: {
    topic: string;
    headline: string;
    universal_truth: string;
  };
  atoms: LessonAtom[];
}

interface ContentAtomRow {
  day_number: number;
  phase: string;
  content_type: string;
  variant: string | null;
  age_bucket: string | null;
  language: string;
  text_content: string;
  metadata: Record<string, any>;
  change_source: string;
}

// ═══════════════════════════════════════════════════════════════════
// PARSING
// ═══════════════════════════════════════════════════════════════════

function parseStaticFile(filePath: string, dayNumber: number): LessonPack | null {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    
    // Create sandbox to execute the JS
    const sandbox = {
      window: {
        CURIOUS_KELLY: {
          LOCAL_PACKS: {}
        }
      }
    };
    
    vm.runInNewContext(content, sandbox);
    
    // Try to find the lesson data
    const dayKey = `DAY_${String(dayNumber).padStart(3, '0')}`;
    const lessonData = (sandbox.window.CURIOUS_KELLY as any)[dayKey];
    
    if (lessonData && lessonData.lesson && lessonData.atoms) {
      return lessonData as LessonPack;
    }
    
    return null;
  } catch (e) {
    console.error(`  ❌ Failed to parse day ${dayNumber}:`, e);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════
// TRANSFORMATION
// ═══════════════════════════════════════════════════════════════════

function transformToContentAtoms(pack: LessonPack): ContentAtomRow[] {
  const atoms: ContentAtomRow[] = [];
  const dayNumber = pack.meta.day_number;
  
  for (const atom of pack.atoms) {
    const phase = atom.phase.toLowerCase();
    const content = atom.content;
    
    // Main talk/script
    if (content.script) {
      atoms.push({
        day_number: dayNumber,
        phase,
        content_type: 'talk',
        variant: null,
        age_bucket: null,
        language: 'en',
        text_content: content.script,
        metadata: {
          kellyPose: content.kellyPose,
          kellyEmotion: content.kellyEmotion,
          visual_cue: content.visual_cue,
          factNumber: content.factNumber,
          factTitle: content.factTitle
        },
        change_source: 'initial_seed'
      });
    }
    
    // Cliff prompt/question
    if (content.cliffPrompt) {
      atoms.push({
        day_number: dayNumber,
        phase,
        content_type: 'prompt',
        variant: null,
        age_bucket: null,
        language: 'en',
        text_content: content.cliffPrompt,
        metadata: {},
        change_source: 'initial_seed'
      });
    }
    
    // Options and responses
    if (content.options && Array.isArray(content.options)) {
      for (const option of content.options) {
        // Option text
        atoms.push({
          day_number: dayNumber,
          phase,
          content_type: 'option',
          variant: option.letter,
          age_bucket: null,
          language: 'en',
          text_content: option.text,
          metadata: {
            quality: option.quality
          },
          change_source: 'initial_seed'
        });
        
        // Response to option
        if (option.response) {
          atoms.push({
            day_number: dayNumber,
            phase,
            content_type: 'response',
            variant: option.letter,
            age_bucket: null,
            language: 'en',
            text_content: option.response,
            metadata: {},
            change_source: 'initial_seed'
          });
        }
      }
    }
  }
  
  return atoms;
}

// ═══════════════════════════════════════════════════════════════════
// DATABASE OPERATIONS
// ═══════════════════════════════════════════════════════════════════

async function insertAtoms(atoms: ContentAtomRow[]): Promise<{ inserted: number; errors: number }> {
  let inserted = 0;
  let errors = 0;
  
  // Insert in batches of 50
  const batchSize = 50;
  for (let i = 0; i < atoms.length; i += batchSize) {
    const batch = atoms.slice(i, i + batchSize);
    
    const { error } = await supabase
      .from('content_atoms')
      .upsert(batch, {
        onConflict: 'day_number,phase,content_type,variant,age_bucket,language',
        ignoreDuplicates: false
      });
    
    if (error) {
      console.error(`    ❌ Batch insert error:`, error.message);
      errors += batch.length;
    } else {
      inserted += batch.length;
    }
  }
  
  return { inserted, errors };
}

async function clearDayAtoms(dayNumber: number): Promise<void> {
  const { error } = await supabase
    .from('content_atoms')
    .delete()
    .eq('day_number', dayNumber);
  
  if (error) {
    console.error(`  ⚠️ Could not clear existing atoms for day ${dayNumber}:`, error.message);
  }
}

// ═══════════════════════════════════════════════════════════════════
// MIGRATION
// ═══════════════════════════════════════════════════════════════════

async function migrateDay(dayNumber: number): Promise<{ success: boolean; atomCount: number }> {
  const paddedDay = String(dayNumber).padStart(3, '0');
  const filePath = path.join(DATA_DIR, `day-${paddedDay}-complete.js`);
  
  // Check if file exists
  if (!fs.existsSync(filePath)) {
    // Try non-padded version
    const altPath = path.join(DATA_DIR, `day-${dayNumber}-complete.js`);
    if (!fs.existsSync(altPath)) {
      console.log(`  ⏭️ Day ${dayNumber}: No static file found`);
      return { success: false, atomCount: 0 };
    }
  }
  
  console.log(`  📦 Day ${dayNumber}: Parsing...`);
  
  const pack = parseStaticFile(
    fs.existsSync(filePath) ? filePath : path.join(DATA_DIR, `day-${dayNumber}-complete.js`),
    dayNumber
  );
  
  if (!pack) {
    console.log(`  ⚠️ Day ${dayNumber}: Could not parse`);
    return { success: false, atomCount: 0 };
  }
  
  const atoms = transformToContentAtoms(pack);
  console.log(`  📝 Day ${dayNumber}: ${atoms.length} atoms to insert`);
  
  if (atoms.length === 0) {
    return { success: true, atomCount: 0 };
  }
  
  // Clear existing and insert new
  await clearDayAtoms(dayNumber);
  const { inserted, errors } = await insertAtoms(atoms);
  
  if (errors > 0) {
    console.log(`  ⚠️ Day ${dayNumber}: ${inserted} inserted, ${errors} errors`);
    return { success: false, atomCount: inserted };
  }
  
  console.log(`  ✅ Day ${dayNumber}: ${inserted} atoms inserted`);
  return { success: true, atomCount: inserted };
}

async function migrateRange(start: number, end: number): Promise<void> {
  console.log(`\n🚀 Migrating days ${start} to ${end}...\n`);
  
  let totalAtoms = 0;
  let successDays = 0;
  let failedDays = 0;
  
  for (let day = start; day <= end; day++) {
    const result = await migrateDay(day);
    totalAtoms += result.atomCount;
    if (result.success) successDays++;
    else if (result.atomCount === 0) { /* no file, skip count */ }
    else failedDays++;
  }
  
  console.log(`\n════════════════════════════════════════`);
  console.log(`📊 Migration Complete`);
  console.log(`   Days processed: ${end - start + 1}`);
  console.log(`   Successful: ${successDays}`);
  console.log(`   Failed: ${failedDays}`);
  console.log(`   Total atoms: ${totalAtoms}`);
  console.log(`════════════════════════════════════════\n`);
}

async function migrateAll(): Promise<void> {
  // Find all day files
  const files = fs.readdirSync(DATA_DIR)
    .filter(f => f.match(/^day-\d+-complete\.js$/))
    .map(f => {
      const match = f.match(/day-(\d+)-complete\.js/);
      return match ? parseInt(match[1]) : 0;
    })
    .filter(d => d > 0)
    .sort((a, b) => a - b);
  
  console.log(`\n🚀 Found ${files.length} lesson files to migrate\n`);
  
  let totalAtoms = 0;
  let successDays = 0;
  let failedDays = 0;
  
  for (const day of files) {
    const result = await migrateDay(day);
    totalAtoms += result.atomCount;
    if (result.success && result.atomCount > 0) successDays++;
    else if (!result.success) failedDays++;
  }
  
  console.log(`\n════════════════════════════════════════`);
  console.log(`📊 Migration Complete`);
  console.log(`   Files processed: ${files.length}`);
  console.log(`   Successful: ${successDays}`);
  console.log(`   Failed: ${failedDays}`);
  console.log(`   Total atoms: ${totalAtoms}`);
  console.log(`════════════════════════════════════════\n`);
}

// ═══════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  
  console.log('\n╔════════════════════════════════════════════════════╗');
  console.log('║   Content Atoms Migration - Static → Supabase      ║');
  console.log('╚════════════════════════════════════════════════════╝\n');
  
  if (args.includes('--all')) {
    await migrateAll();
  } else if (args.includes('--day')) {
    const dayIndex = args.indexOf('--day');
    const dayNumber = parseInt(args[dayIndex + 1]);
    if (isNaN(dayNumber) || dayNumber < 1 || dayNumber > 365) {
      console.error('❌ Invalid day number. Use --day 17');
      process.exit(1);
    }
    await migrateDay(dayNumber);
  } else if (args.includes('--range')) {
    const rangeIndex = args.indexOf('--range');
    const range = args[rangeIndex + 1];
    const [start, end] = range.split('-').map(Number);
    if (isNaN(start) || isNaN(end) || start < 1 || end > 365 || start > end) {
      console.error('❌ Invalid range. Use --range 1-50');
      process.exit(1);
    }
    await migrateRange(start, end);
  } else {
    console.log('Usage:');
    console.log('  npx tsx scripts/migrate-to-content-atoms.ts --all');
    console.log('  npx tsx scripts/migrate-to-content-atoms.ts --day 17');
    console.log('  npx tsx scripts/migrate-to-content-atoms.ts --range 1-50');
    process.exit(0);
  }
}

main().catch(console.error);
