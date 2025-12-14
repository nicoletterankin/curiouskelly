#!/usr/bin/env npx tsx
/**
 * 🔧 RESTORE LESSON ATOMS FROM BACKUP
 * 
 * Restores the original archetype scripts from the November 30 backup
 * after the V1 tone police disaster.
 * 
 * SCOPE:
 * - Days 2-150 (damaged by V1)
 * - ALL archetypes EXCEPT "The Scientist" (will be fixed properly later)
 * - Restores original content from backup JSON
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  BACKUP_FILE: 'backups/lesson_atoms_2025-11-30T12-58-09-241Z.json',
  DAMAGED_DAY_START: 2,
  DAMAGED_DAY_END: 150,
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

interface BackupAtom {
  id: string;
  core_lesson_id: string;
  archetype: string;
  phase: string;
  content: {
    script: string;
    options?: string[];
    responses?: Record<string, string>;
  };
}

async function main() {
  console.log('\n╔══════════════════════════════════════════════════════════════════╗');
  console.log('║  🔧 RESTORE LESSON ATOMS FROM BACKUP                             ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║  Backup: ${CONFIG.BACKUP_FILE.padEnd(52)}║`);
  console.log(`║  Damaged Days: ${CONFIG.DAMAGED_DAY_START}-${CONFIG.DAMAGED_DAY_END}`.padEnd(67) + '║');
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  // Load backup
  console.log('📂 Loading backup file...');
  const backupData: BackupAtom[] = JSON.parse(fs.readFileSync(CONFIG.BACKUP_FILE, 'utf-8'));
  console.log(`   Found ${backupData.length} atoms in backup\n`);

  // Get core_lesson_ids for damaged days
  console.log('🔍 Fetching damaged day lesson IDs...');
  const { data: lessons, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id, day_number')
    .gte('day_number', CONFIG.DAMAGED_DAY_START)
    .lte('day_number', CONFIG.DAMAGED_DAY_END);

  if (lessonError || !lessons) {
    console.error('❌ Failed to fetch lessons:', lessonError);
    return;
  }

  const lessonIdToDay = new Map<string, number>();
  lessons.forEach(l => lessonIdToDay.set(l.id, l.day_number));
  console.log(`   Found ${lessons.length} damaged days\n`);

  // Filter backup atoms to only damaged days, EXCLUDING The Scientist
  const atomsToRestore = backupData.filter(atom => {
    const dayNumber = lessonIdToDay.get(atom.core_lesson_id);
    return (
      dayNumber !== undefined &&
      dayNumber >= CONFIG.DAMAGED_DAY_START &&
      dayNumber <= CONFIG.DAMAGED_DAY_END &&
      atom.archetype !== 'The Scientist'  // Keep The Scientist as-is for proper fix later
    );
  });

  console.log(`📊 Atoms to restore: ${atomsToRestore.length}`);
  console.log(`   (Excluding "The Scientist" - will be fixed properly)\n`);

  // Restore in batches
  const BATCH_SIZE = 100;
  let restored = 0;
  let errors = 0;

  for (let i = 0; i < atomsToRestore.length; i += BATCH_SIZE) {
    const batch = atomsToRestore.slice(i, i + BATCH_SIZE);
    
    // Update each atom
    for (const atom of batch) {
      const { error } = await supabase
        .from('lesson_atoms')
        .update({ content: atom.content })
        .eq('id', atom.id);

      if (error) {
        errors++;
        if (errors <= 5) {
          console.log(`   ❌ Error restoring ${atom.id}: ${error.message}`);
        }
      } else {
        restored++;
      }
    }

    const progress = Math.round((i + batch.length) / atomsToRestore.length * 100);
    process.stdout.write(`\r   Progress: ${progress}% (${restored} restored, ${errors} errors)`);
  }

  console.log('\n');
  console.log('═'.repeat(70));
  console.log('🏁 RESTORE COMPLETE');
  console.log('═'.repeat(70));
  console.log(`✅ Restored: ${restored}`);
  console.log(`❌ Errors:   ${errors}`);
  console.log(`⏭️  Skipped:  The Scientist archetype (for proper fix later)`);
  console.log('═'.repeat(70));
  console.log('\nNext step: Run proper Scientist-only conversational update');
}

main().catch(console.error);





