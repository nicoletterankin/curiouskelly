/**
 * Verify Content Atoms Migration
 * 
 * Checks that content_atoms table is properly populated
 * and matches the static files.
 * 
 * Usage:
 *   npx tsx scripts/verify-content-atoms.ts
 *   npx tsx scripts/verify-content-atoms.ts --day 17
 */

import * as fs from 'fs';
import * as path from 'path';
import * as vm from 'vm';
import { createClient } from '@supabase/supabase-js';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY || '';

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const DATA_DIR = path.join(process.cwd(), 'public', 'data');

// ═══════════════════════════════════════════════════════════════════
// VERIFICATION
// ═══════════════════════════════════════════════════════════════════

async function verifyTableExists(): Promise<boolean> {
  console.log('🔍 Checking if content_atoms table exists...');
  
  const { data, error } = await supabase
    .from('content_atoms')
    .select('id')
    .limit(1);
  
  if (error) {
    if (error.code === '42P01') {
      console.log('❌ Table content_atoms does not exist!');
      console.log('   Run the migration SQL first:');
      console.log('   \\i docs/backend/migrations/002_content_atoms.sql');
      return false;
    }
    console.error('❌ Error checking table:', error.message);
    return false;
  }
  
  console.log('✅ Table content_atoms exists');
  return true;
}

async function getAtomCounts(): Promise<Map<number, number>> {
  console.log('\n📊 Fetching atom counts per day...');
  
  const { data, error } = await supabase
    .from('content_atoms')
    .select('day_number')
    .eq('is_live', true);
  
  if (error) {
    console.error('❌ Error fetching atoms:', error.message);
    return new Map();
  }
  
  const counts = new Map<number, number>();
  for (const row of data || []) {
    counts.set(row.day_number, (counts.get(row.day_number) || 0) + 1);
  }
  
  return counts;
}

async function verifyDay(dayNumber: number): Promise<{ valid: boolean; issues: string[] }> {
  const issues: string[] = [];
  
  // Get atoms from database
  const { data: dbAtoms, error } = await supabase
    .from('content_atoms')
    .select('*')
    .eq('day_number', dayNumber)
    .eq('is_live', true);
  
  if (error) {
    issues.push(`Database error: ${error.message}`);
    return { valid: false, issues };
  }
  
  // Check atom count
  if (!dbAtoms || dbAtoms.length === 0) {
    issues.push('No atoms found in database');
    return { valid: false, issues };
  }
  
  // Verify required phases exist
  const phases = new Set(dbAtoms.map(a => a.phase));
  const requiredPhases = ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom'];
  
  for (const phase of requiredPhases) {
    if (!phases.has(phase)) {
      issues.push(`Missing phase: ${phase}`);
    }
  }
  
  // Verify talk content exists for main phases
  for (const phase of ['hook', 'fact1', 'fact2', 'fact3', 'wisdom']) {
    const hasTalk = dbAtoms.some(a => a.phase === phase && a.content_type === 'talk');
    if (!hasTalk) {
      issues.push(`Missing talk content for ${phase}`);
    }
  }
  
  // Verify cliff has options
  const cliffOptions = dbAtoms.filter(a => a.phase === 'cliff' && a.content_type === 'option');
  if (cliffOptions.length < 2) {
    issues.push(`Cliff phase has ${cliffOptions.length} options (expected at least 2)`);
  }
  
  return {
    valid: issues.length === 0,
    issues
  };
}

async function verifySample(): Promise<void> {
  console.log('\n🔬 Verifying sample days...\n');
  
  const sampleDays = [1, 17, 50, 100, 200, 300, 365];
  
  for (const day of sampleDays) {
    const result = await verifyDay(day);
    if (result.valid) {
      console.log(`  ✅ Day ${day}: Valid`);
    } else {
      console.log(`  ⚠️ Day ${day}: Issues found`);
      for (const issue of result.issues) {
        console.log(`     - ${issue}`);
      }
    }
  }
}

async function generateReport(): Promise<void> {
  console.log('\n╔════════════════════════════════════════════════════╗');
  console.log('║   Content Atoms Verification Report                ║');
  console.log('╚════════════════════════════════════════════════════╝\n');
  
  // Check table exists
  if (!await verifyTableExists()) {
    return;
  }
  
  // Get counts
  const atomCounts = await getAtomCounts();
  const totalAtoms = Array.from(atomCounts.values()).reduce((a, b) => a + b, 0);
  const daysWithAtoms = atomCounts.size;
  
  console.log(`\n📈 Summary:`);
  console.log(`   Total atoms in database: ${totalAtoms}`);
  console.log(`   Days with atoms: ${daysWithAtoms}`);
  console.log(`   Average atoms per day: ${(totalAtoms / Math.max(1, daysWithAtoms)).toFixed(1)}`);
  
  // Check for missing days
  const staticFiles = fs.readdirSync(DATA_DIR)
    .filter(f => f.match(/^day-\d+-complete\.js$/))
    .map(f => {
      const match = f.match(/day-(\d+)-complete\.js/);
      return match ? parseInt(match[1]) : 0;
    })
    .filter(d => d > 0);
  
  const missingDays = staticFiles.filter(d => !atomCounts.has(d));
  
  console.log(`\n📁 Static files: ${staticFiles.length}`);
  console.log(`   Missing in database: ${missingDays.length}`);
  
  if (missingDays.length > 0 && missingDays.length <= 10) {
    console.log(`   Missing days: ${missingDays.join(', ')}`);
  } else if (missingDays.length > 10) {
    console.log(`   First 10 missing: ${missingDays.slice(0, 10).join(', ')}...`);
  }
  
  // Verify sample
  await verifySample();
  
  // Final status
  console.log('\n════════════════════════════════════════════════════');
  if (missingDays.length === 0 && daysWithAtoms >= staticFiles.length) {
    console.log('✅ VERIFICATION PASSED - All content migrated successfully');
  } else if (daysWithAtoms > 0) {
    console.log('⚠️ PARTIAL MIGRATION - Some days are missing');
    console.log('   Run: npx tsx scripts/migrate-to-content-atoms.ts --all');
  } else {
    console.log('❌ NO DATA - Run migration first');
    console.log('   1. Run SQL: \\i docs/backend/migrations/002_content_atoms.sql');
    console.log('   2. Run: npx tsx scripts/migrate-to-content-atoms.ts --all');
  }
  console.log('════════════════════════════════════════════════════\n');
}

// ═══════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  
  if (args.includes('--day')) {
    const dayIndex = args.indexOf('--day');
    const dayNumber = parseInt(args[dayIndex + 1]);
    if (isNaN(dayNumber)) {
      console.error('Invalid day number');
      process.exit(1);
    }
    
    console.log(`\n🔍 Verifying Day ${dayNumber}...\n`);
    const result = await verifyDay(dayNumber);
    
    if (result.valid) {
      console.log('✅ Day is valid');
    } else {
      console.log('❌ Issues found:');
      for (const issue of result.issues) {
        console.log(`   - ${issue}`);
      }
    }
  } else {
    await generateReport();
  }
}

main().catch(console.error);
