#!/usr/bin/env npx tsx
/**
 * KELLY MOTION ID MANAGER
 * 
 * Simple interface to add avatar IDs to the motion library.
 * 
 * Usage:
 *   npx tsx scripts/kelly-motion-ids.ts                    # Show status
 *   npx tsx scripts/kelly-motion-ids.ts scientist A abc123 # Add one ID
 *   npx tsx scripts/kelly-motion-ids.ts scientist abc def ghi  # Add all 3 for archetype
 */

import * as fs from 'fs';

const LIBRARY_PATH = 'generated-images/kelly-motion-library.json';

const ARCHETYPES = [
  'scientist', 'explorer', 'rebel', 'architect',
  'diplomat', 'empath', 'macgyver', 'mystic',
  'provider', 'storyteller', 'strategist', 'survivor'
];

interface Library {
  [key: string]: any;
}

function load(): Library {
  return JSON.parse(fs.readFileSync(LIBRARY_PATH, 'utf-8'));
}

function save(lib: Library): void {
  lib.updated = new Date().toISOString().split('T')[0];
  fs.writeFileSync(LIBRARY_PATH, JSON.stringify(lib, null, 2));
}

function showStatus(): void {
  const lib = load();
  
  console.log('');
  console.log('┌────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐');
  console.log('│ ARCHETYPE      │ A (Warm)                    B (Talk)                     C (Filler)                          │');
  console.log('├────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤');
  
  let total = 0;
  let filled = 0;
  
  for (const arch of ARCHETYPES) {
    if (!lib[arch]) continue;
    
    const a = lib[arch].A || '';
    const b = lib[arch].B || '';
    const c = lib[arch].C || '';
    
    const aDisplay = a ? a.slice(0, 26).padEnd(26) : '❌'.padEnd(26);
    const bDisplay = b ? b.slice(0, 26).padEnd(26) : '❌'.padEnd(26);
    const cDisplay = c ? c.slice(0, 26).padEnd(26) : '❌'.padEnd(26);
    
    console.log(`│ ${arch.padEnd(14)} │ ${aDisplay}  ${bDisplay}  ${cDisplay} │`);
    
    total += 3;
    if (a) filled++;
    if (b) filled++;
    if (c) filled++;
  }
  
  console.log('└────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘');
  console.log(`\n📊 Progress: ${filled}/${total} IDs configured (${Math.round(filled/total*100)}%)`);
  
  if (filled < total) {
    console.log('\n💡 To add IDs:');
    console.log('   npx tsx scripts/kelly-motion-ids.ts <archetype> <A_id> <B_id> <C_id>');
    console.log('   Example: npx tsx scripts/kelly-motion-ids.ts scientist abc123 def456 ghi789');
  }
}

function addIds(archetype: string, ids: string[]): void {
  const lib = load();
  
  if (!ARCHETYPES.includes(archetype)) {
    console.error(`❌ Unknown archetype: ${archetype}`);
    console.log(`   Available: ${ARCHETYPES.join(', ')}`);
    return;
  }
  
  if (ids.length === 1 && ids[0].includes(' ')) {
    // Handle space-separated IDs
    ids = ids[0].split(/\s+/);
  }
  
  if (ids.length === 2) {
    // Format: archetype A|B|C id
    const motion = ids[0].toUpperCase();
    const id = ids[1];
    if (['A', 'B', 'C'].includes(motion)) {
      lib[archetype][motion] = id;
      save(lib);
      console.log(`✅ ${archetype}.${motion} = ${id}`);
      return;
    }
  }
  
  if (ids.length >= 3) {
    // Format: archetype idA idB idC
    lib[archetype].A = ids[0];
    lib[archetype].B = ids[1];
    lib[archetype].C = ids[2];
    save(lib);
    console.log(`✅ ${archetype}:`);
    console.log(`   A (Warm): ${ids[0]}`);
    console.log(`   B (Talk): ${ids[1]}`);
    console.log(`   C (Filler): ${ids[2]}`);
    return;
  }
  
  console.error('❌ Need at least 3 IDs (A, B, C) or specify motion letter');
  console.log('   Examples:');
  console.log('   npx tsx scripts/kelly-motion-ids.ts scientist abc123 def456 ghi789');
  console.log('   npx tsx scripts/kelly-motion-ids.ts scientist A abc123');
}

// Main
const args = process.argv.slice(2);

if (args.length === 0) {
  showStatus();
} else {
  const archetype = args[0].toLowerCase();
  const ids = args.slice(1);
  addIds(archetype, ids);
}
