#!/usr/bin/env npx tsx
/**
 * HEYGEN MOTION LIBRARY UPDATER
 * 
 * Updates the Kelly motion library manifest with avatar IDs.
 * 
 * Usage:
 *   npx tsx scripts/heygen-update-motion-library.ts --add scientist hypothesis abc123
 *   npx tsx scripts/heygen-update-motion-library.ts --show
 *   npx tsx scripts/heygen-update-motion-library.ts --test scientist
 */

import 'dotenv/config';
import * as fs from 'fs';

const MANIFEST_PATH = 'generated-images/kelly-motion-manifest.json';
const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;

interface MotionManifest {
  version: string;
  updated: string;
  archetypes: Record<string, Record<string, string>>;
  motion_to_phase_mapping: Record<string, string[]>;
}

function loadManifest(): MotionManifest {
  const data = fs.readFileSync(MANIFEST_PATH, 'utf-8');
  return JSON.parse(data);
}

function saveManifest(manifest: MotionManifest): void {
  manifest.updated = new Date().toISOString().split('T')[0];
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
}

async function testAvatarId(avatarId: string): Promise<boolean> {
  // Try a simple request to validate the avatar exists
  const response = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: avatarId,
        },
        voice: {
          type: 'text',
          input_text: 'Test',
          voice_id: '0015ce4f932b405b9fc3a5e2f5e92c46',
        },
      }],
      dimension: { width: 512, height: 512 },
      test: true, // Some APIs support this to not actually generate
    }),
  });

  // If it's a talking photo issue, it will say so
  const text = await response.text();
  return !text.includes('talking photo not found') && !text.includes('invalid_parameter');
}

async function main() {
  const args = process.argv.slice(2);
  
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  📚 KELLY MOTION LIBRARY MANAGER                               ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');

  if (args[0] === '--show' || args.length === 0) {
    // Show current state
    const manifest = loadManifest();
    
    console.log('📋 CURRENT MOTION LIBRARY STATUS:\n');
    
    let total = 0;
    let filled = 0;
    
    for (const [archetype, motions] of Object.entries(manifest.archetypes)) {
      const motionList = Object.entries(motions);
      const filledMotions = motionList.filter(([_, id]) => id !== '');
      
      console.log(`${archetype.toUpperCase()} (${filledMotions.length}/${motionList.length}):`);
      for (const [motion, id] of motionList) {
        const status = id ? `✅ ${id}` : '❌ (empty)';
        console.log(`  ${motion}: ${status}`);
      }
      console.log('');
      
      total += motionList.length;
      filled += filledMotions.length;
    }
    
    console.log('═'.repeat(60));
    console.log(`📊 TOTAL: ${filled}/${total} motion IDs configured`);
    
    if (filled < total) {
      console.log(`\n💡 To add an ID:`);
      console.log(`   npx tsx scripts/heygen-update-motion-library.ts --add <archetype> <motion> <id>`);
      console.log(`   Example: npx tsx scripts/heygen-update-motion-library.ts --add scientist hypothesis abc123`);
    }
    
    return;
  }
  
  if (args[0] === '--add' && args.length >= 4) {
    const [_, archetype, motion, avatarId] = args;
    const manifest = loadManifest();
    
    if (!manifest.archetypes[archetype]) {
      console.error(`❌ Unknown archetype: ${archetype}`);
      console.log(`   Available: ${Object.keys(manifest.archetypes).join(', ')}`);
      return;
    }
    
    if (!(motion in manifest.archetypes[archetype])) {
      console.error(`❌ Unknown motion for ${archetype}: ${motion}`);
      console.log(`   Available: ${Object.keys(manifest.archetypes[archetype]).join(', ')}`);
      return;
    }
    
    manifest.archetypes[archetype][motion] = avatarId;
    saveManifest(manifest);
    
    console.log(`✅ Updated ${archetype}.${motion} = ${avatarId}`);
    console.log(`   Saved to: ${MANIFEST_PATH}`);
    return;
  }
  
  if (args[0] === '--test' && args[1]) {
    const archetype = args[1];
    const manifest = loadManifest();
    
    if (!manifest.archetypes[archetype]) {
      console.error(`❌ Unknown archetype: ${archetype}`);
      return;
    }
    
    console.log(`🧪 Testing ${archetype} motion IDs...\n`);
    
    for (const [motion, avatarId] of Object.entries(manifest.archetypes[archetype])) {
      if (!avatarId) {
        console.log(`   ${motion}: ⏭️ (no ID set)`);
        continue;
      }
      
      process.stdout.write(`   ${motion}: testing ${avatarId}... `);
      const valid = await testAvatarId(avatarId);
      console.log(valid ? '✅' : '❌');
    }
    
    return;
  }
  
  if (args[0] === '--bulk') {
    // Bulk update mode - read from stdin or file
    console.log('📥 BULK UPDATE MODE');
    console.log('   Paste avatar IDs in format: archetype.motion=id');
    console.log('   Example: scientist.hypothesis=abc123');
    console.log('   One per line. Press Ctrl+D when done.\n');
    
    const manifest = loadManifest();
    let updated = 0;
    
    // This would need stdin reading - for now just show example
    console.log('💡 Example bulk input:');
    console.log('   scientist.hypothesis=abc123');
    console.log('   scientist.discovery=def456');
    console.log('   scientist.conclusion=ghi789');
    
    return;
  }
  
  // Show usage
  console.log('📖 USAGE:');
  console.log('   --show              Show current library status');
  console.log('   --add A M ID        Add avatar ID for archetype A, motion M');
  console.log('   --test ARCHETYPE    Test avatar IDs for an archetype');
  console.log('');
  console.log('📋 ARCHETYPES:');
  console.log('   scientist, explorer, rebel, architect, diplomat, empath,');
  console.log('   macgyver, mystic, provider, storyteller, strategist, survivor');
}

main().catch(console.error);
