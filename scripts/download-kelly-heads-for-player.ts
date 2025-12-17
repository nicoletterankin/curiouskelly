#!/usr/bin/env npx tsx
/**
 * Download Adult Kelly Head Images from Supabase
 * For use as placeholders in the lesson player
 */

import * as fs from 'fs';
import * as path from 'path';

const SUPABASE_BASE = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates';

// Adult Kelly head images (consistent identity)
const KELLY_HEADS: Record<string, string> = {
  'scientist': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_scientist_head.png`,
  'explorer': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_explorer_head.png`,
  'rebel': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_rebel_head.png`,
  'architect': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_architect_head.png`,
  'diplomat': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_diplomat_head.png`,
  'empath': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_empath_head.png`,
  'macgyver': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_macgyver_head.png`,
  'mystic': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_mystic_head.png`,
  'provider': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_provider_head.png`,
  'storyteller': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_storyteller_head.png`,
  'strategist': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_strategist_head.png`,
  'survivor': `${SUPABASE_BASE}/heygen/archetypes-head-only/kelly_survivor_head.png`,
};

const OUTPUT_DIR = path.join(process.cwd(), 'public', 'kelly', 'heads');

async function downloadImage(url: string, outputPath: string): Promise<boolean> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      console.error(`   ❌ Failed: ${response.status}`);
      return false;
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(outputPath, buffer);
    return true;
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return false;
  }
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  📥 DOWNLOAD ADULT KELLY HEADS FROM SUPABASE                   ║');
  console.log('║  For lesson player placeholders                                ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  // Create output directory
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`\n📁 Output: ${OUTPUT_DIR}`);
  
  let success = 0;
  let failed = 0;
  
  for (const [archetype, url] of Object.entries(KELLY_HEADS)) {
    console.log(`\n🎭 Downloading: ${archetype}`);
    const outputPath = path.join(OUTPUT_DIR, `kelly_${archetype}_head.png`);
    
    if (await downloadImage(url, outputPath)) {
      console.log(`   ✅ Saved: kelly_${archetype}_head.png`);
      success++;
    } else {
      failed++;
    }
  }
  
  // Also create a default/main kelly head (use scientist as default)
  console.log(`\n📌 Creating default kelly head (from scientist)...`);
  const defaultPath = path.join(OUTPUT_DIR, 'kelly_default_head.png');
  const scientistPath = path.join(OUTPUT_DIR, 'kelly_scientist_head.png');
  if (fs.existsSync(scientistPath)) {
    fs.copyFileSync(scientistPath, defaultPath);
    console.log(`   ✅ Created: kelly_default_head.png`);
  }
  
  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log(`✅ Downloaded: ${success}/12`);
  console.log(`❌ Failed: ${failed}/12`);
  console.log('═'.repeat(60));
  
  // Create manifest
  const manifest = {
    generated: new Date().toISOString(),
    source: 'Supabase kelly-templates',
    heads: Object.keys(KELLY_HEADS).map(arch => ({
      archetype: arch,
      file: `kelly_${arch}_head.png`,
      path: `public/kelly/heads/kelly_${arch}_head.png`,
    })),
  };
  
  const manifestPath = path.join(OUTPUT_DIR, 'heads-manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\n💾 Manifest: ${manifestPath}`);
}

main().catch(console.error);
