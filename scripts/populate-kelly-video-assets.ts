#!/usr/bin/env npx tsx
/**
 * 📦 POPULATE KELLY VIDEO ASSETS
 * 
 * Reads existing Day 1 HeyGen video results and populates the kelly_video_assets table.
 * This enables the frontend to look up videos by lesson/phase/age/language.
 * 
 * Usage:
 *   npx tsx scripts/populate-kelly-video-assets.ts
 *   npx tsx scripts/populate-kelly-video-assets.ts --day 1
 *   npx tsx scripts/populate-kelly-video-assets.ts --dry-run
 * 
 * Created: December 11, 2025
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// ES Module compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  DAY1_RESULTS_PATH: path.join(__dirname, '../generated-videos/heygen-production/day1_full_results.json'),
};

if (!CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_KEY) {
  console.error('❌ Missing Supabase credentials');
  console.error('   Required: PUBLIC_SUPABASE_URL (or SUPABASE_URL) + SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// ═══════════════════════════════════════════════════════════════════════════
// MAPPINGS - Based on ACTUAL production schema
// ═══════════════════════════════════════════════════════════════════════════

// Map HeyGen phase names to database phase names
// Database uses: hook, q1, q2, q3, wisdom
const PHASE_MAP: Record<string, string> = {
  'Hook': 'hook',
  'Fact1': 'q1',
  'Fact2': 'q2',
  'Fact3': 'q3',
  'Wisdom': 'wisdom',
};

// Map archetypes to template styles
const ARCHETYPE_TEMPLATE_MAP: Record<string, string> = {
  'The Scientist': 'explaining',
  'The Explorer': 'curious',
  'The Rebel': 'confident',
  'The Architect': 'explaining',
  'The Empath': 'warm',
  'The MacGyver': 'explaining',
  'The Mystic': 'reflective',
  'The Provider': 'warm',
  'The Storyteller': 'excited',
  'The Survivor': 'confident',
  'The Strategist': 'explaining',
  'The Diplomat': 'warm',
};

// ═══════════════════════════════════════════════════════════════════════════
// TYPES - Matching ACTUAL production schema
// ═══════════════════════════════════════════════════════════════════════════

interface VideoResult {
  archetype: string;
  phase: string;
  status: string;
  url: string;
}

interface VideoAssetInsert {
  day_number: number;
  phase: string;
  template: string;
  asset_type: string;
  age_bucket: string | null;
  language: string;
  storage_bucket: string;
  storage_path: string;
  public_url: string;
  resolution: string;
  status: string;
  quality_tier: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

async function loadDay1Results(): Promise<VideoResult[]> {
  if (!fs.existsSync(CONFIG.DAY1_RESULTS_PATH)) {
    throw new Error(`Results file not found: ${CONFIG.DAY1_RESULTS_PATH}`);
  }
  
  const content = fs.readFileSync(CONFIG.DAY1_RESULTS_PATH, 'utf-8');
  const results: VideoResult[] = JSON.parse(content);
  
  // Filter to only successful videos
  return results.filter(r => r.status === 'success' && r.url);
}

function transformToAssetInserts(videos: VideoResult[], dayNumber: number): VideoAssetInsert[] {
  const inserts: VideoAssetInsert[] = [];
  
  for (const video of videos) {
    const dbPhase = PHASE_MAP[video.phase];
    if (!dbPhase) {
      console.warn(`⚠️ Unknown phase: ${video.phase}, skipping`);
      continue;
    }
    
    const template = ARCHETYPE_TEMPLATE_MAP[video.archetype] || 'explaining';
    
    // Extract storage path from URL
    // URL format: https://xxx.supabase.co/storage/v1/object/public/kelly-videos/production/day_001/day_001_fact1_architect.mp4
    const urlParts = video.url.split('/kelly-videos/');
    const storagePath = urlParts[1] || '';
    
    inserts.push({
      day_number: dayNumber,
      phase: dbPhase,
      template: template,
      asset_type: 'video',
      age_bucket: null, // Videos work for all ages
      language: 'en',
      storage_bucket: 'kelly-videos',
      storage_path: storagePath,
      public_url: video.url,
      resolution: '1920x1080',
      status: 'generated',
      quality_tier: 'production',
    });
  }
  
  return inserts;
}

async function insertVideoAssets(assets: VideoAssetInsert[], dryRun: boolean): Promise<number> {
  if (dryRun) {
    console.log('\n🔍 DRY RUN - Would insert:');
    for (const asset of assets) {
      console.log(`   Day ${asset.day_number} | ${asset.phase} | ${asset.template} | ${asset.asset_type}`);
    }
    return assets.length;
  }
  
  let inserted = 0;
  let skipped = 0;
  
  for (const asset of assets) {
    // Check if already exists
    const { data: existing } = await supabase
      .from('kelly_video_assets')
      .select('id')
      .eq('day_number', asset.day_number)
      .eq('phase', asset.phase)
      .eq('template', asset.template)
      .eq('asset_type', asset.asset_type)
      .limit(1);
    
    if (existing && existing.length > 0) {
      console.log(`   ⏭️ Skipping: Day ${asset.day_number} ${asset.phase} ${asset.template} (exists)`);
      skipped++;
      continue;
    }
    
    const { error } = await supabase
      .from('kelly_video_assets')
      .insert(asset);
    
    if (error) {
      console.error(`❌ Failed to insert: ${asset.phase} ${asset.template}`, error.message);
    } else {
      inserted++;
    }
  }
  
  console.log(`   Skipped: ${skipped}`);
  return inserted;
}

async function verifyInserts(dayNumber: number): Promise<void> {
  const { data, error } = await supabase
    .from('kelly_video_assets')
    .select('phase, template, asset_type, status')
    .eq('day_number', dayNumber)
    .eq('asset_type', 'video');
  
  if (error) {
    console.error('❌ Verification query failed:', error.message);
    return;
  }
  
  console.log(`\n✅ Verification: ${data?.length || 0} video assets registered for Day ${dayNumber}`);
  
  // Group by phase
  const byPhase: Record<string, number> = {};
  for (const row of data || []) {
    byPhase[row.phase] = (byPhase[row.phase] || 0) + 1;
  }
  
  console.log('\n📊 Video assets by phase:');
  for (const [phase, count] of Object.entries(byPhase)) {
    console.log(`   ${phase}: ${count} videos`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('═'.repeat(72));
  console.log('📦 POPULATE KELLY VIDEO ASSETS');
  console.log('═'.repeat(72));
  
  // Parse args
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  let dayNumber = 1;
  
  const dayArg = args.find(a => a.startsWith('--day='));
  if (dayArg) {
    dayNumber = parseInt(dayArg.split('=')[1], 10);
  }
  
  console.log(`\n📅 Target: Day ${dayNumber}`);
  console.log(`🔧 Mode: ${dryRun ? 'DRY RUN' : 'LIVE INSERT'}`);
  
  // Load existing results
  console.log('\n📂 Loading video results...');
  const videos = await loadDay1Results();
  console.log(`   Found ${videos.length} successful videos`);
  
  // Show existing counts
  const { data: existingVideos } = await supabase
    .from('kelly_video_assets')
    .select('id')
    .eq('day_number', dayNumber)
    .eq('asset_type', 'video');
  console.log(`   Existing video assets for Day ${dayNumber}: ${existingVideos?.length || 0}`);
  
  // Transform to database format
  console.log('\n🔄 Transforming to database format...');
  const assets = transformToAssetInserts(videos, dayNumber);
  console.log(`   Created ${assets.length} asset records`);
  
  // Insert into database
  console.log('\n💾 Inserting into kelly_video_assets...');
  const insertedCount = await insertVideoAssets(assets, dryRun);
  console.log(`   Inserted: ${insertedCount}`);
  
  // Verify
  if (!dryRun) {
    await verifyInserts(dayNumber);
  }
  
  console.log('\n' + '═'.repeat(72));
  console.log('✅ COMPLETE');
  console.log('═'.repeat(72));
  
  if (dryRun) {
    console.log('\n💡 Run without --dry-run to actually insert records');
  }
}

main().catch(err => {
  console.error('❌ Fatal error:', err.message);
  process.exit(1);
});

















