#!/usr/bin/env npx tsx
/**
 * 🎨 POPULATE PHASE VISUAL URLs IN DATABASE
 * 
 * Updates core_lessons and kelly_video_assets with predictable URLs
 * for phase visuals, so the frontend can start displaying them.
 * 
 * This creates the database records pointing to where assets WILL be stored.
 * Run the asset generation pipeline separately to actually create the images.
 * 
 * Usage:
 *   npx tsx scripts/populate-phase-visual-urls.ts --all
 *   npx tsx scripts/populate-phase-visual-urls.ts --range=1-50
 *   npx tsx scripts/populate-phase-visual-urls.ts --dry-run
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

const STORAGE_BASE = `${SUPABASE_URL}/storage/v1/object/public`;
const VISUALS_BUCKET = 'lesson-visuals';
const PHASES = ['hook', 'q1', 'q2', 'q3', 'wisdom'];

// ═══════════════════════════════════════════════════════════════════════════
// URL GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

function getPhaseVisualUrl(dayNumber: number, phase: string): string {
  const paddedDay = String(dayNumber).padStart(3, '0');
  return `${STORAGE_BASE}/${VISUALS_BUCKET}/day_${paddedDay}/${phase}.png`;
}

function getThumbnailUrl(dayNumber: number): string {
  const paddedDay = String(dayNumber).padStart(3, '0');
  return `${STORAGE_BASE}/lesson-thumbnails/day_${paddedDay}.webp`;
}

// ═══════════════════════════════════════════════════════════════════════════
// DATABASE FUNCTIONS
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

async function upsertVideoAsset(asset: {
  day_number: number;
  phase: string;
  public_url: string;
}) {
  const { error } = await supabase
    .from('kelly_video_assets')
    .upsert({
      day_number: asset.day_number,
      phase: asset.phase,
      template: 'visual',
      asset_type: 'image',
      age_bucket: null,
      language: 'en',
      storage_bucket: VISUALS_BUCKET,
      storage_path: `day_${String(asset.day_number).padStart(3, '0')}/${asset.phase}.png`,
      public_url: asset.public_url,
      resolution: '1344x768',
      status: 'pending', // Will be 'generated' once actual file exists
      quality_tier: 'production',
    }, {
      onConflict: 'day_number,phase,template,asset_type'
    });
  
  return !error;
}

async function updateLessonUrls(dayNumber: number, heroUrl: string, thumbnailUrl: string) {
  const { error } = await supabase
    .from('core_lessons')
    .update({
      hero_image_url: heroUrl,
      thumbnail_url: thumbnailUrl,
    })
    .eq('day_number', dayNumber);
  
  return !error;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

async function processRange(start: number, end: number, dryRun: boolean) {
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  🎨 POPULATE PHASE VISUAL URLs`);
  console.log(`  Days ${start} to ${end} ${dryRun ? '(DRY RUN)' : ''}`);
  console.log(`${'█'.repeat(60)}\n`);
  
  const lessons = await getLessonsInRange(start, end);
  console.log(`Found ${lessons.length} lessons\n`);
  
  let assetsCreated = 0;
  let lessonsUpdated = 0;
  
  for (const lesson of lessons) {
    const { day_number, topic } = lesson;
    console.log(`📚 Day ${day_number}: ${topic}`);
    
    // Create asset records for each phase
    for (const phase of PHASES) {
      const url = getPhaseVisualUrl(day_number, phase);
      
      if (dryRun) {
        console.log(`   ${phase}: ${url.substring(url.length - 30)}`);
      } else {
        const success = await upsertVideoAsset({
          day_number,
          phase,
          public_url: url,
        });
        if (success) assetsCreated++;
      }
    }
    
    // Update core_lessons with hero and thumbnail URLs
    const heroUrl = getPhaseVisualUrl(day_number, 'hook');
    const thumbnailUrl = getThumbnailUrl(day_number);
    
    if (!dryRun) {
      const success = await updateLessonUrls(day_number, heroUrl, thumbnailUrl);
      if (success) lessonsUpdated++;
    }
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📊 RESULTS`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`📅 Days processed: ${lessons.length}`);
  
  if (!dryRun) {
    console.log(`✅ Asset records created: ${assetsCreated}`);
    console.log(`✅ Lessons updated: ${lessonsUpdated}`);
  } else {
    console.log(`\n💡 Run without --dry-run to update database`);
  }
}

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
🎨 POPULATE PHASE VISUAL URLs

This script creates database records pointing to where phase visuals
will be stored. Run the asset generation pipeline to create the actual images.

Usage:
  npx tsx scripts/populate-phase-visual-urls.ts --all          # All 365 days
  npx tsx scripts/populate-phase-visual-urls.ts --range=1-50   # Day range
  npx tsx scripts/populate-phase-visual-urls.ts --day=1        # Single day
  npx tsx scripts/populate-phase-visual-urls.ts --dry-run      # Preview

Environment Required:
  SUPABASE_URL (or PUBLIC_SUPABASE_URL)
  SUPABASE_SERVICE_ROLE_KEY
    `);
  }
}

main().catch(console.error);
