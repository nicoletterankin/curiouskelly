#!/usr/bin/env npx tsx
/**
 * 🔄 SYNC HEYGEN VIDEOS TO DATABASE
 * 
 * Reads existing video URLs from production results and inserts them into
 * the kelly_video_assets table so the lesson player can find them.
 * 
 * Usage:
 *   npx tsx scripts/sync-videos-to-database.ts
 *   npx tsx scripts/sync-videos-to-database.ts --dry-run
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// Parse .env file manually to get the CORRECT service role key (first occurrence)
import * as dotenv from 'dotenv';
const envContent = fs.readFileSync(path.join(process.cwd(), '.env'), 'utf-8');
const envLines = envContent.split('\n');
let serviceRoleKey = '';
for (const line of envLines) {
  if (line.startsWith('SUPABASE_SERVICE_ROLE_KEY=') && !serviceRoleKey) {
    serviceRoleKey = line.split('=')[1].trim();
    break;
  }
}

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: serviceRoleKey || process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

interface VideoResult {
  archetype: string;
  phase: string;
  status: string;
  url: string;
}

// Map archetype names to template IDs used in the player
function archetypeToTemplateId(archetype: string): string {
  return archetype
    .replace(/^The\s+/i, '')
    .toLowerCase()
    .replace(/\s+/g, '_');
}

// Map phase names to database phase names
// Database uses: hook, cliff, q1, q2, q3, wisdom, outro (not fact1/fact2/fact3)
function normalizePhase(phase: string): string {
  const phaseMap: Record<string, string> = {
    'fact1': 'q1',
    'fact2': 'q2', 
    'fact3': 'q3',
    'hook': 'hook',
    'cliff': 'cliff',
    'wisdom': 'wisdom',
    'outro': 'outro',
  };
  const lower = phase.toLowerCase();
  return phaseMap[lower] || lower;
}

async function main() {
  const dryRun = process.argv.includes('--dry-run');
  
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🔄 SYNC HEYGEN VIDEOS TO DATABASE                         ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  
  if (dryRun) {
    console.log('🏃 DRY RUN MODE - No changes will be made\n');
  }

  // Load all results files
  const resultsDir = path.join(process.cwd(), 'generated-videos', 'heygen-production');
  const files = fs.readdirSync(resultsDir).filter(f => f.endsWith('.json'));
  
  console.log(`📁 Found ${files.length} result files in ${resultsDir}\n`);
  
  let totalVideos = 0;
  let insertedVideos = 0;
  let skippedVideos = 0;
  let failedVideos = 0;
  
  for (const file of files) {
    const filePath = path.join(resultsDir, file);
    const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    
    // Handle both array format and object format
    const results: VideoResult[] = Array.isArray(data) ? data : [];
    if (!Array.isArray(data)) {
      console.log(`   ⏭️ Skipping ${file} (not an array format)`);
      continue;
    }
    
    // Extract day number from filename (e.g., day1_full_results.json -> 1)
    const dayMatch = file.match(/day(\d+)/i);
    const dayNumber = dayMatch ? parseInt(dayMatch[1]) : 1;
    
    console.log(`\n📄 Processing ${file} (Day ${dayNumber})...`);
    console.log(`   ${results.length} videos in file`);
    
    for (const video of results) {
      if (video.status !== 'success' || !video.url) {
        console.log(`   ⏭️ Skipping ${video.archetype} ${video.phase} (status: ${video.status})`);
        skippedVideos++;
        continue;
      }
      
      totalVideos++;
      
      // Extract storage path from the public URL
      // URL format: https://...supabase.co/storage/v1/object/public/kelly-videos/production/day_001/day_001_fact1_architect.mp4
      const urlParts = video.url.split('/public/');
      const storagePath = urlParts.length > 1 ? urlParts[1] : `production/day_${String(dayNumber).padStart(3, '0')}/${video.phase.toLowerCase()}_${archetypeToTemplateId(video.archetype)}.mp4`;
      
      const record = {
        day_number: dayNumber,
        phase: normalizePhase(video.phase),
        template: archetypeToTemplateId(video.archetype),
        age_bucket: 'adult', // Default age bucket
        asset_type: 'video',
        language: 'en',
        public_url: video.url,
        storage_path: storagePath,
        status: 'validated',
      };
      
      if (dryRun) {
        console.log(`   📝 Would insert: Day ${record.day_number}, ${record.phase}, ${record.template}`);
        insertedVideos++;
        continue;
      }
      
      // Check if already exists
      const { data: existing } = await supabase
        .from('kelly_video_assets')
        .select('id')
        .eq('day_number', record.day_number)
        .eq('phase', record.phase)
        .eq('template', record.template)
        .eq('age_bucket', record.age_bucket)
        .eq('asset_type', 'video')
        .eq('language', record.language)
        .limit(1);
      
      if (existing && existing.length > 0) {
        console.log(`   ⏭️ Already exists: Day ${record.day_number}, ${record.phase}, ${record.template}`);
        skippedVideos++;
        continue;
      }
      
      // Insert new record
      const { error } = await supabase
        .from('kelly_video_assets')
        .insert(record);
      
      if (error) {
        console.log(`   ❌ Failed: ${record.template} ${record.phase} - ${error.message}`);
        failedVideos++;
      } else {
        console.log(`   ✅ Inserted: Day ${record.day_number}, ${record.phase}, ${record.template}`);
        insertedVideos++;
      }
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(60));
  console.log(`   Total videos found:  ${totalVideos}`);
  console.log(`   Inserted:            ${insertedVideos}`);
  console.log(`   Skipped:             ${skippedVideos}`);
  console.log(`   Failed:              ${failedVideos}`);
  
  if (!dryRun && insertedVideos > 0) {
    console.log('\n🎉 Videos synced! Test at: https://curiouskelly.com/learn.html?debug&day=1');
  }
}

main().catch(console.error);
