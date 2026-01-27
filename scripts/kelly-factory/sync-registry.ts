#!/usr/bin/env npx tsx
/**
 * 🔄 SYNC KELLY LESSON ASSETS REGISTRY
 * 
 * Scans the kelly-videos storage bucket for lipsync videos and ensures
 * every video has a corresponding kelly_lesson_assets row.
 * 
 * Usage:
 *   npx tsx scripts/kelly-factory/sync-registry.ts              # Full sync
 *   npx tsx scripts/kelly-factory/sync-registry.ts --find-orphans   # Find videos not in DB
 *   npx tsx scripts/kelly-factory/sync-registry.ts --repair         # Create missing records
 *   npx tsx scripts/kelly-factory/sync-registry.ts --dry-run        # Preview only
 *   npx tsx scripts/kelly-factory/sync-registry.ts --status         # Show registry status
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

// =============================================================================
// CONFIGURATION
// =============================================================================

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY;
const BUCKET_NAME = 'kelly-videos';
const LIPSYNC_PREFIX = 'lipsync';
const PAGE_SIZE = 1000;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error('❌ Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// =============================================================================
// TYPES
// =============================================================================

interface ParsedVideo {
  storagePath: string;
  publicUrl: string;
  dayNumber: number;
  phase: string;
  ageGroup: number;
  language: string;
  year: number;
}

interface RegistryStats {
  pending: number;
  audio_ready: number;
  script_ready: number;
  complete: number;
  error: number;
  [key: string]: number;
}

// =============================================================================
// PARSING
// =============================================================================

// Pattern 1 (new): lipsync/{year}/{lang}/day-{NNN}/{phase}-age{X}-{lang}.mp4
// Example: lipsync/2026/en/day-014/hook-age16-en.mp4
const VIDEO_PATH_REGEX_NEW = /^lipsync\/(\d{4})\/(\w+)\/day-(\d{3})\/(\w+)-age(\d+)-(\w+)\.mp4$/;

// Pattern 2 (old): lipsync/day-{N}/{phase}-age{X}-{lang}.mp4
// Example: lipsync/day-39/hook-age16-en.mp4
const VIDEO_PATH_REGEX_OLD = /^lipsync\/day-(\d+)\/(\w+)-age(\d+)-(\w+)\.mp4$/;

function parseVideoPath(storagePath: string): ParsedVideo | null {
  // Try new format first
  let match = storagePath.match(VIDEO_PATH_REGEX_NEW);
  if (match) {
    const [, year, lang1, dayStr, phase, ageStr, lang2] = match;
    const { data: urlData } = supabase.storage
      .from(BUCKET_NAME)
      .getPublicUrl(storagePath);

    return {
      storagePath,
      publicUrl: urlData.publicUrl,
      dayNumber: parseInt(dayStr, 10),
      phase,
      ageGroup: parseInt(ageStr, 10),
      language: lang2,
      year: parseInt(year, 10),
    };
  }

  // Try old format
  match = storagePath.match(VIDEO_PATH_REGEX_OLD);
  if (match) {
    const [, dayStr, phase, ageStr, lang] = match;
    const { data: urlData } = supabase.storage
      .from(BUCKET_NAME)
      .getPublicUrl(storagePath);

    return {
      storagePath,
      publicUrl: urlData.publicUrl,
      dayNumber: parseInt(dayStr, 10),
      phase,
      ageGroup: parseInt(ageStr, 10),
      language: lang,
      year: new Date().getFullYear(),
    };
  }

  return null;
}

// =============================================================================
// STORAGE LISTING
// =============================================================================

async function listAllLipsyncVideos(): Promise<string[]> {
  const results: string[] = [];
  const queue = [LIPSYNC_PREFIX];

  while (queue.length > 0) {
    const prefix = queue.pop()!;
    const { data, error } = await supabase.storage.from(BUCKET_NAME).list(prefix, {
      limit: PAGE_SIZE,
      sortBy: { column: 'name', order: 'asc' },
    });

    if (error) {
      throw new Error(`Storage list failed for "${prefix}": ${error.message}`);
    }

    for (const entry of data || []) {
      if (!entry.name || entry.name === '.' || entry.name === '..') continue;
      const fullPath = prefix ? `${prefix}/${entry.name}` : entry.name;
      
      // Check if it's a file (has metadata.size) or folder
      if (entry.metadata && typeof entry.metadata.size === 'number') {
        if (fullPath.endsWith('.mp4')) {
          results.push(fullPath);
        }
      } else {
        queue.push(fullPath);
      }
    }
  }

  return results;
}

// =============================================================================
// DATABASE OPERATIONS
// =============================================================================

async function getRegisteredVideos(): Promise<Map<string, any>> {
  const registered = new Map<string, any>();
  let from = 0;

  while (true) {
    const { data, error } = await supabase
      .from('kelly_lesson_assets')
      .select('id, day_number, phase, age_group, language, video_url, status')
      .not('video_url', 'is', null)
      .range(from, from + PAGE_SIZE - 1);

    if (error) throw new Error(`DB query failed: ${error.message}`);
    if (!data || data.length === 0) break;

    for (const row of data) {
      // Create a key for matching: day-phase-age-lang
      const key = `${row.day_number}-${row.phase}-${row.age_group}-${row.language}`;
      registered.set(key, row);
    }

    if (data.length < PAGE_SIZE) break;
    from += PAGE_SIZE;
  }

  return registered;
}

async function getRegistryStatus(): Promise<RegistryStats> {
  const { data, error } = await supabase
    .from('kelly_lesson_assets')
    .select('status');

  if (error) throw new Error(`Status query failed: ${error.message}`);

  const stats: RegistryStats = {
    pending: 0,
    audio_ready: 0,
    script_ready: 0,
    complete: 0,
    error: 0,
  };

  for (const row of data || []) {
    stats[row.status] = (stats[row.status] || 0) + 1;
  }

  return stats;
}

async function createMissingRecord(video: ParsedVideo, dryRun: boolean): Promise<boolean> {
  if (dryRun) {
    console.log(`   [DRY RUN] Would create: day=${video.dayNumber}, phase=${video.phase}, age=${video.ageGroup}`);
    return true;
  }

  const { error } = await supabase.from('kelly_lesson_assets').insert({
    day_number: video.dayNumber,
    phase: video.phase,
    age_group: video.ageGroup,
    language: video.language,
    video_url: video.publicUrl,
    video_source: 'sadtalker',
    status: 'complete',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });

  if (error) {
    if (error.message.includes('duplicate key')) {
      return true; // Already exists
    }
    console.error(`   ❌ Insert failed: ${error.message}`);
    return false;
  }

  return true;
}

// =============================================================================
// COMMANDS
// =============================================================================

async function showStatus() {
  console.log('\n' + '═'.repeat(60));
  console.log('  📊 KELLY LESSON ASSETS REGISTRY STATUS');
  console.log('═'.repeat(60));

  const stats = await getRegistryStatus();
  const total = Object.values(stats).reduce((a, b) => a + b, 0);

  console.log(`\n   Total records: ${total}\n`);
  
  const statusOrder = ['complete', 'audio_ready', 'script_ready', 'pending', 'error'];
  for (const status of statusOrder) {
    if (stats[status] > 0) {
      const emoji = status === 'complete' ? '✅' : 
                    status === 'audio_ready' ? '🔊' : 
                    status === 'script_ready' ? '📝' :
                    status === 'pending' ? '⏳' : '❌';
      const pct = ((stats[status] / total) * 100).toFixed(1);
      console.log(`   ${emoji} ${status.padEnd(12)} ${stats[status].toString().padStart(5)}  (${pct}%)`);
    }
  }
  console.log('═'.repeat(60));
}

async function findOrphans() {
  console.log('\n' + '═'.repeat(60));
  console.log('  🔍 FINDING ORPHANED VIDEOS (in storage, not in DB)');
  console.log('═'.repeat(60));

  console.log('\n📂 Scanning storage bucket...');
  const storageVideos = await listAllLipsyncVideos();
  console.log(`   Found ${storageVideos.length} videos in storage`);

  console.log('\n📊 Loading registered videos...');
  const registered = await getRegisteredVideos();
  console.log(`   Found ${registered.size} registered videos in DB`);

  const orphans: ParsedVideo[] = [];

  for (const path of storageVideos) {
    const parsed = parseVideoPath(path);
    if (!parsed) {
      console.log(`   ⚠️ Unparseable path: ${path}`);
      continue;
    }

    const key = `${parsed.dayNumber}-${parsed.phase}-${parsed.ageGroup}-${parsed.language}`;
    if (!registered.has(key)) {
      orphans.push(parsed);
    }
  }

  console.log('\n' + '─'.repeat(60));
  console.log(`   Orphaned videos: ${orphans.length}`);
  console.log('─'.repeat(60));

  if (orphans.length > 0) {
    console.log('\n   Orphan list:');
    for (const v of orphans.slice(0, 20)) {
      console.log(`   • Day ${v.dayNumber} / ${v.phase} / age${v.ageGroup} / ${v.language}`);
    }
    if (orphans.length > 20) {
      console.log(`   ... and ${orphans.length - 20} more`);
    }
    console.log('\n   💡 Run with --repair to create missing records');
  }

  return orphans;
}

async function repair(dryRun: boolean) {
  console.log('\n' + '═'.repeat(60));
  console.log(`  🔧 REPAIR MISSING RECORDS ${dryRun ? '(DRY RUN)' : ''}`);
  console.log('═'.repeat(60));

  const orphans = await findOrphans();
  
  if (orphans.length === 0) {
    console.log('\n   ✅ No repairs needed - all videos registered!');
    return;
  }

  console.log(`\n   Creating ${orphans.length} missing records...`);
  
  let created = 0;
  let failed = 0;

  for (const video of orphans) {
    const success = await createMissingRecord(video, dryRun);
    if (success) {
      created++;
    } else {
      failed++;
    }
  }

  console.log('\n' + '─'.repeat(60));
  console.log(`   ✅ Created: ${created}`);
  console.log(`   ❌ Failed:  ${failed}`);
  console.log('─'.repeat(60));

  if (dryRun) {
    console.log('\n   💡 Run without --dry-run to apply changes');
  }
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const findOrphansFlag = args.includes('--find-orphans');
  const repairFlag = args.includes('--repair');
  const statusFlag = args.includes('--status');

  if (args.includes('--help')) {
    console.log(`
🔄 Sync Kelly Lesson Assets Registry

Usage:
  npx tsx scripts/kelly-factory/sync-registry.ts [options]

Options:
  --status       Show current registry status
  --find-orphans Find videos in storage not registered in DB
  --repair       Create missing records for orphaned videos
  --dry-run      Preview changes without applying
  --help         Show this help

Examples:
  npx tsx scripts/kelly-factory/sync-registry.ts --status
  npx tsx scripts/kelly-factory/sync-registry.ts --find-orphans
  npx tsx scripts/kelly-factory/sync-registry.ts --repair --dry-run
  npx tsx scripts/kelly-factory/sync-registry.ts --repair
`);
    process.exit(0);
  }

  console.log('\n🔄 KELLY LESSON ASSETS SYNC');
  console.log('━'.repeat(60));
  console.log(`Bucket: ${BUCKET_NAME}`);
  console.log(`Prefix: ${LIPSYNC_PREFIX}/`);
  console.log(`Mode:   ${dryRun ? 'DRY RUN' : 'LIVE'}`);
  console.log('━'.repeat(60));

  // Always show status
  await showStatus();

  if (findOrphansFlag) {
    await findOrphans();
  } else if (repairFlag) {
    await repair(dryRun);
  } else if (!statusFlag) {
    // Default: show status + find orphans
    await findOrphans();
  }
}

main().catch(err => {
  console.error('❌ Sync failed:', err);
  process.exit(1);
});
