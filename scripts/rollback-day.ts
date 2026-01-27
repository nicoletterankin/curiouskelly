#!/usr/bin/env npx tsx
/**
 * Rollback an approved day back to pending_review status.
 * 
 * Usage:
 *   npx tsx scripts/rollback-day.ts --day=354
 *   npx tsx scripts/rollback-day.ts --day=354 --delete-videos  # Also remove video assets
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as readline from 'readline';
import { alert } from './alert.js';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error('❌ Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

function parseArgs(): { day: number; deleteVideos: boolean; yes: boolean; archetype: string; age: string } {
  const args = process.argv.slice(2);
  let day = 0;
  let deleteVideos = false;
  let yes = false;
  let archetype = 'The Explorer';
  let age = 'adult';
  
  for (const arg of args) {
    if (arg.startsWith('--day=')) {
      day = parseInt(arg.split('=')[1], 10);
    } else if (arg === '--delete-videos') {
      deleteVideos = true;
    } else if (arg === '--yes' || arg === '-y') {
      yes = true;
    } else if (arg.startsWith('--archetype=')) {
      archetype = arg.split('=')[1];
    } else if (arg.startsWith('--age=')) {
      age = arg.split('=')[1];
    }
  }
  
  return { day, deleteVideos, yes, archetype, age };
}

async function prompt(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer.toLowerCase().trim());
    });
  });
}

async function rollbackDay(day: number, archetype: string, age: string) {
  const { error } = await supabase
    .from('generation_status')
    .update({
      status: 'pending_review',
      approved: false,
      approved_at: null,
      published: false,
      published_at: null,
    })
    .eq('lesson_day', day)
    .eq('archetype', archetype)
    .eq('age_bucket', age);
  
  if (error) {
    throw new Error(`Failed to rollback: ${error.message}`);
  }
  
  await alert('ROLLBACK_TRIGGERED', { day });
}

async function deleteVideos(day: number) {
  // Delete from kelly_video_assets
  const { error, count } = await supabase
    .from('kelly_video_assets')
    .delete()
    .eq('lesson_day', day);
  
  if (error) {
    throw new Error(`Failed to delete videos: ${error.message}`);
  }
  
  return count || 0;
}

async function main() {
  const args = parseArgs();
  
  if (!args.day || args.day < 1 || args.day > 365) {
    console.log(`
Usage: npx tsx scripts/rollback-day.ts --day=<DAY_NUMBER> [options]

Options:
  --day=N           Day number to rollback (1-365) [required]
  --delete-videos   Also delete video assets from database
  --yes, -y         Skip confirmation prompt
  --archetype=NAME  Archetype (default: "The Explorer")
  --age=BUCKET      Age bucket (default: "adult")

Examples:
  npx tsx scripts/rollback-day.ts --day=354
  npx tsx scripts/rollback-day.ts --day=354 --delete-videos
`);
    process.exit(1);
  }
  
  console.log(`\n⏪ Rolling back Day ${args.day}...`);
  
  if (args.deleteVideos) {
    console.log('⚠️  WARNING: This will also DELETE all video assets for this day!');
  }
  
  if (!args.yes) {
    const answer = await prompt(`\nProceed with rollback? (y/n): `);
    if (answer !== 'y' && answer !== 'yes') {
      console.log('❌ Cancelled.');
      process.exit(0);
    }
  }
  
  // Rollback status
  await rollbackDay(args.day, args.archetype, args.age);
  console.log(`✅ Day ${args.day} status rolled back to 'pending_review'`);
  
  // Delete videos if requested
  if (args.deleteVideos) {
    const count = await deleteVideos(args.day);
    console.log(`🗑️  Deleted ${count} video assets`);
  }
  
  console.log(`\n🔄 Day ${args.day} is now ready for regeneration.`);
}

main().catch((error) => {
  console.error('❌ Error:', error.message);
  process.exit(1);
});







