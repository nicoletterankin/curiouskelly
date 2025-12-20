#!/usr/bin/env npx tsx
/**
 * Approve a generated day for publishing.
 * 
 * Usage: 
 *   npx tsx scripts/approve-day.ts --day=354
 *   npx tsx scripts/approve-day.ts --day=354 --yes  # Skip confirmation
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

const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];

function parseArgs(): { day: number; yes: boolean; archetype: string; age: string } {
  const args = process.argv.slice(2);
  let day = 0;
  let yes = false;
  let archetype = 'The Explorer';
  let age = 'adult';
  
  for (const arg of args) {
    if (arg.startsWith('--day=')) {
      day = parseInt(arg.split('=')[1], 10);
    } else if (arg === '--yes' || arg === '-y') {
      yes = true;
    } else if (arg.startsWith('--archetype=')) {
      archetype = arg.split('=')[1];
    } else if (arg.startsWith('--age=')) {
      age = arg.split('=')[1];
    }
  }
  
  return { day, yes, archetype, age };
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

async function getDayStatus(day: number, archetype: string, age: string) {
  // Get generation_status
  const { data: status } = await supabase
    .from('generation_status')
    .select('*')
    .eq('lesson_day', day)
    .eq('archetype', archetype)
    .eq('age_bucket', age)
    .single();
  
  // Get video counts
  const { data: videos } = await supabase
    .from('kelly_video_assets')
    .select('phase, public_url, status')
    .eq('lesson_day', day);
  
  const videoCount = videos?.filter(v => v.public_url)?.length || 0;
  const phases = videos?.map(v => v.phase) || [];
  
  return { status, videoCount, phases, videos };
}

async function approveDay(day: number, archetype: string, age: string) {
  const now = new Date().toISOString();
  
  const { error } = await supabase
    .from('generation_status')
    .upsert({
      lesson_day: day,
      archetype,
      age_bucket: age,
      status: 'approved',
      approved: true,
      approved_at: now,
      reviewed: true,
      reviewed_at: now,
      year: 2025,
    }, {
      onConflict: 'lesson_day,archetype,age_bucket,year'
    });
  
  if (error) {
    throw new Error(`Failed to approve: ${error.message}`);
  }
  
  await alert('DAY_APPROVED', { day });
}

async function main() {
  const { day, yes, archetype, age } = parseArgs();
  
  if (!day || day < 1 || day > 365) {
    console.log(`
Usage: npx tsx scripts/approve-day.ts --day=<DAY_NUMBER> [options]

Options:
  --day=N           Day number to approve (1-365) [required]
  --yes, -y         Skip confirmation prompt
  --archetype=NAME  Archetype (default: "The Explorer")
  --age=BUCKET      Age bucket (default: "adult")

Examples:
  npx tsx scripts/approve-day.ts --day=354
  npx tsx scripts/approve-day.ts --day=354 --yes
  npx tsx scripts/approve-day.ts --day=354 --archetype="The Scientist" --age=teen
`);
    process.exit(1);
  }
  
  console.log(`\n📋 Checking Day ${day} (${archetype} / ${age})...\n`);
  
  const { status, videoCount, phases, videos } = await getDayStatus(day, archetype, age);
  
  // Display current status
  console.log(`Current Status: ${status?.status || 'not_started'}`);
  console.log(`Videos: ${videoCount}/7 phases`);
  console.log(`Phases with video: ${phases.join(', ') || 'none'}`);
  
  if (status?.approved) {
    console.log(`\n✅ Day ${day} is already approved (at ${status.approved_at})`);
    return;
  }
  
  // Show video details
  console.log('\n📹 Video Details:');
  for (const phase of PHASES) {
    const video = videos?.find(v => v.phase === phase);
    const icon = video?.public_url ? '✅' : '❌';
    const url = video?.public_url ? video.public_url.substring(0, 50) + '...' : 'No video';
    console.log(`  ${icon} ${phase}: ${url}`);
  }
  
  // Warn if incomplete
  if (videoCount < 7) {
    console.log(`\n⚠️  Warning: Only ${videoCount}/7 phases have videos!`);
  }
  
  // Confirm
  if (!yes) {
    const answer = await prompt(`\nApprove Day ${day}? (y/n): `);
    if (answer !== 'y' && answer !== 'yes') {
      console.log('❌ Cancelled.');
      process.exit(0);
    }
  }
  
  // Approve
  console.log(`\n✅ Approving Day ${day}...`);
  await approveDay(day, archetype, age);
  console.log(`🎉 Day ${day} approved and ready for learners!`);
}

main().catch((error) => {
  console.error('❌ Error:', error.message);
  process.exit(1);
});
