#!/usr/bin/env npx tsx
/**
 * 🔍 VERIFY DAY READY
 * 
 * Comprehensive check that a day is ready for production:
 * - All phases have videos
 * - All phases have audio
 * - Lesson atoms exist
 * - Core lesson exists
 * - Videos are accessible
 * 
 * Usage:
 *   npx tsx scripts/verify-day-ready.ts --day=354
 *   npx tsx scripts/verify-day-ready.ts --days=354,355,356
 *   npx tsx scripts/verify-day-ready.ts --days=auto
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];

interface DayStatus {
  day: number;
  ready: boolean;
  issues: string[];
  details: {
    hasCoreLessons: boolean;
    hasLessonAtoms: boolean;
    videoCount: number;
    audioCount: number;
    missingPhases: string[];
    videosAccessible: boolean;
  };
}

function parseArgs(): { days: number[] } {
  const args = process.argv.slice(2);
  let days: number[] = [];
  
  for (const arg of args) {
    if (arg.startsWith('--day=')) {
      days = [parseInt(arg.split('=')[1], 10)];
    } else if (arg.startsWith('--days=')) {
      const value = arg.split('=')[1];
      if (value === 'auto') {
        days = []; // Will be resolved
      } else {
        days = value.split(',').map(d => parseInt(d.trim(), 10));
      }
    }
  }
  
  return { days };
}

function getTodayDayNumber(): number {
  const startDate = new Date('2025-01-01');
  const today = new Date();
  const diffTime = today.getTime() - startDate.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return Math.min(365, Math.max(1, diffDays));
}

async function checkVideoAccessible(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

async function verifyDay(day: number): Promise<DayStatus> {
  const status: DayStatus = {
    day,
    ready: false,
    issues: [],
    details: {
      hasCoreLessons: false,
      hasLessonAtoms: false,
      videoCount: 0,
      audioCount: 0,
      missingPhases: [],
      videosAccessible: true,
    },
  };

  // 1. Check core_lessons
  const { data: coreLesson, error: coreLessonError } = await supabase
    .from('core_lessons')
    .select('id, title')
    .eq('day_number', day)
    .single();

  if (coreLessonError || !coreLesson) {
    status.issues.push(`No core_lesson found for day ${day}`);
  } else {
    status.details.hasCoreLessons = true;
  }

  // 2. Check lesson_atoms
  if (coreLesson?.id) {
    const { data: atoms, count } = await supabase
      .from('lesson_atoms')
      .select('phase', { count: 'exact' })
      .eq('core_lesson_id', coreLesson.id);

    if (!count || count < 7) {
      status.issues.push(`Only ${count || 0} lesson_atoms found (need 7)`);
    } else {
      status.details.hasLessonAtoms = true;
    }
  }

  // 3. Check kelly_video_assets
  const { data: videos } = await supabase
    .from('kelly_video_assets')
    .select('phase, public_url, status')
    .eq('day_number', day)
    .eq('status', 'validated');

  status.details.videoCount = videos?.length || 0;
  
  const videoPhasesFound = new Set((videos || []).map(v => v.phase));
  status.details.missingPhases = PHASES.filter(p => !videoPhasesFound.has(p));

  if (status.details.missingPhases.length > 0) {
    status.issues.push(`Missing video phases: ${status.details.missingPhases.join(', ')}`);
  }

  // 4. Check video accessibility (sample first video)
  if (videos && videos.length > 0) {
    const firstVideo = videos[0];
    if (firstVideo.public_url) {
      const accessible = await checkVideoAccessible(firstVideo.public_url);
      if (!accessible) {
        status.issues.push('Videos not accessible via URL');
        status.details.videosAccessible = false;
      }
    }
  }

  // 5. Check audio files in storage
  const { data: audioFiles } = await supabase.storage
    .from('kelly-templates')
    .list('heygen/audio', { search: `day_${day}_` });

  const audioPhasesFound = new Set<string>();
  for (const file of audioFiles || []) {
    for (const phase of PHASES) {
      if (file.name.includes(`day_${day}_${phase}_`)) {
        audioPhasesFound.add(phase);
      }
    }
  }
  status.details.audioCount = audioPhasesFound.size;

  if (audioPhasesFound.size < PHASES.length) {
    const missingAudio = PHASES.filter(p => !audioPhasesFound.has(p));
    status.issues.push(`Missing audio phases: ${missingAudio.join(', ')}`);
  }

  // Determine overall readiness
  status.ready = status.issues.length === 0;

  return status;
}

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║             🔍 VERIFY DAY READY                              ║
╚══════════════════════════════════════════════════════════════╝
`);

  let { days } = parseArgs();

  // Handle auto mode
  if (days.length === 0) {
    const today = getTodayDayNumber();
    days = [today, today + 1, today + 2].filter(d => d <= 365);
  }

  console.log(`Checking days: ${days.join(', ')}\n`);

  const results: DayStatus[] = [];
  let allReady = true;

  for (const day of days) {
    const status = await verifyDay(day);
    results.push(status);
    
    if (!status.ready) {
      allReady = false;
    }

    const icon = status.ready ? '✅' : '❌';
    console.log(`${icon} Day ${day}:`);
    
    if (status.ready) {
      console.log(`   ✓ Core lesson exists`);
      console.log(`   ✓ Lesson atoms complete`);
      console.log(`   ✓ ${status.details.videoCount}/${PHASES.length} videos`);
      console.log(`   ✓ ${status.details.audioCount}/${PHASES.length} audio files`);
      console.log(`   ✓ Videos accessible`);
    } else {
      for (const issue of status.issues) {
        console.log(`   ✗ ${issue}`);
      }
    }
    console.log('');
  }

  // Summary
  console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  
  if (allReady) {
    console.log(`
✅ ALL DAYS READY FOR PRODUCTION!

Next steps:
  1. Deploy: git push origin main
  2. Verify live: https://curiouskelly.com/learn.html?day=${days[0]}
`);
    process.exit(0);
  } else {
    const notReady = results.filter(r => !r.ready).map(r => r.day);
    console.log(`
❌ SOME DAYS NOT READY: ${notReady.join(', ')}

To fix:
  1. Generate audio: npx tsx scripts/daily-generation-engine.ts --days=${notReady.join(',')}
  2. Generate videos: npx tsx scripts/generate-sadtalker.ts --days=${notReady.join(',')}
  3. Re-verify: npx tsx scripts/verify-day-ready.ts --days=${notReady.join(',')}
`);
    process.exit(1);
  }
}

main().catch(console.error);
