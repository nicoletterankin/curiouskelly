#!/usr/bin/env npx tsx
/**
 * HEYGEN BATCH GENERATOR
 * 
 * Generates Kelly videos for all 12 archetypes for a given day.
 * Includes rate limiting, cost estimation, and resume capability.
 * 
 * Usage:
 *   npx tsx scripts/heygen-batch-generate.ts --day 351
 *   npx tsx scripts/heygen-batch-generate.ts --day 351 --dry-run
 *   npx tsx scripts/heygen-batch-generate.ts --day 351 --skip scientist,explorer
 *   npx tsx scripts/heygen-batch-generate.ts --day 351 --only scientist,explorer
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const KELLY_VOICE_ID = '0015ce4f932b405b9fc3a5e2f5e92c46';
const MAX_SCENE_SECONDS = 8;
const WORDS_PER_SECOND = 2.5;

// Rate limiting: wait between API calls
const DELAY_BETWEEN_VIDEOS_MS = 5000;

// Credit estimation (~0.5 credits per 30s of video)
const CREDITS_PER_30_SECONDS = 0.5;

const ARCHETYPES = [
  'scientist', 'explorer', 'rebel', 'architect',
  'diplomat', 'empath', 'macgyver', 'mystic',
  'provider', 'storyteller', 'strategist', 'survivor'
] as const;

type Archetype = typeof ARCHETYPES[number];
type MotionKey = 'A' | 'B' | 'C';

interface MotionLibrary {
  [archetype: string]: { A: string; B: string; C: string; };
}

interface LessonPhase {
  script: string;
  duration: number;
}

interface Lesson {
  meta: { day: number; topic: string };
  phases: { [key: string]: LessonPhase };
  phaseOrder: string[];
  totalDuration: number;
}

interface ScriptSegment {
  text: string;
  motion: MotionKey;
  avatarId: string;
  estimatedDuration: number;
}

interface BatchResult {
  archetype: string;
  success: boolean;
  videoId?: string;
  error?: string;
  scenes: number;
  duration: number;
}

// ═══════════════════════════════════════════════════════════════════
// LOADERS
// ═══════════════════════════════════════════════════════════════════

function loadMotionLibrary(): MotionLibrary {
  const libraryPath = path.join(process.cwd(), 'generated-images', 'kelly-motion-library.json');
  return JSON.parse(fs.readFileSync(libraryPath, 'utf-8'));
}

function loadLesson(day: number): Lesson {
  const lessonPath = path.join(process.cwd(), 'public', 'lessons', `day-${day}.json`);
  return JSON.parse(fs.readFileSync(lessonPath, 'utf-8'));
}

function loadManifest(day: number): any {
  const manifestPath = path.join(process.cwd(), 'generated-videos', `day-${day}-manifest.json`);
  if (fs.existsSync(manifestPath)) {
    return JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  }
  return { day, generated: new Date().toISOString(), videos: {} };
}

function saveManifest(manifest: any): void {
  const manifestDir = path.join(process.cwd(), 'generated-videos');
  if (!fs.existsSync(manifestDir)) {
    fs.mkdirSync(manifestDir, { recursive: true });
  }
  const manifestPath = path.join(manifestDir, `day-${manifest.day}-manifest.json`);
  manifest.updated = new Date().toISOString();
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}

// ═══════════════════════════════════════════════════════════════════
// SCRIPT SPLITTING (same logic as video-generator)
// ═══════════════════════════════════════════════════════════════════

function estimateDuration(text: string): number {
  return text.split(/\s+/).filter(w => w.length > 0).length / WORDS_PER_SECOND;
}

function findNaturalBreak(text: string, targetPosition: number): number {
  const breakChars = ['.', '!', '?', '—', ';', ':'];
  const softBreakChars = [','];
  const searchStart = Math.floor(targetPosition * 0.6);
  const searchEnd = Math.ceil(targetPosition * 1.4);
  
  let bestBreak = -1;
  let bestDistance = Infinity;
  
  for (let i = searchStart; i < Math.min(searchEnd, text.length); i++) {
    if (breakChars.includes(text[i])) {
      const distance = Math.abs(i - targetPosition);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestBreak = i + 1;
      }
    }
  }
  
  if (bestBreak === -1) {
    for (let i = searchStart; i < Math.min(searchEnd, text.length); i++) {
      if (softBreakChars.includes(text[i])) {
        const distance = Math.abs(i - targetPosition);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestBreak = i + 1;
        }
      }
    }
  }
  
  if (bestBreak === -1) {
    for (let i = targetPosition; i < text.length; i++) {
      if (text[i] === ' ') return i;
    }
    return text.length;
  }
  
  return bestBreak;
}

function getMotionPattern(phaseType: string): MotionKey[] {
  switch (phaseType) {
    case 'hook': return ['A', 'B', 'A'];
    case 'cliff': return ['B', 'A', 'B'];
    case 'fact1':
    case 'fact2':
    case 'fact3': return ['B', 'C', 'B'];
    case 'wisdom': return ['A', 'C', 'A'];
    case 'outro': return ['A', 'C', 'A'];
    default: return ['B', 'A', 'B'];
  }
}

function splitPhaseScript(
  script: string,
  phaseType: string,
  archetype: Archetype,
  motionLibrary: MotionLibrary
): ScriptSegment[] {
  const totalDuration = estimateDuration(script);
  const motionPattern = getMotionPattern(phaseType);
  
  if (totalDuration <= MAX_SCENE_SECONDS) {
    const motion = motionPattern[0];
    return [{
      text: script,
      motion,
      avatarId: motionLibrary[archetype][motion],
      estimatedDuration: totalDuration,
    }];
  }
  
  const numSegments = Math.ceil(totalDuration / MAX_SCENE_SECONDS);
  const segments: ScriptSegment[] = [];
  let remaining = script;
  let segmentIndex = 0;
  
  while (remaining.length > 0 && segmentIndex < numSegments) {
    const targetLength = Math.floor(script.length / numSegments);
    const motion = motionPattern[segmentIndex % motionPattern.length];
    
    if (segmentIndex === numSegments - 1) {
      segments.push({
        text: remaining.trim(),
        motion,
        avatarId: motionLibrary[archetype][motion],
        estimatedDuration: estimateDuration(remaining),
      });
      break;
    }
    
    const breakPoint = findNaturalBreak(remaining, targetLength);
    const segmentText = remaining.slice(0, breakPoint).trim();
    
    segments.push({
      text: segmentText,
      motion,
      avatarId: motionLibrary[archetype][motion],
      estimatedDuration: estimateDuration(segmentText),
    });
    
    remaining = remaining.slice(breakPoint).trim();
    segmentIndex++;
  }
  
  return segments;
}

function buildSegmentsForArchetype(
  lesson: Lesson,
  archetype: Archetype,
  motionLibrary: MotionLibrary
): ScriptSegment[] {
  const allSegments: ScriptSegment[] = [];
  
  for (const phaseName of lesson.phaseOrder) {
    const phase = lesson.phases[phaseName];
    if (!phase?.script) continue;
    
    const segments = splitPhaseScript(phase.script, phaseName, archetype, motionLibrary);
    allSegments.push(...segments);
  }
  
  return allSegments;
}

// ═══════════════════════════════════════════════════════════════════
// VIDEO GENERATION
// ═══════════════════════════════════════════════════════════════════

async function generateVideo(segments: ScriptSegment[]): Promise<string | null> {
  const videoInputs = segments.map(segment => ({
    character: {
      type: 'talking_photo',
      talking_photo_id: segment.avatarId,
    },
    voice: {
      type: 'text',
      input_text: segment.text,
      voice_id: KELLY_VOICE_ID,
      speed: 1.0,
    },
    background: {
      type: 'color',
      value: '#1a1a2e',
    },
  }));

  const maxRetries = 3;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    if (attempt > 1) {
      await new Promise(r => setTimeout(r, 5000 * attempt));
    }

    try {
      const response = await fetch('https://api.heygen.com/v2/video/generate', {
        method: 'POST',
        headers: {
          'X-Api-Key': HEYGEN_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_inputs: videoInputs,
          dimension: { width: 1920, height: 1080 },
        }),
      });

      const text = await response.text();
      let data: any;
      
      try {
        data = JSON.parse(text);
      } catch (e) {
        if (attempt < maxRetries) continue;
        return null;
      }
      
      if (!response.ok) {
        if (response.status >= 500 && attempt < maxRetries) continue;
        return null;
      }

      return data.data?.video_id;
      
    } catch (error) {
      if (attempt < maxRetries) continue;
      return null;
    }
  }

  return null;
}

// ═══════════════════════════════════════════════════════════════════
// BATCH EXECUTION
// ═══════════════════════════════════════════════════════════════════

async function runBatch(
  day: number,
  archetypesToRun: Archetype[],
  dryRun: boolean
): Promise<BatchResult[]> {
  const motionLibrary = loadMotionLibrary();
  const lesson = loadLesson(day);
  const manifest = loadManifest(day);
  
  console.log(`\n📋 Day ${day}: "${lesson.meta?.topic}"`);
  console.log(`   Total lesson duration: ${lesson.totalDuration}s`);
  console.log(`   Archetypes to generate: ${archetypesToRun.length}`);
  
  // Calculate estimates
  let totalScenes = 0;
  let totalDuration = 0;
  
  const archetypeData: { archetype: Archetype; segments: ScriptSegment[] }[] = [];
  
  for (const archetype of archetypesToRun) {
    const segments = buildSegmentsForArchetype(lesson, archetype, motionLibrary);
    const duration = segments.reduce((sum, s) => sum + s.estimatedDuration, 0);
    
    archetypeData.push({ archetype, segments });
    totalScenes += segments.length;
    totalDuration += duration;
  }
  
  const estimatedCredits = (totalDuration / 30) * CREDITS_PER_30_SECONDS;
  
  console.log(`\n📊 Batch Estimates:`);
  console.log(`   Total scenes: ${totalScenes}`);
  console.log(`   Total duration: ~${Math.round(totalDuration)}s (~${Math.round(totalDuration/60)}min)`);
  console.log(`   Estimated credits: ~${estimatedCredits.toFixed(1)}`);
  
  if (dryRun) {
    console.log('\n🔍 DRY RUN - Showing what would be generated:\n');
    
    for (const { archetype, segments } of archetypeData) {
      const duration = segments.reduce((sum, s) => sum + s.estimatedDuration, 0);
      console.log(`   ${archetype.padEnd(12)} │ ${segments.length} scenes │ ~${duration.toFixed(0)}s`);
    }
    
    console.log('\n✅ Dry run complete. Run without --dry-run to generate.');
    return [];
  }
  
  // Execute batch
  console.log('\n🚀 Starting batch generation...\n');
  
  const results: BatchResult[] = [];
  
  for (let i = 0; i < archetypeData.length; i++) {
    const { archetype, segments } = archetypeData[i];
    const duration = segments.reduce((sum, s) => sum + s.estimatedDuration, 0);
    
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`[${i + 1}/${archetypeData.length}] ${archetype.toUpperCase()}`);
    console.log(`   ${segments.length} scenes, ~${duration.toFixed(0)}s`);
    
    const videoId = await generateVideo(segments);
    
    if (videoId) {
      console.log(`   ✅ Started: ${videoId}`);
      
      manifest.videos[archetype] = {
        video_id: videoId,
        status: 'pending',
        phases: lesson.phaseOrder.filter(p => lesson.phases[p]?.script),
        total_scenes: segments.length,
        estimated_duration: duration,
        submitted: new Date().toISOString(),
      };
      
      results.push({
        archetype,
        success: true,
        videoId,
        scenes: segments.length,
        duration,
      });
    } else {
      console.log(`   ❌ Failed to start`);
      
      results.push({
        archetype,
        success: false,
        error: 'Generation failed',
        scenes: segments.length,
        duration,
      });
    }
    
    // Save manifest after each video (for resume capability)
    saveManifest(manifest);
    
    // Rate limiting delay (except for last video)
    if (i < archetypeData.length - 1) {
      console.log(`   ⏳ Waiting ${DELAY_BETWEEN_VIDEOS_MS / 1000}s before next...`);
      await new Promise(r => setTimeout(r, DELAY_BETWEEN_VIDEOS_MS));
    }
  }
  
  return results;
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

function parseArgs(): {
  day: number;
  dryRun: boolean;
  skip: Archetype[];
  only: Archetype[];
} {
  const args = process.argv.slice(2);
  
  let day: number | undefined;
  let dryRun = false;
  let skip: Archetype[] = [];
  let only: Archetype[] = [];
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      day = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--dry-run') {
      dryRun = true;
    } else if (args[i] === '--skip' && args[i + 1]) {
      skip = args[i + 1].split(',').map(s => s.trim() as Archetype);
      i++;
    } else if (args[i] === '--only' && args[i + 1]) {
      only = args[i + 1].split(',').map(s => s.trim() as Archetype);
      i++;
    }
  }
  
  if (!day) {
    console.log('');
    console.log('Usage:');
    console.log('  npx tsx scripts/heygen-batch-generate.ts --day 351');
    console.log('  npx tsx scripts/heygen-batch-generate.ts --day 351 --dry-run');
    console.log('  npx tsx scripts/heygen-batch-generate.ts --day 351 --skip scientist,explorer');
    console.log('  npx tsx scripts/heygen-batch-generate.ts --day 351 --only scientist,explorer');
    console.log('');
    process.exit(1);
  }
  
  return { day, dryRun, skip, only };
}

async function main(): Promise<void> {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN BATCH GENERATOR                                     ║');
  console.log('║  Generate Kelly videos for all 12 archetypes                  ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  const { day, dryRun, skip, only } = parseArgs();
  
  // Determine which archetypes to run
  let archetypesToRun: Archetype[];
  
  if (only.length > 0) {
    archetypesToRun = only.filter(a => ARCHETYPES.includes(a));
    console.log(`\n📋 Running ONLY: ${archetypesToRun.join(', ')}`);
  } else {
    archetypesToRun = ARCHETYPES.filter(a => !skip.includes(a));
    if (skip.length > 0) {
      console.log(`\n📋 Skipping: ${skip.join(', ')}`);
    }
  }
  
  // Check for already-completed in manifest
  const manifest = loadManifest(day);
  const alreadyDone = archetypesToRun.filter(a => 
    manifest.videos[a]?.status === 'completed' ||
    manifest.videos[a]?.status === 'pending' ||
    manifest.videos[a]?.status === 'processing'
  );
  
  if (alreadyDone.length > 0 && !dryRun) {
    console.log(`\n⚠️  Already in manifest: ${alreadyDone.join(', ')}`);
    console.log(`   Add --skip ${alreadyDone.join(',')} to exclude, or remove from manifest to regenerate`);
  }
  
  const results = await runBatch(day, archetypesToRun, dryRun);
  
  if (results.length > 0) {
    // Summary
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);
    
    console.log('\n════════════════════════════════════════════════════════════════');
    console.log('📊 BATCH COMPLETE');
    console.log(`   ✅ Successful: ${successful.length}`);
    console.log(`   ❌ Failed: ${failed.length}`);
    console.log('');
    console.log('   Check status with:');
    console.log(`   npx tsx scripts/heygen-check-status.ts --day ${day}`);
    console.log('');
    console.log('   Or poll until complete:');
    console.log(`   npx tsx scripts/heygen-check-status.ts --day ${day} --poll`);
    console.log('════════════════════════════════════════════════════════════════');
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
