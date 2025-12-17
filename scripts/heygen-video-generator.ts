#!/usr/bin/env npx tsx
/**
 * HEYGEN VIDEO GENERATOR
 * 
 * Production-ready video generation using the Kelly Motion Library.
 * Eliminates the "uncanny valley" by splitting scripts into 8-second scenes
 * with different motion variants, avoiding HeyGen's 10-second loop seam.
 * 
 * Usage:
 *   npx tsx scripts/heygen-video-generator.ts --day 351 --archetype scientist
 *   npx tsx scripts/heygen-video-generator.ts --day 351 --archetype scientist --phase hook
 *   npx tsx scripts/heygen-video-generator.ts --day 351 --archetype scientist --dry-run
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const KELLY_VOICE_ID = '0015ce4f932b405b9fc3a5e2f5e92c46';
const MAX_SCENE_SECONDS = 8;
const WORDS_PER_SECOND = 2.5;

const ARCHETYPES = [
  'scientist', 'explorer', 'rebel', 'architect',
  'diplomat', 'empath', 'macgyver', 'mystic',
  'provider', 'storyteller', 'strategist', 'survivor'
] as const;

type Archetype = typeof ARCHETYPES[number];
type MotionKey = 'A' | 'B' | 'C';

interface MotionLibrary {
  [archetype: string]: {
    A: string;
    B: string;
    C: string;
  };
}

interface LessonPhase {
  script: string;
  duration: number;
  title?: string;
  prompt?: string;
}

interface Lesson {
  meta: { day: number; topic: string };
  phases: { [key: string]: LessonPhase };
  phaseOrder: string[];
}

interface ScriptSegment {
  text: string;
  motion: MotionKey;
  avatarId: string;
  estimatedDuration: number;
}

interface GenerationResult {
  day: number;
  archetype: string;
  videoId: string;
  phases: string[];
  totalScenes: number;
  estimatedDuration: number;
  timestamp: string;
}

// ═══════════════════════════════════════════════════════════════════
// MOTION LIBRARY LOADER
// ═══════════════════════════════════════════════════════════════════

function loadMotionLibrary(): MotionLibrary {
  const libraryPath = path.join(process.cwd(), 'generated-images', 'kelly-motion-library.json');
  
  if (!fs.existsSync(libraryPath)) {
    throw new Error(`Motion library not found at ${libraryPath}`);
  }
  
  const data = JSON.parse(fs.readFileSync(libraryPath, 'utf-8'));
  
  // Validate all archetypes have all motions
  for (const arch of ARCHETYPES) {
    if (!data[arch]) {
      throw new Error(`Missing archetype in motion library: ${arch}`);
    }
    for (const motion of ['A', 'B', 'C'] as MotionKey[]) {
      if (!data[arch][motion]) {
        throw new Error(`Missing motion ${motion} for archetype ${arch}`);
      }
    }
  }
  
  console.log('✅ Motion library loaded (36 avatar IDs)');
  return data;
}

// ═══════════════════════════════════════════════════════════════════
// LESSON LOADER
// ═══════════════════════════════════════════════════════════════════

function loadLesson(day: number): Lesson {
  const lessonPath = path.join(process.cwd(), 'public', 'lessons', `day-${day}.json`);
  
  if (!fs.existsSync(lessonPath)) {
    throw new Error(`Lesson not found at ${lessonPath}`);
  }
  
  const lesson = JSON.parse(fs.readFileSync(lessonPath, 'utf-8'));
  console.log(`✅ Lesson loaded: Day ${day} - "${lesson.meta?.topic || 'Unknown'}"`);
  return lesson;
}

// ═══════════════════════════════════════════════════════════════════
// SCRIPT SPLITTER
// ═══════════════════════════════════════════════════════════════════

function estimateDuration(text: string): number {
  const words = text.split(/\s+/).filter(w => w.length > 0).length;
  return words / WORDS_PER_SECOND;
}

function findNaturalBreak(text: string, targetPosition: number): number {
  const breakChars = ['.', '!', '?', '—', ';', ':'];
  const softBreakChars = [','];
  
  // Search within 40% of target position for hard breaks
  const searchStart = Math.floor(targetPosition * 0.6);
  const searchEnd = Math.ceil(targetPosition * 1.4);
  
  let bestBreak = -1;
  let bestDistance = Infinity;
  
  // First try hard breaks (sentence ends)
  for (let i = searchStart; i < Math.min(searchEnd, text.length); i++) {
    if (breakChars.includes(text[i])) {
      const distance = Math.abs(i - targetPosition);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestBreak = i + 1;
      }
    }
  }
  
  // If no hard break found, try soft breaks (commas)
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
  
  // Last resort: break at word boundary
  if (bestBreak === -1) {
    for (let i = targetPosition; i < text.length; i++) {
      if (text[i] === ' ') {
        return i;
      }
    }
    return text.length;
  }
  
  return bestBreak;
}

/**
 * Get the motion rotation pattern for a phase type
 * A = Warm Welcoming, B = Talk Talk Talk, C = Filler
 */
function getMotionPattern(phaseType: string): MotionKey[] {
  switch (phaseType) {
    case 'hook':
      return ['A', 'B', 'A']; // Warm open, engaging middle, warm close
    case 'cliff':
      return ['B', 'A', 'B']; // Provocative energy
    case 'fact1':
    case 'fact2':
    case 'fact3':
      return ['B', 'C', 'B']; // Teaching with grounded pauses
    case 'wisdom':
      return ['A', 'C', 'A']; // Warm insight, centered reflection
    case 'outro':
      return ['A', 'C', 'A']; // Warm close, settled ending
    default:
      return ['B', 'A', 'B']; // Default teaching pattern
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
  
  // Short scripts: single scene
  if (totalDuration <= MAX_SCENE_SECONDS) {
    const motion = motionPattern[0];
    return [{
      text: script,
      motion,
      avatarId: motionLibrary[archetype][motion],
      estimatedDuration: totalDuration,
    }];
  }
  
  // Calculate number of segments needed
  const numSegments = Math.ceil(totalDuration / MAX_SCENE_SECONDS);
  const segments: ScriptSegment[] = [];
  
  let remaining = script;
  let segmentIndex = 0;
  
  while (remaining.length > 0 && segmentIndex < numSegments) {
    const targetLength = Math.floor(script.length / numSegments);
    const motion = motionPattern[segmentIndex % motionPattern.length];
    
    if (segmentIndex === numSegments - 1) {
      // Last segment gets everything remaining
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

// ═══════════════════════════════════════════════════════════════════
// HEYGEN API
// ═══════════════════════════════════════════════════════════════════

interface VideoInput {
  character: {
    type: 'talking_photo';
    talking_photo_id: string;
  };
  voice: {
    type: 'text';
    input_text: string;
    voice_id: string;
    speed: number;
  };
  background: {
    type: 'color';
    value: string;
  };
}

async function generateVideo(segments: ScriptSegment[], dryRun: boolean = false): Promise<string | null> {
  const videoInputs: VideoInput[] = segments.map(segment => ({
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

  console.log(`\n📤 Video request: ${segments.length} scene(s)`);
  segments.forEach((seg, i) => {
    console.log(`   Scene ${i + 1}: Motion ${seg.motion} (~${seg.estimatedDuration.toFixed(1)}s)`);
    console.log(`            "${seg.text.slice(0, 60)}${seg.text.length > 60 ? '...' : ''}"`);
  });

  if (dryRun) {
    console.log('\n🔍 DRY RUN - No API call made');
    return 'dry-run-video-id';
  }

  const maxRetries = 3;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    if (attempt > 1) {
      console.log(`\n🔄 Retry attempt ${attempt}/${maxRetries}...`);
      await new Promise(r => setTimeout(r, 5000 * attempt)); // Exponential backoff
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
        console.error(`❌ Non-JSON response (${response.status}):`, text.slice(0, 200));
        if (attempt < maxRetries) continue;
        return null;
      }
      
      if (!response.ok) {
        console.error(`❌ Generation failed (${response.status}):`, data.error?.message || JSON.stringify(data));
        if (response.status >= 500 && attempt < maxRetries) continue;
        return null;
      }

      const videoId = data.data?.video_id;
      console.log('✅ Video job started:', videoId);
      return videoId;
      
    } catch (error) {
      console.error(`❌ Network error:`, error);
      if (attempt < maxRetries) continue;
      return null;
    }
  }

  return null;
}

// ═══════════════════════════════════════════════════════════════════
// MANIFEST MANAGEMENT
// ═══════════════════════════════════════════════════════════════════

function saveResult(result: GenerationResult): void {
  const manifestDir = path.join(process.cwd(), 'generated-videos');
  const manifestPath = path.join(manifestDir, `day-${result.day}-manifest.json`);
  
  // Create directory if needed
  if (!fs.existsSync(manifestDir)) {
    fs.mkdirSync(manifestDir, { recursive: true });
  }
  
  // Load existing manifest or create new
  let manifest: any = { day: result.day, generated: new Date().toISOString(), videos: {} };
  if (fs.existsSync(manifestPath)) {
    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  }
  
  // Add/update this archetype
  manifest.videos[result.archetype] = {
    video_id: result.videoId,
    status: 'pending',
    phases: result.phases,
    total_scenes: result.totalScenes,
    estimated_duration: result.estimatedDuration,
    submitted: result.timestamp,
  };
  
  manifest.updated = new Date().toISOString();
  
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\n📝 Manifest saved: ${manifestPath}`);
}

// ═══════════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════

function parseArgs(): { day: number; archetype: Archetype; phase?: string; dryRun: boolean } {
  const args = process.argv.slice(2);
  
  let day: number | undefined;
  let archetype: Archetype | undefined;
  let phase: string | undefined;
  let dryRun = false;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      day = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--archetype' && args[i + 1]) {
      archetype = args[i + 1] as Archetype;
      i++;
    } else if (args[i] === '--phase' && args[i + 1]) {
      phase = args[i + 1];
      i++;
    } else if (args[i] === '--dry-run') {
      dryRun = true;
    }
  }
  
  if (!day || !archetype) {
    console.log('');
    console.log('Usage:');
    console.log('  npx tsx scripts/heygen-video-generator.ts --day 351 --archetype scientist');
    console.log('  npx tsx scripts/heygen-video-generator.ts --day 351 --archetype scientist --phase hook');
    console.log('  npx tsx scripts/heygen-video-generator.ts --day 351 --archetype scientist --dry-run');
    console.log('');
    console.log('Archetypes:', ARCHETYPES.join(', '));
    console.log('');
    process.exit(1);
  }
  
  if (!ARCHETYPES.includes(archetype)) {
    console.error(`Invalid archetype: ${archetype}`);
    console.error('Valid archetypes:', ARCHETYPES.join(', '));
    process.exit(1);
  }
  
  return { day, archetype, phase, dryRun };
}

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN VIDEO GENERATOR                                     ║');
  console.log('║  Multi-motion scene stitching for natural Kelly videos        ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  const { day, archetype, phase: singlePhase, dryRun } = parseArgs();
  
  console.log(`\n📋 Configuration:`);
  console.log(`   Day: ${day}`);
  console.log(`   Archetype: ${archetype}`);
  console.log(`   Phase: ${singlePhase || 'ALL'}`);
  console.log(`   Dry Run: ${dryRun}`);
  
  // Load resources
  const motionLibrary = loadMotionLibrary();
  const lesson = loadLesson(day);
  
  // Determine which phases to process
  const phasesToProcess = singlePhase 
    ? [singlePhase] 
    : lesson.phaseOrder.filter(p => lesson.phases[p]?.script);
  
  console.log(`\n📚 Phases to process: ${phasesToProcess.join(', ')}`);
  
  // Build all segments for all phases
  const allSegments: ScriptSegment[] = [];
  
  for (const phaseName of phasesToProcess) {
    const phase = lesson.phases[phaseName];
    if (!phase?.script) {
      console.log(`   ⏭️  Skipping ${phaseName} (no script)`);
      continue;
    }
    
    console.log(`\n📝 Processing ${phaseName} (${phase.duration}s actual, ~${estimateDuration(phase.script).toFixed(1)}s estimated):`);
    console.log(`   "${phase.script.slice(0, 80)}..."`);
    
    const segments = splitPhaseScript(phase.script, phaseName, archetype, motionLibrary);
    
    segments.forEach((seg, i) => {
      console.log(`   → Scene ${allSegments.length + i + 1}: Motion ${seg.motion} (~${seg.estimatedDuration.toFixed(1)}s)`);
    });
    
    allSegments.push(...segments);
  }
  
  console.log(`\n📊 Total: ${allSegments.length} scenes, ~${allSegments.reduce((sum, s) => sum + s.estimatedDuration, 0).toFixed(1)}s`);
  
  // Generate video
  const videoId = await generateVideo(allSegments, dryRun);
  
  if (videoId && !dryRun) {
    const result: GenerationResult = {
      day,
      archetype,
      videoId,
      phases: phasesToProcess,
      totalScenes: allSegments.length,
      estimatedDuration: allSegments.reduce((sum, s) => sum + s.estimatedDuration, 0),
      timestamp: new Date().toISOString(),
    };
    
    saveResult(result);
    
    console.log('\n════════════════════════════════════════════════════════════════');
    console.log('✅ VIDEO GENERATION STARTED');
    console.log(`   Video ID: ${videoId}`);
    console.log('');
    console.log('   Check status with:');
    console.log(`   npx tsx scripts/heygen-check-status.ts ${videoId}`);
    console.log('════════════════════════════════════════════════════════════════');
  } else if (dryRun) {
    console.log('\n✅ Dry run complete - no video generated');
  } else {
    console.error('\n❌ Video generation failed');
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
