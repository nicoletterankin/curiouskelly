#!/usr/bin/env npx tsx
/**
 * HEYGEN PHASE GENERATOR
 * 
 * Generates individual phase clips for the app (not full day summaries).
 * Each phase becomes its own video file for interactive playback.
 * 
 * Usage:
 *   npx tsx scripts/heygen-phase-generator.ts --day 351 --archetype scientist
 *   npx tsx scripts/heygen-phase-generator.ts --day 351 --archetype scientist --phase hook
 *   npx tsx scripts/heygen-phase-generator.ts --day 351 --archetype scientist --dry-run
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

const PHASES = ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro'] as const;

type Archetype = typeof ARCHETYPES[number];
type Phase = typeof PHASES[number];
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
}

interface ScriptSegment {
  text: string;
  motion: MotionKey;
  avatarId: string;
  estimatedDuration: number;
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

// ═══════════════════════════════════════════════════════════════════
// SCRIPT SPLITTING
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

// ═══════════════════════════════════════════════════════════════════
// VIDEO GENERATION
// ═══════════════════════════════════════════════════════════════════

async function generatePhaseVideo(
  segments: ScriptSegment[],
  dryRun: boolean = false
): Promise<string | null> {
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

  if (dryRun) {
    return 'dry-run-video-id';
  }

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
        console.error(`   ❌ Non-JSON response:`, text.slice(0, 100));
        if (attempt < maxRetries) continue;
        return null;
      }
      
      if (!response.ok) {
        console.error(`   ❌ Error (${response.status}):`, data.error?.message || JSON.stringify(data));
        if (response.status >= 500 && attempt < maxRetries) continue;
        return null;
      }

      return data.data?.video_id;
      
    } catch (error) {
      console.error(`   ❌ Network error:`, error);
      if (attempt < maxRetries) continue;
      return null;
    }
  }

  return null;
}

// ═══════════════════════════════════════════════════════════════════
// MANIFEST
// ═══════════════════════════════════════════════════════════════════

interface PhaseManifest {
  day: number;
  archetype: string;
  generated: string;
  phases: {
    [phase: string]: {
      video_id: string;
      status: string;
      scenes: number;
      duration: number;
      submitted: string;
    };
  };
}

function savePhaseManifest(day: number, archetype: string, phase: string, videoId: string, scenes: number, duration: number): void {
  const manifestDir = path.join(process.cwd(), 'generated-videos', 'phases');
  const manifestPath = path.join(manifestDir, `day-${day}-${archetype}.json`);
  
  if (!fs.existsSync(manifestDir)) {
    fs.mkdirSync(manifestDir, { recursive: true });
  }
  
  let manifest: PhaseManifest = {
    day,
    archetype,
    generated: new Date().toISOString(),
    phases: {}
  };
  
  if (fs.existsSync(manifestPath)) {
    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  }
  
  manifest.phases[phase] = {
    video_id: videoId,
    status: 'pending',
    scenes,
    duration,
    submitted: new Date().toISOString(),
  };
  
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  
  let day: number | undefined;
  let archetype: Archetype | undefined;
  let phase: Phase | undefined;
  let dryRun = false;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      day = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--archetype' && args[i + 1]) {
      archetype = args[i + 1] as Archetype;
      i++;
    } else if (args[i] === '--phase' && args[i + 1]) {
      phase = args[i + 1] as Phase;
      i++;
    } else if (args[i] === '--dry-run') {
      dryRun = true;
    }
  }
  
  if (!day || !archetype) {
    console.log('');
    console.log('Usage:');
    console.log('  npx tsx scripts/heygen-phase-generator.ts --day 351 --archetype scientist');
    console.log('  npx tsx scripts/heygen-phase-generator.ts --day 351 --archetype scientist --phase hook');
    console.log('  npx tsx scripts/heygen-phase-generator.ts --day 351 --archetype scientist --dry-run');
    console.log('');
    process.exit(1);
  }
  
  return { day, archetype, phase, dryRun };
}

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎬 HEYGEN PHASE GENERATOR                                     ║');
  console.log('║  Individual phase clips for app playback                       ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  const { day, archetype, phase: singlePhase, dryRun } = parseArgs();
  
  console.log(`\n📋 Configuration:`);
  console.log(`   Day: ${day}`);
  console.log(`   Archetype: ${archetype}`);
  console.log(`   Phase: ${singlePhase || 'ALL (7 phases)'}`);
  console.log(`   Dry Run: ${dryRun}`);
  
  const motionLibrary = loadMotionLibrary();
  const lesson = loadLesson(day);
  
  console.log(`\n✅ Lesson: Day ${day} - "${lesson.meta?.topic}"`);
  
  const phasesToProcess = singlePhase ? [singlePhase] : PHASES.filter(p => lesson.phases[p]?.script);
  
  console.log(`\n📚 Generating ${phasesToProcess.length} phase clip(s):\n`);
  
  let totalScenes = 0;
  let totalDuration = 0;
  let successCount = 0;
  
  for (const phaseName of phasesToProcess) {
    const phaseData = lesson.phases[phaseName];
    if (!phaseData?.script) {
      console.log(`   ⏭️  ${phaseName}: No script, skipping`);
      continue;
    }
    
    const segments = splitPhaseScript(phaseData.script, phaseName, archetype, motionLibrary);
    const duration = segments.reduce((sum, s) => sum + s.estimatedDuration, 0);
    
    console.log(`   📹 ${phaseName.toUpperCase()}`);
    console.log(`      ${segments.length} scene(s), ~${duration.toFixed(1)}s`);
    segments.forEach((seg, i) => {
      console.log(`      Scene ${i + 1}: Motion ${seg.motion} (~${seg.estimatedDuration.toFixed(1)}s)`);
    });
    
    const videoId = await generatePhaseVideo(segments, dryRun);
    
    if (videoId) {
      console.log(`      ✅ Video ID: ${videoId}`);
      
      if (!dryRun) {
        savePhaseManifest(day, archetype, phaseName, videoId, segments.length, duration);
      }
      
      successCount++;
      totalScenes += segments.length;
      totalDuration += duration;
    } else {
      console.log(`      ❌ Failed to generate`);
    }
    
    // Rate limiting delay between phases
    if (!dryRun && phasesToProcess.indexOf(phaseName as Phase) < phasesToProcess.length - 1) {
      await new Promise(r => setTimeout(r, 2000));
    }
    
    console.log('');
  }
  
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`📊 SUMMARY: ${successCount}/${phasesToProcess.length} phases generated`);
  console.log(`   Total scenes: ${totalScenes}`);
  console.log(`   Total duration: ~${totalDuration.toFixed(1)}s`);
  
  if (!dryRun && successCount > 0) {
    console.log(`\n   Manifest: generated-videos/phases/day-${day}-${archetype}.json`);
  }
  console.log('════════════════════════════════════════════════════════════════');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
