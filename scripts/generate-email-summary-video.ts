#!/usr/bin/env npx tsx
/**
 * EMAIL SUMMARY VIDEO GENERATOR
 * 
 * Creates ~100 second summary videos for email that include:
 * - LEARN track core content
 * - GROW track activity
 * - Connection between the two
 * 
 * Usage:
 *   npx tsx scripts/generate-email-summary-video.ts --day 351
 *   npx tsx scripts/generate-email-summary-video.ts --day 351 --dry-run
 *   npx tsx scripts/generate-email-summary-video.ts --generate-december
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const KELLY_VOICE_ID = '0015ce4f932b405b9fc3a5e2f5e92c46';

// Using "storyteller" archetype for email summaries - warm and engaging
const SUMMARY_ARCHETYPE = 'storyteller';

interface MotionLibrary {
  [archetype: string]: {
    A: string;
    B: string;
    C: string;
  };
}

interface LessonData {
  meta: { day: number; topic: string; emoji?: string };
  headline?: string;
  phases: {
    hook?: { script: string; duration: number };
    fact1?: { script: string; duration: number };
    fact2?: { script: string; duration: number };
    wisdom?: { script: string; duration: number };
    [key: string]: { script?: string; duration?: number } | undefined;
  };
  growTrack?: {
    title: string;
    emoji?: string;
    learning_objective: string;
    activity: string;
  };
}

interface SummaryScript {
  phase: string;
  text: string;
  motion: 'A' | 'B' | 'C';
  duration: number;
}

// ═══════════════════════════════════════════════════════════════════
// LOADERS
// ═══════════════════════════════════════════════════════════════════

function loadMotionLibrary(): MotionLibrary {
  const libraryPath = path.join(process.cwd(), 'generated-images', 'kelly-motion-library.json');
  if (!fs.existsSync(libraryPath)) {
    throw new Error(`Motion library not found at ${libraryPath}`);
  }
  return JSON.parse(fs.readFileSync(libraryPath, 'utf-8'));
}

function loadLesson(day: number): LessonData | null {
  const lessonPath = path.join(process.cwd(), 'public', 'lessons', `day-${day}.json`);
  if (!fs.existsSync(lessonPath)) {
    console.warn(`⚠️  Lesson not found: ${lessonPath}`);
    return null;
  }
  return JSON.parse(fs.readFileSync(lessonPath, 'utf-8'));
}

function getDateForDay(day: number): string {
  const months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December'];
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  
  let remaining = day;
  let monthIndex = 0;
  
  while (remaining > monthDays[monthIndex]) {
    remaining -= monthDays[monthIndex];
    monthIndex++;
  }
  
  return `${months[monthIndex]} ${remaining}`;
}

// ═══════════════════════════════════════════════════════════════════
// SCRIPT GENERATOR
// ═══════════════════════════════════════════════════════════════════

function condenseSentences(text: string, maxSentences: number): string {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  return sentences.slice(0, maxSentences).join(' ').trim();
}

function generateSummaryScript(lesson: LessonData): SummaryScript[] {
  const day = lesson.meta.day;
  const dateStr = getDateForDay(day);
  const topic = lesson.meta.topic;
  const emoji = lesson.meta.emoji || '📚';
  const grow = lesson.growTrack;
  
  const scripts: SummaryScript[] = [];
  
  // 1. INTRO
  scripts.push({
    phase: 'intro',
    text: `Good morning! I'm Kelly, and this is your daily lesson for ${dateStr}. Today we're exploring two powerful ideas together.`,
    motion: 'A',
    duration: 8
  });
  
  // 2. LEARN HOOK (condensed)
  if (lesson.phases.hook?.script) {
    const hookText = condenseSentences(lesson.phases.hook.script, 3);
    scripts.push({
      phase: 'learn_hook',
      text: hookText,
      motion: 'B',
      duration: Math.ceil(hookText.split(/\s+/).length / 2.5)
    });
  }
  
  // 3. LEARN CORE (fact1 + condensed fact2)
  let coreText = '';
  if (lesson.phases.fact1?.script) {
    coreText += lesson.phases.fact1.script + ' ';
  }
  if (lesson.phases.fact2?.script) {
    coreText += condenseSentences(lesson.phases.fact2.script, 2);
  }
  if (coreText) {
    scripts.push({
      phase: 'learn_core',
      text: coreText.trim(),
      motion: 'B',
      duration: Math.ceil(coreText.split(/\s+/).length / 2.5)
    });
  }
  
  // 4. LEARN WISDOM (condensed)
  if (lesson.phases.wisdom?.script) {
    const wisdomText = condenseSentences(lesson.phases.wisdom.script, 2);
    scripts.push({
      phase: 'learn_wisdom',
      text: `Today's wisdom: ${wisdomText}`,
      motion: 'A',
      duration: Math.ceil(wisdomText.split(/\s+/).length / 2.5) + 2
    });
  }
  
  // 5. TRANSITION
  scripts.push({
    phase: 'transition',
    text: "Now let's take what we learned and put it into practice.",
    motion: 'C',
    duration: 4
  });
  
  // 6. GROW INTRO
  if (grow) {
    scripts.push({
      phase: 'grow_intro',
      text: `Today's growth challenge is ${grow.title}.`,
      motion: 'A',
      duration: 6
    });
    
    // 7. GROW ACTIVITY
    scripts.push({
      phase: 'grow_activity',
      text: `Here's your activity: ${grow.activity} ${grow.learning_objective ? `This helps you ${grow.learning_objective.toLowerCase()}.` : ''}`,
      motion: 'A',
      duration: Math.ceil((grow.activity + (grow.learning_objective || '')).split(/\s+/).length / 2.5) + 2
    });
    
    // 8. CONNECTION
    scripts.push({
      phase: 'grow_connect',
      text: `And here's how today's lessons connect: What you learned about ${topic.toLowerCase()} can be applied to ${grow.title.toLowerCase()}. Practice makes progress.`,
      motion: 'B',
      duration: 10
    });
  }
  
  // 9. CLOSE
  scripts.push({
    phase: 'close',
    text: "That's today's lesson. Learn something, grow a little. I'll see you tomorrow with something new. Stay curious!",
    motion: 'A',
    duration: 8
  });
  
  return scripts;
}

// ═══════════════════════════════════════════════════════════════════
// HEYGEN API
// ═══════════════════════════════════════════════════════════════════

async function generateVideo(
  scripts: SummaryScript[],
  motionLibrary: MotionLibrary,
  dryRun: boolean
): Promise<string | null> {
  const archetype = SUMMARY_ARCHETYPE;
  
  const videoInputs = scripts.map(script => ({
    character: {
      type: 'talking_photo',
      talking_photo_id: motionLibrary[archetype][script.motion],
    },
    voice: {
      type: 'text',
      input_text: script.text,
      voice_id: KELLY_VOICE_ID,
      speed: 1.0,
    },
    background: {
      type: 'color',
      value: '#1a1a2e',
    },
  }));

  console.log(`\n📤 Summary video: ${scripts.length} scenes`);
  scripts.forEach((s, i) => {
    console.log(`   ${i + 1}. [${s.phase}] Motion ${s.motion} (~${s.duration}s)`);
    console.log(`      "${s.text.slice(0, 60)}${s.text.length > 60 ? '...' : ''}"`);
  });

  const totalDuration = scripts.reduce((sum, s) => sum + s.duration, 0);
  console.log(`\n📊 Estimated total: ${totalDuration}s`);

  if (dryRun) {
    console.log('\n🔍 DRY RUN - No API call made');
    return 'dry-run-summary-id';
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

    const data = await response.json();
    
    if (!response.ok) {
      console.error(`❌ Generation failed (${response.status}):`, data.error?.message || JSON.stringify(data));
      return null;
    }

    const videoId = data.data?.video_id;
    console.log('✅ Summary video job started:', videoId);
    return videoId;
    
  } catch (error) {
    console.error('❌ Network error:', error);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════
// MANIFEST
// ═══════════════════════════════════════════════════════════════════

function saveSummaryManifest(day: number, scripts: SummaryScript[], videoId: string | null): void {
  const outputDir = path.join(process.cwd(), 'content', 'email-summary-video');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  const manifest = {
    day,
    date: getDateForDay(day),
    type: 'email-summary',
    generated: new Date().toISOString(),
    videoId,
    status: videoId ? 'submitted' : 'script-only',
    totalDuration: scripts.reduce((sum, s) => sum + s.duration, 0),
    scenes: scripts.length,
    script: scripts,
  };
  
  const outputPath = path.join(outputDir, `day-${String(day).padStart(3, '0')}-summary-manifest.json`);
  fs.writeFileSync(outputPath, JSON.stringify(manifest, null, 2));
  console.log(`\n📝 Manifest saved: ${outputPath}`);
}

// ═══════════════════════════════════════════════════════════════════
// DECEMBER BATCH
// ═══════════════════════════════════════════════════════════════════

async function generateDecemberScripts(): Promise<void> {
  console.log('\n📅 Generating December summary scripts (Days 335-365)...\n');
  
  const outputDir = path.join(process.cwd(), 'content', 'email-summary-video', 'december');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  const results: { day: number; date: string; topic: string; hasGrow: boolean; scenes: number }[] = [];
  
  for (let day = 335; day <= 365; day++) {
    const lesson = loadLesson(day);
    if (!lesson) {
      console.log(`⏭️  Day ${day}: No lesson found`);
      continue;
    }
    
    const scripts = generateSummaryScript(lesson);
    const dateStr = getDateForDay(day);
    
    const summary = {
      day,
      date: dateStr,
      topic: lesson.meta.topic,
      emoji: lesson.meta.emoji,
      headline: lesson.headline,
      growTrack: lesson.growTrack,
      scripts,
      totalDuration: scripts.reduce((sum, s) => sum + s.duration, 0),
    };
    
    const outputPath = path.join(outputDir, `day-${String(day).padStart(3, '0')}-summary.json`);
    fs.writeFileSync(outputPath, JSON.stringify(summary, null, 2));
    
    results.push({
      day,
      date: dateStr,
      topic: lesson.meta.topic,
      hasGrow: !!lesson.growTrack,
      scenes: scripts.length,
    });
    
    console.log(`✅ Day ${day} (${dateStr}): "${lesson.meta.topic}" - ${scripts.length} scenes`);
  }
  
  // Summary report
  console.log('\n════════════════════════════════════════════════════════════════');
  console.log('📊 DECEMBER SUMMARY SCRIPTS GENERATED');
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`Total days: ${results.length}`);
  console.log(`With GROW track: ${results.filter(r => r.hasGrow).length}`);
  console.log(`Output: ${outputDir}`);
  console.log('════════════════════════════════════════════════════════════════');
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  
  let day: number | undefined;
  let dryRun = false;
  let generateDecember = false;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day' && args[i + 1]) {
      day = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--dry-run') {
      dryRun = true;
    } else if (args[i] === '--generate-december') {
      generateDecember = true;
    }
  }
  
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  📧 EMAIL SUMMARY VIDEO GENERATOR                              ║');
  console.log('║  Combined LEARN + GROW for email delivery                      ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  if (generateDecember) {
    await generateDecemberScripts();
    return;
  }
  
  if (!day) {
    console.log('\nUsage:');
    console.log('  npx tsx scripts/generate-email-summary-video.ts --day 351');
    console.log('  npx tsx scripts/generate-email-summary-video.ts --day 351 --dry-run');
    console.log('  npx tsx scripts/generate-email-summary-video.ts --generate-december');
    process.exit(1);
  }
  
  const motionLibrary = loadMotionLibrary();
  const lesson = loadLesson(day);
  
  if (!lesson) {
    console.error(`\n❌ Could not load lesson for day ${day}`);
    process.exit(1);
  }
  
  console.log(`\n📋 Day ${day}: "${lesson.meta.topic}"`);
  console.log(`   LEARN: ${lesson.meta.emoji || '📚'} ${lesson.meta.topic}`);
  console.log(`   GROW: ${lesson.growTrack ? `${lesson.growTrack.emoji || '🎯'} ${lesson.growTrack.title}` : '(none)'}`);
  
  const scripts = generateSummaryScript(lesson);
  
  const videoId = await generateVideo(scripts, motionLibrary, dryRun);
  
  saveSummaryManifest(day, scripts, videoId);
  
  if (videoId && !dryRun) {
    console.log('\n════════════════════════════════════════════════════════════════');
    console.log('✅ SUMMARY VIDEO GENERATION STARTED');
    console.log(`   Video ID: ${videoId}`);
    console.log('');
    console.log('   Check status with:');
    console.log(`   npx tsx scripts/heygen-check-status.ts ${videoId}`);
    console.log('════════════════════════════════════════════════════════════════');
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
