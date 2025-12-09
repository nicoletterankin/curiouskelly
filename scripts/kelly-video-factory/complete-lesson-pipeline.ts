#!/usr/bin/env npx tsx
/**
 * 🎬 COMPLETE LESSON PIPELINE
 * 
 * Generates ALL videos for the complete 5-phase journey:
 * - Main script videos (5 per archetype)
 * - Response videos for each option (12 per archetype)
 * - Total: 17 videos per archetype, 51 per day
 * 
 * USAGE:
 *   npx tsx scripts/kelly-video-factory/complete-lesson-pipeline.ts --day 1
 *   npx tsx scripts/kelly-video-factory/complete-lesson-pipeline.ts --day 1 --archetype "The Explorer"
 *   npx tsx scripts/kelly-video-factory/complete-lesson-pipeline.ts --day 1 --phase Hook
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import { generateHDVideo, CONFIG as HD_CONFIG } from './hd-golden-lesson-pipeline.js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  OUTPUT_DIR: path.join(process.cwd(), 'generated-videos', 'complete-lessons'),
};

// =============================================================================
// TYPES
// =============================================================================

interface VideoTask {
  dayNumber: number;
  archetype: string;
  phase: string;
  videoType: 'main' | 'response_A' | 'response_B' | 'response_C';
  script: string;
  outputPath: string;
}

interface GenerationResult {
  task: VideoTask;
  success: boolean;
  videoPath?: string;
  error?: string;
  duration: number;
}

interface LessonContent {
  script: string;
  options: Array<{
    text: string;
    letter: string;
    quality: string;
    response: string;
  }>;
}

// =============================================================================
// SUPABASE CLIENT
// =============================================================================

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function log(emoji: string, message: string, indent = 0): void {
  const prefix = '  '.repeat(indent);
  console.log(`${prefix}${emoji} ${message}`);
}

// =============================================================================
// FETCH LESSON CONTENT
// =============================================================================

async function fetchLessonContent(
  dayNumber: number,
  archetype: string,
  phase: string
): Promise<LessonContent | null> {
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();

  if (!lesson) {
    log('❌', `No lesson found for day ${dayNumber}`, 1);
    return null;
  }

  const { data: atom } = await supabase
    .from('lesson_atoms')
    .select('content')
    .eq('core_lesson_id', lesson.id)
    .eq('archetype', archetype)
    .eq('phase', phase)
    .single();

  if (!atom || !atom.content) {
    log('❌', `No content found for ${archetype} / ${phase}`, 1);
    return null;
  }

  return atom.content as LessonContent;
}

// =============================================================================
// CREATE VIDEO TASKS
// =============================================================================

async function createVideoTasks(
  dayNumber: number,
  archetypeFilter?: string,
  phaseFilter?: string
): Promise<VideoTask[]> {
  const tasks: VideoTask[] = [];
  
  const archetypes = archetypeFilter 
    ? [archetypeFilter] 
    : ['The Explorer', 'The Rebel', 'The Scientist'];
  
  const phases = phaseFilter 
    ? [phaseFilter] 
    : ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'];

  for (const archetype of archetypes) {
    for (const phase of phases) {
      const content = await fetchLessonContent(dayNumber, archetype, phase);
      
      if (!content) continue;

      const baseDir = path.join(
        CONFIG.OUTPUT_DIR,
        `day_${String(dayNumber).padStart(3, '0')}`,
        archetype.replace(/\s+/g, '_'),
        phase
      );

      // Main script video
      tasks.push({
        dayNumber,
        archetype,
        phase,
        videoType: 'main',
        script: content.script,
        outputPath: path.join(baseDir, 'main.mp4'),
      });

      // Response videos (except for Wisdom phase)
      if (phase !== 'Wisdom' && content.options) {
        for (const option of content.options) {
          tasks.push({
            dayNumber,
            archetype,
            phase,
            videoType: `response_${option.letter}` as any,
            script: option.response,
            outputPath: path.join(baseDir, `response_${option.letter}.mp4`),
          });
        }
      }
    }
  }

  return tasks;
}

// =============================================================================
// GENERATE VIDEO
// =============================================================================

async function generateVideo(task: VideoTask): Promise<GenerationResult> {
  const startTime = Date.now();
  
  log('🎬', `${task.archetype} / ${task.phase} / ${task.videoType}`, 1);
  log('📝', `"${task.script.substring(0, 60)}..."`, 2);

  try {
    // Use the existing HD pipeline for generation
    // We'll temporarily write the script to a temp file
    const tempDir = path.join(CONFIG.OUTPUT_DIR, 'temp');
    fs.mkdirSync(tempDir, { recursive: true });
    
    const tempScriptFile = path.join(tempDir, `${Date.now()}_script.txt`);
    fs.writeFileSync(tempScriptFile, task.script);

    // Generate video using HD pipeline
    // Note: This uses the existing generateHDVideo function
    // which handles ElevenLabs → Flux → MiniMax → Sync Labs
    const result = await generateHDVideo(
      task.archetype,
      task.phase,
      task.dayNumber
    );

    // Clean up temp file
    fs.unlinkSync(tempScriptFile);

    if (result.success && result.steps.finalVideo) {
      // Move video to correct location
      fs.mkdirSync(path.dirname(task.outputPath), { recursive: true});
      fs.copyFileSync(result.steps.finalVideo.path, task.outputPath);

      log('✅', `Saved: ${task.outputPath}`, 2);

      // Update database
      await updateVideoStatus(task, 'completed', task.outputPath);

      return {
        task,
        success: true,
        videoPath: task.outputPath,
        duration: (Date.now() - startTime) / 1000,
      };
    } else {
      throw new Error(result.error || 'Generation failed');
    }
  } catch (error: any) {
    log('❌', `Failed: ${error.message}`, 2);

    // Update database
    await updateVideoStatus(task, 'failed', null, error.message);

    return {
      task,
      success: false,
      error: error.message,
      duration: (Date.now() - startTime) / 1000,
    };
  }
}

// =============================================================================
// UPDATE DATABASE
// =============================================================================

async function updateVideoStatus(
  task: VideoTask,
  status: 'pending' | 'generating' | 'completed' | 'failed',
  videoUrl: string | null,
  errorMessage?: string
): Promise<void> {
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', task.dayNumber)
    .single();

  if (!lesson) return;

  const updateData: any = {
    status,
    video_url: videoUrl,
    error_message: errorMessage || null,
  };

  if (status === 'generating') {
    updateData.started_at = new Date().toISOString();
  } else if (status === 'completed' || status === 'failed') {
    updateData.completed_at = new Date().toISOString();
  }

  await supabase
    .from('lesson_video_generation_status')
    .upsert({
      core_lesson_id: lesson.id,
      archetype: task.archetype,
      phase: task.phase,
      video_type: task.videoType,
      ...updateData,
    }, {
      onConflict: 'core_lesson_id,archetype,phase,video_type',
    });
}

// =============================================================================
// MAIN PIPELINE
// =============================================================================

async function generateCompleteLesson(
  dayNumber: number,
  archetypeFilter?: string,
  phaseFilter?: string
): Promise<void> {
  console.log('\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log(`║  🎬 COMPLETE LESSON PIPELINE - Day ${dayNumber}`.padEnd(71) + '║');
  console.log('║  Generating ALL videos for the 5-phase journey                       ║');
  console.log('╚' + '═'.repeat(70) + '╝');
  console.log('');

  // Create video tasks
  log('📋', 'Creating video generation tasks...');
  const tasks = await createVideoTasks(dayNumber, archetypeFilter, phaseFilter);
  
  log('✅', `Created ${tasks.length} video tasks`, 1);
  console.log('');

  // Show breakdown
  const mainTasks = tasks.filter(t => t.videoType === 'main');
  const responseTasks = tasks.filter(t => t.videoType !== 'main');
  
  log('📊', 'Task Breakdown:', 1);
  log('🎥', `Main script videos: ${mainTasks.length}`, 2);
  log('💬', `Response videos: ${responseTasks.length}`, 2);
  log('📦', `Total videos: ${tasks.length}`, 2);
  console.log('');

  // Estimate time
  const estimatedMinutes = tasks.length * 5; // ~5 min per video
  log('⏱️', `Estimated time: ${estimatedMinutes} minutes (${(estimatedMinutes / 60).toFixed(1)} hours)`, 1);
  console.log('');

  // Confirm
  console.log('⏳ Starting in 5 seconds... (Ctrl+C to cancel)');
  await new Promise(resolve => setTimeout(resolve, 5000));
  console.log('');

  // Generate videos
  const results: GenerationResult[] = [];
  let completed = 0;

  for (const task of tasks) {
    completed++;
    console.log(`\n[${completed}/${tasks.length}] Generating...`);

    // Mark as generating in database
    await updateVideoStatus(task, 'generating', null);

    const result = await generateVideo(task);
    results.push(result);

    // Save progress
    const progressPath = path.join(CONFIG.OUTPUT_DIR, `day_${dayNumber}_progress.json`);
    fs.writeFileSync(progressPath, JSON.stringify(results, null, 2));

    // Brief pause between generations
    if (completed < tasks.length) {
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  // Final summary
  console.log('\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log('║  📊 GENERATION COMPLETE                                              ║');
  console.log('╚' + '═'.repeat(70) + '╝');

  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;

  console.log(`\n   ✅ Successful: ${successful}/${tasks.length}`);
  console.log(`   ❌ Failed: ${failed}/${tasks.length}`);

  // Show breakdown by type
  const mainSuccess = results.filter(r => r.success && r.task.videoType === 'main').length;
  const responseSuccess = results.filter(r => r.success && r.task.videoType !== 'main').length;

  console.log(`\n   🎥 Main videos: ${mainSuccess}/${mainTasks.length}`);
  console.log(`   💬 Response videos: ${responseSuccess}/${responseTasks.length}`);

  // Save final results
  const resultsPath = path.join(CONFIG.OUTPUT_DIR, `day_${dayNumber}_results.json`);
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  console.log(`\n📁 Results saved: ${resultsPath}`);
  console.log('');
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);

  let dayNumber = 1;
  let archetype: string | undefined;
  let phase: string | undefined;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--day':
        dayNumber = parseInt(args[++i]);
        break;
      case '--archetype':
        archetype = args[++i];
        break;
      case '--phase':
        phase = args[++i];
        break;
      case '--help':
        console.log(`
🎬 Complete Lesson Pipeline

Generates ALL videos for the complete 5-phase journey.

Usage:
  npx tsx scripts/kelly-video-factory/complete-lesson-pipeline.ts [options]

Options:
  --day <number>       Day number (1-365)
  --archetype <name>   Filter to specific archetype
  --phase <name>       Filter to specific phase
  --help               Show this help

Examples:
  # Generate all videos for Day 1 (51 videos)
  npx tsx scripts/kelly-video-factory/complete-lesson-pipeline.ts --day 1

  # Generate only Explorer videos for Day 1 (17 videos)
  npx tsx scripts/kelly-video-factory/complete-lesson-pipeline.ts --day 1 --archetype "The Explorer"

  # Generate only Hook phase for Day 1 (12 videos)
  npx tsx scripts/kelly-video-factory/complete-lesson-pipeline.ts --day 1 --phase Hook
`);
        process.exit(0);
    }
  }

  try {
    await generateCompleteLesson(dayNumber, archetype, phase);
  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message);
    process.exit(1);
  }
}

main();

export { generateCompleteLesson, createVideoTasks };






