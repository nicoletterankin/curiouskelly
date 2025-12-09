/**
 * Kelly Video Batch Generator
 * 
 * Batch generates lip-synced Kelly videos using ElevenLabs Omnihuman 1.5
 * Videos are stored in Supabase Storage and tracked in kelly_video_assets table
 * 
 * Usage:
 *   npx ts-node scripts/generate-kelly-videos.ts --start=1 --end=31 --age=young_adult
 *   npx ts-node scripts/generate-kelly-videos.ts --start=1 --end=7 --age=young_adult --dry-run
 *   npx ts-node scripts/generate-kelly-videos.ts --day=1 --phase=welcome --age=young_adult
 * 
 * Options:
 *   --start=N       Start day number (1-365)
 *   --end=N         End day number (1-365)
 *   --day=N         Single day to generate (alternative to start/end)
 *   --phase=X       Single phase to generate (welcome, q1, q2, q3, wisdom)
 *   --age=X         Age bucket (toddler, child, teen, young_adult, adult, elder)
 *   --lang=X        Language code (en, es, fr) - default: en
 *   --force         Regenerate even if video exists
 *   --dry-run       Preview what would be generated without making API calls
 *   --delay=N       Delay between generations in ms (default: 3000)
 */

import * as dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });
dotenv.config();

import { createClient } from '@supabase/supabase-js';

// Configuration
const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const BASE_URL = process.env.PUBLIC_SITE_URL || 'http://localhost:3000';

const PHASES = ['welcome', 'q1', 'q2', 'q3', 'wisdom'] as const;
const AGE_BUCKETS = ['toddler', 'child', 'teen', 'young_adult', 'adult', 'elder'] as const;

// Phase to lesson_atoms phase mapping
const PHASE_TO_ATOM: Record<string, string> = {
  'welcome': 'Welcome',
  'q1': 'Fact1',
  'q2': 'Fact2',
  'q3': 'Fact3',
  'wisdom': 'Hook'
};

interface GenerationConfig {
  startDay: number;
  endDay: number;
  phases: string[];
  ageBucket: string;
  language: string;
  force: boolean;
  dryRun: boolean;
  delayMs: number;
}

interface GenerationResult {
  day: number;
  phase: string;
  success: boolean;
  videoUrl?: string;
  error?: string;
  cached?: boolean;
  durationMs?: number;
}

// Initialize Supabase
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

/**
 * Parse command line arguments
 */
function parseArgs(): GenerationConfig {
  const args = process.argv.slice(2);
  
  const getArg = (name: string): string | undefined => {
    const arg = args.find(a => a.startsWith(`--${name}=`));
    return arg ? arg.split('=')[1] : undefined;
  };
  
  const hasFlag = (name: string): boolean => {
    return args.includes(`--${name}`);
  };

  const singleDay = getArg('day');
  const singlePhase = getArg('phase');

  return {
    startDay: singleDay ? parseInt(singleDay) : parseInt(getArg('start') || '1'),
    endDay: singleDay ? parseInt(singleDay) : parseInt(getArg('end') || '31'),
    phases: singlePhase ? [singlePhase] : [...PHASES],
    ageBucket: getArg('age') || 'young_adult',
    language: getArg('lang') || 'en',
    force: hasFlag('force'),
    dryRun: hasFlag('dry-run'),
    delayMs: parseInt(getArg('delay') || '3000')
  };
}

/**
 * Get script content for a lesson phase
 */
async function getPhaseScript(
  lessonId: string,
  phase: string,
  archetype: string = 'The Explorer'
): Promise<string | null> {
  const atomPhase = PHASE_TO_ATOM[phase] || phase;
  
  const { data: atom, error } = await supabase
    .from('lesson_atoms')
    .select('content')
    .eq('core_lesson_id', lessonId)
    .eq('phase', atomPhase)
    .eq('archetype', archetype)
    .single();

  if (error || !atom?.content) {
    // Try without archetype filter
    const { data: fallbackAtom } = await supabase
      .from('lesson_atoms')
      .select('content')
      .eq('core_lesson_id', lessonId)
      .eq('phase', atomPhase)
      .limit(1)
      .single();
    
    if (fallbackAtom?.content?.script) {
      return fallbackAtom.content.script;
    }
    return null;
  }

  return atom.content.script || atom.content.text || null;
}

/**
 * Check if video already exists
 */
async function videoExists(
  lessonDay: number,
  phase: string,
  ageBucket: string,
  language: string
): Promise<boolean> {
  const { data } = await supabase
    .from('kelly_video_assets')
    .select('id')
    .eq('lesson_day', lessonDay)
    .eq('phase', phase)
    .eq('age_bucket', ageBucket)
    .eq('language', language)
    .eq('status', 'completed')
    .single();

  return !!data;
}

/**
 * Generate a single video
 */
async function generateVideo(
  lessonDay: number,
  phase: string,
  ageBucket: string,
  language: string,
  text: string,
  force: boolean
): Promise<GenerationResult> {
  try {
    const response = await fetch(`${BASE_URL}/api/elevenlabs-video`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lessonDay,
        phase,
        ageBucket,
        language,
        text,
        forceRegenerate: force
      })
    });

    const result = await response.json();

    if (result.success) {
      return {
        day: lessonDay,
        phase,
        success: true,
        videoUrl: result.videoUrl,
        cached: result.cached,
        durationMs: result.durationMs
      };
    } else {
      return {
        day: lessonDay,
        phase,
        success: false,
        error: result.error
      };
    }
  } catch (error) {
    return {
      day: lessonDay,
      phase,
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Main batch generation function
 */
async function main() {
  const config = parseArgs();

  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   🎬 KELLY VIDEO BATCH GENERATOR');
  console.log('   ElevenLabs Omnihuman 1.5 Integration');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');
  console.log(`📅 Days: ${config.startDay} - ${config.endDay}`);
  console.log(`🎭 Phases: ${config.phases.join(', ')}`);
  console.log(`👶 Age bucket: ${config.ageBucket}`);
  console.log(`🌐 Language: ${config.language}`);
  console.log(`⚡ Force regenerate: ${config.force}`);
  console.log(`🔍 Dry run: ${config.dryRun}`);
  console.log(`⏱️  Delay: ${config.delayMs}ms`);
  console.log('');

  // Validate age bucket
  if (!AGE_BUCKETS.includes(config.ageBucket as any)) {
    console.error(`❌ Invalid age bucket: ${config.ageBucket}`);
    console.error(`   Valid options: ${AGE_BUCKETS.join(', ')}`);
    process.exit(1);
  }

  // Calculate totals
  const totalDays = config.endDay - config.startDay + 1;
  const totalVideos = totalDays * config.phases.length;
  
  console.log(`📊 Total videos to process: ${totalVideos}`);
  console.log('');

  // Estimate time and cost
  const estimatedTimeMinutes = (totalVideos * (config.delayMs + 30000)) / 60000; // 30s avg generation + delay
  const estimatedCredits = totalVideos * 750; // Rough estimate
  
  console.log(`⏰ Estimated time: ~${Math.round(estimatedTimeMinutes)} minutes`);
  console.log(`💰 Estimated credits: ~${estimatedCredits.toLocaleString()}`);
  console.log('');

  if (config.dryRun) {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('   🔍 DRY RUN MODE - No videos will be generated');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('');
  }

  // Fetch lessons for the date range
  const { data: lessons, error: lessonsError } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic')
    .gte('day_number', config.startDay)
    .lte('day_number', config.endDay)
    .order('day_number');

  if (lessonsError || !lessons?.length) {
    console.error('❌ Failed to fetch lessons:', lessonsError?.message || 'No lessons found');
    process.exit(1);
  }

  console.log(`📚 Found ${lessons.length} lessons`);
  console.log('');

  // Track results
  const results: GenerationResult[] = [];
  let generated = 0;
  let cached = 0;
  let skipped = 0;
  let failed = 0;

  // Process each lesson and phase
  for (const lesson of lessons) {
    console.log(`\n📖 Day ${lesson.day_number}: ${lesson.topic}`);
    console.log('   ' + '─'.repeat(50));

    for (const phase of config.phases) {
      const prefix = `   [${phase.padEnd(7)}]`;

      // Check if video already exists (unless force regenerate)
      if (!config.force) {
        const exists = await videoExists(
          lesson.day_number,
          phase,
          config.ageBucket,
          config.language
        );

        if (exists) {
          console.log(`${prefix} ⏭️  Already exists (skipping)`);
          skipped++;
          results.push({
            day: lesson.day_number,
            phase,
            success: true,
            cached: true
          });
          continue;
        }
      }

      // Get script content
      const script = await getPhaseScript(lesson.id, phase);
      
      if (!script) {
        console.log(`${prefix} ⚠️  No script found (skipping)`);
        skipped++;
        results.push({
          day: lesson.day_number,
          phase,
          success: false,
          error: 'No script content'
        });
        continue;
      }

      // Dry run - just show what would be generated
      if (config.dryRun) {
        console.log(`${prefix} 📝 Would generate (${script.length} chars)`);
        continue;
      }

      // Generate video
      console.log(`${prefix} 🔄 Generating...`);
      
      const result = await generateVideo(
        lesson.day_number,
        phase,
        config.ageBucket,
        config.language,
        script,
        config.force
      );

      results.push(result);

      if (result.success) {
        if (result.cached) {
          console.log(`${prefix} 📦 Cached: ${result.videoUrl}`);
          cached++;
        } else {
          console.log(`${prefix} ✅ Generated: ${result.videoUrl}`);
          generated++;
        }
      } else {
        console.log(`${prefix} ❌ Failed: ${result.error}`);
        failed++;
      }

      // Rate limiting delay
      if (!config.dryRun) {
        await new Promise(r => setTimeout(r, config.delayMs));
      }
    }
  }

  // Print summary
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   📊 GENERATION SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');
  console.log(`   ✅ Generated:  ${generated}`);
  console.log(`   📦 Cached:     ${cached}`);
  console.log(`   ⏭️  Skipped:    ${skipped}`);
  console.log(`   ❌ Failed:     ${failed}`);
  console.log(`   ─────────────────────`);
  console.log(`   📊 Total:      ${results.length}`);
  console.log('');

  // List failures if any
  if (failed > 0) {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('   ❌ FAILURES');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('');
    
    const failures = results.filter(r => !r.success);
    for (const failure of failures) {
      console.log(`   Day ${failure.day}, ${failure.phase}: ${failure.error}`);
    }
    console.log('');
  }

  // Exit with appropriate code
  process.exit(failed > 0 ? 1 : 0);
}

// Run
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});



