#!/usr/bin/env node
/**
 * Pre-Compute Expression Data for All 365 Daily Lessons
 * 
 * This script generates facial expressions and gestures for all lessons
 * and stores them in Supabase lesson_atoms table as expression_data (JSONB).
 * 
 * Usage:
 *   node scripts/precompute-expressions.js                    # Process all lessons
 *   node scripts/precompute-expressions.js --day 1            # Process single day
 *   node scripts/precompute-expressions.js --days 1-30        # Process range
 *   node scripts/precompute-expressions.js --archetype "The Scientist"  # Single archetype
 *   node scripts/precompute-expressions.js --dry-run          # Preview without saving
 *   node scripts/precompute-expressions.js --output ./output  # Save to files instead
 * 
 * @module precompute-expressions
 */

import { createClient } from '@supabase/supabase-js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

// Import expression generator
import {
  ExpressionGenerator,
  BatchExpressionGenerator,
  ARCHETYPE_PROFILES,
  AGE_PROFILES,
  TONE_MODIFIERS,
} from '../app/expression-generator.js';

// =============================================================================
// CONFIGURATION
// =============================================================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONFIG = {
  // Supabase configuration
  supabaseUrl: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL,
  supabaseKey: process.env.SUPABASE_SERVICE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY,
  
  // Processing settings
  batchSize: 10,
  delayBetweenBatches: 100, // ms
  retryAttempts: 3,
  retryDelay: 1000, // ms
  
  // Output settings
  lessonsDir: path.resolve(__dirname, '../lessons'),
  outputDir: path.resolve(__dirname, '../output/expressions'),
  
  // Default archetypes to process
  defaultArchetypes: [
    'The Scientist',
    'The Explorer',
    'The Storyteller',
  ],
  
  // All age buckets
  ageBuckets: ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'],
};

// =============================================================================
// SUPABASE CLIENT
// =============================================================================

let supabase = null;

function initSupabase() {
  if (!CONFIG.supabaseUrl || !CONFIG.supabaseKey) {
    console.warn('⚠️  Supabase credentials not found. Running in file-only mode.');
    return null;
  }
  
  supabase = createClient(CONFIG.supabaseUrl, CONFIG.supabaseKey);
  return supabase;
}

// =============================================================================
// LESSON LOADING
// =============================================================================

/**
 * Load lesson DNA from local files or Supabase
 */
async function loadLessonDNA(dayNumber) {
  // Try local file first
  const possibleFiles = [
    path.join(CONFIG.lessonsDir, `day-${dayNumber}-dna.json`),
    path.join(CONFIG.lessonsDir, `lesson-${dayNumber}.json`),
  ];
  
  for (const filePath of possibleFiles) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return JSON.parse(content);
    } catch (e) {
      // File not found, try next
    }
  }
  
  // Try to find by scanning lessons directory
  try {
    const files = await fs.readdir(CONFIG.lessonsDir);
    const dnaFiles = files.filter(f => f.endsWith('-dna.json'));
    
    for (const file of dnaFiles) {
      const filePath = path.join(CONFIG.lessonsDir, file);
      const content = await fs.readFile(filePath, 'utf-8');
      const data = JSON.parse(content);
      
      if (data.calendar?.day === dayNumber) {
        return data;
      }
    }
  } catch (e) {
    // Directory error
  }
  
  // Try Supabase
  if (supabase) {
    const { data, error } = await supabase
      .from('core_lessons')
      .select('*')
      .eq('day_number', dayNumber)
      .single();
    
    if (data && !error) {
      return data;
    }
  }
  
  return null;
}

/**
 * Load all available lesson DNAs from local files
 */
async function loadAllLocalLessons() {
  const lessons = [];
  
  try {
    const files = await fs.readdir(CONFIG.lessonsDir);
    const dnaFiles = files.filter(f => f.endsWith('-dna.json') || f.endsWith('.json'));
    
    for (const file of dnaFiles) {
      try {
        const filePath = path.join(CONFIG.lessonsDir, file);
        const content = await fs.readFile(filePath, 'utf-8');
        const data = JSON.parse(content);
        
        if (data.id && data.ageVariants) {
          lessons.push({
            ...data,
            _sourceFile: file,
          });
        }
      } catch (e) {
        console.warn(`⚠️  Could not parse ${file}: ${e.message}`);
      }
    }
  } catch (e) {
    console.error('Error reading lessons directory:', e.message);
  }
  
  // Sort by day number if available
  lessons.sort((a, b) => {
    const dayA = a.calendar?.day || 999;
    const dayB = b.calendar?.day || 999;
    return dayA - dayB;
  });
  
  return lessons;
}

// =============================================================================
// EXPRESSION GENERATION
// =============================================================================

/**
 * Generate expression data for a single lesson
 */
async function generateExpressionsForLesson(lessonDNA, options = {}) {
  const {
    archetypes = CONFIG.defaultArchetypes,
    verbose = false,
  } = options;
  
  const batchGenerator = new BatchExpressionGenerator();
  const results = {
    lessonId: lessonDNA.id,
    title: lessonDNA.title || lessonDNA.universal_concept_translations?.en,
    dayNumber: lessonDNA.calendar?.day,
    generatedAt: new Date().toISOString(),
    archetypes: {},
  };
  
  for (const archetype of archetypes) {
    if (verbose) {
      console.log(`    → Generating for archetype: ${archetype}`);
    }
    
    try {
      results.archetypes[archetype] = batchGenerator.generateForLesson(lessonDNA, archetype);
    } catch (e) {
      console.error(`    ✗ Error generating for ${archetype}: ${e.message}`);
      results.archetypes[archetype] = { error: e.message };
    }
  }
  
  // Calculate statistics
  results.stats = calculateExpressionStats(results);
  
  return results;
}

/**
 * Calculate statistics for generated expressions
 */
function calculateExpressionStats(results) {
  const stats = {
    totalExpressions: 0,
    totalGestures: 0,
    archetypeCount: Object.keys(results.archetypes).length,
    ageVariants: 0,
    phases: 0,
  };
  
  for (const archetypeData of Object.values(results.archetypes)) {
    if (archetypeData.variants) {
      for (const variant of Object.values(archetypeData.variants)) {
        stats.ageVariants++;
        for (const phaseData of Object.values(variant)) {
          stats.phases++;
          if (phaseData.expressions) {
            stats.totalExpressions += phaseData.expressions.length;
          }
          if (phaseData.gestures) {
            stats.totalGestures += phaseData.gestures.length;
          }
        }
      }
    }
  }
  
  return stats;
}

// =============================================================================
// STORAGE
// =============================================================================

/**
 * Save expression data to Supabase
 */
async function saveToSupabase(lessonId, expressionData) {
  if (!supabase) {
    console.warn('⚠️  Supabase not initialized. Skipping database save.');
    return false;
  }
  
  const { error } = await supabase
    .from('lesson_atoms')
    .upsert({
      core_lesson_id: lessonId,
      archetype: 'expression_data',
      phase: 'all',
      content: expressionData,
      updated_at: new Date().toISOString(),
    }, {
      onConflict: 'core_lesson_id,archetype,phase',
    });
  
  if (error) {
    console.error(`✗ Supabase error for ${lessonId}: ${error.message}`);
    return false;
  }
  
  return true;
}

/**
 * Save expression data to local file
 */
async function saveToFile(lessonId, expressionData, outputDir) {
  await fs.mkdir(outputDir, { recursive: true });
  
  const filename = `${lessonId}-expressions.json`;
  const filePath = path.join(outputDir, filename);
  
  await fs.writeFile(filePath, JSON.stringify(expressionData, null, 2));
  
  return filePath;
}

// =============================================================================
// BATCH PROCESSING
// =============================================================================

/**
 * Process lessons in batches
 */
async function processLessonBatch(lessons, options = {}) {
  const {
    archetypes = CONFIG.defaultArchetypes,
    outputDir = CONFIG.outputDir,
    dryRun = false,
    saveToDb = true,
    saveToFiles = true,
    verbose = false,
  } = options;
  
  const results = {
    processed: 0,
    succeeded: 0,
    failed: 0,
    skipped: 0,
    errors: [],
  };
  
  for (let i = 0; i < lessons.length; i++) {
    const lesson = lessons[i];
    const lessonId = lesson.id || `lesson-${i}`;
    const dayNumber = lesson.calendar?.day || i + 1;
    
    console.log(`\n[${i + 1}/${lessons.length}] Processing: ${lessonId} (Day ${dayNumber})`);
    
    if (!lesson.ageVariants || Object.keys(lesson.ageVariants).length === 0) {
      console.log('  ⏭️  Skipped: No age variants found');
      results.skipped++;
      continue;
    }
    
    try {
      // Generate expressions
      const expressionData = await generateExpressionsForLesson(lesson, {
        archetypes,
        verbose,
      });
      
      if (dryRun) {
        console.log('  🔍 Dry run - would generate:');
        console.log(`     Expressions: ${expressionData.stats.totalExpressions}`);
        console.log(`     Gestures: ${expressionData.stats.totalGestures}`);
        console.log(`     Age variants: ${expressionData.stats.ageVariants}`);
        results.succeeded++;
      } else {
        // Save to files
        if (saveToFiles) {
          const filePath = await saveToFile(lessonId, expressionData, outputDir);
          console.log(`  📁 Saved to: ${path.basename(filePath)}`);
        }
        
        // Save to Supabase
        if (saveToDb && supabase) {
          const saved = await saveToSupabase(lessonId, expressionData);
          if (saved) {
            console.log('  💾 Saved to Supabase');
          }
        }
        
        console.log(`  ✅ Generated: ${expressionData.stats.totalExpressions} expressions, ${expressionData.stats.totalGestures} gestures`);
        results.succeeded++;
      }
    } catch (error) {
      console.error(`  ✗ Error: ${error.message}`);
      results.errors.push({ lessonId, error: error.message });
      results.failed++;
    }
    
    results.processed++;
    
    // Delay between lessons
    if (i < lessons.length - 1) {
      await sleep(CONFIG.delayBetweenBatches);
    }
  }
  
  return results;
}

// =============================================================================
// CLI
// =============================================================================

function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    day: null,
    days: null,
    archetype: null,
    archetypes: null,
    dryRun: false,
    outputDir: CONFIG.outputDir,
    verbose: false,
    all: false,
    noDb: false,
    noFiles: false,
    help: false,
  };
  
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    
    switch (arg) {
      case '--day':
      case '-d':
        options.day = parseInt(args[++i], 10);
        break;
      case '--days':
        options.days = args[++i]; // e.g., "1-30"
        break;
      case '--archetype':
      case '-a':
        options.archetype = args[++i];
        break;
      case '--archetypes':
        options.archetypes = args[++i].split(',').map(s => s.trim());
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      case '--output':
      case '-o':
        options.outputDir = path.resolve(args[++i]);
        break;
      case '--verbose':
      case '-v':
        options.verbose = true;
        break;
      case '--all':
        options.all = true;
        break;
      case '--no-db':
        options.noDb = true;
        break;
      case '--no-files':
        options.noFiles = true;
        break;
      case '--help':
      case '-h':
        options.help = true;
        break;
    }
  }
  
  return options;
}

function printHelp() {
  console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║         Curious Kelly - Expression Pre-Computation Script                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
  node scripts/precompute-expressions.js [options]

OPTIONS:
  --day, -d <number>       Process a single day (1-365)
  --days <range>           Process a range of days (e.g., "1-30")
  --archetype, -a <name>   Process only one archetype (e.g., "The Scientist")
  --archetypes <list>      Process multiple archetypes (comma-separated)
  --dry-run                Preview without saving any data
  --output, -o <dir>       Output directory for JSON files
  --no-db                  Skip Supabase storage
  --no-files               Skip file storage
  --verbose, -v            Show detailed output
  --all                    Process all 12 archetypes
  --help, -h               Show this help message

EXAMPLES:
  # Process all available lessons with default archetypes
  node scripts/precompute-expressions.js

  # Process only Day 1
  node scripts/precompute-expressions.js --day 1

  # Process Days 1-30 with all archetypes
  node scripts/precompute-expressions.js --days 1-30 --all

  # Dry run for The Explorer archetype
  node scripts/precompute-expressions.js --archetype "The Explorer" --dry-run

  # Save only to files (no database)
  node scripts/precompute-expressions.js --no-db --output ./my-expressions

ARCHETYPES:
${Object.keys(ARCHETYPE_PROFILES).map(a => `  - ${a}`).join('\n')}

AGE BUCKETS:
${CONFIG.ageBuckets.map(a => `  - ${a}`).join('\n')}
`);
}

// =============================================================================
// UTILITIES
// =============================================================================

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function parseDayRange(rangeStr) {
  const [start, end] = rangeStr.split('-').map(s => parseInt(s.trim(), 10));
  const days = [];
  for (let i = start; i <= end; i++) {
    days.push(i);
  }
  return days;
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║         ✨ Curious Kelly - Expression Pre-Computation ✨                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
`);
  
  const options = parseArgs();
  
  if (options.help) {
    printHelp();
    process.exit(0);
  }
  
  // Initialize Supabase
  if (!options.noDb) {
    initSupabase();
  }
  
  // Determine archetypes to process
  let archetypes = CONFIG.defaultArchetypes;
  if (options.all) {
    archetypes = Object.keys(ARCHETYPE_PROFILES);
  } else if (options.archetypes) {
    archetypes = options.archetypes;
  } else if (options.archetype) {
    archetypes = [options.archetype];
  }
  
  console.log(`📋 Configuration:`);
  console.log(`   Archetypes: ${archetypes.length} (${archetypes.slice(0, 3).join(', ')}${archetypes.length > 3 ? '...' : ''})`);
  console.log(`   Output: ${options.outputDir}`);
  console.log(`   Mode: ${options.dryRun ? 'Dry Run' : 'Production'}`);
  console.log(`   Database: ${options.noDb ? 'Disabled' : (supabase ? 'Connected' : 'Not available')}`);
  console.log(`   Files: ${options.noFiles ? 'Disabled' : 'Enabled'}`);
  
  // Load lessons
  let lessons = [];
  
  if (options.day) {
    console.log(`\n📚 Loading Day ${options.day}...`);
    const lesson = await loadLessonDNA(options.day);
    if (lesson) {
      lessons.push(lesson);
    } else {
      console.error(`✗ Could not find lesson for Day ${options.day}`);
      process.exit(1);
    }
  } else if (options.days) {
    const dayNumbers = parseDayRange(options.days);
    console.log(`\n📚 Loading Days ${options.days} (${dayNumbers.length} lessons)...`);
    
    for (const dayNum of dayNumbers) {
      const lesson = await loadLessonDNA(dayNum);
      if (lesson) {
        lessons.push(lesson);
      } else {
        console.warn(`⚠️  Day ${dayNum} not found`);
      }
    }
  } else {
    console.log(`\n📚 Loading all available lessons...`);
    lessons = await loadAllLocalLessons();
  }
  
  console.log(`   Found ${lessons.length} lessons`);
  
  if (lessons.length === 0) {
    console.error('\n✗ No lessons found to process!');
    console.log('\nMake sure lesson DNA files exist in:');
    console.log(`  ${CONFIG.lessonsDir}`);
    process.exit(1);
  }
  
  // Process lessons
  console.log(`\n🚀 Starting expression generation...`);
  const startTime = Date.now();
  
  const results = await processLessonBatch(lessons, {
    archetypes,
    outputDir: options.outputDir,
    dryRun: options.dryRun,
    saveToDb: !options.noDb && !!supabase,
    saveToFiles: !options.noFiles,
    verbose: options.verbose,
  });
  
  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  
  // Print summary
  console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SUMMARY                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Total Processed: ${results.processed}
  Succeeded: ${results.succeeded} ✅
  Failed: ${results.failed} ✗
  Skipped: ${results.skipped} ⏭️
  
  Duration: ${duration}s
  Avg per lesson: ${(duration / Math.max(results.processed, 1)).toFixed(2)}s
`);
  
  if (results.errors.length > 0) {
    console.log('  Errors:');
    for (const err of results.errors) {
      console.log(`    - ${err.lessonId}: ${err.error}`);
    }
  }
  
  if (!options.dryRun && !options.noFiles) {
    console.log(`\n📁 Output saved to: ${options.outputDir}`);
  }
  
  process.exit(results.failed > 0 ? 1 : 0);
}

// Run main
main().catch(error => {
  console.error('\n💥 Fatal error:', error);
  process.exit(1);
});


