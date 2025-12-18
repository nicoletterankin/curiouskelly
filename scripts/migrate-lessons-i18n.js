#!/usr/bin/env node
/**
 * Migrate Lesson JSON to i18n Structure
 * 
 * This script transforms v5.0-full-choices lessons to v5.0-full-choices-i18n:
 * - Converts string fields to { en, es, pt } objects
 * - Preserves English content
 * - Marks ES/PT as "[NEEDS TRANSLATION]" for human review
 * 
 * Usage:
 *   node scripts/migrate-lessons-i18n.js --day=1
 *   node scripts/migrate-lessons-i18n.js --range=1-30
 *   node scripts/migrate-lessons-i18n.js --all
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const LESSONS_DIR = path.join(__dirname, '..', 'public', 'lessons');
const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];

// ============================================================================
// I18N HELPERS
// ============================================================================

/**
 * Convert a string to i18n object
 * If already an object with 'en', return as-is
 */
function toI18n(value, placeholder = '[NEEDS TRANSLATION]') {
  if (value === null || value === undefined) {
    return null;
  }
  
  // Already i18n object
  if (typeof value === 'object' && value.en !== undefined) {
    return {
      en: value.en,
      es: value.es || placeholder,
      pt: value.pt || placeholder
    };
  }
  
  // String - convert to i18n
  if (typeof value === 'string') {
    return {
      en: value,
      es: placeholder,
      pt: placeholder
    };
  }
  
  // Number - for duration, convert to per-language
  if (typeof value === 'number') {
    return {
      en: value,
      es: Math.round(value * 1.15), // Spanish ~15% longer
      pt: Math.round(value * 1.08)  // Portuguese ~8% longer
    };
  }
  
  // Array of strings - convert each
  if (Array.isArray(value)) {
    return value.map(item => {
      if (typeof item === 'string') {
        return { en: item, es: placeholder, pt: placeholder };
      }
      return item;
    });
  }
  
  return value;
}

/**
 * Convert option to i18n structure
 */
function optionToI18n(option) {
  return {
    letter: option.letter,
    text: toI18n(option.text),
    quality: option.quality,
    response: toI18n(option.response)
  };
}

/**
 * Convert phase to i18n structure
 */
function phaseToI18n(phase) {
  if (!phase) return null;
  
  return {
    title: toI18n(phase.title),
    script: toI18n(phase.script),
    duration: toI18n(phase.duration),
    prompt: phase.prompt ? toI18n(phase.prompt) : null,
    options: (phase.options || []).map(optionToI18n)
  };
}

// ============================================================================
// LESSON MIGRATION
// ============================================================================

function migrateLesson(lesson) {
  // Skip if already i18n
  if (lesson.meta?.version === 'v5.0-full-choices-i18n') {
    return { migrated: false, reason: 'Already i18n' };
  }
  
  // Build i18n lesson
  const i18nLesson = {
    meta: {
      ...lesson.meta,
      topic: toI18n(lesson.meta?.topic),
      version: 'v5.0-full-choices-i18n',
      languages: ['en', 'es', 'pt']
    },
    headline: toI18n(lesson.headline),
    universal_truth: toI18n(lesson.universal_truth),
    fun_facts: (lesson.fun_facts || []).map(fact => toI18n(fact)),
    discussion_questions: (lesson.discussion_questions || []).map(q => toI18n(q)),
    phases: {},
    phaseOrder: lesson.phaseOrder,
    totalDuration: toI18n(lesson.totalDuration || 94),
    growTrack: lesson.growTrack ? {
      title: toI18n(lesson.growTrack.title),
      emoji: lesson.growTrack.emoji,
      learning_objective: toI18n(lesson.growTrack.learning_objective),
      activity: toI18n(lesson.growTrack.activity)
    } : null
  };
  
  // Convert each phase
  for (const phaseName of PHASES) {
    const phase = lesson.phases?.[phaseName];
    if (phase) {
      i18nLesson.phases[phaseName] = phaseToI18n(phase);
    }
  }
  
  return { migrated: true, lesson: i18nLesson };
}

function loadLesson(dayNumber) {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  if (!fs.existsSync(filePath)) return null;
  
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  } catch (e) {
    console.error(`❌ Failed to parse day-${dayNumber}.json: ${e.message}`);
    return null;
  }
}

function saveLesson(dayNumber, lesson) {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  fs.writeFileSync(filePath, JSON.stringify(lesson, null, 2) + '\n');
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  let days = [];
  let dryRun = args.includes('--dry-run');
  
  for (const arg of args) {
    if (arg === '--all') {
      for (let d = 1; d <= 365; d++) days.push(d);
    } else if (arg.startsWith('--range=')) {
      const [start, end] = arg.split('=')[1].split('-').map(n => parseInt(n, 10));
      for (let d = start; d <= end; d++) days.push(d);
    } else if (arg.startsWith('--days=')) {
      days = arg.split('=')[1].split(',').map(n => parseInt(n.trim(), 10));
    } else if (arg.startsWith('--day=')) {
      days.push(parseInt(arg.split('=')[1], 10));
    }
  }
  
  if (days.length === 0) {
    console.log('Usage:');
    console.log('  node scripts/migrate-lessons-i18n.js --day=1');
    console.log('  node scripts/migrate-lessons-i18n.js --range=1-30');
    console.log('  node scripts/migrate-lessons-i18n.js --all');
    console.log('  node scripts/migrate-lessons-i18n.js --all --dry-run');
    console.log('');
    console.log('This converts lessons to i18n structure (EN/ES/PT)');
    console.log('ES/PT are marked as "[NEEDS TRANSLATION]" for later filling');
    process.exit(0);
  }
  
  console.log('═'.repeat(60));
  console.log('🌍 LESSON I18N MIGRATION');
  console.log('═'.repeat(60));
  console.log(`📅 Days to process: ${days.length}`);
  console.log(`🔍 Mode: ${dryRun ? 'DRY RUN (no changes)' : 'LIVE'}`);
  
  const totals = { migrated: 0, skipped: 0, errors: 0 };
  
  for (const day of days) {
    const lesson = loadLesson(day);
    if (!lesson) {
      console.log(`⚠️ Day ${day}: No file`);
      totals.errors++;
      continue;
    }
    
    const result = migrateLesson(lesson);
    
    if (!result.migrated) {
      console.log(`⏭️ Day ${day}: ${result.reason}`);
      totals.skipped++;
      continue;
    }
    
    if (!dryRun) {
      saveLesson(day, result.lesson);
    }
    
    console.log(`✅ Day ${day}: "${lesson.meta?.topic}" → i18n`);
    totals.migrated++;
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Migrated: ${totals.migrated}`);
  console.log(`⏭️  Skipped: ${totals.skipped}`);
  console.log(`❌ Errors: ${totals.errors}`);
  
  if (dryRun) {
    console.log('\n⚠️ DRY RUN - No files were changed');
  }
}

main().catch(console.error);
