#!/usr/bin/env node
/**
 * Sync Lesson JSON Files to Supabase
 * 
 * This script reads the v5.0-full-choices lesson JSON files and:
 * 1. Updates core_lessons table with lesson metadata
 * 2. Creates/updates lesson_atoms with phase content including choices
 * 
 * Usage:
 *   node scripts/sync-lessons-to-supabase.js --day=1
 *   node scripts/sync-lessons-to-supabase.js --range=1-30
 *   node scripts/sync-lessons-to-supabase.js --all
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '..', '.env.local') });

// ============================================================================
// CONFIGURATION
// ============================================================================

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ Missing Supabase credentials (need SERVICE_ROLE_KEY for writes)');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const LESSONS_DIR = path.join(__dirname, '..', 'public', 'lessons');
const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];

// ============================================================================
// LESSON PROCESSING
// ============================================================================

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

async function syncLesson(dayNumber) {
  const lesson = loadLesson(dayNumber);
  if (!lesson) {
    console.log(`   ⚠️ No lesson file for day ${dayNumber}`);
    return { success: false, error: 'No file' };
  }
  
  const topic = lesson.meta?.topic || 'Unknown Topic';
  console.log(`📚 Day ${dayNumber}: "${topic}"`);
  
  // 1. Check if core_lesson exists
  const { data: existing } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();
  
  let lessonId = existing?.id;
  
  if (!lessonId) {
    // Create new core_lesson
    const { data: newLesson, error: createError } = await supabase
      .from('core_lessons')
      .insert({
        day_number: dayNumber,
        topic: topic,
        universal_truth: lesson.universal_truth || '',
        marketing_headline: lesson.headline || topic,
        marketing_tagline: lesson.meta?.category || '',
        icon_emoji: lesson.meta?.emoji || '📚'
      })
      .select('id')
      .single();
    
    if (createError) {
      console.log(`   ❌ Failed to create lesson: ${createError.message}`);
      return { success: false, error: createError.message };
    }
    
    lessonId = newLesson.id;
    console.log(`   ✅ Created core_lesson (${lessonId})`);
  } else {
    // Update existing
    const { error: updateError } = await supabase
      .from('core_lessons')
      .update({
        topic: topic,
        universal_truth: lesson.universal_truth || '',
        marketing_headline: lesson.headline || topic,
        icon_emoji: lesson.meta?.emoji || '📚'
      })
      .eq('id', lessonId);
    
    if (updateError) {
      console.log(`   ❌ Failed to update lesson: ${updateError.message}`);
    } else {
      console.log(`   ✅ Updated core_lesson`);
    }
  }
  
  // 2. Sync atoms for each phase
  const phases = lesson.phases || {};
  let atomsCreated = 0;
  let atomsUpdated = 0;
  
  for (const phase of PHASES) {
    const phaseData = phases[phase];
    if (!phaseData) continue;
    
    // Build content object with v5.0 full-choices structure
    const content = {
      title: phaseData.title || phase,
      script: phaseData.script || '',
      duration: phaseData.duration || 15,
      // v5.0 choice fields
      prompt: phaseData.prompt || null,
      options: phaseData.options || [],
      // Legacy compatibility
      choice_intro: phaseData.prompt || null,
      option_a: phaseData.options?.[0]?.text || null,
      option_b: phaseData.options?.[1]?.text || null,
      success_response: phaseData.options?.[0]?.response || null,
      alt_response: phaseData.options?.[1]?.response || null,
    };
    
    // Check if atom exists (for default archetype)
    const { data: existingAtom } = await supabase
      .from('lesson_atoms')
      .select('id')
      .eq('core_lesson_id', lessonId)
      .eq('phase', phase)
      .eq('archetype', 'The Scientist')
      .single();
    
    if (existingAtom) {
      // Update
      const { error } = await supabase
        .from('lesson_atoms')
        .update({ content })
        .eq('id', existingAtom.id);
      
      if (!error) atomsUpdated++;
    } else {
      // Insert
      const { error } = await supabase
        .from('lesson_atoms')
        .insert({
          core_lesson_id: lessonId,
          day_number: dayNumber,
          phase,
          archetype: 'The Scientist',
          content
        });
      
      if (!error) atomsCreated++;
    }
  }
  
  console.log(`   📝 Atoms: ${atomsCreated} created, ${atomsUpdated} updated`);
  
  return { success: true, lessonId, atomsCreated, atomsUpdated };
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  let days = [];
  
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
    console.log('  node scripts/sync-lessons-to-supabase.js --day=1');
    console.log('  node scripts/sync-lessons-to-supabase.js --range=1-30');
    console.log('  node scripts/sync-lessons-to-supabase.js --all');
    console.log('');
    console.log('This syncs v5.0-full-choices lesson JSON to Supabase');
    process.exit(0);
  }
  
  console.log('═'.repeat(60));
  console.log('🔄 LESSON SYNC TO SUPABASE');
  console.log('═'.repeat(60));
  console.log(`📅 Days to sync: ${days.length}`);
  
  const totals = { success: 0, failed: 0, atomsCreated: 0, atomsUpdated: 0 };
  
  for (const day of days) {
    const result = await syncLesson(day);
    if (result.success) {
      totals.success++;
      totals.atomsCreated += result.atomsCreated || 0;
      totals.atomsUpdated += result.atomsUpdated || 0;
    } else {
      totals.failed++;
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 SYNC SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Lessons synced: ${totals.success}`);
  console.log(`❌ Failed: ${totals.failed}`);
  console.log(`📝 Atoms created: ${totals.atomsCreated}`);
  console.log(`📝 Atoms updated: ${totals.atomsUpdated}`);
}

main().catch(console.error);
