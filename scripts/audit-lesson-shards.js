#!/usr/bin/env node
/**
 * LESSON SHARDS AUDIT
 * 
 * Audits all 56,134 lesson_shards for content corruption.
 * Checks if shard content matches the core_lesson topic.
 * 
 * Strategy:
 * 1. Load all core_lessons (365) with their topics
 * 2. For each lesson, fetch all its shards
 * 3. Extract script content from each shard
 * 4. Check if script mentions the correct topic
 * 5. Record mismatches in lesson_audits table
 * 
 * This is a LONG-RUNNING operation (56K rows).
 */

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing environment variables');
  console.error('   Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

// Known topic keywords for pattern matching
const TOPIC_PATTERNS = {
  'Starting Fresh': ['start', 'fresh', 'begin', 'new'],
  'Pulleys': ['pulley', 'rope', 'lift', 'mechanical advantage'],
  'Microscopes': ['microscope', 'tiny', 'magnify', 'cells'],
  'Leaves': ['leaf', 'leaves', 'photosynthesis', 'chlorophyll'],
  'Wind Power': ['wind', 'turbine', 'renewable', 'energy'],
  // Add more as needed
};

async function auditAllShards() {
  console.log('🔍 LESSON SHARDS AUDIT - Starting...\n');
  console.log('📊 Auditing 56,134 shards across 365 lessons');
  console.log('⏱️  This will take 5-10 minutes...\n');

  // Step 1: Load all core lessons
  console.log('📖 Loading all core lessons...');
  const { data: lessons, error: lessonsError } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic')
    .order('day_number');

  if (lessonsError) {
    console.error('❌ Error loading lessons:', lessonsError);
    process.exit(1);
  }

  console.log(`✅ Loaded ${lessons.length} lessons\n`);

  let totalShards = 0;
  let corruptedShards = 0;
  let lessonsWithCorruption = 0;
  const corruptedLessons = [];

  // Step 2: Audit each lesson's shards
  for (let i = 0; i < lessons.length; i++) {
    const lesson = lessons[i];
    const progress = `[${i + 1}/${lessons.length}]`;

    // Fetch all shards for this lesson
    const { data: shards, error: shardsError } = await supabase
      .from('lesson_shards')
      .select('id, age, tone, script_content')
      .eq('core_lesson_id', lesson.id);

    if (shardsError) {
      console.error(`${progress} ❌ Day ${lesson.day_number}: Error fetching shards`);
      continue;
    }

    if (!shards || shards.length === 0) {
      console.log(`${progress} ⚠️  Day ${lesson.day_number} (${lesson.topic}): No shards found`);
      continue;
    }

    totalShards += shards.length;

    // Check each shard for content mismatch
    let lessonCorruptCount = 0;
    const corruptedShardIds = [];

    for (const shard of shards) {
      if (!shard.script_content || !shard.script_content.script) {
        continue;
      }

      const scriptText = shard.script_content.script.toLowerCase();
      const topic = lesson.topic.toLowerCase();

      // Simple check: does the script mention the topic?
      // This is a heuristic - we'll flag potential issues
      const topicWords = topic.split(' ');
      const mentionsTopic = topicWords.some(word => 
        word.length > 3 && scriptText.includes(word.toLowerCase())
      );

      // Also check for known wrong topics (e.g., microscope in pulley lesson)
      const wrongTopicDetected = Object.entries(TOPIC_PATTERNS).some(([wrongTopic, keywords]) => {
        if (wrongTopic.toLowerCase() === topic) return false; // Skip if it's the correct topic
        return keywords.some(keyword => scriptText.includes(keyword.toLowerCase()));
      });

      if (!mentionsTopic || wrongTopicDetected) {
        lessonCorruptCount++;
        corruptedShards++;
        corruptedShardIds.push(shard.id);
      }
    }

    if (lessonCorruptCount > 0) {
      lessonsWithCorruption++;
      const percentage = ((lessonCorruptCount / shards.length) * 100).toFixed(1);
      console.log(`${progress} 🚨 Day ${lesson.day_number} (${lesson.topic}): ${lessonCorruptCount}/${shards.length} shards corrupted (${percentage}%)`);
      
      corruptedLessons.push({
        day_number: lesson.day_number,
        topic: lesson.topic,
        total_shards: shards.length,
        corrupted_shards: lessonCorruptCount,
        percentage: percentage,
        core_lesson_id: lesson.id
      });

      // Record in audit table (one record per lesson, not per shard)
      await supabase.from('lesson_audits').insert({
        day_number: lesson.day_number,
        audit_type: 'content_completeness',
        status: 'fail',
        field_name: 'lesson_shards',
        actual_issue: `${lessonCorruptCount} of ${shards.length} shards have content mismatches`,
        audited_by: 'shard_audit_v1',
        confidence_score: 0.7
      });
    } else {
      // Show progress every 10 lessons
      if ((i + 1) % 10 === 0) {
        console.log(`${progress} ✅ Day ${lesson.day_number} (${lesson.topic}): All ${shards.length} shards OK`);
      }
    }
  }

  // Final summary
  console.log('\n' + '='.repeat(80));
  console.log('📊 AUDIT COMPLETE\n');
  console.log(`Total shards audited: ${totalShards.toLocaleString()}`);
  console.log(`Corrupted shards: ${corruptedShards.toLocaleString()}`);
  console.log(`Lessons with corruption: ${lessonsWithCorruption}/${lessons.length}`);
  console.log(`Corruption rate: ${((corruptedShards / totalShards) * 100).toFixed(2)}%`);
  console.log('='.repeat(80));

  if (corruptedLessons.length > 0) {
    console.log('\n🚨 TOP 20 WORST OFFENDERS:\n');
    corruptedLessons
      .sort((a, b) => b.corrupted_shards - a.corrupted_shards)
      .slice(0, 20)
      .forEach((lesson, idx) => {
        console.log(`${idx + 1}. Day ${lesson.day_number} (${lesson.topic}): ${lesson.corrupted_shards}/${lesson.total_shards} shards (${lesson.percentage}%)`);
      });
  }

  console.log('\n✅ Audit results recorded in lesson_audits table');
}

// Run the audit
auditAllShards().catch(err => {
  console.error('💥 Fatal error:', err);
  process.exit(1);
});

