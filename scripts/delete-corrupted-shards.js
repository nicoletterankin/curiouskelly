import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function deleteCorruptedShards() {
  console.log('🔍 Identifying corrupted shards...\n');

  // Step 1: Get all core lessons with their topics
  const { data: lessons, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic')
    .order('day_number');

  if (lessonError) {
    console.error('Error fetching lessons:', lessonError);
    return;
  }

  console.log(`✅ Loaded ${lessons.length} lessons\n`);

  let totalDeleted = 0;
  let corruptedLessons = [];

  // Step 2: For each lesson, check its shards
  for (const lesson of lessons) {
    const { data: shards, error: shardError } = await supabase
      .from('lesson_shards')
      .select('id, script_content')
      .eq('core_lesson_id', lesson.id);

    if (shardError) {
      console.error(`Error fetching shards for Day ${lesson.day_number}:`, shardError);
      continue;
    }

    if (!shards || shards.length === 0) continue;

    // Check if any shard content is corrupted
    let hasCorruption = false;
    const shardIdsToDelete = [];

    for (const shard of shards) {
      if (!shard.script_content) continue;

      // Convert script_content to string for searching
      const contentStr = JSON.stringify(shard.script_content).toLowerCase();
      const topicLower = lesson.topic.toLowerCase();

      // Simple heuristic: if the content doesn't contain any words from the topic,
      // it's likely corrupted. We'll use a more sophisticated check:
      // - Check if topic keywords appear in the content
      // - Check for known corruption patterns (leaf, wind, etc.)

      const topicWords = topicLower.split(' ').filter(w => w.length > 3);
      const hasTopicMatch = topicWords.some(word => contentStr.includes(word));

      // Known corruption patterns
      const corruptionPatterns = [
        'leaf', 'leaves', 'photosynthesis', 'chlorophyll',
        'wind power', 'turbine', 'renewable energy',
        'fossil', 'dinosaur', 'paleontology'
      ];

      const hasCorruptionPattern = corruptionPatterns.some(pattern => 
        contentStr.includes(pattern) && !topicLower.includes(pattern)
      );

      if (!hasTopicMatch || hasCorruptionPattern) {
        hasCorruption = true;
        shardIdsToDelete.push(shard.id);
      }
    }

    if (hasCorruption) {
      corruptedLessons.push({
        day: lesson.day_number,
        topic: lesson.topic,
        shardsToDelete: shardIdsToDelete.length
      });

      // Delete the corrupted shards
      const { error: deleteError } = await supabase
        .from('lesson_shards')
        .delete()
        .in('id', shardIdsToDelete);

      if (deleteError) {
        console.error(`❌ Error deleting shards for Day ${lesson.day_number}:`, deleteError);
      } else {
        totalDeleted += shardIdsToDelete.length;
        console.log(`🗑️  Day ${lesson.day_number} (${lesson.topic}): Deleted ${shardIdsToDelete.length} corrupted shards`);
      }
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('📊 DELETION SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total corrupted shards deleted: ${totalDeleted}`);
  console.log(`Lessons affected: ${corruptedLessons.length}`);
  console.log('\nCorrupted lessons:');
  corruptedLessons.forEach(l => {
    console.log(`  Day ${l.day}: ${l.topic} (${l.shardsToDelete} shards)`);
  });
}

deleteCorruptedShards().catch(console.error);


