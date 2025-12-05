/**
 * Query Full Lesson Structure
 * Gets ONE complete lesson with ALL atoms and ALL shards as JSON
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function queryFullLesson() {
  const result = {
    lesson: null,
    atoms: [],
    shards: []
  };

  // 1. Get first populated lesson (day_number = 1, or first available)
  console.error('Querying core_lessons...');
  let { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', 1)
    .single();

  if (lessonError || !lesson) {
    // Try to get the first available lesson
    console.error('Day 1 not found, trying first available...');
    const { data: firstLesson, error: firstError } = await supabase
      .from('core_lessons')
      .select('*')
      .order('day_number', { ascending: true })
      .limit(1)
      .single();
    
    if (firstError) {
      console.error('Error fetching lesson:', firstError);
      return;
    }
    lesson = firstLesson;
  }

  result.lesson = lesson;
  console.error(`Found lesson: Day ${lesson.day_number} - ${lesson.topic}`);

  // 2. Get ALL atoms for this lesson
  console.error('Querying lesson_atoms...');
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id);

  if (atomsError) {
    console.error('Error fetching atoms:', atomsError);
  } else {
    result.atoms = atoms || [];
    console.error(`Found ${result.atoms.length} atoms`);
  }

  // 3. Get ALL shards for this lesson
  console.error('Querying lesson_shards...');
  const { data: shards, error: shardsError } = await supabase
    .from('lesson_shards')
    .select('*')
    .eq('core_lesson_id', lesson.id);

  if (shardsError) {
    console.error('Error fetching shards:', shardsError);
  } else {
    result.shards = shards || [];
    console.error(`Found ${result.shards.length} shards`);
  }

  // Output the full structure as JSON
  console.log(JSON.stringify(result, null, 2));
}

queryFullLesson().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});








