/**
 * Sync 365 lessons from JSON to Supabase
 * Run: node tools/sync-lessons-to-supabase.js
 */

const fs = require('fs');
const { createClient } = require('@supabase/supabase-js');

// Load environment
require('dotenv').config();

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error('Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

async function syncLessons() {
  console.log('📚 Loading lessons from JSON...');
  
  const calendarPath = 'lessons/365_day_calendar.json';
  const calendarData = JSON.parse(fs.readFileSync(calendarPath, 'utf8'));
  
  console.log(`Found ${calendarData.lessons.length} lessons`);
  
  // Transform lessons for database
  const lessonsToInsert = calendarData.lessons.map(lesson => ({
    day_number: lesson.day,
    title: lesson.title,
    slug: lesson.lesson_id,
    emoji: lesson.icon || '📚',
    category: mapCategory(lesson.category, lesson.title),
    description: lesson.learning_objective,
    learning_objectives: lesson.learning_objectives || [],
    difficulty: lesson.difficulty || 'beginner',
    duration_min: lesson.duration?.min || 5,
    duration_max: lesson.duration?.max || 15,
    marketing_headline: lesson.marketing_headline || null,
    marketing_tagline: lesson.marketing_tagline || null,
    date_label: lesson.date,
    is_active: true
  }));
  
  console.log('🗑️ Clearing existing lessons...');
  const { error: deleteError } = await supabase
    .from('lessons')
    .delete()
    .neq('day_number', 0); // Delete all
  
  if (deleteError) {
    console.error('Delete error:', deleteError);
    process.exit(1);
  }
  
  console.log('📤 Inserting lessons in batches...');
  
  // Insert in batches of 50
  const BATCH_SIZE = 50;
  let inserted = 0;
  
  for (let i = 0; i < lessonsToInsert.length; i += BATCH_SIZE) {
    const batch = lessonsToInsert.slice(i, i + BATCH_SIZE);
    
    const { error: insertError } = await supabase
      .from('lessons')
      .insert(batch);
    
    if (insertError) {
      console.error(`Batch ${Math.floor(i/BATCH_SIZE)+1} error:`, insertError);
      // Try individual inserts
      for (const lesson of batch) {
        const { error } = await supabase.from('lessons').insert(lesson);
        if (error) {
          console.error(`  Failed: Day ${lesson.day_number} - ${lesson.title}:`, error.message);
        } else {
          inserted++;
        }
      }
    } else {
      inserted += batch.length;
      console.log(`  Batch ${Math.floor(i/BATCH_SIZE)+1}: ${batch.length} lessons inserted`);
    }
  }
  
  console.log(`\n✅ Done! ${inserted} lessons synced to Supabase`);
  
  // Verify
  const { count } = await supabase.from('lessons').select('*', { count: 'exact', head: true });
  console.log(`📊 Total lessons in database: ${count}`);
}

function mapCategory(category, title) {
  // Smart category mapping based on title keywords
  const titleLower = title.toLowerCase();
  
  if (titleLower.includes('star') || titleLower.includes('planet') || titleLower.includes('space') || titleLower.includes('moon') || titleLower.includes('sun')) return 'Space';
  if (titleLower.includes('animal') || titleLower.includes('bird') || titleLower.includes('fish') || titleLower.includes('insect') || titleLower.includes('ocean')) return 'Nature';
  if (titleLower.includes('friend') || titleLower.includes('kind') || titleLower.includes('emotion') || titleLower.includes('feel') || titleLower.includes('happy')) return 'Life Skills';
  if (titleLower.includes('music') || titleLower.includes('art') || titleLower.includes('paint') || titleLower.includes('draw') || titleLower.includes('dance')) return 'Arts';
  if (titleLower.includes('math') || titleLower.includes('number') || titleLower.includes('count') || titleLower.includes('shape')) return 'Math';
  if (titleLower.includes('history') || titleLower.includes('ancient') || titleLower.includes('war') || titleLower.includes('king') || titleLower.includes('queen')) return 'History';
  if (titleLower.includes('body') || titleLower.includes('brain') || titleLower.includes('heart') || titleLower.includes('health') || titleLower.includes('food')) return 'Health';
  if (titleLower.includes('code') || titleLower.includes('computer') || titleLower.includes('internet') || titleLower.includes('robot') || titleLower.includes('email')) return 'Technology';
  if (titleLower.includes('water') || titleLower.includes('cloud') || titleLower.includes('light') || titleLower.includes('sound') || titleLower.includes('energy')) return 'Science';
  if (titleLower.includes('seed') || titleLower.includes('plant') || titleLower.includes('tree') || titleLower.includes('flower') || titleLower.includes('leaf')) return 'Nature';
  
  return category || 'General';
}

syncLessons().catch(console.error);


