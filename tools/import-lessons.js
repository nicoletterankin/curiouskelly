/**
 * Import 365 lessons from JSON to Supabase
 */
const fs = require('fs');
const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function importLessons() {
  const data = JSON.parse(fs.readFileSync('lessons/365_day_calendar.json', 'utf8'));
  console.log(`Importing ${data.lessons.length} lessons...`);
  
  // Clear existing
  await supabase.from('lessons').delete().neq('day_number', -999);
  
  // Prepare lessons
  const lessons = data.lessons.map(l => ({
    day_number: l.day,
    title: l.title,
    emoji: l.icon || '📚',
    content: { description: l.learning_objective }
  }));
  
  // Insert in batches
  for (let i = 0; i < lessons.length; i += 50) {
    const batch = lessons.slice(i, i + 50);
    const { error } = await supabase.from('lessons').upsert(batch, { onConflict: 'day_number' });
    if (error) console.error(`Batch ${i/50 + 1} error:`, error.message);
    else console.log(`Batch ${Math.floor(i/50) + 1}: ${batch.length} lessons`);
  }
  
  const { count } = await supabase.from('lessons').select('*', { count: 'exact', head: true });
  console.log(`Done! ${count} lessons in database.`);
}

importLessons().catch(console.error);


