require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function check() {
  console.log('=== CHECKING CORE_LESSONS SCHEMA ===\n');
  
  // Get one row to see the structure
  const { data, error } = await supabase
    .from('core_lessons')
    .select('*')
    .limit(1);
    
  if (error) {
    console.log('Error:', error.message);
    return;
  }
  
  if (data && data.length > 0) {
    console.log('Columns in core_lessons:');
    Object.keys(data[0]).forEach(k => console.log('  -', k, ':', typeof data[0][k]));
    console.log('\nSample row:');
    console.log(JSON.stringify(data[0], null, 2));
  }
  
  // Also check Day 34
  console.log('\n=== DAY 34 DATA ===');
  const { data: day34, error: err34 } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', 34);
    
  if (err34) {
    console.log('Error:', err34.message);
  } else if (day34) {
    day34.forEach(l => console.log(JSON.stringify(l, null, 2)));
  }
  
  // Check lesson_perspectives for Day 34
  console.log('\n=== LESSON PERSPECTIVES DAY 34 ===');
  const { data: persp } = await supabase
    .from('lesson_perspectives')
    .select('*')
    .eq('day_number', 34)
    .limit(3);
    
  if (persp) {
    persp.forEach(p => console.log('Day', p.day_number, p.title || p.topic, '- Age:', p.age_group));
  }
}

check().catch(console.error);
