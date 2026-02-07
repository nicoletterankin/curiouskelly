require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function check() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║       LESSON VERIFICATION: February 3, 2026                ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
  
  // Calculate what day February 3, 2026 is
  const jan1 = new Date('2026-01-01');
  const feb3 = new Date('2026-02-03');
  const dayOfYear = Math.floor((feb3 - jan1) / (1000 * 60 * 60 * 24)) + 1;
  console.log(`📅 DATE MAPPING:`);
  console.log(`   February 3, 2026 = Day ${dayOfYear} of the year\n`);
  
  // What is Day 34 in the database?
  console.log('📚 WHAT IS DAY 34 IN THE DATABASE?');
  const { data: day34, error: err34 } = await supabase
    .from('core_lessons')
    .select('day_number, title, topic, track, theme')
    .eq('day_number', 34)
    .order('track');
    
  if (err34) {
    console.log('   Error:', err34.message);
  } else if (!day34 || day34.length === 0) {
    console.log('   ❌ NO LESSONS FOUND FOR DAY 34');
  } else {
    day34.forEach(l => {
      console.log(`   Day ${l.day_number}: "${l.title}"`);
      console.log(`      Topic: ${l.topic || 'N/A'}`);
      console.log(`      Theme: ${l.theme || 'N/A'}`);
      console.log(`      Track: ${l.track || 'default'}`);
      console.log('');
    });
  }
  
  // What about magnets?
  console.log('🧲 LESSONS ABOUT MAGNETS:');
  const { data: magnets } = await supabase
    .from('core_lessons')
    .select('day_number, title, topic, track')
    .or('title.ilike.%magnet%,topic.ilike.%magnet%')
    .order('day_number');
    
  if (!magnets || magnets.length === 0) {
    console.log('   No lessons found with "magnet" in title or topic');
  } else {
    magnets.forEach(l => console.log(`   Day ${l.day_number}: "${l.title}" (${l.track || 'default'})`));
  }
  
  // What about popcorn?
  console.log('\n🍿 LESSONS ABOUT POPCORN:');
  const { data: popcorn } = await supabase
    .from('core_lessons')
    .select('day_number, title, topic, track')
    .or('title.ilike.%popcorn%,topic.ilike.%popcorn%')
    .order('day_number');
    
  if (!popcorn || popcorn.length === 0) {
    console.log('   No lessons found with "popcorn" in title or topic');
  } else {
    popcorn.forEach(l => console.log(`   Day ${l.day_number}: "${l.title}" (${l.track || 'default'})`));
  }
  
  // Show nearby days for context
  console.log('\n📋 LESSONS FOR DAYS 30-40:');
  const { data: nearby } = await supabase
    .from('core_lessons')
    .select('day_number, title, track')
    .gte('day_number', 30)
    .lte('day_number', 40)
    .order('day_number')
    .order('track');
    
  if (nearby) {
    nearby.forEach(l => console.log(`   Day ${l.day_number}: "${l.title}" (${l.track || 'default'})`));
  }
  
  // Check Day 19 specifically (user mentioned magnets might be Day 19)
  console.log('\n📋 WHAT IS DAY 19?');
  const { data: day19 } = await supabase
    .from('core_lessons')
    .select('day_number, title, topic, track')
    .eq('day_number', 19);
    
  if (day19) {
    day19.forEach(l => console.log(`   Day ${l.day_number}: "${l.title}" - ${l.topic || 'N/A'} (${l.track || 'default'})`));
  }
  
  console.log('\n════════════════════════════════════════════════════════════');
}

check().catch(console.error);
