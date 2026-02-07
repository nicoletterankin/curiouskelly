const { neon } = require('@neondatabase/serverless');
const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

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
  const day34 = await sql`
    SELECT day_number, title, topic, track, theme
    FROM core_lessons 
    WHERE day_number = 34
    ORDER BY track
  `;
  if (day34.length === 0) {
    console.log('   ❌ NO LESSONS FOUND FOR DAY 34');
  } else {
    day34.forEach(l => {
      console.log(`   Day ${l.day_number}: "${l.title}"`);
      console.log(`      Topic: ${l.topic || 'N/A'}`);
      console.log(`      Theme: ${l.theme || 'N/A'}`);
      console.log(`      Track: ${l.track || 'default'}`);
    });
  }
  
  // What about magnets?
  console.log('\n🧲 LESSONS ABOUT MAGNETS:');
  const magnets = await sql`
    SELECT day_number, title, topic, track
    FROM core_lessons 
    WHERE LOWER(title) LIKE '%magnet%' OR LOWER(topic) LIKE '%magnet%'
    ORDER BY day_number
  `;
  if (magnets.length === 0) {
    console.log('   No lessons found with "magnet" in title or topic');
  } else {
    magnets.forEach(l => console.log(`   Day ${l.day_number}: "${l.title}" (${l.track || 'default'})`));
  }
  
  // What about popcorn?
  console.log('\n🍿 LESSONS ABOUT POPCORN:');
  const popcorn = await sql`
    SELECT day_number, title, topic, track
    FROM core_lessons 
    WHERE LOWER(title) LIKE '%popcorn%' OR LOWER(topic) LIKE '%popcorn%'
    ORDER BY day_number
  `;
  if (popcorn.length === 0) {
    console.log('   No lessons found with "popcorn" in title or topic');
  } else {
    popcorn.forEach(l => console.log(`   Day ${l.day_number}: "${l.title}" (${l.track || 'default'})`));
  }
  
  // Show nearby days for context
  console.log('\n📋 LESSONS FOR DAYS 30-40:');
  const nearby = await sql`
    SELECT day_number, title, track
    FROM core_lessons 
    WHERE day_number BETWEEN 30 AND 40
    ORDER BY day_number, track
  `;
  nearby.forEach(l => console.log(`   Day ${l.day_number}: "${l.title}" (${l.track || 'default'})`));
  
  console.log('\n════════════════════════════════════════════════════════════');
}

check().catch(console.error);
