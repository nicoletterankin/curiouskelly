/**
 * Sprint G: Validate Generated Scripts
 * Checks every lesson has 14 scripts (7 phases × 2 options)
 */
require('dotenv').config();
const { Client } = require('pg');

async function validate() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log('=== Script Validation ===\n');
  
  // Get script counts per day
  const res = await client.query(`
    SELECT cl.day_number, cl.title, COUNT(ls.id) as script_count,
           SUM(ls.word_count) as total_words,
           COUNT(CASE WHEN ls.word_count < 10 THEN 1 END) as short_scripts,
           COUNT(CASE WHEN ls.content IS NULL OR ls.content = '' THEN 1 END) as empty_scripts
    FROM core_lessons_v2 cl
    LEFT JOIN lesson_atoms la ON la.lesson_id = cl.id AND la.age_group = 'adult' AND la.language = 'en'
    LEFT JOIN lesson_scripts ls ON ls.atom_id = la.id
    GROUP BY cl.id, cl.day_number, cl.title
    ORDER BY cl.day_number
  `);
  
  let fullyScripted = 0;
  let partiallyScripted = 0;
  let noScripts = 0;
  let totalWords = 0;
  let totalScripts = 0;
  const issues = [];
  
  for (const row of res.rows) {
    const count = parseInt(row.script_count);
    totalScripts += count;
    totalWords += parseInt(row.total_words) || 0;
    
    if (count >= 14) {
      fullyScripted++;
    } else if (count > 0) {
      partiallyScripted++;
      if (count < 7) {
        issues.push(`Day ${row.day_number} (${row.title}): only ${count}/14 scripts`);
      }
    } else {
      noScripts++;
      issues.push(`Day ${row.day_number} (${row.title}): NO scripts`);
    }
    
    if (parseInt(row.empty_scripts) > 0) {
      issues.push(`Day ${row.day_number}: ${row.empty_scripts} empty scripts`);
    }
    if (parseInt(row.short_scripts) > 0) {
      issues.push(`Day ${row.day_number}: ${row.short_scripts} scripts under 10 words`);
    }
  }
  
  const avgDuration = Math.round(totalWords / 150); // 150 wpm
  
  console.log(`=== Summary ===`);
  console.log(`${fullyScripted} of 365 days fully scripted (14/14)`);
  console.log(`${partiallyScripted} partially scripted`);
  console.log(`${noScripts} with no scripts`);
  console.log(`\nTotal scripts: ${totalScripts}`);
  console.log(`Total word count: ${totalWords.toLocaleString()}`);
  console.log(`Average script duration: ${avgDuration} minutes (@ 150 wpm)`);
  console.log(`Estimated total audio duration: ${Math.round(totalWords / 150)} minutes`);
  console.log(`Estimated ElevenLabs characters: ~${Math.round(totalWords * 5.5).toLocaleString()}`);
  
  if (issues.length > 0) {
    console.log(`\n=== Issues (${issues.length}) ===`);
    issues.slice(0, 20).forEach(i => console.log(`  ${i}`));
    if (issues.length > 20) console.log(`  ...and ${issues.length - 20} more`);
  }
  
  await client.end();
}

validate().catch(e => { console.error(e); process.exit(1); });
