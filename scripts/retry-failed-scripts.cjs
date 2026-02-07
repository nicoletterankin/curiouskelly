/**
 * Retry script generation for days that are incomplete
 */
require('dotenv').config();
const { Client } = require('pg');
const { processLesson } = require('./generate-phase-scripts.cjs');

async function retryFailed() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  // Find days with fewer than 14 scripts
  const res = await client.query(`
    SELECT cl.id, cl.day_number, cl.title, COUNT(ls.id) as script_count
    FROM core_lessons_v2 cl
    LEFT JOIN lesson_atoms la ON la.lesson_id = cl.id AND la.age_group = 'adult' AND la.language = 'en'
    LEFT JOIN lesson_scripts ls ON ls.atom_id = la.id
    GROUP BY cl.id, cl.day_number, cl.title
    HAVING COUNT(ls.id) < 14
    ORDER BY cl.day_number
  `);
  
  console.log(`Found ${res.rows.length} days needing retry`);
  
  let success = 0, failed = 0;
  
  for (const row of res.rows) {
    console.log(`Retrying Day ${row.day_number} (${row.title}) - currently has ${row.script_count} scripts`);
    
    const result = await processLesson(client, row.id, row.day_number);
    if (result.success) {
      success++;
      console.log(`  SUCCESS: ${result.scriptsWritten} scripts`);
    } else {
      failed++;
      console.log(`  FAILED: ${result.error?.substring(0, 80)}`);
    }
    
    await new Promise(r => setTimeout(r, 500));
  }
  
  console.log(`\nRetry complete: ${success} success, ${failed} failed`);
  await client.end();
}

retryFailed().catch(e => { console.error(e); process.exit(1); });
