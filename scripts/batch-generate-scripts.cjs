/**
 * Sprint B: Batch Script Generation
 * Processes lessons in batches of 10
 * Tracks progress in generation_jobs table
 * Handles rate limiting and can resume from where it left off
 */
require('dotenv').config();
const { Client } = require('pg');
const { processLesson } = require('./generate-phase-scripts.cjs');
const fs = require('fs');
const path = require('path');

async function batchGenerate(startDay, endDay, batchSize = 10) {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log(`\n=== Batch Script Generation: Days ${startDay}-${endDay} ===`);
  console.log(`Batch size: ${batchSize}`);
  
  // Get all lessons in range that need script generation
  const lessons = await client.query(
    `SELECT cl.id, cl.day_number, cl.title
     FROM core_lessons_v2 cl
     WHERE cl.day_number >= $1 AND cl.day_number <= $2
     ORDER BY cl.day_number`,
    [startDay, endDay]
  );
  
  console.log(`Found ${lessons.rows.length} lessons to process`);
  
  // Check which already have complete scripts (14 scripts = fully done)
  const completed = await client.query(
    `SELECT cl.day_number, COUNT(ls.id) as script_count
     FROM core_lessons_v2 cl
     JOIN lesson_atoms la ON la.lesson_id = cl.id
     JOIN lesson_scripts ls ON ls.atom_id = la.id
     WHERE cl.day_number >= $1 AND cl.day_number <= $2
     GROUP BY cl.day_number
     HAVING COUNT(ls.id) >= 14`,
    [startDay, endDay]
  );
  
  const completedDays = new Set(completed.rows.map(r => r.day_number));
  const toProcess = lessons.rows.filter(l => !completedDays.has(l.day_number));
  
  console.log(`Already complete: ${completedDays.size}, Remaining: ${toProcess.length}`);
  
  if (toProcess.length === 0) {
    console.log('All lessons already have 14 scripts. Nothing to do.');
    await client.end();
    return;
  }
  
  let totalSuccess = 0;
  let totalFailed = 0;
  let totalScripts = 0;
  const batchCount = Math.ceil(toProcess.length / batchSize);
  const startTime = Date.now();
  
  for (let batch = 0; batch < batchCount; batch++) {
    const batchStart = batch * batchSize;
    const batchLessons = toProcess.slice(batchStart, batchStart + batchSize);
    
    console.log(`\nBatch ${batch + 1}/${batchCount}: Days ${batchLessons.map(l => l.day_number).join(', ')}`);
    
    let batchScripts = 0;
    let batchFails = 0;
    
    for (const lesson of batchLessons) {
      try {
        // Log job start
        await client.query(
          `INSERT INTO generation_jobs (atom_id, job_type, provider, status, input_params, started_at)
           VALUES (NULL, 'script', 'openai', 'running', $1, NOW())`,
          [JSON.stringify({ day_number: lesson.day_number, title: lesson.title })]
        );
        
        const result = await processLesson(client, lesson.id, lesson.day_number);
        
        if (result.success) {
          totalSuccess++;
          totalScripts += result.scriptsWritten;
          batchScripts += result.scriptsWritten;
          
          // Update job status
          await client.query(
            `UPDATE generation_jobs SET status = 'complete', completed_at = NOW()
             WHERE job_type = 'script' AND input_params->>'day_number' = $1 AND status = 'running'`,
            [String(lesson.day_number)]
          );
          
          process.stdout.write(`  Day ${lesson.day_number}: ${result.scriptsWritten} scripts ✓\n`);
        } else {
          totalFailed++;
          batchFails++;
          
          await client.query(
            `UPDATE generation_jobs SET status = 'failed', error = $1, completed_at = NOW()
             WHERE job_type = 'script' AND input_params->>'day_number' = $2 AND status = 'running'`,
            [result.error, String(lesson.day_number)]
          );
          
          process.stdout.write(`  Day ${lesson.day_number}: FAILED - ${result.error.substring(0, 80)}\n`);
        }
        
        // Rate limit: ~200ms between requests
        await new Promise(r => setTimeout(r, 200));
        
      } catch (e) {
        totalFailed++;
        batchFails++;
        console.error(`  Day ${lesson.day_number}: ERROR - ${e.message}`);
      }
    }
    
    const elapsed = Math.round((Date.now() - startTime) / 1000);
    console.log(`Batch ${batch + 1} complete: ${batchScripts} scripts, ${batchFails} failures (${elapsed}s elapsed)`);
  }
  
  // Final summary
  const elapsed = Math.round((Date.now() - startTime) / 1000);
  console.log(`\n=== Generation Complete ===`);
  console.log(`Lessons: ${totalSuccess} success, ${totalFailed} failed`);
  console.log(`Scripts generated: ${totalScripts}`);
  console.log(`Time: ${elapsed}s`);
  
  // Verify
  const verify = await client.query(
    `SELECT COUNT(DISTINCT cl.day_number) as days_with_scripts
     FROM core_lessons_v2 cl
     JOIN lesson_atoms la ON la.lesson_id = cl.id
     JOIN lesson_scripts ls ON ls.atom_id = la.id
     WHERE cl.day_number >= $1 AND cl.day_number <= $2`,
    [startDay, endDay]
  );
  console.log(`Days with scripts: ${verify.rows[0].days_with_scripts}/${endDay - startDay + 1}`);
  
  // Log first 3 scripts for quality check
  console.log(`\n=== Sample Scripts (Day ${startDay}) ===`);
  const samples = await client.query(
    `SELECT la.phase, la.variant, ls.option_number, ls.content, ls.word_count
     FROM core_lessons_v2 cl
     JOIN lesson_atoms la ON la.lesson_id = cl.id
     JOIN lesson_scripts ls ON ls.atom_id = la.id
     WHERE cl.day_number = $1
     ORDER BY la.phase, ls.option_number
     LIMIT 6`,
    [startDay]
  );
  for (const s of samples.rows) {
    console.log(`  Phase ${s.phase} (${s.variant}) Option ${s.option_number} [${s.word_count} words]:`);
    console.log(`    "${s.content.substring(0, 120)}..."`);
  }
  
  await client.end();
}

// CLI: node batch-generate-scripts.cjs [startDay] [endDay] [batchSize]
const startDay = parseInt(process.argv[2]) || 61;
const endDay = parseInt(process.argv[3]) || 100;
const batchSize = parseInt(process.argv[4]) || 10;

batchGenerate(startDay, endDay, batchSize).catch(e => {
  console.error('Fatal error:', e);
  process.exit(1);
});
