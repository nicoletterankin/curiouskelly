/**
 * Sprint C: Batch Audio Generation
 * Processes scripts in batches, respecting ElevenLabs rate limits
 */
require('dotenv').config();
const { Client } = require('pg');
const { processScript } = require('./generate-audio.cjs');

const API_KEY = process.env.ELEVENLABS_API_KEY;

async function checkCredits() {
  try {
    const res = await fetch('https://api.elevenlabs.io/v1/user/subscription', {
      headers: { 'xi-api-key': API_KEY }
    });
    if (res.ok) {
      const sub = await res.json();
      return {
        used: sub.character_count,
        limit: sub.character_limit,
        remaining: sub.character_limit - sub.character_count,
        tier: sub.tier,
      };
    }
  } catch (e) {}
  return null;
}

async function batchGenerate(startDay, endDay, batchSize = 5) {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log(`\n=== Batch Audio Generation: Days ${startDay}-${endDay} ===`);
  
  // Check credits first
  const credits = await checkCredits();
  if (credits) {
    console.log(`ElevenLabs Credits: ${credits.remaining.toLocaleString()} remaining (${credits.tier})`);
    console.log(`  Used: ${credits.used.toLocaleString()} / ${credits.limit.toLocaleString()}`);
  }
  
  // Get scripts that need audio
  const scripts = await client.query(`
    SELECT ls.id, ls.content, ls.phase, ls.option_number, ls.word_count,
           cl.day_number, cl.title
    FROM lesson_scripts ls
    JOIN lesson_atoms la ON la.id = ls.atom_id
    JOIN core_lessons_v2 cl ON cl.id = la.lesson_id
    WHERE cl.day_number >= $1 AND cl.day_number <= $2
      AND ls.content IS NOT NULL AND ls.content != ''
      AND la.audio_url IS NULL
    ORDER BY cl.day_number, ls.phase, ls.option_number
  `, [startDay, endDay]);
  
  console.log(`Found ${scripts.rows.length} scripts needing audio`);
  
  // Estimate chars needed
  const totalChars = scripts.rows.reduce((sum, s) => sum + (s.content?.length || 0), 0);
  console.log(`Estimated characters needed: ${totalChars.toLocaleString()}`);
  
  if (credits && totalChars > credits.remaining) {
    console.log(`WARNING: Need ${totalChars.toLocaleString()} chars but only ${credits.remaining.toLocaleString()} remaining`);
    console.log(`Will process as many as possible before running out`);
  }
  
  let totalSuccess = 0;
  let totalFailed = 0;
  let totalSize = 0;
  let charsUsed = 0;
  const startTime = Date.now();
  
  for (let i = 0; i < scripts.rows.length; i += batchSize) {
    const batch = scripts.rows.slice(i, i + batchSize);
    const batchNum = Math.floor(i / batchSize) + 1;
    const totalBatches = Math.ceil(scripts.rows.length / batchSize);
    
    process.stdout.write(`\nBatch ${batchNum}/${totalBatches}: `);
    
    for (const script of batch) {
      const result = await processScript(
        client, script.id, script.content,
        script.day_number, script.phase, script.option_number
      );
      
      if (result.success) {
        totalSuccess++;
        totalSize += result.size;
        charsUsed += script.content.length;
        process.stdout.write('.');
      } else {
        totalFailed++;
        if (result.error?.includes('quota') || result.error?.includes('limit') || result.error?.includes('429')) {
          console.log(`\n\nCREDIT LIMIT REACHED after ${totalSuccess} files`);
          console.log(`Characters used this run: ${charsUsed.toLocaleString()}`);
          break;
        }
        process.stdout.write('X');
      }
      
      // Rate limit: 500ms between requests
      await new Promise(r => setTimeout(r, 500));
    }
    
    const elapsed = Math.round((Date.now() - startTime) / 1000);
    process.stdout.write(` (${elapsed}s)`);
  }
  
  const elapsed = Math.round((Date.now() - startTime) / 1000);
  console.log(`\n\n=== Audio Generation Complete ===`);
  console.log(`Success: ${totalSuccess}, Failed: ${totalFailed}`);
  console.log(`Total size: ${Math.round(totalSize / 1024 / 1024 * 100) / 100} MB`);
  console.log(`Characters used: ${charsUsed.toLocaleString()}`);
  console.log(`Time: ${elapsed}s`);
  
  // Final credit check
  const finalCredits = await checkCredits();
  if (finalCredits) {
    console.log(`\nRemaining credits: ${finalCredits.remaining.toLocaleString()} characters`);
  }
  
  await client.end();
}

const startDay = parseInt(process.argv[2]) || 61;
const endDay = parseInt(process.argv[3]) || 100;
const batchSize = parseInt(process.argv[4]) || 5;

batchGenerate(startDay, endDay, batchSize).catch(e => {
  console.error('Fatal:', e);
  process.exit(1);
});
