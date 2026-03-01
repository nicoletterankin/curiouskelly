/**
 * FULL BATCH: Generate TTS audio for ALL 365 days x 5 phases = 1,825 files
 * 
 * Pipeline: kellyos_lessons (text) -> TTS worker -> Vercel Blob -> kellyos_audio (URL update)
 * 
 * Rate limit: 5 requests/second max (200ms between requests)
 * Logging: Every 50 files
 * Error handling: Log failures, continue batch
 */
require('dotenv').config();
const { neon } = require('@neondatabase/serverless');
const { put } = require('@vercel/blob');

const sql = neon('postgresql://neondb_owner:npg_Nsq0oHSkb8yO@ep-fragrant-scene-a4lk0xwx-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require');

const TTS_URL = 'https://tts.curiouskelly.com/tts';
const VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
const RATE_LIMIT_MS = 220; // ~4.5 req/sec to stay safely under 5/sec

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function run() {
  const startTime = Date.now();
  
  // 1. Get ALL 1,825 English mentor-tone scripts
  console.log('=== LOADING ALL SCRIPTS FROM kellyos_lessons ===');
  const scripts = await sql`
    SELECT day_number, phase, content_text
    FROM kellyos_lessons
    WHERE language = 'en' AND tone = 'mentor'
    ORDER BY day_number, phase
  `;
  console.log(`Loaded ${scripts.length} scripts`);

  if (scripts.length !== 1825) {
    console.log(`WARNING: Expected 1825, got ${scripts.length}`);
  }

  // 2. Check which already have web URLs (skip those)
  const existing = await sql`
    SELECT day_number, phase 
    FROM kellyos_audio 
    WHERE audio_url LIKE 'http%'
  `;
  const existingSet = new Set(existing.map(r => `${r.day_number}-${r.phase}`));
  console.log(`Already have ${existingSet.size} web URLs — will skip those`);

  const toProcess = scripts.filter(s => !existingSet.has(`${s.day_number}-${s.phase}`));
  console.log(`Need to process: ${toProcess.length} files\n`);

  // 3. Process in order with rate limiting
  let completed = 0;
  let failed = 0;
  let skipped = 0;
  const failures = [];

  for (let i = 0; i < toProcess.length; i++) {
    const { day_number, phase, content_text } = toProcess[i];
    const dayPad = String(day_number).padStart(3, '0');

    try {
      // Call TTS
      const ttsRes = await fetch(TTS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          text: content_text, 
          voice_id: VOICE_ID, 
          phase, 
          day: day_number, 
          language: 'en' 
        })
      });

      if (!ttsRes.ok) {
        const errText = await ttsRes.text().catch(() => '');
        failures.push({ day_number, phase, error: `TTS ${ttsRes.status}: ${errText.substring(0, 200)}` });
        failed++;
        // Rate limit even on failure
        await sleep(RATE_LIMIT_MS);
        continue;
      }

      const audioBuffer = Buffer.from(await ttsRes.arrayBuffer());

      // Upload to Vercel Blob
      const filename = `audio/2026/en/day-${dayPad}/${phase}.mp3`;
      const blob = await put(filename, audioBuffer, {
        access: 'public',
        contentType: 'audio/mpeg',
        addRandomSuffix: false,
        allowOverwrite: true,
      });

      // Update kellyos_audio
      await sql`
        UPDATE kellyos_audio 
        SET audio_url = ${blob.url}
        WHERE day_number = ${day_number} AND phase = ${phase}
      `;

      completed++;
    } catch (e) {
      failures.push({ day_number, phase, error: e.message });
      failed++;
    }

    // Progress logging every 50 files
    if ((completed + failed) % 50 === 0 || i === toProcess.length - 1) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const pct = (((completed + failed + skipped) / toProcess.length) * 100).toFixed(1);
      const rate = (completed / (elapsed / 60)).toFixed(1);
      console.log(`[${elapsed}s] ${pct}% | ${completed} done, ${failed} failed | Day ${day_number} ${phase} | ${rate}/min`);
    }

    // Rate limit
    await sleep(RATE_LIMIT_MS);
  }

  // 4. Summary
  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log('\n========================================');
  console.log('BATCH COMPLETE');
  console.log(`  Total time: ${totalTime}s (${(totalTime / 60).toFixed(1)} min)`);
  console.log(`  Completed: ${completed}`);
  console.log(`  Failed: ${failed}`);
  console.log(`  Previously had web URL: ${existingSet.size}`);
  console.log('========================================');

  if (failures.length > 0) {
    console.log('\nFAILED ITEMS:');
    for (const f of failures) {
      console.log(`  Day ${f.day_number} | ${f.phase} | ${f.error}`);
    }
  }

  // 5. Final verification
  console.log('\n=== FINAL VERIFICATION ===');
  const webCount = await sql`SELECT COUNT(*)::int as cnt FROM kellyos_audio WHERE audio_url LIKE 'http%'`;
  const localCount = await sql`SELECT COUNT(*)::int as cnt FROM kellyos_audio WHERE audio_url NOT LIKE 'http%'`;
  const totalCount = await sql`SELECT COUNT(*)::int as cnt FROM kellyos_audio`;
  console.log(`  Total rows: ${totalCount[0].cnt}`);
  console.log(`  Web URLs: ${webCount[0].cnt}`);
  console.log(`  Local/other: ${localCount[0].cnt}`);
  console.log(`  TARGET: 1,825 web URLs`);
  console.log(`  STATUS: ${webCount[0].cnt >= 1825 ? 'COMPLETE' : 'INCOMPLETE — ' + (1825 - webCount[0].cnt) + ' remaining'}`);
}

run().catch(e => {
  console.error('FATAL:', e.message);
  process.exit(1);
});
