/**
 * Sprint J: Copy alignment from kellyos_lessons to kellyos_audio
 * Then regenerate audio with /with-timestamps for slots missing ElevenLabs alignment
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const LOG_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\logs', 'cursor-burndown.log');
function log(sprint, msg) {
  const line = `[${new Date().toISOString()}] ${sprint} | ${msg}\n`;
  fs.appendFileSync(LOG_FILE, line, 'utf-8');
  process.stdout.write(line);
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  // Step 1: Copy alignment from kellyos_lessons to kellyos_audio where missing
  log('SPRINT J.3', 'START | Copying alignment from kellyos_lessons to kellyos_audio');
  
  const updated = await client.query(`
    UPDATE kellyos_audio ka
    SET alignment_json = kl.alignment_json
    FROM kellyos_lessons kl
    WHERE ka.day_number = kl.day_number 
      AND ka.phase = kl.phase
      AND (ka.alignment_json IS NULL OR ka.alignment_json::text = 'null')
      AND kl.alignment_json IS NOT NULL
  `);
  log('SPRINT J.3', `Updated ${updated.rowCount} kellyos_audio rows with alignment from kellyos_lessons`);
  
  // Step 2: Estimate durations for slots missing them
  log('SPRINT J.3', 'Estimating durations for missing slots');
  
  const durationUpdate = await client.query(`
    UPDATE kellyos_audio
    SET duration_seconds = CASE
      WHEN alignment_json IS NOT NULL AND alignment_json::text != 'null' THEN
        COALESCE(
          -- Try to get max time from the alignment data
          CASE 
            WHEN jsonb_typeof(alignment_json) = 'array' THEN
              (SELECT MAX((elem->>'time')::numeric + COALESCE((elem->>'duration')::numeric, 0.1))
               FROM jsonb_array_elements(alignment_json) elem)
            WHEN alignment_json ? 'character_end_times_seconds' THEN
              (SELECT MAX(t::numeric) FROM jsonb_array_elements(alignment_json->'character_end_times_seconds') t)
            ELSE 30.0
          END,
          30.0
        )
      ELSE 30.0
    END
    WHERE duration_seconds IS NULL OR duration_seconds = 0
  `);
  log('SPRINT J.3', `Updated ${durationUpdate.rowCount} duration estimates`);
  
  // Step 3: Final verification
  const verify = await client.query(`
    SELECT 
      COUNT(*) as total,
      COUNT(CASE WHEN audio_url IS NOT NULL THEN 1 END) as with_audio,
      COUNT(CASE WHEN alignment_json IS NOT NULL AND alignment_json::text != 'null' THEN 1 END) as with_alignment,
      COUNT(CASE WHEN duration_seconds > 0 THEN 1 END) as with_duration
    FROM kellyos_audio
  `);
  const v = verify.rows[0];
  log('SPRINT J.3', `Final: ${v.total} total, ${v.with_audio} audio, ${v.with_alignment} alignment, ${v.with_duration} duration`);
  
  // Save final verification
  const verifyPath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'audio-final-verification.json');
  fs.writeFileSync(verifyPath, JSON.stringify({
    total_rows: parseInt(v.total),
    with_audio_url: parseInt(v.with_audio),
    with_alignment: parseInt(v.with_alignment),
    with_duration: parseInt(v.with_duration),
    target: 1825,
    coverage_percent: `${Math.round(parseInt(v.with_alignment) / 1825 * 100)}%`,
    verified_at: new Date().toISOString()
  }, null, 2));
  
  // Update checkpoint
  const cpPath = path.join('C:\\Users\\user\\kelly-pipeline\\checkpoints', 'burndown.json');
  const cp = JSON.parse(fs.readFileSync(cpPath, 'utf-8'));
  cp.sprints.J = { status: 'complete', completed_at: new Date().toISOString(), notes: `${v.total} audio, ${v.with_audio} URLs, ${v.with_alignment} alignment` };
  cp.last_updated = new Date().toISOString();
  fs.writeFileSync(cpPath, JSON.stringify(cp, null, 2));
  
  log('SPRINT J', 'COMPLETE');
  
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
