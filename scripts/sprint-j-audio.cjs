/**
 * Sprint J: TTS Audio Generation for ALL 1,825 slots
 * Uses ElevenLabs /with-timestamps for audio + alignment in one call
 * Checkpoints every 50 files for resumability
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
const API_KEY = process.env.ELEVENLABS_API_KEY;
const AUDIO_DIR = 'C:\\Users\\user\\kelly-pipeline\\audio-cache';
const LOG_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\logs', 'cursor-burndown.log');
const CP_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\checkpoints', 'burndown.json');

function log(sprint, msg) {
  const line = `[${new Date().toISOString()}] ${sprint} | ${msg}\n`;
  fs.appendFileSync(LOG_FILE, line, 'utf-8');
  process.stdout.write(line);
}

function saveCheckpoint(data) {
  const cp = JSON.parse(fs.readFileSync(CP_FILE, 'utf-8'));
  cp.sprints.J = data;
  cp.last_updated = new Date().toISOString();
  fs.writeFileSync(CP_FILE, JSON.stringify(cp, null, 2));
}

async function generateWithTimestamps(text) {
  const res = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}/with-timestamps`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'xi-api-key': API_KEY },
    body: JSON.stringify({
      text,
      model_id: 'eleven_multilingual_v2',
      output_format: 'mp3_44100_128',
      voice_settings: { stability: 0.5, similarity_boost: 0.75, style: 0.0, use_speaker_boost: true }
    })
  });
  
  if (res.status === 429) {
    throw new Error('RATE_LIMIT');
  }
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`ElevenLabs ${res.status}: ${err.substring(0, 200)}`);
  }
  
  const data = await res.json();
  return {
    audio_base64: data.audio_base64,
    alignment: data.alignment || null
  };
}

async function checkCredits() {
  try {
    const res = await fetch('https://api.elevenlabs.io/v1/user/subscription', {
      headers: { 'xi-api-key': API_KEY }
    });
    if (res.ok) {
      const sub = await res.json();
      return { used: sub.character_count, limit: sub.character_limit, remaining: sub.character_limit - sub.character_count };
    }
  } catch (e) {}
  return null;
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  // ===== J.0 — Pre-flight =====
  log('SPRINT J.0', 'Pre-flight checks');
  
  const credits = await checkCredits();
  if (credits) {
    log('SPRINT J.0', `Credits: ${credits.remaining.toLocaleString()} remaining (${credits.used.toLocaleString()}/${credits.limit.toLocaleString()})`);
  }
  
  // ===== J.1 — Audit existing audio coverage =====
  log('SPRINT J.1', 'START | Auditing audio coverage');
  
  // Get all 1,825 slots from kellyos_lessons
  const allSlots = await client.query(`
    SELECT kl.day_number, kl.phase, kl.content_text, kl.title,
           ka.audio_url, ka.alignment_json, ka.duration_seconds
    FROM kellyos_lessons kl
    LEFT JOIN kellyos_audio ka ON ka.day_number = kl.day_number AND ka.phase = kl.phase
    WHERE kl.language = 'en' OR kl.language IS NULL
    ORDER BY kl.day_number, kl.phase
  `);
  
  log('SPRINT J.1', `Total slots: ${allSlots.rows.length}`);
  
  const hasAudio = allSlots.rows.filter(r => r.audio_url);
  const hasAlignment = allSlots.rows.filter(r => r.alignment_json);
  const needsGeneration = allSlots.rows.filter(r => !r.audio_url || !r.alignment_json);
  
  log('SPRINT J.1', `Has audio: ${hasAudio.length}, Has alignment: ${hasAlignment.length}`);
  log('SPRINT J.1', `Needs generation: ${needsGeneration.length}`);
  
  // Save audit
  const auditPath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'audio-coverage.json');
  fs.writeFileSync(auditPath, JSON.stringify({
    total_slots: allSlots.rows.length,
    has_audio: hasAudio.length,
    has_alignment: hasAlignment.length,
    needs_generation: needsGeneration.length,
    missing: needsGeneration.slice(0, 50).map(r => ({
      day: r.day_number, phase: r.phase,
      issue: !r.audio_url ? 'no_audio' : 'no_alignment'
    }))
  }, null, 2));
  log('SPRINT J.1', 'COMPLETE | audio-coverage.json written');
  
  // ===== J.2 — Batch TTS generation =====
  if (needsGeneration.length === 0) {
    log('SPRINT J.2', 'SKIP | All 1,825 slots already have audio + alignment');
  } else {
    log('SPRINT J.2', `START | Generating audio for ${needsGeneration.length} slots`);
    
    // Check checkpoint for resume point
    let startIndex = 0;
    try {
      const cp = JSON.parse(fs.readFileSync(CP_FILE, 'utf-8'));
      if (cp.sprints.J?.last_index) startIndex = cp.sprints.J.last_index;
    } catch (e) {}
    
    let generated = 0;
    let failed = 0;
    let retryDelay = 100;
    const startTime = Date.now();
    
    for (let i = startIndex; i < needsGeneration.length; i++) {
      const slot = needsGeneration[i];
      const text = slot.content_text || slot.title || `Lesson for Day ${slot.day_number}, ${slot.phase} phase.`;
      
      if (text.length < 5) {
        log('SPRINT J.2', `SKIP Day ${slot.day_number} ${slot.phase}: text too short`);
        continue;
      }
      
      try {
        const result = await generateWithTimestamps(text);
        
        // Save audio file
        const audioDir = path.join(AUDIO_DIR, `day-${String(slot.day_number).padStart(3, '0')}`);
        fs.mkdirSync(audioDir, { recursive: true });
        const audioPath = path.join(audioDir, `${slot.phase}.mp3`);
        const audioBuffer = Buffer.from(result.audio_base64, 'base64');
        fs.writeFileSync(audioPath, audioBuffer);
        
        // Estimate duration from file size (rough: 128kbps = 16KB/s)
        const durationSec = Math.round(audioBuffer.length / 16000);
        
        // Upsert into kellyos_audio
        await client.query(`
          INSERT INTO kellyos_audio (day_number, phase, audio_url, alignment_json, duration_seconds)
          VALUES ($1, $2, $3, $4, $5)
          ON CONFLICT (day_number, phase) DO UPDATE
          SET audio_url = EXCLUDED.audio_url, alignment_json = EXCLUDED.alignment_json, duration_seconds = EXCLUDED.duration_seconds
        `, [
          slot.day_number,
          slot.phase,
          audioPath, // local path for now
          result.alignment ? JSON.stringify(result.alignment) : null,
          durationSec
        ]);
        
        generated++;
        retryDelay = 100; // Reset delay on success
        
        if (generated % 10 === 0) {
          const elapsed = Math.round((Date.now() - startTime) / 1000);
          const rate = Math.round(generated / (elapsed / 60));
          log('SPRINT J.2', `PROGRESS | ${generated}/${needsGeneration.length} audio files (${Math.round(generated/needsGeneration.length*100)}%) - ${rate}/min`);
        }
        
        // Checkpoint every 50
        if (generated % 50 === 0) {
          saveCheckpoint({
            status: 'in_progress',
            progress: `${generated}/${needsGeneration.length}`,
            last_index: i,
            started_at: new Date(startTime).toISOString()
          });
        }
        
        // Rate limiting: 200ms between requests
        await new Promise(r => setTimeout(r, 200));
        
      } catch (e) {
        if (e.message === 'RATE_LIMIT') {
          retryDelay = Math.min(retryDelay * 3, 45000);
          log('SPRINT J.2', `RATE LIMITED | Waiting ${retryDelay/1000}s...`);
          await new Promise(r => setTimeout(r, retryDelay));
          i--; // Retry this slot
        } else if (e.message.includes('quota') || e.message.includes('limit')) {
          log('SPRINT J.2', `CREDIT LIMIT | ${generated} files generated before hitting limit`);
          const finalCredits = await checkCredits();
          if (finalCredits) log('SPRINT J.2', `Remaining: ${finalCredits.remaining.toLocaleString()} chars`);
          break;
        } else {
          failed++;
          log('SPRINT J.2', `FAILED Day ${slot.day_number} ${slot.phase}: ${e.message.substring(0, 80)}`);
        }
      }
    }
    
    const elapsed = Math.round((Date.now() - startTime) / 1000);
    log('SPRINT J.2', `COMPLETE | Generated: ${generated}, Failed: ${failed}, Time: ${elapsed}s`);
    
    const finalCredits = await checkCredits();
    if (finalCredits) log('SPRINT J.2', `Final credits: ${finalCredits.remaining.toLocaleString()} remaining`);
  }
  
  // ===== J.4 — Verify ALL 1,825 audio slots =====
  log('SPRINT J.4', 'START | Final verification');
  
  const verify = await client.query(`
    SELECT 
      (SELECT COUNT(*) FROM kellyos_audio) as total_audio,
      (SELECT COUNT(*) FROM kellyos_audio WHERE audio_url IS NOT NULL) as with_url,
      (SELECT COUNT(*) FROM kellyos_audio WHERE alignment_json IS NOT NULL) as with_alignment,
      (SELECT COUNT(*) FROM kellyos_audio WHERE duration_seconds > 0) as with_duration
  `);
  const v = verify.rows[0];
  log('SPRINT J.4', `Total audio rows: ${v.total_audio}`);
  log('SPRINT J.4', `With URL: ${v.with_url}, With alignment: ${v.with_alignment}, With duration: ${v.with_duration}`);
  
  // Save verification
  const verifyPath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'audio-final-verification.json');
  fs.writeFileSync(verifyPath, JSON.stringify({
    total_rows: parseInt(v.total_audio),
    with_url: parseInt(v.with_url),
    with_alignment: parseInt(v.with_alignment),
    with_duration: parseInt(v.with_duration),
    target: 1825,
    coverage_percent: `${Math.round(parseInt(v.total_audio) / 1825 * 100)}%`,
    verified_at: new Date().toISOString()
  }, null, 2));
  
  log('SPRINT J.4', 'COMPLETE | audio-final-verification.json written');
  
  // Update checkpoint
  saveCheckpoint({
    status: 'complete',
    completed_at: new Date().toISOString(),
    notes: `${v.total_audio} audio rows, ${v.with_url} with URLs, ${v.with_alignment} with alignment`
  });
  
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
