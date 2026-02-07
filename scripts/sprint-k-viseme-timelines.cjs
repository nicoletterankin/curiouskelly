/**
 * Sprint K: Viseme Timeline Pre-computation
 * Convert alignment_json -> proper viseme timeline arrays
 * Store in kellyos_audio.viseme_timeline (add column if needed)
 * Also save as JSON files
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

// Phoneme-to-viseme mapping (matches lib/lip-sync/phoneme-to-viseme.ts)
const CHAR_TO_VISEME = {
  'a': 'aa', 'e': 'E', 'i': 'ih', 'o': 'oh', 'u': 'U',
  'p': 'PP', 'b': 'PP', 'm': 'PP',
  'f': 'FF', 'v': 'FF',
  't': 'DD', 'd': 'DD', 'n': 'DD', 'l': 'DD',
  'k': 'kk', 'g': 'kk',
  's': 'SS', 'z': 'SS',
  'r': 'RR',
  'h': 'sil',
  'w': 'U', 'y': 'ih', 'j': 'CH',
  'c': 'kk', 'q': 'kk', 'x': 'kk',
  ' ': 'sil', '.': 'sil', ',': 'sil', '!': 'sil', '?': 'sil',
  "'": 'sil', '"': 'sil', '-': 'sil', ':': 'sil', ';': 'sil',
};

const DIGRAPH_TO_VISEME = {
  'th': 'TH', 'sh': 'CH', 'ch': 'CH', 'zh': 'CH',
  'ng': 'kk', 'ph': 'FF', 'wh': 'U',
};

function alignmentToVisemeTimeline(alignment) {
  if (!alignment) return [];
  
  // Case 1: Already an array of {time, viseme} objects
  if (Array.isArray(alignment)) {
    if (alignment.length > 0 && alignment[0].viseme) return alignment;
    if (alignment.length > 0 && alignment[0].time !== undefined) return alignment;
  }
  
  // Case 2: ElevenLabs format {characters, character_start_times_seconds, character_end_times_seconds}
  if (alignment.characters && alignment.character_start_times_seconds) {
    const chars = alignment.characters;
    const starts = alignment.character_start_times_seconds;
    const ends = alignment.character_end_times_seconds;
    
    const timeline = [];
    let lastViseme = 'sil';
    
    for (let i = 0; i < chars.length; i++) {
      const char = chars[i].toLowerCase();
      const startTime = starts[i];
      const endTime = ends[i];
      const duration = endTime - startTime;
      
      // Check digraphs
      let viseme;
      if (i < chars.length - 1) {
        const digraph = char + chars[i + 1].toLowerCase();
        if (DIGRAPH_TO_VISEME[digraph]) {
          viseme = DIGRAPH_TO_VISEME[digraph];
          i++; // Skip next char
        }
      }
      if (!viseme) {
        viseme = CHAR_TO_VISEME[char] || 'sil';
      }
      
      // Only add if viseme changed or significant time gap
      if (viseme !== lastViseme || duration > 0.15) {
        timeline.push({
          time: Math.round(startTime * 1000) / 1000,
          viseme,
          duration: Math.round(duration * 1000) / 1000
        });
        lastViseme = viseme;
      }
    }
    
    return timeline;
  }
  
  // Case 3: Pre-computed viseme array from Sprint 4 (text-to-viseme)
  if (alignment.visemes || alignment.timeline) {
    return alignment.visemes || alignment.timeline;
  }
  
  return [];
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  log('SPRINT K.1', 'START | Building viseme timelines for all 1,825 slots');
  
  // Add viseme_timeline column if missing
  try {
    await client.query('ALTER TABLE kellyos_audio ADD COLUMN IF NOT EXISTS viseme_timeline JSONB');
    log('SPRINT K.1', 'Added/verified viseme_timeline column');
  } catch (e) {
    log('SPRINT K.1', `Column note: ${e.message.substring(0, 60)}`);
  }
  
  // Get all alignment data
  const slots = await client.query(`
    SELECT id, day_number, phase, alignment_json, duration_seconds
    FROM kellyos_audio
    WHERE alignment_json IS NOT NULL
    ORDER BY day_number, phase
  `);
  
  log('SPRINT K.1', `Processing ${slots.rows.length} slots`);
  
  let processed = 0;
  let valid = 0;
  let invalid = 0;
  const visemeDir = path.join('C:\\Users\\user\\kelly-pipeline\\viseme-timelines');
  
  for (const slot of slots.rows) {
    let alignment = slot.alignment_json;
    if (typeof alignment === 'string') {
      try { alignment = JSON.parse(alignment); } catch (e) { alignment = null; }
    }
    
    const timeline = alignmentToVisemeTimeline(alignment);
    
    // Validate
    const isValid = timeline.length >= 5 &&
      timeline[0].time >= 0 &&
      new Set(timeline.map(t => t.viseme)).size >= 2;
    
    if (isValid) valid++;
    else invalid++;
    
    // Save to DB
    await client.query(
      'UPDATE kellyos_audio SET viseme_timeline = $1 WHERE id = $2',
      [JSON.stringify(timeline), slot.id]
    );
    
    // Also update kellyos_lessons
    await client.query(
      'UPDATE kellyos_lessons SET alignment_json = $1 WHERE day_number = $2 AND phase = $3',
      [JSON.stringify(timeline), slot.day_number, slot.phase]
    );
    
    // Save to file
    const dayDir = path.join(visemeDir, `day-${String(slot.day_number).padStart(3, '0')}`);
    fs.mkdirSync(dayDir, { recursive: true });
    fs.writeFileSync(path.join(dayDir, `${slot.phase}.json`), JSON.stringify(timeline, null, 2));
    
    processed++;
    if (processed % 200 === 0) {
      log('SPRINT K.1', `PROGRESS | ${processed}/${slots.rows.length} (${valid} valid, ${invalid} invalid)`);
    }
  }
  
  log('SPRINT K.1', `COMPLETE | Processed: ${processed}, Valid: ${valid}, Invalid: ${invalid}`);
  
  // K.2 — Validate
  log('SPRINT K.2', 'START | Validating all timelines');
  
  const validation = await client.query(`
    SELECT day_number, phase, viseme_timeline
    FROM kellyos_audio
    WHERE viseme_timeline IS NOT NULL
    ORDER BY day_number, phase
  `);
  
  let validated = 0;
  let failedValidation = 0;
  const failures = [];
  
  for (const row of validation.rows) {
    let tl = row.viseme_timeline;
    if (typeof tl === 'string') tl = JSON.parse(tl);
    
    const hasEvents = Array.isArray(tl) && tl.length >= 5;
    const chronological = !hasEvents || tl.every((e, i) => i === 0 || e.time >= tl[i-1].time);
    const variety = !hasEvents || new Set(tl.map(e => e.viseme)).size >= 2;
    
    if (hasEvents && chronological && variety) {
      validated++;
    } else {
      failedValidation++;
      if (failures.length < 20) {
        failures.push({
          day: row.day_number, phase: row.phase,
          events: Array.isArray(tl) ? tl.length : 0,
          reason: !hasEvents ? 'too_few_events' : !chronological ? 'not_chronological' : 'no_variety'
        });
      }
    }
  }
  
  const validationResult = {
    total: validation.rows.length,
    valid: validated,
    invalid: failedValidation,
    failures,
    validated_at: new Date().toISOString()
  };
  
  fs.writeFileSync(
    path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'viseme-validation.json'),
    JSON.stringify(validationResult, null, 2)
  );
  
  log('SPRINT K.2', `COMPLETE | Valid: ${validated}/${validation.rows.length}, Invalid: ${failedValidation}`);
  
  // Update checkpoint
  const cpPath = path.join('C:\\Users\\user\\kelly-pipeline\\checkpoints', 'burndown.json');
  const cp = JSON.parse(fs.readFileSync(cpPath, 'utf-8'));
  cp.sprints.K = { status: 'complete', completed_at: new Date().toISOString(), notes: `${validated} valid timelines, ${failedValidation} invalid` };
  cp.last_updated = new Date().toISOString();
  fs.writeFileSync(cpPath, JSON.stringify(cp, null, 2));
  
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
