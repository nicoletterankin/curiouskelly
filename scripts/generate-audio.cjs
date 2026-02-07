/**
 * Sprint C: ElevenLabs Audio Generation
 * Generates TTS audio for lesson scripts
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
const API_KEY = process.env.ELEVENLABS_API_KEY;

async function generateAudio(text, outputPath) {
  if (!API_KEY) throw new Error('ELEVENLABS_API_KEY not set');
  
  const res = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'xi-api-key': API_KEY,
    },
    body: JSON.stringify({
      text,
      model_id: 'eleven_turbo_v2_5',
      voice_settings: {
        stability: 0.5,
        similarity_boost: 0.8,
        style: 0.3,
        use_speaker_boost: true,
      }
    })
  });
  
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`ElevenLabs error ${res.status}: ${err.substring(0, 200)}`);
  }
  
  const buffer = Buffer.from(await res.arrayBuffer());
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, buffer);
  
  return { size: buffer.length, path: outputPath };
}

async function processScript(client, scriptId, content, dayNumber, phase, option) {
  const audioDir = path.join('C:\\Users\\user\\kelly-pipeline\\audio-cache');
  const filename = `day-${String(dayNumber).padStart(3, '0')}-phase-${phase}-opt-${option}.mp3`;
  const outputPath = path.join(audioDir, `day-${String(dayNumber).padStart(3, '0')}`, filename);
  
  try {
    const result = await generateAudio(content, outputPath);
    
    // Update generation_jobs
    await client.query(
      `INSERT INTO generation_jobs (atom_id, job_type, provider, status, output_url, completed_at, input_params)
       VALUES (NULL, 'audio', 'elevenlabs', 'complete', $1, NOW(), $2)`,
      [outputPath, JSON.stringify({ script_id: scriptId, day: dayNumber, phase, option })]
    );
    
    return { success: true, size: result.size, path: outputPath };
  } catch (e) {
    await client.query(
      `INSERT INTO generation_jobs (atom_id, job_type, provider, status, error, completed_at, input_params)
       VALUES (NULL, 'audio', 'elevenlabs', 'failed', $1, NOW(), $2)`,
      [e.message, JSON.stringify({ script_id: scriptId, day: dayNumber, phase, option })]
    );
    return { success: false, error: e.message };
  }
}

module.exports = { generateAudio, processScript };

// CLI mode
if (require.main === module) {
  (async () => {
    const client = new Client({ connectionString: process.env.DATABASE_URL });
    await client.connect();
    
    // Test with first available script
    const script = await client.query(`
      SELECT ls.id, ls.content, ls.phase, ls.option_number, cl.day_number
      FROM lesson_scripts ls
      JOIN lesson_atoms la ON la.id = ls.atom_id
      JOIN core_lessons_v2 cl ON cl.id = la.lesson_id
      WHERE ls.content IS NOT NULL AND ls.content != ''
      ORDER BY cl.day_number, ls.phase
      LIMIT 1
    `);
    
    if (script.rows.length === 0) {
      console.log('No scripts found');
      process.exit(0);
    }
    
    const s = script.rows[0];
    console.log(`Generating audio for Day ${s.day_number}, Phase ${s.phase}, Option ${s.option_number}`);
    console.log(`Script: "${s.content.substring(0, 80)}..."`);
    
    const result = await processScript(client, s.id, s.content, s.day_number, s.phase, s.option_number);
    console.log(result.success ? `SUCCESS: ${result.size} bytes -> ${result.path}` : `FAILED: ${result.error}`);
    
    // Check remaining credits
    try {
      const credRes = await fetch('https://api.elevenlabs.io/v1/user/subscription', {
        headers: { 'xi-api-key': API_KEY }
      });
      if (credRes.ok) {
        const sub = await credRes.json();
        console.log(`\nElevenLabs credits: ${sub.character_count}/${sub.character_limit} characters used`);
        console.log(`Remaining: ${sub.character_limit - sub.character_count} characters`);
      }
    } catch (e) {
      console.log('Could not check credits:', e.message);
    }
    
    await client.end();
  })();
}
