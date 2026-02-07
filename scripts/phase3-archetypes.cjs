/**
 * Phase 3: Archetype Personality System
 * Generates hook + wisdom for 12 archetypes × 365 days = 8,760 scripts.
 * Each archetype has a distinct teaching voice and emotional signature.
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

function log(msg) {
  console.log(`[${new Date().toISOString()}] ARCHETYPE | ${msg}`);
}

const ARCHETYPES = [
  { name: 'mentor', emotion: 'warm', opener: 'Let me guide you through something fascinating today...' },
  { name: 'scientist', emotion: 'focused', opener: "Here's what the data tells us about..." },
  { name: 'storyteller', emotion: 'animated', opener: 'Once upon a time, in a world much like ours...' },
  { name: 'explorer', emotion: 'excited', opener: 'I discovered something incredible, and I have to share it...' },
  { name: 'philosopher', emotion: 'contemplative', opener: 'Have you ever stopped to wonder why...' },
  { name: 'artist', emotion: 'dreamy', opener: 'Close your eyes. Imagine a world where...' },
  { name: 'coach', emotion: 'energetic', opener: "Alright, team! Today we're going to crush..." },
  { name: 'librarian', emotion: 'precise', opener: 'I found the most remarkable passage in...' },
  { name: 'inventor', emotion: 'curious', opener: "What if I told you there's a way to..." },
  { name: 'historian', emotion: 'reverent', opener: '2,000 years ago, someone had the exact same question...' },
  { name: 'naturalist', emotion: 'peaceful', opener: 'Step outside with me. Notice how...' },
  { name: 'futurist', emotion: 'visionary', opener: 'By 2050, this will change everything...' },
];

async function rewriteInArchetype(texts, archetype, phase, retries = 3) {
  const phaseDesc = phase === 'hook'
    ? 'opening hook that captures curiosity and sets the stage for learning'
    : 'closing wisdom that provides a memorable takeaway and inspires reflection';

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            {
              role: 'system',
              content: `You are Kelly, an AI teacher, speaking in the voice of the "${archetype.name}" archetype. Your emotional signature is "${archetype.emotion}".

Your style: ${archetype.opener}

Rewrite each ${phaseDesc} in this archetype's voice. Keep the same factual content but change the delivery style, opening, and emotional tone to match the ${archetype.name} archetype.

Return ONLY a JSON array of rewritten strings matching the input order. No extra text.`
            },
            { role: 'user', content: JSON.stringify(texts) }
          ],
          temperature: 0.5,
          max_tokens: 8000
        })
      });

      if (res.status === 429) {
        await new Promise(r => setTimeout(r, (attempt + 1) * 10000));
        continue;
      }
      if (!res.ok) throw new Error(`OpenAI ${res.status}`);

      const data = await res.json();
      const content = data.choices[0].message.content;
      const match = content.match(/\[[\s\S]*\]/);
      if (!match) throw new Error('No JSON array');
      const parsed = JSON.parse(match[0]);
      if (parsed.length !== texts.length) throw new Error(`Got ${parsed.length}, expected ${texts.length}`);
      return parsed;
    } catch (e) {
      if (attempt === retries - 1) throw e;
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}

// lesson_atoms schema: lesson_id (=day_number), phase (INT 1-7), variant (TEXT), age_group, language, script, status
const PHASE_INT = { hook: 1, wisdom: 5 };

async function processArchetypePhase(client, archetype, phase) {
  // This version sources from core_lessons_v2 columns, but they may not exist.
  // Redirects to processArchetypePhaseFromLessons anyway.
  return processArchetypePhaseFromLessons(client, archetype, phase);
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  log('Connected. Starting archetype generation...');

  // Check if core_lessons_v2 has hook_text and wisdom_text columns
  const colCheck = await client.query(`
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'core_lessons_v2' AND column_name IN ('hook_text', 'wisdom_text')
  `);
  const cols = colCheck.rows.map(r => r.column_name);
  
  let useAlternateSource = false;
  if (!cols.includes('hook_text') || !cols.includes('wisdom_text')) {
    log('core_lessons_v2 missing hook_text/wisdom_text — will source from kellyos_lessons');
    useAlternateSource = true;
  }

  const results = {};
  const phases = ['hook', 'wisdom'];

  for (const archetype of ARCHETYPES) {
    results[archetype.name] = {};
    for (const phase of phases) {
      if (useAlternateSource) {
        // Alternative: source from kellyos_lessons
        results[archetype.name][phase] = await processArchetypePhaseFromLessons(client, archetype, phase);
      } else {
        results[archetype.name][phase] = await processArchetypePhase(client, archetype, phase);
      }
    }
    log(`${archetype.name}: hooks=${results[archetype.name].hook.done}, wisdom=${results[archetype.name].wisdom.done}`);
  }

  // Save results
  const auditDir = path.join(__dirname, '..', 'kelly-pipeline', 'audit');
  if (!fs.existsSync(auditDir)) fs.mkdirSync(auditDir, { recursive: true });
  fs.writeFileSync(
    path.join(auditDir, 'archetype-results.json'),
    JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2)
  );

  log('ARCHETYPE GENERATION COMPLETE');
  await client.end();
}

// Sources from kellyos_lessons, writes to lesson_atoms with correct schema
async function processArchetypePhaseFromLessons(client, archetype, phase) {
  const phaseInt = PHASE_INT[phase] || 1;
  const variant = archetype.name; // Use archetype name as variant

  const existingRes = await client.query(
    'SELECT COUNT(*) as cnt FROM lesson_atoms WHERE variant = $1 AND phase = $2 AND age_group = $3',
    [variant, phaseInt, 'adult']
  );
  const existing = parseInt(existingRes.rows[0].cnt);

  if (existing >= 360) {
    log(`${archetype.name}/${phase}: SKIP — ${existing}/365 already done`);
    return { done: existing, new: 0 };
  }

  // Get existing set for fast lookup
  const existingAtoms = await client.query(
    'SELECT lesson_id FROM lesson_atoms WHERE variant = $1 AND phase = $2 AND age_group = $3',
    [variant, phaseInt, 'adult']
  );
  const existingSet = new Set(existingAtoms.rows.map(r => r.lesson_id));

  const allRes = await client.query(`
    SELECT k.day_number, k.content_text as source_text, k.title
    FROM kellyos_lessons k
    WHERE k.phase = $1
      AND (k.language = 'en' OR k.language IS NULL)
      AND k.content_text IS NOT NULL
      AND length(k.content_text) > 10
    ORDER BY k.day_number
  `, [phase]);

  const missing = allRes.rows.filter(r => !existingSet.has(r.day_number));

  if (missing.length === 0) {
    log(`${archetype.name}/${phase}: Nothing to do`);
    return { done: existing, new: 0 };
  }

  log(`${archetype.name}/${phase}: ${missing.length} needed`);

  let done = 0;
  let failed = 0;
  const batchSize = 10;

  for (let i = 0; i < missing.length; i += batchSize) {
    const batch = missing.slice(i, i + batchSize);
    const texts = batch.map(r => r.source_text);

    try {
      const rewritten = await rewriteInArchetype(texts, archetype, phase);
      for (let j = 0; j < batch.length && j < rewritten.length; j++) {
        try {
          await client.query(`
            INSERT INTO lesson_atoms (lesson_id, phase, variant, age_group, language, script, status)
            VALUES ($1, $2, $3, 'adult', 'en', $4, 'script_complete')
            ON CONFLICT (lesson_id, phase, variant, age_group, language) DO UPDATE
            SET script = EXCLUDED.script, status = 'script_complete'
          `, [batch[j].day_number, phaseInt, variant, rewritten[j]]);
          done++;
        } catch (dbErr) {
          failed++;
          log(`${archetype.name}/${phase}: DB error: ${dbErr.message.substring(0, 80)}`);
        }
      }
    } catch (e) {
      failed += batch.length;
      log(`${archetype.name}/${phase}: Error: ${e.message}`);
    }

    if (done % 50 === 0 && done > 0) log(`${archetype.name}/${phase}: ${done}/${missing.length}`);
    await new Promise(r => setTimeout(r, 200));
  }

  return { done: existing + done, new: done, failed };
}

main().catch(e => { console.error('[ARCHETYPE ERROR]', e); process.exit(1); });
