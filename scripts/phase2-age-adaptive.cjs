/**
 * Phase 2: Age-Adaptive Content Generation
 * Generates kid (2-7), teen (13-17), and elder (65+) versions of all 1,825 English scripts.
 * Uses OpenAI GPT-4o-mini for rewrites.
 * Stores in lesson_atoms table with age_group field.
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

function log(msg) {
  console.log(`[${new Date().toISOString()}] AGE-ADAPT | ${msg}`);
}

const AGE_GROUPS = [
  {
    name: 'kid',
    label: 'Ages 2-7',
    systemPrompt: `You are rewriting educational scripts for children ages 2-7. Rules:
- Use simple, concrete vocabulary only (no abstract concepts)
- Maximum 8 words per sentence
- Warm, playful, encouraging tone
- Add sensory details and familiar comparisons ("as big as a school bus!")
- Remove complex metaphors and jargon
- Keep it 40-60% of the original length
- Maintain all factual accuracy
- Return ONLY the rewritten script, no explanations.`
  },
  {
    name: 'teen',
    label: 'Ages 13-17',
    systemPrompt: `You are rewriting educational scripts for teenagers ages 13-17. Rules:
- Age-appropriate vocabulary, no dumbing down
- Peer-like tone that respects their intelligence
- Add real-world relevance and "why should I care?" angle
- Connect to technology, social media, and current culture where relevant
- Remove any condescension or over-simplification
- Keep it 80-100% of the original length
- Maintain all factual accuracy
- Return ONLY the rewritten script, no explanations.`
  },
  {
    name: 'elder',
    label: 'Ages 65+',
    systemPrompt: `You are rewriting educational scripts for adults ages 65+. Rules:
- Full vocabulary complexity, respectful and dignified tone
- Warm, unhurried, reflective delivery
- Add historical context and life experience connections
- Include intergenerational perspective ("imagine sharing this with a grandchild")
- Slightly slower pacing, more contemplative
- Keep it 90-110% of the original length
- Maintain all factual accuracy
- Return ONLY the rewritten script, no explanations.`
  }
];

async function rewriteBatch(texts, ageGroup, retries = 3) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const userContent = texts.length === 1
        ? texts[0]
        : JSON.stringify(texts);
      
      const wrapInArray = texts.length > 1;

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
              content: ageGroup.systemPrompt + (wrapInArray
                ? '\n\nYou will receive a JSON array of scripts. Return a JSON array of rewritten scripts in the same order. ONLY output the JSON array.'
                : '')
            },
            { role: 'user', content: userContent }
          ],
          temperature: 0.4,
          max_tokens: 8000
        })
      });

      if (res.status === 429) {
        const wait = Math.min(30000, (attempt + 1) * 10000);
        log(`Rate limited, waiting ${wait / 1000}s`);
        await new Promise(r => setTimeout(r, wait));
        continue;
      }
      if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).substring(0, 200)}`);

      const data = await res.json();
      const content = data.choices[0].message.content;

      if (wrapInArray) {
        const match = content.match(/\[[\s\S]*\]/);
        if (!match) throw new Error('No JSON array in response');
        const parsed = JSON.parse(match[0]);
        if (parsed.length !== texts.length) throw new Error(`Got ${parsed.length}, expected ${texts.length}`);
        return parsed;
      } else {
        return [content.trim()];
      }
    } catch (e) {
      if (attempt === retries - 1) throw e;
      log(`Retry ${attempt + 1}: ${e.message}`);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}

// Phase map: kellyos_lessons phase names → lesson_atoms phase integers
const PHASE_MAP = { hook: 1, story: 2, wonder: 3, action: 4, wisdom: 5, example: 3, practice: 4, reflect: 5 };
const PHASE_NAMES = { 1: 'hook', 2: 'story', 3: 'wonder', 4: 'action', 5: 'wisdom' };
const VARIANT_FOR_AGE = { kid: 'kid_adapt', teen: 'teen_adapt', elder: 'elder_adapt' };

async function processAgeGroup(client, ageGroup) {
  const variant = VARIANT_FOR_AGE[ageGroup.name];

  // Count existing
  const existingRes = await client.query(
    'SELECT COUNT(*) as cnt FROM lesson_atoms WHERE age_group = $1 AND language = $2',
    [ageGroup.name, 'en']
  );
  const existing = parseInt(existingRes.rows[0].cnt);
  
  if (existing >= 1820) {
    log(`${ageGroup.name}: SKIP — already have ${existing}/1825`);
    return { done: existing, new: 0 };
  }

  log(`${ageGroup.name}: ${existing} existing, finding gaps...`);

  // Find English scripts from kellyos_lessons that don't have this age adaptation yet
  // lesson_id maps 1:1 with day_number in core_lessons_v2
  const missing = await client.query(`
    SELECT e.day_number, e.phase as phase_name, e.content_text, e.title
    FROM kellyos_lessons e
    WHERE (e.language = 'en' OR e.language IS NULL)
      AND e.content_text IS NOT NULL
      AND length(e.content_text) > 20
    ORDER BY e.day_number, e.phase
  `, []);

  // Filter out those that already exist in lesson_atoms
  const existingAtoms = await client.query(
    'SELECT lesson_id, phase FROM lesson_atoms WHERE age_group = $1 AND language = $2',
    [ageGroup.name, 'en']
  );
  const existingSet = new Set(existingAtoms.rows.map(r => `${r.lesson_id}-${r.phase}`));

  const missingRows = missing.rows.filter(r => {
    const phaseInt = PHASE_MAP[r.phase_name] || 1;
    return !existingSet.has(`${r.day_number}-${phaseInt}`);
  });

  log(`${ageGroup.name}: ${missingRows.length} adaptations needed`);

  let done = 0;
  let failed = 0;
  const batchSize = 5;

  for (let i = 0; i < missingRows.length; i += batchSize) {
    const batch = missingRows.slice(i, i + batchSize);
    const texts = batch.map(r => r.content_text);

    try {
      const rewritten = await rewriteBatch(texts, ageGroup);

      for (let j = 0; j < batch.length && j < rewritten.length; j++) {
        const phaseInt = PHASE_MAP[batch[j].phase_name] || 1;
        try {
          await client.query(`
            INSERT INTO lesson_atoms (lesson_id, phase, variant, age_group, language, script, status)
            VALUES ($1, $2, $3, $4, 'en', $5, 'script_complete')
            ON CONFLICT (lesson_id, phase, variant, age_group, language) DO UPDATE
            SET script = EXCLUDED.script, status = 'script_complete'
          `, [batch[j].day_number, phaseInt, variant, ageGroup.name, rewritten[j]]);
          done++;
        } catch (dbErr) {
          log(`${ageGroup.name}: DB error day ${batch[j].day_number}: ${dbErr.message.substring(0, 80)}`);
          failed++;
        }
      }
    } catch (e) {
      failed += batch.length;
      log(`${ageGroup.name}: Error at batch ${i}: ${e.message}`);
    }

    if (done % 50 === 0 && done > 0) {
      log(`${ageGroup.name}: ${done}/${missingRows.length} done (${failed} failed)`);
    }

    await new Promise(r => setTimeout(r, 200));
  }

  log(`${ageGroup.name} COMPLETE: ${done} new adaptations (${failed} failed)`);
  return { done: existing + done, new: done, failed };
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  log('Connected. Starting age-adaptive content generation...');

  const results = {};
  for (const ageGroup of AGE_GROUPS) {
    results[ageGroup.name] = await processAgeGroup(client, ageGroup);
  }

  log('=== AGE-ADAPTIVE RESULTS ===');
  for (const [group, result] of Object.entries(results)) {
    log(`${group}: ${result.done} total (${result.new} new)`);
  }

  const auditDir = path.join(__dirname, '..', 'kelly-pipeline', 'audit');
  if (!fs.existsSync(auditDir)) fs.mkdirSync(auditDir, { recursive: true });
  fs.writeFileSync(
    path.join(auditDir, 'age-adaptive-results.json'),
    JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2)
  );

  await client.end();
}

main().catch(e => { console.error('[AGE-ADAPT ERROR]', e); process.exit(1); });
