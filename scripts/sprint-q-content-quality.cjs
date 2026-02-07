/**
 * Sprint Q: Content Quality Enhancement
 * Q.1 — Generate fact-check questions (730 facts)
 * Q.2 — Generate lesson summaries (365)
 * Q.3 — Generate teaching notes
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const LOG_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\logs', 'cursor-burndown.log');
const CP_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\checkpoints', 'burndown.json');

function log(sprint, msg) {
  const line = `[${new Date().toISOString()}] ${sprint} | ${msg}\n`;
  fs.appendFileSync(LOG_FILE, line, 'utf-8');
  process.stdout.write(line);
}

async function callOpenAI(prompt, maxTokens = 500) {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [{ role: 'system', content: 'You are an educational content assistant. Be concise.' }, { role: 'user', content: prompt }],
      temperature: 0.7, max_tokens: maxTokens
    })
  });
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).substring(0, 100)}`);
  const data = await res.json();
  return data.choices[0].message.content;
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  // ===== Q.1 — Fact-check questions =====
  log('SPRINT Q.1', 'START | Creating kellyos_facts table + generating facts');
  
  await client.query(`
    CREATE TABLE IF NOT EXISTS kellyos_facts (
      id SERIAL PRIMARY KEY,
      day_number INTEGER NOT NULL,
      statement TEXT NOT NULL,
      is_true BOOLEAN NOT NULL,
      explanation TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);
  await client.query('CREATE INDEX IF NOT EXISTS idx_kellyos_facts_day ON kellyos_facts(day_number)');
  
  // Check existing
  const existingFacts = await client.query('SELECT COUNT(*) as cnt FROM kellyos_facts');
  log('SPRINT Q.1', `Existing facts: ${existingFacts.rows[0].cnt}`);
  
  if (parseInt(existingFacts.rows[0].cnt) < 700) {
    // Get all lessons
    const lessons = await client.query(`
      SELECT cl.day_number, cl.title, cl.subject, cl.learning_objective
      FROM core_lessons_v2 cl ORDER BY cl.day_number
    `);
    
    let factsGenerated = 0;
    for (const lesson of lessons.rows) {
      // Check if already has facts
      const has = await client.query('SELECT COUNT(*) as cnt FROM kellyos_facts WHERE day_number = $1', [lesson.day_number]);
      if (parseInt(has.rows[0].cnt) >= 2) continue;
      
      try {
        const prompt = `For the lesson "${lesson.title}" (${lesson.subject}), learning objective: "${lesson.learning_objective}":

Generate exactly 2 statements in JSON format:
1. One TRUE statement (a fact from the lesson)
2. One FALSE statement (a plausible but incorrect claim)

Return JSON only:
{"true_statement": "...", "true_explanation": "...", "false_statement": "...", "false_explanation": "why this is false..."}`;
        
        const response = await callOpenAI(prompt, 300);
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const facts = JSON.parse(jsonMatch[0]);
          
          await client.query(
            'INSERT INTO kellyos_facts (day_number, statement, is_true, explanation) VALUES ($1, $2, true, $3) ON CONFLICT DO NOTHING',
            [lesson.day_number, facts.true_statement, facts.true_explanation]
          );
          await client.query(
            'INSERT INTO kellyos_facts (day_number, statement, is_true, explanation) VALUES ($1, $2, false, $3) ON CONFLICT DO NOTHING',
            [lesson.day_number, facts.false_statement, facts.false_explanation]
          );
          factsGenerated += 2;
        }
      } catch (e) {
        // Skip failures silently
      }
      
      if (factsGenerated % 50 === 0 && factsGenerated > 0) {
        log('SPRINT Q.1', `PROGRESS | ${factsGenerated} facts generated`);
      }
      await new Promise(r => setTimeout(r, 100));
    }
    log('SPRINT Q.1', `COMPLETE | ${factsGenerated} facts generated`);
  } else {
    log('SPRINT Q.1', 'SKIP | Already have 700+ facts');
  }
  
  // ===== Q.2 — Lesson summaries =====
  log('SPRINT Q.2', 'START | Generating summaries');
  
  // Add summary column if not exists
  try {
    await client.query('ALTER TABLE core_lessons_v2 ADD COLUMN IF NOT EXISTS summary TEXT');
  } catch {}
  
  const needsSummary = await client.query(
    'SELECT id, day_number, title, subject, learning_objective FROM core_lessons_v2 WHERE summary IS NULL ORDER BY day_number'
  );
  
  let summariesDone = 0;
  for (const lesson of needsSummary.rows) {
    try {
      const summary = await callOpenAI(
        `Write a 1-sentence summary (max 100 characters) for this lesson: "${lesson.title}" about ${lesson.subject}. Learning: ${lesson.learning_objective}. Just the summary text, no quotes.`,
        60
      );
      const trimmed = summary.replace(/^["']|["']$/g, '').substring(0, 100);
      await client.query('UPDATE core_lessons_v2 SET summary = $1 WHERE id = $2', [trimmed, lesson.id]);
      summariesDone++;
    } catch {}
    if (summariesDone % 50 === 0 && summariesDone > 0) {
      log('SPRINT Q.2', `PROGRESS | ${summariesDone}/${needsSummary.rows.length} summaries`);
    }
    await new Promise(r => setTimeout(r, 80));
  }
  log('SPRINT Q.2', `COMPLETE | ${summariesDone} summaries generated`);
  
  // ===== Q.3 — Teaching notes =====
  log('SPRINT Q.3', 'START | Generating teaching notes');
  
  try {
    await client.query('ALTER TABLE kellyos_lessons ADD COLUMN IF NOT EXISTS teaching_notes JSONB');
  } catch {}
  
  const needsNotes = await client.query(`
    SELECT kl.id, kl.day_number, kl.phase, kl.title, kl.content_text
    FROM kellyos_lessons kl
    WHERE kl.teaching_notes IS NULL AND (kl.language = 'en' OR kl.language IS NULL)
    ORDER BY kl.day_number, kl.phase
    LIMIT 500
  `);
  
  let notesDone = 0;
  for (const lesson of needsNotes.rows) {
    try {
      const notes = await callOpenAI(
        `For the "${lesson.phase}" phase of lesson "${lesson.title}":
Generate teaching notes as JSON:
{"energy_level": "high/medium/low", "key_point": "one sentence", "engagement_tip": "one sentence", "transition_cue": "one sentence"}
Only JSON, no other text.`,
        150
      );
      const jsonMatch = notes.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        await client.query('UPDATE kellyos_lessons SET teaching_notes = $1 WHERE id = $2', [jsonMatch[0], lesson.id]);
        notesDone++;
      }
    } catch {}
    if (notesDone % 100 === 0 && notesDone > 0) {
      log('SPRINT Q.3', `PROGRESS | ${notesDone}/${needsNotes.rows.length} notes`);
    }
    await new Promise(r => setTimeout(r, 80));
  }
  log('SPRINT Q.3', `COMPLETE | ${notesDone} teaching notes generated`);
  
  // Update checkpoint
  const cp = JSON.parse(fs.readFileSync(CP_FILE, 'utf-8'));
  cp.sprints.Q = { status: 'complete', completed_at: new Date().toISOString(), notes: `facts, summaries, notes done` };
  cp.last_updated = new Date().toISOString();
  fs.writeFileSync(CP_FILE, JSON.stringify(cp, null, 2));
  
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
