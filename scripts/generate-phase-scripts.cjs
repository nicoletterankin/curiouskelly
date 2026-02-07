/**
 * Sprint B: Generate Phase Scripts for Lessons
 * Input: lesson seed from core_lessons_v2
 * Output: 14 script segments per lesson (7 phases × 2 options)
 * Uses OpenAI API (GPT-4o-mini for speed) - falls back to Anthropic
 */
require('dotenv').config();
const { Client } = require('pg');

const PHASE_NAMES = {
  1: 'Hook',
  2: 'Teach',
  3: 'Example',
  4: 'Practice',
  5: 'Reflect',
  6: 'Apply',
  7: 'Close'
};

const PHASE_DESCRIPTIONS = {
  1: 'A compelling opening question or statement that grabs attention and creates curiosity. 1-2 sentences, 10-20 words.',
  2: 'The main teaching content. Explain the core concept clearly. 3-5 sentences, 40-80 words.',
  3: 'A vivid real-world example that illustrates the concept. 2-4 sentences, 30-60 words.',
  4: 'An interactive practice activity or thought experiment the learner can try. 2-3 sentences, 20-50 words.',
  5: 'A reflective question or prompt that encourages deeper thinking. 1-2 sentences, 15-30 words.',
  6: 'A practical application the learner can do today in their daily life. 2-3 sentences, 20-40 words.',
  7: 'A memorable closing thought or wisdom quote that encapsulates the lesson. 1-2 sentences, 10-25 words.'
};

async function callOpenAI(prompt) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY not set');
  
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are Kelly, a warm, curious AI teacher who makes learning magical for all ages. Write in first person as Kelly. Be enthusiastic but not over-the-top. Use simple, clear language.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.8,
      max_tokens: 2000
    })
  });
  
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI API error ${res.status}: ${err.substring(0, 200)}`);
  }
  
  const data = await res.json();
  return data.choices[0].message.content;
}

async function generateScriptsForLesson(lesson) {
  const seedData = lesson.seed_data || {};
  const existingScripts = seedData.scripts || {};
  
  const prompt = `Generate lesson scripts for this educational lesson:

LESSON: Day ${lesson.day_number} - "${lesson.title}"
SUBJECT: ${lesson.subject || 'general'}
LEARNING OBJECTIVE: ${lesson.learning_objective || 'Explore this topic'}
EXISTING CONTEXT: ${JSON.stringify(seedData.existing || {}).substring(0, 300)}

Generate ALL 7 phases, each with 2 OPTIONS (Option A and Option B are different approaches to the same phase).

For each phase, respond in this exact JSON format:
{
  "phases": {
    "1": { "name": "Hook", "option_1": "...", "option_2": "..." },
    "2": { "name": "Teach", "option_1": "...", "option_2": "..." },
    "3": { "name": "Example", "option_1": "...", "option_2": "..." },
    "4": { "name": "Practice", "option_1": "...", "option_2": "..." },
    "5": { "name": "Reflect", "option_1": "...", "option_2": "..." },
    "6": { "name": "Apply", "option_1": "...", "option_2": "..." },
    "7": { "name": "Close", "option_1": "...", "option_2": "..." }
  }
}

Phase guidelines:
1. Hook: ${PHASE_DESCRIPTIONS[1]}
2. Teach: ${PHASE_DESCRIPTIONS[2]}
3. Example: ${PHASE_DESCRIPTIONS[3]}
4. Practice: ${PHASE_DESCRIPTIONS[4]}
5. Reflect: ${PHASE_DESCRIPTIONS[5]}
6. Apply: ${PHASE_DESCRIPTIONS[6]}
7. Close: ${PHASE_DESCRIPTIONS[7]}

IMPORTANT: 
- Write as Kelly speaking directly to the learner
- Option 1 and Option 2 should take different angles/approaches
- Each script segment should be natural spoken language (it will be turned into audio)
- Do NOT include stage directions or [brackets]
- Return ONLY valid JSON, no other text`;

  const response = await callOpenAI(prompt);
  
  // Parse JSON from response
  let parsed;
  try {
    // Try to extract JSON from response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error('No JSON found in response');
    parsed = JSON.parse(jsonMatch[0]);
  } catch (e) {
    throw new Error(`Failed to parse script response: ${e.message}`);
  }
  
  return parsed.phases;
}

async function processLesson(client, lessonId, dayNumber) {
  // Get lesson data
  const lessonRes = await client.query(
    'SELECT * FROM core_lessons_v2 WHERE id = $1',
    [lessonId]
  );
  if (lessonRes.rows.length === 0) return { success: false, error: 'Lesson not found' };
  
  const lesson = lessonRes.rows[0];
  
  try {
    const phases = await generateScriptsForLesson(lesson);
    
    let scriptsWritten = 0;
    
    for (let phase = 1; phase <= 7; phase++) {
      const phaseData = phases[String(phase)];
      if (!phaseData) continue;
      
      // Get or create atom for this phase
      let atomRes = await client.query(
        `SELECT id FROM lesson_atoms WHERE lesson_id = $1 AND phase = $2 AND age_group = 'adult' AND language = 'en'`,
        [lessonId, phase]
      );
      
      let atomId;
      if (atomRes.rows.length === 0) {
        const newAtom = await client.query(
          `INSERT INTO lesson_atoms (lesson_id, phase, variant, age_group, language, status)
           VALUES ($1, $2, $3, 'adult', 'en', 'script_complete') RETURNING id`,
          [lessonId, phase, PHASE_NAMES[phase].toLowerCase()]
        );
        atomId = newAtom.rows[0].id;
      } else {
        atomId = atomRes.rows[0].id;
        await client.query(
          `UPDATE lesson_atoms SET status = 'script_complete', script = $1 WHERE id = $2`,
          [phaseData.option_1, atomId]
        );
      }
      
      // Upsert option 1
      const content1 = phaseData.option_1 || '';
      const wc1 = content1.split(/\s+/).filter(w => w).length;
      await client.query(
        `INSERT INTO lesson_scripts (atom_id, phase, option_number, content, duration_seconds, word_count)
         VALUES ($1, $2, 1, $3, $4, $5)
         ON CONFLICT (atom_id, phase, option_number) DO UPDATE
         SET content = EXCLUDED.content, duration_seconds = EXCLUDED.duration_seconds, word_count = EXCLUDED.word_count`,
        [atomId, phase, content1, Math.round(wc1 / 2.5), wc1]
      );
      scriptsWritten++;
      
      // Upsert option 2
      const content2 = phaseData.option_2 || '';
      const wc2 = content2.split(/\s+/).filter(w => w).length;
      await client.query(
        `INSERT INTO lesson_scripts (atom_id, phase, option_number, content, duration_seconds, word_count)
         VALUES ($1, $2, 2, $3, $4, $5)
         ON CONFLICT (atom_id, phase, option_number) DO UPDATE
         SET content = EXCLUDED.content, duration_seconds = EXCLUDED.duration_seconds, word_count = EXCLUDED.word_count`,
        [atomId, phase, content2, Math.round(wc2 / 2.5), wc2]
      );
      scriptsWritten++;
    }
    
    return { success: true, scriptsWritten, dayNumber };
  } catch (e) {
    return { success: false, error: e.message, dayNumber };
  }
}

// Export for batch use
module.exports = { processLesson, generateScriptsForLesson, PHASE_NAMES };

// CLI mode
if (require.main === module) {
  (async () => {
    const client = new Client({ connectionString: process.env.DATABASE_URL });
    await client.connect();
    
    const dayArg = parseInt(process.argv[2]) || 1;
    console.log(`Generating scripts for Day ${dayArg}...`);
    
    const lesson = await client.query('SELECT id FROM core_lessons_v2 WHERE day_number = $1', [dayArg]);
    if (lesson.rows.length === 0) {
      console.error(`Day ${dayArg} not found`);
      process.exit(1);
    }
    
    const result = await processLesson(client, lesson.rows[0].id, dayArg);
    console.log(result.success ? `Done: ${result.scriptsWritten} scripts written` : `Failed: ${result.error}`);
    
    await client.end();
  })();
}
