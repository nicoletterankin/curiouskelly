/**
 * Phase 5: Accessibility — SRT Subtitles, Transcripts, Teacher Guides
 * 
 * 5A: Generate SRT subtitle files from alignment_json
 * 5B: Generate full lesson transcripts
 * 5C: Generate teacher guides for all 365 days
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

function log(msg) {
  console.log(`[${new Date().toISOString()}] ACCESS | ${msg}`);
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// ========== TASK 5A: SRT Subtitles ==========

function alignmentToSRT(alignment) {
  if (!alignment || !alignment.characters || !alignment.character_start_times_seconds) return null;

  const chars = alignment.characters;
  const starts = alignment.character_start_times_seconds;
  const ends = alignment.character_end_times_seconds;

  // Group characters into words
  const words = [];
  let currentWord = '';
  let wordStart = 0;
  let wordEnd = 0;

  for (let i = 0; i < chars.length; i++) {
    if (chars[i] === ' ' || chars[i] === '\n') {
      if (currentWord) {
        words.push({ text: currentWord, start: wordStart, end: wordEnd });
        currentWord = '';
      }
    } else {
      if (!currentWord) wordStart = starts[i] || wordEnd;
      currentWord += chars[i];
      wordEnd = ends[i] || starts[i] || wordEnd;
    }
  }
  if (currentWord) words.push({ text: currentWord, start: wordStart, end: wordEnd });

  // Group words into subtitle lines (max 42 chars, 2 lines per subtitle)
  const subtitles = [];
  let subText = '';
  let subStart = 0;
  let subEnd = 0;
  let lineCount = 0;
  let subIdx = 1;

  for (const word of words) {
    const newText = subText ? subText + ' ' + word.text : word.text;
    const currentLineLength = (subText.split('\n').pop() || '').length + word.text.length + 1;

    if (currentLineLength > 42 && lineCount < 1) {
      // Wrap to second line
      subText = subText + '\n' + word.text;
      subEnd = word.end;
      lineCount++;
    } else if (currentLineLength > 42 || lineCount >= 1 && newText.split('\n').pop().length > 42) {
      // Emit current subtitle
      if (subText) {
        subtitles.push({ idx: subIdx++, start: subStart, end: subEnd, text: subText });
      }
      subText = word.text;
      subStart = word.start;
      subEnd = word.end;
      lineCount = 0;
    } else {
      if (!subText) subStart = word.start;
      subText = newText;
      subEnd = word.end;
    }
  }
  if (subText) subtitles.push({ idx: subIdx++, start: subStart, end: subEnd, text: subText });

  // Format SRT
  return subtitles.map(s => {
    const formatTime = (t) => {
      const h = Math.floor(t / 3600);
      const m = Math.floor((t % 3600) / 60);
      const sec = Math.floor(t % 60);
      const ms = Math.floor((t % 1) * 1000);
      return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
    };
    return `${s.idx}\n${formatTime(s.start)} --> ${formatTime(s.end)}\n${s.text}\n`;
  }).join('\n');
}

async function generateSRTSubtitles(client) {
  log('5A: SRT Subtitle Generation');

  const audioRows = await client.query(`
    SELECT id, day_number, phase, alignment_json, language
    FROM kellyos_audio
    WHERE alignment_json IS NOT NULL
      AND (srt_text IS NULL OR srt_text = '')
    ORDER BY day_number, phase
  `);
  log(`5A: ${audioRows.rows.length} audio files need SRT generation`);

  let done = 0;
  let failed = 0;
  const srtDir = path.join(__dirname, '..', 'kelly-pipeline', 'subtitles');

  for (const row of audioRows.rows) {
    try {
      const alignment = typeof row.alignment_json === 'string'
        ? JSON.parse(row.alignment_json) : row.alignment_json;
      
      const srt = alignmentToSRT(alignment);
      if (!srt) { failed++; continue; }

      // Save to database
      await client.query(
        'UPDATE kellyos_audio SET srt_text = $1 WHERE id = $2',
        [srt, row.id]
      );

      // Save to file
      const lang = row.language || 'en';
      const langDir = path.join(srtDir, lang);
      ensureDir(langDir);
      fs.writeFileSync(
        path.join(langDir, `day-${String(row.day_number).padStart(3, '0')}-${row.phase}.srt`),
        srt
      );
      done++;
    } catch (e) {
      failed++;
    }

    if (done % 100 === 0 && done > 0) log(`5A: ${done}/${audioRows.rows.length}`);
  }

  log(`5A DONE: ${done} SRT files generated (${failed} failed)`);
  return done;
}

// ========== TASK 5B: Transcripts ==========

async function generateTranscripts(client) {
  log('5B: Transcript Generation');

  const transcriptDir = path.join(__dirname, '..', 'kelly-pipeline', 'transcripts', 'en');
  ensureDir(transcriptDir);

  // Get all days
  const days = await client.query(`
    SELECT DISTINCT day_number FROM kellyos_lessons
    WHERE (language = 'en' OR language IS NULL)
    ORDER BY day_number
  `);
  log(`5B: ${days.rows.length} days to generate transcripts for`);

  let done = 0;
  const phases = ['hook', 'story', 'wonder', 'action', 'wisdom'];

  for (const dayRow of days.rows) {
    const day = dayRow.day_number;
    
    // Get title
    let title = `Day ${day}`;
    try {
      const titleRes = await client.query(
        'SELECT title FROM core_lessons_v2 WHERE day_number = $1', [day]
      );
      if (titleRes.rows.length > 0) title = titleRes.rows[0].title || title;
    } catch {}

    // Get all phases
    const phaseTexts = {};
    for (const phase of phases) {
      try {
        const phaseRes = await client.query(`
          SELECT content_text FROM kellyos_lessons
          WHERE day_number = $1 AND phase = $2 AND (language = 'en' OR language IS NULL)
          LIMIT 1
        `, [day, phase]);
        phaseTexts[phase] = phaseRes.rows[0]?.content_text || '';
      } catch {
        phaseTexts[phase] = '';
      }
    }

    // Get Kelly quote
    let kellyQuote = '';
    try {
      const quoteRes = await client.query(
        "SELECT kelly_quote FROM core_lessons_v2 WHERE day_number = $1", [day]
      );
      kellyQuote = quoteRes.rows[0]?.kelly_quote || '';
    } catch {}

    // Generate transcript
    const transcript = `LESSON ${day}: ${title}
Date: Day ${day} of 365

=== HOOK ===
${phaseTexts.hook || '[No hook content]'}

=== STORY ===
${phaseTexts.story || '[No story content]'}

=== WONDER ===
${phaseTexts.wonder || '[No wonder content]'}

=== ACTION ===
${phaseTexts.action || '[No action content]'}

=== WISDOM ===
${phaseTexts.wisdom || '[No wisdom content]'}

--- Kelly's Quote ---
"${kellyQuote || 'Every day is a chance to learn something new.'}"
`;

    fs.writeFileSync(
      path.join(transcriptDir, `day-${String(day).padStart(3, '0')}-transcript.txt`),
      transcript
    );
    done++;
  }

  log(`5B DONE: ${done} transcripts generated`);
  return done;
}

// ========== TASK 5C: Teacher Guides ==========

async function generateTeacherGuides(client) {
  log('5C: Teacher Guide Generation');

  const existing = await client.query('SELECT COUNT(*) as cnt FROM kellyos_teacher_guides');
  const existingCount = parseInt(existing.rows[0].cnt);

  if (existingCount >= 360) {
    log(`5C: SKIP — ${existingCount} guides already exist`);
    return existingCount;
  }

  const needed = await client.query(`
    SELECT c.day_number, c.title, c.subject, c.category, c.learning_objective
    FROM core_lessons_v2 c
    WHERE NOT EXISTS (SELECT 1 FROM kellyos_teacher_guides g WHERE g.day_number = c.day_number)
    ORDER BY c.day_number
  `);
  log(`5C: ${needed.rows.length} teacher guides needed`);

  let done = 0;
  const batchSize = 5;

  for (let i = 0; i < needed.rows.length; i += batchSize) {
    const batch = needed.rows.slice(i, i + batchSize);

    for (const row of batch) {
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
                content: `Generate a teacher's guide for this lesson. Return JSON: {
                  "grade_range": "K-2 / 3-5 / 6-8 / 9-12",
                  "standards": ["Standard 1", "Standard 2"],
                  "prep_notes": "preparation notes",
                  "discussion_questions": ["Q1", "Q2", "Q3", "Q4", "Q5"],
                  "extension_activities": ["Activity 1", "Activity 2", "Activity 3"],
                  "assessment_rubric": {"excellent": "...", "proficient": "...", "developing": "..."},
                  "materials": "materials needed",
                  "time_15min": "15-minute version plan",
                  "time_30min": "30-minute version plan",
                  "time_45min": "45-minute version plan"
                }`
              },
              {
                role: 'user',
                content: `Day ${row.day_number}: "${row.title}" (${row.subject || row.category || 'general'}). Objective: ${row.learning_objective || 'learn about ' + row.title}`
              }
            ],
            temperature: 0.3,
            max_tokens: 2000,
            response_format: { type: 'json_object' }
          })
        });

        if (!res.ok) throw new Error(`OpenAI ${res.status}`);
        const data = await res.json();
        const guide = JSON.parse(data.choices[0].message.content);

        await client.query(`
          INSERT INTO kellyos_teacher_guides (
            day_number, grade_range, standards_alignment, prep_notes,
            discussion_questions, extension_activities, assessment_rubric,
            materials, time_15min, time_30min, time_45min
          ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
          ON CONFLICT (day_number) DO UPDATE SET
            grade_range = EXCLUDED.grade_range,
            standards_alignment = EXCLUDED.standards_alignment,
            prep_notes = EXCLUDED.prep_notes,
            discussion_questions = EXCLUDED.discussion_questions,
            extension_activities = EXCLUDED.extension_activities,
            assessment_rubric = EXCLUDED.assessment_rubric,
            materials = EXCLUDED.materials,
            time_15min = EXCLUDED.time_15min,
            time_30min = EXCLUDED.time_30min,
            time_45min = EXCLUDED.time_45min
        `, [
          row.day_number,
          guide.grade_range,
          JSON.stringify(guide.standards || guide.standards_alignment || []),
          guide.prep_notes,
          JSON.stringify(guide.discussion_questions || []),
          JSON.stringify(guide.extension_activities || []),
          JSON.stringify(guide.assessment_rubric || {}),
          guide.materials,
          guide.time_15min,
          guide.time_30min,
          guide.time_45min
        ]);
        done++;
      } catch (e) {
        log(`5C: Error day ${row.day_number}: ${e.message}`);
      }

      await new Promise(r => setTimeout(r, 300));
    }

    if (done % 25 === 0 && done > 0) log(`5C: ${done}/${needed.rows.length}`);
  }

  log(`5C DONE: ${done} teacher guides generated`);
  return done;
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  log('Connected. Starting accessibility pipeline...');

  const results = {};
  results.srt = await generateSRTSubtitles(client);
  results.transcripts = await generateTranscripts(client);
  results.teacherGuides = await generateTeacherGuides(client);

  log('=== ACCESSIBILITY RESULTS ===');
  log(`SRT files: ${results.srt}`);
  log(`Transcripts: ${results.transcripts}`);
  log(`Teacher guides: ${results.teacherGuides}`);

  const auditDir = path.join(__dirname, '..', 'kelly-pipeline', 'audit');
  ensureDir(auditDir);
  fs.writeFileSync(
    path.join(auditDir, 'accessibility-results.json'),
    JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2)
  );

  await client.end();
}

main().catch(e => { console.error('[ACCESS ERROR]', e); process.exit(1); });
