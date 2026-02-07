/**
 * Phase 4: Content Enrichment
 * Generates rich metadata for every lesson:
 * - Learning objectives (3 per day)
 * - Difficulty ratings (vocabulary, complexity, prior knowledge)
 * - Topic tags (5-8 per day)
 * - Kelly quotes (3 per day: hook, wonder, wisdom)
 * - Lesson summaries (short, teaser, SEO)
 * - "Is This True?" facts (5 per day)
 * - Lesson graph connections (prerequisites, followups, related)
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

function log(msg) {
  console.log(`[${new Date().toISOString()}] ENRICH | ${msg}`);
}

async function callOpenAI(systemPrompt, userContent, retries = 3) {
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
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userContent }
          ],
          temperature: 0.3,
          max_tokens: 4000,
          response_format: { type: 'json_object' }
        })
      });

      if (res.status === 429) {
        await new Promise(r => setTimeout(r, (attempt + 1) * 10000));
        continue;
      }
      if (!res.ok) throw new Error(`OpenAI ${res.status}`);

      const data = await res.json();
      return JSON.parse(data.choices[0].message.content);
    } catch (e) {
      if (attempt === retries - 1) throw e;
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}

// Task 4A: Learning Objectives
async function generateLearningObjectives(client) {
  log('4A: Learning Objectives');
  
  const needed = await client.query(`
    SELECT day_number, title, subject, learning_objective
    FROM core_lessons_v2
    WHERE learning_objectives IS NULL
    ORDER BY day_number
  `);
  log(`4A: ${needed.rows.length} days need objectives`);

  let done = 0;
  const batchSize = 5;

  for (let i = 0; i < needed.rows.length; i += batchSize) {
    const batch = needed.rows.slice(i, i + batchSize);
    
    for (const row of batch) {
      try {
        const result = await callOpenAI(
          `Generate 3 measurable learning objectives using Bloom's taxonomy for this lesson. Return JSON: {"objectives": [{"level": "remember|understand|apply", "verb": "identify|explain|apply|etc", "text": "After this lesson, learners will be able to..."}]}`,
          `Day ${row.day_number}: "${row.title}" (${row.subject || 'general'}). Current objective: ${row.learning_objective || 'none'}`
        );

        await client.query(
          'UPDATE core_lessons_v2 SET learning_objectives = $1 WHERE day_number = $2',
          [JSON.stringify(result.objectives), row.day_number]
        );
        done++;
      } catch (e) {
        log(`4A: Error day ${row.day_number}: ${e.message}`);
      }
    }

    if (done % 25 === 0 && done > 0) log(`4A: ${done}/${needed.rows.length}`);
    await new Promise(r => setTimeout(r, 300));
  }
  log(`4A DONE: ${done} objectives generated`);
  return done;
}

// Task 4B: Difficulty Ratings
async function generateDifficultyRatings(client) {
  log('4C: Difficulty Ratings');

  const needed = await client.query(`
    SELECT day_number, title, subject, learning_objective
    FROM core_lessons_v2
    WHERE difficulty_data IS NULL
    ORDER BY day_number
  `);
  log(`4C: ${needed.rows.length} days need ratings`);

  let done = 0;
  const batchSize = 10;

  for (let i = 0; i < needed.rows.length; i += batchSize) {
    const batch = needed.rows.slice(i, i + batchSize);
    const batchInput = batch.map(r => ({
      day: r.day_number,
      title: r.title,
      subject: r.subject
    }));

    try {
      const result = await callOpenAI(
        `Rate the difficulty of these lessons on 1-10 scales. Return JSON: {"ratings": [{"day": N, "vocabulary": 1-10, "complexity": 1-10, "prior_knowledge": 1-10, "overall": 1-10, "recommended_min_age": N}]}`,
        JSON.stringify(batchInput)
      );

      for (const rating of (result.ratings || [])) {
        await client.query(
          'UPDATE core_lessons_v2 SET difficulty_data = $1 WHERE day_number = $2',
          [JSON.stringify(rating), rating.day]
        );
        done++;
      }
    } catch (e) {
      log(`4C: Error batch ${i}: ${e.message}`);
    }

    if (done % 50 === 0 && done > 0) log(`4C: ${done}/${needed.rows.length}`);
    await new Promise(r => setTimeout(r, 200));
  }
  log(`4C DONE: ${done} ratings generated`);
  return done;
}

// Task 4D: Topic Tags
async function generateTopicTags(client) {
  log('4D: Topic Tags');

  const existingTags = await client.query('SELECT COUNT(DISTINCT day_number) as cnt FROM kellyos_tags');
  const existingCount = parseInt(existingTags.rows[0].cnt);

  if (existingCount >= 360) {
    log(`4D: SKIP — ${existingCount} days already tagged`);
    return existingCount;
  }

  const needed = await client.query(`
    SELECT c.day_number, c.title, c.subject, c.category
    FROM core_lessons_v2 c
    WHERE NOT EXISTS (SELECT 1 FROM kellyos_tags t WHERE t.day_number = c.day_number)
    ORDER BY c.day_number
  `);
  log(`4D: ${needed.rows.length} days need tags`);

  let done = 0;
  const batchSize = 10;
  const categories = 'Science, History, Philosophy, Arts, Technology, Nature, Society, Mathematics, Health, Geography, Psychology, Economics, Literature, Music, Sports';

  for (let i = 0; i < needed.rows.length; i += batchSize) {
    const batch = needed.rows.slice(i, i + batchSize);
    const batchInput = batch.map(r => ({
      day: r.day_number,
      title: r.title,
      subject: r.subject,
      category: r.category
    }));

    try {
      const result = await callOpenAI(
        `Generate 5-8 topic tags for each lesson. Assign primary and secondary categories from: ${categories}. Return JSON: {"lessons": [{"day": N, "tags": ["tag1", ...], "primary_category": "...", "secondary_category": "..."}]}`,
        JSON.stringify(batchInput)
      );

      for (const lesson of (result.lessons || [])) {
        for (let t = 0; t < (lesson.tags || []).length; t++) {
          await client.query(`
            INSERT INTO kellyos_tags (day_number, tag, category, is_primary)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
          `, [lesson.day, lesson.tags[t], t === 0 ? lesson.primary_category : lesson.secondary_category, t === 0]);
        }
        done++;
      }
    } catch (e) {
      log(`4D: Error batch ${i}: ${e.message}`);
    }

    if (done % 50 === 0 && done > 0) log(`4D: ${done}/${needed.rows.length}`);
    await new Promise(r => setTimeout(r, 200));
  }
  log(`4D DONE: ${done} days tagged`);
  return done;
}

// Task 4E: Kelly Quotes
async function generateKellyQuotes(client) {
  log('4E: Kelly Quotes');

  const existingQuotes = await client.query('SELECT COUNT(DISTINCT day_number) as cnt FROM kellyos_quotes');
  const existingCount = parseInt(existingQuotes.rows[0].cnt);

  if (existingCount >= 360) {
    log(`4E: SKIP — ${existingCount} days already have quotes`);
    return existingCount;
  }

  const needed = await client.query(`
    SELECT c.day_number, c.title, c.subject
    FROM core_lessons_v2 c
    WHERE NOT EXISTS (SELECT 1 FROM kellyos_quotes q WHERE q.day_number = c.day_number)
    ORDER BY c.day_number
  `);
  log(`4E: ${needed.rows.length} days need quotes`);

  let done = 0;
  const batchSize = 10;

  for (let i = 0; i < needed.rows.length; i += batchSize) {
    const batch = needed.rows.slice(i, i + batchSize);
    const batchInput = batch.map(r => ({
      day: r.day_number,
      title: r.title,
      subject: r.subject
    }));

    try {
      const result = await callOpenAI(
        `Generate 3 memorable Kelly quotes for each lesson. Each quote 10-25 words, quotable, shareable, no clichés. Types: hook (curiosity-sparking), wonder (mind-expanding), wisdom (reflection/takeaway). Return JSON: {"quotes": [{"day": N, "hook": "...", "wonder": "...", "wisdom": "..."}]}`,
        JSON.stringify(batchInput)
      );

      for (const q of (result.quotes || [])) {
        for (const type of ['hook', 'wonder', 'wisdom']) {
          if (q[type]) {
            await client.query(`
              INSERT INTO kellyos_quotes (day_number, quote_type, quote_text)
              VALUES ($1, $2, $3)
              ON CONFLICT DO NOTHING
            `, [q.day, type, q[type]]);
          }
        }
        done++;
      }
    } catch (e) {
      log(`4E: Error batch ${i}: ${e.message}`);
    }

    if (done % 50 === 0 && done > 0) log(`4E: ${done}/${needed.rows.length}`);
    await new Promise(r => setTimeout(r, 200));
  }
  log(`4E DONE: ${done} days with quotes`);
  return done;
}

// Task 4F: Lesson Summaries
async function generateSummaries(client) {
  log('4F: Lesson Summaries');

  const needed = await client.query(`
    SELECT day_number, title, subject, learning_objective
    FROM core_lessons_v2
    WHERE summary_short IS NULL
    ORDER BY day_number
  `);
  log(`4F: ${needed.rows.length} days need summaries`);

  let done = 0;
  const batchSize = 10;

  for (let i = 0; i < needed.rows.length; i += batchSize) {
    const batch = needed.rows.slice(i, i + batchSize);
    const batchInput = batch.map(r => ({
      day: r.day_number,
      title: r.title,
      subject: r.subject,
      objective: r.learning_objective
    }));

    try {
      const result = await callOpenAI(
        `Generate summaries for each lesson. Return JSON: {"summaries": [{"day": N, "short": "max 100 chars one-liner", "teaser": "Two-sentence engaging teaser", "meta": "SEO description max 160 chars"}]}`,
        JSON.stringify(batchInput)
      );

      for (const s of (result.summaries || [])) {
        await client.query(`
          UPDATE core_lessons_v2 SET
            summary_short = $1,
            summary_teaser = $2,
            meta_description = $3
          WHERE day_number = $4
        `, [s.short, s.teaser, s.meta, s.day]);
        done++;
      }
    } catch (e) {
      log(`4F: Error batch ${i}: ${e.message}`);
    }

    if (done % 50 === 0 && done > 0) log(`4F: ${done}/${needed.rows.length}`);
    await new Promise(r => setTimeout(r, 200));
  }
  log(`4F DONE: ${done} summaries generated`);
  return done;
}

// Task 4G: Is This True? Facts
async function generateFacts(client) {
  log('4G: Is This True? Facts');

  const existingFacts = await client.query('SELECT COUNT(DISTINCT day_number) as cnt FROM kellyos_facts_v2');
  const existingCount = parseInt(existingFacts.rows[0].cnt);

  if (existingCount >= 360) {
    log(`4G: SKIP — ${existingCount} days already have facts`);
    return existingCount;
  }

  const needed = await client.query(`
    SELECT c.day_number, c.title, c.subject
    FROM core_lessons_v2 c
    WHERE NOT EXISTS (SELECT 1 FROM kellyos_facts_v2 f WHERE f.day_number = c.day_number)
    ORDER BY c.day_number
  `);
  log(`4G: ${needed.rows.length} days need facts`);

  let done = 0;
  const batchSize = 5;

  for (let i = 0; i < needed.rows.length; i += batchSize) {
    const batch = needed.rows.slice(i, i + batchSize);

    for (const row of batch) {
      try {
        const result = await callOpenAI(
          `Generate 5 true/false statements about this lesson topic for a quiz. Include: 2 TRUE statements, 2 FALSE but plausible statements, and 1 TRICKY (partially true/nuanced) statement. Each must have an explanation. Return JSON: {"facts": [{"statement": "...", "is_true": bool, "is_tricky": bool, "explanation": "...", "difficulty": 1-10}]}`,
          `Day ${row.day_number}: "${row.title}" (${row.subject || 'general'})`
        );

        for (const fact of (result.facts || [])) {
          await client.query(`
            INSERT INTO kellyos_facts_v2 (day_number, statement, is_true, is_tricky, explanation, difficulty)
            VALUES ($1, $2, $3, $4, $5, $6)
          `, [row.day_number, fact.statement, fact.is_true, fact.is_tricky || false, fact.explanation, fact.difficulty || 5]);
        }
        done++;
      } catch (e) {
        log(`4G: Error day ${row.day_number}: ${e.message}`);
      }
    }

    if (done % 25 === 0 && done > 0) log(`4G: ${done}/${needed.rows.length}`);
    await new Promise(r => setTimeout(r, 300));
  }
  log(`4G DONE: ${done} days with facts`);
  return done;
}

// Task 4B: Prerequisites and connections
async function generateLessonGraph(client) {
  log('4B: Lesson Graph');

  const existingEdges = await client.query('SELECT COUNT(*) as cnt FROM kellyos_lesson_graph');
  const existingCount = parseInt(existingEdges.rows[0].cnt);

  if (existingCount >= 2000) {
    log(`4B: SKIP — ${existingCount} edges already exist`);
    return existingCount;
  }

  // Get all lessons
  const lessons = await client.query(
    'SELECT day_number, title, subject, category FROM core_lessons_v2 ORDER BY day_number'
  );
  log(`4B: Building graph for ${lessons.rows.length} lessons`);

  const batchSize = 20;
  let totalEdges = 0;

  for (let i = 0; i < lessons.rows.length; i += batchSize) {
    const batch = lessons.rows.slice(i, i + batchSize);
    const allLessons = lessons.rows.map(r => `Day ${r.day_number}: ${r.title} (${r.subject || r.category})`).join('\n');
    const batchLessons = batch.map(r => r.day_number);

    try {
      const result = await callOpenAI(
        `Given this curriculum of 365 daily lessons, identify connections for the specified days. For each day, find 2-3 prerequisite days (should come before), 2-3 follow-up days (build on this), and 2-3 related days (same theme, different angle). Return JSON: {"connections": [{"day": N, "prerequisites": [N,...], "followups": [N,...], "related": [N,...]}]}`,
        `ALL LESSONS:\n${allLessons}\n\nFIND CONNECTIONS FOR DAYS: ${batchLessons.join(', ')}`
      );

      for (const conn of (result.connections || [])) {
        for (const pre of (conn.prerequisites || [])) {
          if (pre >= 1 && pre <= 365 && pre !== conn.day) {
            await client.query(
              'INSERT INTO kellyos_lesson_graph (from_day, to_day, relationship, strength) VALUES ($1, $2, $3, 0.7) ON CONFLICT DO NOTHING',
              [pre, conn.day, 'prerequisite']
            );
            totalEdges++;
          }
        }
        for (const fup of (conn.followups || [])) {
          if (fup >= 1 && fup <= 365 && fup !== conn.day) {
            await client.query(
              'INSERT INTO kellyos_lesson_graph (from_day, to_day, relationship, strength) VALUES ($1, $2, $3, 0.7) ON CONFLICT DO NOTHING',
              [conn.day, fup, 'followup']
            );
            totalEdges++;
          }
        }
        for (const rel of (conn.related || [])) {
          if (rel >= 1 && rel <= 365 && rel !== conn.day) {
            await client.query(
              'INSERT INTO kellyos_lesson_graph (from_day, to_day, relationship, strength) VALUES ($1, $2, $3, 0.5) ON CONFLICT DO NOTHING',
              [conn.day, rel, 'related']
            );
            totalEdges++;
          }
        }
      }
    } catch (e) {
      log(`4B: Error batch ${i}: ${e.message}`);
    }

    if (i % 100 === 0) log(`4B: Processed ${i}/${lessons.rows.length} days, ${totalEdges} edges`);
    await new Promise(r => setTimeout(r, 500));
  }
  log(`4B DONE: ${totalEdges} graph edges created`);
  return totalEdges;
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  log('Connected. Starting content enrichment...');

  const results = {};

  // Run sequentially to avoid rate limits
  results.objectives = await generateLearningObjectives(client);
  results.difficulty = await generateDifficultyRatings(client);
  results.tags = await generateTopicTags(client);
  results.quotes = await generateKellyQuotes(client);
  results.summaries = await generateSummaries(client);
  results.facts = await generateFacts(client);
  results.graph = await generateLessonGraph(client);

  log('=== ENRICHMENT RESULTS ===');
  for (const [key, value] of Object.entries(results)) {
    log(`${key}: ${value}`);
  }

  const auditDir = path.join(__dirname, '..', 'kelly-pipeline', 'audit');
  if (!fs.existsSync(auditDir)) fs.mkdirSync(auditDir, { recursive: true });
  fs.writeFileSync(
    path.join(auditDir, 'enrichment-results.json'),
    JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2)
  );

  await client.end();
}

main().catch(e => { console.error('[ENRICH ERROR]', e); process.exit(1); });
