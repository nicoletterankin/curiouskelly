/**
 * Sprint A: Seed core_lessons_v2 from multiple data sources
 * 1. lessons/365_day_calendar.json (metadata)
 * 2. Existing `lessons` table in Neon (scripts)
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

async function seed() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  console.log('Connected to database');
  
  // Load 365-day calendar JSON
  const calendarPath = path.join(__dirname, '..', 'lessons', '365_day_calendar.json');
  const calendar = JSON.parse(fs.readFileSync(calendarPath, 'utf-8'));
  console.log(`Loaded ${calendar.lessons.length} lessons from calendar`);
  
  // Load existing lessons from Neon (has title, topic, scripts)
  let existingLessons = {};
  try {
    const res = await client.query(`
      SELECT day_of_year, title, topic, theme, 
             hook_script, story_script, wonder_script, action_script, wisdom_script,
             track, subtitle, summary, quote, quote_author
      FROM lessons ORDER BY day_of_year
    `);
    for (const r of res.rows) {
      existingLessons[r.day_of_year] = r;
    }
    console.log(`Loaded ${Object.keys(existingLessons).length} existing lessons from DB`);
  } catch (e) {
    console.log('No existing lessons table or error:', e.message);
  }
  
  // Clear and seed core_lessons_v2
  await client.query('DELETE FROM lesson_scripts');
  await client.query('DELETE FROM lesson_atoms');
  await client.query('DELETE FROM core_lessons_v2');
  console.log('Cleared existing data');
  
  let inserted = 0;
  let atomsInserted = 0;
  
  for (const lesson of calendar.lessons) {
    const existing = existingLessons[lesson.day] || {};
    
    const seedData = {
      calendar: {
        date: lesson.date,
        lesson_id: lesson.lesson_id,
        icon: lesson.icon,
        difficulty: lesson.difficulty,
        duration: lesson.duration,
        learning_objectives: lesson.learning_objectives,
        marketing_headline: lesson.marketing_headline,
        marketing_tagline: lesson.marketing_tagline,
        tags: lesson.tags,
      },
      existing: {
        topic: existing.topic || null,
        theme: existing.theme || null,
        track: existing.track || null,
        subtitle: existing.subtitle || null,
        summary: existing.summary || null,
        quote: existing.quote || null,
        quote_author: existing.quote_author || null,
      },
      scripts: {
        hook: existing.hook_script || null,
        story: existing.story_script || null,
        wonder: existing.wonder_script || null,
        action: existing.action_script || null,
        wisdom: existing.wisdom_script || null,
      }
    };
    
    // Insert core lesson
    const coreRes = await client.query(
      `INSERT INTO core_lessons_v2 (day_number, title, subject, learning_objective, category, difficulty, seed_data)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING id`,
      [
        lesson.day,
        existing.title || lesson.title,
        existing.topic || lesson.category || 'general',
        lesson.learning_objective || '',
        lesson.category || 'general',
        lesson.difficulty || 'beginner',
        JSON.stringify(seedData)
      ]
    );
    inserted++;
    const lessonId = coreRes.rows[0].id;
    
    // Create lesson_atoms for the 7-phase structure (adult, English)
    // Map existing 5-phase scripts to 7-phase structure
    const phaseMapping = {
      1: { name: 'hook', script: existing.hook_script },
      2: { name: 'teach', script: existing.story_script },
      3: { name: 'example', script: existing.wonder_script },
      4: { name: 'practice', script: null },
      5: { name: 'reflect', script: existing.action_script },
      6: { name: 'apply', script: null },
      7: { name: 'close', script: existing.wisdom_script },
    };
    
    for (let phase = 1; phase <= 7; phase++) {
      const pm = phaseMapping[phase];
      const status = pm.script ? 'script_complete' : 'pending';
      
      const atomRes = await client.query(
        `INSERT INTO lesson_atoms (lesson_id, phase, variant, age_group, language, script, status)
         VALUES ($1, $2, $3, 'adult', 'en', $4, $5)
         RETURNING id`,
        [lessonId, phase, pm.name, pm.script || null, status]
      );
      atomsInserted++;
      
      // If we have a script, create the lesson_scripts entry
      if (pm.script) {
        const wordCount = pm.script.split(/\s+/).length;
        const duration = Math.round(wordCount / 2.5); // ~150 wpm
        
        await client.query(
          `INSERT INTO lesson_scripts (atom_id, phase, option_number, content, duration_seconds, word_count)
           VALUES ($1, $2, 1, $3, $4, $5)`,
          [atomRes.rows[0].id, phase, pm.script, duration, wordCount]
        );
      }
    }
    
    if (inserted % 50 === 0) {
      console.log(`Seeded ${inserted}/365 lessons...`);
    }
  }
  
  console.log(`\nSeeding complete:`);
  console.log(`  core_lessons_v2: ${inserted} rows`);
  console.log(`  lesson_atoms: ${atomsInserted} rows`);
  
  // Verify
  const counts = await client.query(`
    SELECT 
      (SELECT COUNT(*) FROM core_lessons_v2) as core_count,
      (SELECT COUNT(*) FROM lesson_atoms) as atom_count,
      (SELECT COUNT(*) FROM lesson_scripts) as script_count,
      (SELECT COUNT(*) FROM lesson_atoms WHERE status = 'script_complete') as with_scripts
  `);
  const c = counts.rows[0];
  console.log(`\nVerification:`);
  console.log(`  core_lessons_v2: ${c.core_count}`);
  console.log(`  lesson_atoms: ${c.atom_count}`);
  console.log(`  lesson_scripts: ${c.script_count}`);
  console.log(`  atoms with scripts: ${c.with_scripts}`);
  
  await client.end();
}

seed().catch(e => { console.error(e); process.exit(1); });
