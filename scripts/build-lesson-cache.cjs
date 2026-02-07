/**
 * Sprint F: Build Lesson Cache
 * Pre-compute lesson objects for all 365 days
 * Store as static JSON files in kelly-pipeline/cache/
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const PHASE_NAMES = { 1: 'hook', 2: 'teach', 3: 'example', 4: 'practice', 5: 'reflect', 6: 'apply', 7: 'close' };

const DEFAULT_SCRIPTS = {
  hook: "Welcome back to today's lesson! I have something really exciting to share with you.",
  teach: "Let me tell you about something fascinating that will change how you see the world.",
  example: "Here's a real-world example that brings this concept to life.",
  practice: "Now it's your turn! Try this thought experiment.",
  reflect: "Take a moment to think about what you've learned today.",
  apply: "Here's how you can use this knowledge in your daily life.",
  close: "Remember, every day is an opportunity to learn something new. See you tomorrow!",
};

async function buildCache() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  console.log('=== Building Lesson Cache ===\n');
  
  const cacheDir = path.join('C:\\Users\\user\\kelly-pipeline\\cache');
  fs.mkdirSync(cacheDir, { recursive: true });
  
  // Get all lessons
  const lessons = await client.query(`
    SELECT cl.*, la.phase, la.variant, la.audio_url, la.video_url,
           ls.option_number, ls.content, ls.word_count, ls.duration_seconds
    FROM core_lessons_v2 cl
    LEFT JOIN lesson_atoms la ON la.lesson_id = cl.id AND la.age_group = 'adult' AND la.language = 'en'
    LEFT JOIN lesson_scripts ls ON ls.atom_id = la.id
    ORDER BY cl.day_number, la.phase, ls.option_number
  `);
  
  console.log(`Loaded ${lessons.rows.length} records`);
  
  // Group by day
  const byDay = {};
  for (const row of lessons.rows) {
    if (!byDay[row.day_number]) {
      byDay[row.day_number] = {
        day_number: row.day_number,
        title: row.title,
        subject: row.subject,
        learning_objective: row.learning_objective,
        category: row.category,
        difficulty: row.difficulty,
        seed_data: row.seed_data,
        phases: {},
      };
    }
    
    if (row.phase) {
      const key = `${row.phase}`;
      if (!byDay[row.day_number].phases[key]) {
        byDay[row.day_number].phases[key] = {
          phase: row.phase,
          name: PHASE_NAMES[row.phase] || `phase_${row.phase}`,
          audio_url: row.audio_url,
          video_url: row.video_url,
          options: [],
        };
      }
      if (row.content) {
        byDay[row.day_number].phases[key].options.push({
          option: row.option_number || 1,
          script: row.content,
          word_count: row.word_count || 0,
          duration_seconds: row.duration_seconds || 0,
        });
      }
    }
  }
  
  let cached = 0;
  let withFullAudio = 0;
  let withVideo = 0;
  
  for (let day = 1; day <= 365; day++) {
    const lessonData = byDay[day];
    
    // Build complete lesson with 7 phases
    const lesson = {
      day_number: day,
      title: lessonData?.title || `Lesson ${day}`,
      subject: lessonData?.subject || 'general',
      learning_objective: lessonData?.learning_objective || '',
      category: lessonData?.category || 'general',
      difficulty: lessonData?.difficulty || 'beginner',
      phases: [],
    };
    
    let phaseAudio = 0;
    let phaseVideo = 0;
    
    for (let p = 1; p <= 7; p++) {
      const pName = PHASE_NAMES[p];
      const existing = lessonData?.phases?.[String(p)];
      
      const phase = {
        phase: p,
        name: pName,
        options: existing?.options?.length > 0 
          ? existing.options 
          : [{ option: 1, script: DEFAULT_SCRIPTS[pName], word_count: DEFAULT_SCRIPTS[pName].split(/\s+/).length, duration_seconds: 15, is_default: true }],
        audio_url: existing?.audio_url || null,
        video_url: existing?.video_url || null,
      };
      
      if (phase.audio_url) phaseAudio++;
      if (phase.video_url) phaseVideo++;
      
      lesson.phases.push(phase);
    }
    
    if (phaseAudio === 7) withFullAudio++;
    if (phaseVideo > 0) withVideo++;
    
    // Write to cache
    const cachePath = path.join(cacheDir, `day-${String(day).padStart(3, '0')}.json`);
    fs.writeFileSync(cachePath, JSON.stringify(lesson, null, 2));
    cached++;
  }
  
  // Write index
  const indexPath = path.join(cacheDir, 'index.json');
  const index = {
    total: 365,
    cached,
    with_full_audio: withFullAudio,
    with_video: withVideo,
    generated_at: new Date().toISOString(),
    files: Array.from({ length: 365 }, (_, i) => `day-${String(i + 1).padStart(3, '0')}.json`),
  };
  fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));
  
  console.log(`Cached ${cached} lessons`);
  console.log(`  ${withFullAudio} with full audio (7/7)`);
  console.log(`  ${withVideo} with video`);
  console.log(`\nSaved to: ${cacheDir}`);
  
  await client.end();
}

buildCache().catch(e => { console.error(e); process.exit(1); });
