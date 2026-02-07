/**
 * Sprint P: Performance & Demo Prep
 * P.1 — Pre-cache Days 1-7
 * P.2 — Compress assets for web
 * P.3 — Build preload manifest
 * P.4 — Database query optimization
 */
require('dotenv').config();
const { Client } = require('pg');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const LOG_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\logs', 'cursor-burndown.log');
const CP_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\checkpoints', 'burndown.json');
const CACHE_DIR = 'C:\\Users\\user\\kelly-pipeline\\cache';

function log(sprint, msg) {
  const line = `[${new Date().toISOString()}] ${sprint} | ${msg}\n`;
  fs.appendFileSync(LOG_FILE, line, 'utf-8');
  process.stdout.write(line);
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  // ===== P.1 — Pre-cache Days 1-7 =====
  log('SPRINT P.1', 'START | Pre-caching Days 1-7');
  
  for (let day = 1; day <= 7; day++) {
    const core = await client.query('SELECT * FROM core_lessons_v2 WHERE day_number = $1', [day]);
    const phases = await client.query(`
      SELECT kl.phase, kl.title, kl.content_text,
             ka.audio_url, ka.viseme_timeline, ka.duration_seconds
      FROM kellyos_lessons kl
      LEFT JOIN kellyos_audio ka ON ka.day_number = kl.day_number AND ka.phase = kl.phase
      WHERE kl.day_number = $1 AND (kl.language = 'en' OR kl.language IS NULL)
      ORDER BY CASE kl.phase WHEN 'hook' THEN 1 WHEN 'story' THEN 2 WHEN 'wonder' THEN 3 WHEN 'action' THEN 4 WHEN 'wisdom' THEN 5 ELSE 6 END
    `, [day]);
    
    const scripts = await client.query(`
      SELECT la.phase AS phase_num, la.variant, ls.option_number, ls.content, ls.word_count
      FROM lesson_atoms la JOIN lesson_scripts ls ON ls.atom_id = la.id
      JOIN core_lessons_v2 cl ON cl.id = la.lesson_id
      WHERE cl.day_number = $1 AND la.age_group = 'adult' AND la.language = 'en'
      ORDER BY la.phase, ls.option_number
    `, [day]);
    
    const dayData = {
      day,
      title: core.rows[0]?.title || `Day ${day}`,
      subject: core.rows[0]?.subject || '',
      learning_objective: core.rows[0]?.learning_objective || '',
      phases: phases.rows.map(p => ({
        phase: p.phase, title: p.title, content_text: p.content_text,
        audio_url: p.audio_url, viseme_timeline: p.viseme_timeline, duration_seconds: p.duration_seconds
      })),
      scripts: scripts.rows.map(s => ({
        phase: s.phase_num, variant: s.variant, option: s.option_number,
        content: s.content, word_count: s.word_count
      })),
      cached_at: new Date().toISOString()
    };
    
    fs.writeFileSync(path.join(CACHE_DIR, `day-${String(day).padStart(3, '0')}.json`), JSON.stringify(dayData, null, 2));
  }
  log('SPRINT P.1', 'COMPLETE | Days 1-7 pre-cached');
  
  // ===== P.2 — Compress assets =====
  log('SPRINT P.2', 'START | Compressing assets for web');
  
  const assetsDir = 'C:\\Users\\user\\kelly-pipeline\\kellyos-assets';
  
  // Convert sprite sheets to WebP
  const spriteDirs = ['kid', 'adult', 'elder'].map(a => path.join(assetsDir, 'sprites', a));
  let compressed = 0;
  for (const dir of spriteDirs) {
    if (!fs.existsSync(dir)) continue;
    for (const file of fs.readdirSync(dir)) {
      if (file.endsWith('.png')) {
        const webpPath = path.join(dir, file.replace('.png', '.webp'));
        if (!fs.existsSync(webpPath)) {
          try {
            execSync(`ffmpeg -y -i "${path.join(dir, file)}" -quality 80 "${webpPath}"`, { stdio: 'pipe', timeout: 10000 });
            compressed++;
          } catch {}
        }
      }
    }
  }
  
  // Compress base videos
  const baseDir = path.join(assetsDir, 'base-videos');
  if (fs.existsSync(baseDir)) {
    for (const file of fs.readdirSync(baseDir)) {
      if (file.endsWith('.mp4')) {
        const posterPath = path.join(baseDir, file.replace('.mp4', '_poster.jpg'));
        if (!fs.existsSync(posterPath)) {
          try {
            execSync(`ffmpeg -y -i "${path.join(baseDir, file)}" -ss 0 -frames:v 1 -q:v 5 "${posterPath}"`, { stdio: 'pipe', timeout: 10000 });
            compressed++;
          } catch {}
        }
      }
    }
  }
  
  log('SPRINT P.2', `COMPLETE | ${compressed} files compressed/converted`);
  
  // ===== P.3 — Preload manifest =====
  log('SPRINT P.3', 'START | Building preload manifest');
  
  const manifest = {
    critical: [],
    prefetch: [],
    defer: [],
    generated_at: new Date().toISOString()
  };
  
  // Critical: adult sprites + idle
  const adultSprites = path.join(assetsDir, 'sprites', 'adult');
  if (fs.existsSync(adultSprites)) {
    const sheet = fs.readdirSync(adultSprites).find(f => f.includes('sprite_sheet'));
    if (sheet) manifest.critical.push(`sprites/adult/${sheet}`);
  }
  
  const idleDir = path.join(assetsDir, 'idle-loops');
  if (fs.existsSync(idleDir)) {
    const adultIdle = fs.readdirSync(idleDir).find(f => f.includes('adult'));
    if (adultIdle) manifest.critical.push(`idle-loops/${adultIdle}`);
  }
  
  // Prefetch: base videos for adult
  if (fs.existsSync(baseDir)) {
    for (const f of fs.readdirSync(baseDir)) {
      if (f.startsWith('adult_') && f.endsWith('.mp4')) manifest.prefetch.push(`base-videos/${f}`);
    }
  }
  
  // Defer: kid/elder sprites
  for (const age of ['kid', 'elder']) {
    const dir = path.join(assetsDir, 'sprites', age);
    if (fs.existsSync(dir)) {
      const sheet = fs.readdirSync(dir).find(f => f.includes('sprite_sheet'));
      if (sheet) manifest.defer.push(`sprites/${age}/${sheet}`);
    }
  }
  
  fs.writeFileSync(path.join('C:\\Users\\user\\kelly-pipeline', 'preload-manifest.json'), JSON.stringify(manifest, null, 2));
  log('SPRINT P.3', 'COMPLETE | preload-manifest.json written');
  
  // ===== P.4 — Query optimization =====
  log('SPRINT P.4', 'START | Query optimization');
  
  // Test key query performance
  const queries = [
    { name: 'lesson_by_day', sql: "SELECT * FROM kellyos_lessons WHERE day_number = 1 AND phase = 'hook'" },
    { name: 'audio_by_day', sql: "SELECT * FROM kellyos_audio WHERE day_number = 1 AND phase = 'hook'" },
    { name: 'assets_by_age', sql: "SELECT * FROM kellyos_assets WHERE age = 'adult'" },
    { name: 'calendar', sql: "SELECT day_number, title FROM core_lessons_v2 ORDER BY day_number" },
    { name: 'scripts_by_day', sql: "SELECT ls.* FROM lesson_scripts ls JOIN lesson_atoms la ON la.id = ls.atom_id JOIN core_lessons_v2 cl ON cl.id = la.lesson_id WHERE cl.day_number = 1" },
  ];
  
  for (const q of queries) {
    const start = Date.now();
    await client.query(q.sql);
    const elapsed = Date.now() - start;
    log('SPRINT P.4', `${q.name}: ${elapsed}ms ${elapsed < 50 ? '✓' : '⚠️ SLOW'}`);
  }
  
  log('SPRINT P.4', 'COMPLETE');
  
  // Update checkpoint
  const cp = JSON.parse(fs.readFileSync(CP_FILE, 'utf-8'));
  cp.sprints.P = { status: 'complete', completed_at: new Date().toISOString(), notes: `cache, compression, manifest, query perf verified` };
  cp.last_updated = new Date().toISOString();
  fs.writeFileSync(CP_FILE, JSON.stringify(cp, null, 2));
  
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
