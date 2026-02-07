/**
 * Sprint N: Multi-Language Script Generation
 * N.1 — Translate 1,825 English scripts to 5 languages = 9,125 translations
 * Uses OpenAI GPT-4o-mini for speed + cost efficiency
 * Priority: Spanish, French, Mandarin, Arabic, Hindi
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

const LANGUAGES = [
  { code: 'es', name: 'Spanish', native: 'Español' },
  { code: 'fr', name: 'French', native: 'Français' },
  { code: 'zh', name: 'Mandarin Chinese', native: '中文' },
  { code: 'ar', name: 'Arabic', native: 'العربية' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
];

async function translateBatch(texts, targetLang) {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: `You are a professional translator for educational content. Translate into ${targetLang.name} (${targetLang.native}). Maintain the teaching tone. Return JSON array of translated strings matching input order. Only JSON, no other text.` },
        { role: 'user', content: JSON.stringify(texts) }
      ],
      temperature: 0.3,
      max_tokens: 4000
    })
  });
  
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).substring(0, 100)}`);
  const data = await res.json();
  const content = data.choices[0].message.content;
  
  // Parse JSON array
  const match = content.match(/\[[\s\S]*\]/);
  if (!match) throw new Error('No JSON array in response');
  return JSON.parse(match[0]);
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  log('SPRINT N.1', 'START | Multi-language translation for 5 languages');
  
  // Get all English lessons
  const english = await client.query(`
    SELECT day_number, phase, title, content_text
    FROM kellyos_lessons
    WHERE (language = 'en' OR language IS NULL)
    ORDER BY day_number, phase
  `);
  log('SPRINT N.1', `Found ${english.rows.length} English lessons to translate`);
  
  const totalTranslations = {};
  
  for (const lang of LANGUAGES) {
    log('SPRINT N.1', `STARTING ${lang.name} (${lang.code})`);
    
    // Check how many already exist
    const existing = await client.query(
      'SELECT COUNT(*) as cnt FROM kellyos_lessons WHERE language = $1', [lang.code]
    );
    const existingCount = parseInt(existing.rows[0].cnt);
    log('SPRINT N.1', `${lang.name}: ${existingCount} existing translations`);
    
    if (existingCount >= 1800) {
      log('SPRINT N.1', `${lang.name}: SKIP — already have ${existingCount} translations`);
      totalTranslations[lang.code] = existingCount;
      continue;
    }
    
    // Find which are missing
    const missingResult = await client.query(`
      SELECT e.day_number, e.phase, e.title, e.content_text
      FROM kellyos_lessons e
      WHERE (e.language = 'en' OR e.language IS NULL)
        AND NOT EXISTS (
          SELECT 1 FROM kellyos_lessons t 
          WHERE t.day_number = e.day_number AND t.phase = e.phase AND t.language = $1
        )
      ORDER BY e.day_number, e.phase
    `, [lang.code]);
    
    const missing = missingResult.rows;
    log('SPRINT N.1', `${lang.name}: ${missing.length} translations needed`);
    
    let translated = 0;
    let failed = 0;
    const batchSize = 10; // Translate 10 at a time
    
    for (let i = 0; i < missing.length; i += batchSize) {
      const batch = missing.slice(i, i + batchSize);
      const texts = batch.map(r => r.content_text || r.title || `Lesson for Day ${r.day_number}`);
      const titles = batch.map(r => r.title || `Day ${r.day_number}`);
      
      try {
        const translatedTexts = await translateBatch(texts, lang);
        let translatedTitles;
        try {
          translatedTitles = await translateBatch(titles, lang);
        } catch {
          translatedTitles = titles; // Keep English titles as fallback
        }
        
        for (let j = 0; j < batch.length && j < translatedTexts.length; j++) {
          await client.query(`
            INSERT INTO kellyos_lessons (day_number, phase, title, content_text, language)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
          `, [batch[j].day_number, batch[j].phase, translatedTitles[j] || batch[j].title, translatedTexts[j], lang.code]);
          translated++;
        }
        
      } catch (e) {
        failed += batch.length;
        if (e.message.includes('429')) {
          log('SPRINT N.1', `${lang.name}: Rate limited, waiting 10s...`);
          await new Promise(r => setTimeout(r, 10000));
          i -= batchSize; // Retry
        }
      }
      
      if (translated % 100 === 0 && translated > 0) {
        log('SPRINT N.1', `${lang.name}: ${translated}/${missing.length} translated (${failed} failed)`);
      }
      
      await new Promise(r => setTimeout(r, 150));
    }
    
    totalTranslations[lang.code] = translated + existingCount;
    log('SPRINT N.1', `${lang.name} COMPLETE | ${translated} new translations, ${failed} failed`);
  }
  
  // N.3 — Verify
  log('SPRINT N.3', 'START | Verifying multi-language coverage');
  
  const coverage = {};
  for (const lang of ['en', ...LANGUAGES.map(l => l.code)]) {
    const cnt = await client.query(
      'SELECT COUNT(*) as scripts FROM kellyos_lessons WHERE language = $1 OR (language IS NULL AND $1 = \'en\')',
      [lang]
    );
    const audioCnt = await client.query(
      'SELECT COUNT(*) as audio FROM kellyos_audio WHERE language = $1 OR (language IS NULL AND $1 = \'en\')',
      [lang]
    );
    coverage[lang] = {
      scripts: parseInt(cnt.rows[0].scripts),
      audio: parseInt(audioCnt.rows[0].audio),
    };
  }
  
  const coveragePath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'multilang-coverage.json');
  fs.writeFileSync(coveragePath, JSON.stringify(coverage, null, 2));
  log('SPRINT N.3', `COMPLETE | Coverage: ${JSON.stringify(coverage)}`);
  
  // Update checkpoint
  const cp = JSON.parse(fs.readFileSync(CP_FILE, 'utf-8'));
  cp.sprints.N = { status: 'complete', completed_at: new Date().toISOString(), notes: `Translations: ${JSON.stringify(totalTranslations)}` };
  cp.last_updated = new Date().toISOString();
  fs.writeFileSync(CP_FILE, JSON.stringify(cp, null, 2));
  
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
