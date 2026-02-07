/**
 * Phase 1: MEGA Translation Pipeline
 * Translates ALL English lesson scripts to ALL target languages.
 * Picks up where Sprint N left off (Spanish done, French partial, rest at 0).
 * 
 * Languages: fr (finish), pt, zh, de, ja, ko, it, hi, ar, ru
 * Total target: ~18,250 translations (10 langs x 1825 slots)
 * 
 * Uses OpenAI GPT-4o-mini for speed and cost efficiency.
 * Batch size: 10 scripts per API call.
 * Checkpoint every 50 translations.
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const CHECKPOINT_DIR = path.join(__dirname, '..', 'kelly-pipeline', 'checkpoints');
const CHECKPOINT_FILE = path.join(CHECKPOINT_DIR, 'mega-translate.json');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function log(msg) {
  const line = `[${new Date().toISOString()}] TRANSLATE | ${msg}`;
  console.log(line);
}

function loadCheckpoint() {
  try {
    if (fs.existsSync(CHECKPOINT_FILE)) return JSON.parse(fs.readFileSync(CHECKPOINT_FILE, 'utf-8'));
  } catch {}
  return { completedLanguages: [], inProgress: null, stats: {} };
}

function saveCheckpoint(cp) {
  ensureDir(CHECKPOINT_DIR);
  fs.writeFileSync(CHECKPOINT_FILE, JSON.stringify(cp, null, 2));
}

const ALL_LANGUAGES = [
  { code: 'fr', name: 'French', native: 'Français' },
  { code: 'pt', name: 'Portuguese', native: 'Português' },
  { code: 'zh', name: 'Mandarin Chinese', native: '中文' },
  { code: 'de', name: 'German', native: 'Deutsch' },
  { code: 'ja', name: 'Japanese', native: '日本語' },
  { code: 'ko', name: 'Korean', native: '한국어' },
  { code: 'it', name: 'Italian', native: 'Italiano' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
  { code: 'ar', name: 'Arabic', native: 'العربية' },
  { code: 'ru', name: 'Russian', native: 'Русский' },
];

async function translateBatch(texts, targetLang, retries = 3) {
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
              content: `You are a professional translator for educational content aimed at lifelong learners. Translate the following texts into ${targetLang.name} (${targetLang.native}). Maintain the warm, curious teaching tone. Keep proper nouns, scientific terms, and Kelly's name unchanged. Return ONLY a JSON array of translated strings matching the input order. No extra text or explanation.`
            },
            { role: 'user', content: JSON.stringify(texts) }
          ],
          temperature: 0.3,
          max_tokens: 8000
        })
      });

      if (res.status === 429) {
        const wait = Math.min(30000, (attempt + 1) * 10000);
        log(`Rate limited, waiting ${wait / 1000}s (attempt ${attempt + 1}/${retries})`);
        await new Promise(r => setTimeout(r, wait));
        continue;
      }

      if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).substring(0, 200)}`);

      const data = await res.json();
      const content = data.choices[0].message.content;
      const match = content.match(/\[[\s\S]*\]/);
      if (!match) throw new Error('No JSON array in response');
      
      const parsed = JSON.parse(match[0]);
      if (!Array.isArray(parsed) || parsed.length !== texts.length) {
        throw new Error(`Expected ${texts.length} translations, got ${parsed.length}`);
      }
      return parsed;
    } catch (e) {
      if (attempt === retries - 1) throw e;
      log(`Retry ${attempt + 1}: ${e.message}`);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
}

async function translateLanguage(client, lang) {
  // Check how many already exist
  const existing = await client.query(
    'SELECT COUNT(*) as cnt FROM kellyos_lessons WHERE language = $1', [lang.code]
  );
  const existingCount = parseInt(existing.rows[0].cnt);
  
  if (existingCount >= 1820) {
    log(`${lang.name}: SKIP — already have ${existingCount}/1825 translations`);
    return { translated: 0, existing: existingCount, skipped: true };
  }

  log(`${lang.name}: ${existingCount} existing, finding gaps...`);

  // Find missing translations
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
  log(`${lang.name}: ${missing.length} translations needed`);

  let translated = 0;
  let failed = 0;
  const batchSize = 10;

  for (let i = 0; i < missing.length; i += batchSize) {
    const batch = missing.slice(i, i + batchSize);
    const texts = batch.map(r => r.content_text || r.title || `Lesson for Day ${r.day_number}`);
    const titles = batch.map(r => r.title || `Day ${r.day_number}`);

    try {
      // Translate content
      const translatedTexts = await translateBatch(texts, lang);
      
      // Translate titles (non-critical)
      let translatedTitles;
      try {
        translatedTitles = await translateBatch(titles, lang);
      } catch {
        translatedTitles = titles;
      }

      // Insert into database
      for (let j = 0; j < batch.length && j < translatedTexts.length; j++) {
        try {
          await client.query(`
            INSERT INTO kellyos_lessons (day_number, phase, title, content_text, language)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
          `, [
            batch[j].day_number,
            batch[j].phase,
            translatedTitles[j] || batch[j].title,
            translatedTexts[j],
            lang.code
          ]);
          translated++;
        } catch (dbErr) {
          // Try upsert on conflict
          try {
            await client.query(`
              INSERT INTO kellyos_lessons (day_number, phase, title, content_text, language)
              VALUES ($1, $2, $3, $4, $5)
              ON CONFLICT (day_number, phase, language) DO UPDATE
              SET content_text = EXCLUDED.content_text, title = EXCLUDED.title
            `, [
              batch[j].day_number,
              batch[j].phase,
              translatedTitles[j] || batch[j].title,
              translatedTexts[j],
              lang.code
            ]);
            translated++;
          } catch {
            failed++;
          }
        }
      }
    } catch (e) {
      failed += batch.length;
      log(`${lang.name}: Batch error at ${i}: ${e.message}`);
      if (e.message.includes('429')) {
        log(`Rate limited — waiting 15s`);
        await new Promise(r => setTimeout(r, 15000));
        i -= batchSize; // Retry this batch
      }
    }

    // Log progress
    if ((translated + failed) % 50 === 0 || translated + failed === missing.length) {
      log(`${lang.name}: ${translated}/${missing.length} done, ${failed} failed`);
    }

    // Rate limit spacing
    await new Promise(r => setTimeout(r, 200));
  }

  log(`${lang.name} COMPLETE: ${translated} new (${failed} failed), total = ${existingCount + translated}`);
  return { translated, failed, existing: existingCount, total: existingCount + translated };
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  log('Connected to database');

  // Check unique constraint exists on kellyos_lessons
  try {
    await client.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS idx_kellyos_lessons_unique 
      ON kellyos_lessons(day_number, phase, language)
    `);
    log('Unique index ensured on kellyos_lessons(day_number, phase, language)');
  } catch (e) {
    log(`Index note: ${e.message}`);
  }

  const cp = loadCheckpoint();
  const results = {};

  for (const lang of ALL_LANGUAGES) {
    if (cp.completedLanguages.includes(lang.code)) {
      log(`${lang.name}: Already completed in previous run, skipping`);
      continue;
    }

    cp.inProgress = lang.code;
    saveCheckpoint(cp);

    const result = await translateLanguage(client, lang);
    results[lang.code] = result;

    if (!result.skipped || result.existing >= 1820) {
      cp.completedLanguages.push(lang.code);
    }
    cp.stats[lang.code] = result;
    cp.inProgress = null;
    saveCheckpoint(cp);
  }

  // Final coverage report
  log('=== FINAL COVERAGE ===');
  const coverage = {};
  for (const lang of ['en', 'es', ...ALL_LANGUAGES.map(l => l.code)]) {
    const cnt = await client.query(
      "SELECT COUNT(*) as scripts FROM kellyos_lessons WHERE language = $1 OR (language IS NULL AND $1 = 'en')",
      [lang]
    );
    coverage[lang] = parseInt(cnt.rows[0].scripts);
    log(`${lang}: ${coverage[lang]} scripts`);
  }

  const auditDir = path.join(__dirname, '..', 'kelly-pipeline', 'audit');
  ensureDir(auditDir);
  fs.writeFileSync(
    path.join(auditDir, 'translation-coverage.json'),
    JSON.stringify({ timestamp: new Date().toISOString(), coverage, results }, null, 2)
  );

  log('MEGA TRANSLATION COMPLETE');
  await client.end();
}

main().catch(e => { console.error('[TRANSLATE ERROR]', e); process.exit(1); });
