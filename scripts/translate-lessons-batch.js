#!/usr/bin/env node
/**
 * Batch Translate Lessons to Target Language
 * 
 * Uses OpenAI GPT-4o-mini to translate all 365 lessons while preserving Kelly's voice.
 * Translations are saved back to the lesson JSON files (i18n structure).
 * 
 * Usage:
 *   node scripts/translate-lessons-batch.js --lang=es --day=1
 *   node scripts/translate-lessons-batch.js --lang=es --range=1-10
 *   node scripts/translate-lessons-batch.js --lang=es --all
 *   node scripts/translate-lessons-batch.js --lang=pt --all --dry-run
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';
import * as dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '..', '.env.local') });

// ============================================================================
// CONFIGURATION
// ============================================================================

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
  console.error('❌ Missing OPENAI_API_KEY in .env.local');
  process.exit(1);
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const LESSONS_DIR = path.join(__dirname, '..', 'public', 'lessons');

// Language configurations
const LANGUAGES = {
  es: {
    name: 'Spanish',
    nativeName: 'Español',
    formalYou: 'tú (informal)',
    lengthFactor: 1.15,
    systemPrompt: `You are Kelly, a warm, curious, intelligent AI teacher. You are translating your lessons into Spanish.

## Your Personality in Spanish
- Warm and welcoming: "¡Bienvenidos!" not cold "Bienvenido"
- Use informal "tú" address, never "usted"
- Keep rhetorical questions as questions
- Preserve enthusiasm markers (!, ...)
- Maintain pause markers (—)
- Sound like a friend, not a lecturer

## Spanish-Specific Rules
- Convert Fahrenheit to Celsius
- Use metric units
- Keep proper nouns in original form
- Use Latin American Spanish by default (neutral)
- "You're doing great!" → "¡Lo estás haciendo genial!"
- "Have you ever wondered..." → "¿Alguna vez te has preguntado..."
- "Here's today's wisdom:" → "Esta es la sabiduría de hoy:"
- "Welcome to Day One" → "Bienvenidos al Día Uno"
`
  },
  pt: {
    name: 'Portuguese',
    nativeName: 'Português',
    formalYou: 'você (informal)',
    lengthFactor: 1.08,
    systemPrompt: `You are Kelly, a warm, curious, intelligent AI teacher. You are translating your lessons into Portuguese.

## Your Personality in Portuguese
- Warm and welcoming: "Bem-vindos!" 
- Use informal "você" address (Brazilian Portuguese)
- Keep rhetorical questions as questions
- Preserve enthusiasm markers (!, ...)
- Maintain pause markers (—)
- Sound like a friend, not a lecturer

## Portuguese-Specific Rules
- Convert Fahrenheit to Celsius
- Use metric units
- Keep proper nouns in original form
- Use Brazilian Portuguese
- "You're doing great!" → "Você está indo muito bem!"
- "Have you ever wondered..." → "Você já se perguntou..."
- "Here's today's wisdom:" → "Eis a sabedoria de hoje:"
- "Welcome to Day One" → "Bem-vindos ao Dia Um"
`
  },
  fr: {
    name: 'French',
    nativeName: 'Français',
    formalYou: 'tu (informal)',
    lengthFactor: 1.20,
    systemPrompt: `You are Kelly, a warm, curious, intelligent AI teacher. You are translating your lessons into French.

## Your Personality in French
- Warm and welcoming: "Bienvenue !"
- Use informal "tu" address, never "vous"
- Keep rhetorical questions as questions
- Preserve enthusiasm markers (!, ...)
- Maintain pause markers (—)

## French-Specific Rules
- Convert Fahrenheit to Celsius
- Use metric units
- Keep proper nouns in original form
`
  }
};

// Track API usage
let totalTokens = 0;
let totalCost = 0;

// ============================================================================
// TRANSLATION FUNCTIONS
// ============================================================================

function getEnglishValue(value) {
  if (!value) return '';
  if (typeof value === 'object' && value.en !== undefined) return value.en;
  if (typeof value === 'string') return value;
  return '';
}

async function translateText(text, lang, context = '') {
  if (!text || text === '[NEEDS TRANSLATION]') return '';
  
  const langConfig = LANGUAGES[lang];
  if (!langConfig) throw new Error(`Unsupported language: ${lang}`);
  
  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: langConfig.systemPrompt },
        { 
          role: 'user', 
          content: `${context ? `Context: ${context}\n\n` : ''}Translate this to ${langConfig.name}. Respond with ONLY the translation, no explanation:\n\n${text}` 
        }
      ],
      temperature: 0.3,
      max_tokens: 1000
    });
    
    const usage = response.usage;
    totalTokens += usage.total_tokens;
    // GPT-4o-mini: $0.15/1M input, $0.60/1M output
    totalCost += (usage.prompt_tokens * 0.00000015) + (usage.completion_tokens * 0.0000006);
    
    return response.choices[0].message.content.trim();
  } catch (error) {
    console.error(`    ❌ Translation error: ${error.message}`);
    return '[TRANSLATION_FAILED]';
  }
}

async function translatePhase(phase, lang, topic) {
  const result = {};
  
  // Title
  if (phase.title) {
    const enTitle = getEnglishValue(phase.title);
    if (enTitle && enTitle !== '[NEEDS TRANSLATION]') {
      result.title = await translateText(enTitle, lang, `Phase title for lesson about "${topic}"`);
    }
  }
  
  // Script
  if (phase.script) {
    const enScript = getEnglishValue(phase.script);
    if (enScript && enScript !== '[NEEDS TRANSLATION]') {
      result.script = await translateText(enScript, lang, `Kelly speaking about "${topic}"`);
    }
  }
  
  // Prompt
  if (phase.prompt) {
    const enPrompt = getEnglishValue(phase.prompt);
    if (enPrompt && enPrompt !== '[NEEDS TRANSLATION]') {
      result.prompt = await translateText(enPrompt, lang, `Question for learner about "${topic}"`);
    }
  }
  
  // Options
  if (phase.options && Array.isArray(phase.options)) {
    result.options = [];
    for (const opt of phase.options) {
      const translatedOpt = { letter: opt.letter, quality: opt.quality };
      
      if (opt.text) {
        const enText = getEnglishValue(opt.text);
        if (enText && enText !== '[NEEDS TRANSLATION]') {
          translatedOpt.text = await translateText(enText, lang, `Answer choice for "${topic}"`);
        }
      }
      
      if (opt.response) {
        const enResponse = getEnglishValue(opt.response);
        if (enResponse && enResponse !== '[NEEDS TRANSLATION]') {
          translatedOpt.response = await translateText(enResponse, lang, `Kelly's response to learner's choice about "${topic}"`);
        }
      }
      
      result.options.push(translatedOpt);
    }
  }
  
  return result;
}

function mergeTranslation(original, translation, lang) {
  // Helper to merge i18n field
  const mergeField = (origField, translatedValue) => {
    if (!origField) return origField;
    
    if (typeof origField === 'object' && origField.en !== undefined) {
      return {
        ...origField,
        [lang]: translatedValue || origField[lang]
      };
    }
    
    // Convert to i18n structure
    return {
      en: origField,
      [lang]: translatedValue || '[NEEDS TRANSLATION]'
    };
  };
  
  const result = { ...original };
  
  if (translation.title) {
    result.title = mergeField(original.title, translation.title);
  }
  if (translation.script) {
    result.script = mergeField(original.script, translation.script);
  }
  if (translation.prompt) {
    result.prompt = mergeField(original.prompt, translation.prompt);
  }
  
  if (translation.options && original.options) {
    result.options = original.options.map((origOpt, i) => {
      const transOpt = translation.options[i] || {};
      return {
        ...origOpt,
        text: mergeField(origOpt.text, transOpt.text),
        response: mergeField(origOpt.response, transOpt.response)
      };
    });
  }
  
  return result;
}

// ============================================================================
// LESSON PROCESSING
// ============================================================================

function loadLesson(dayNumber) {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  if (!fs.existsSync(filePath)) return null;
  
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch (e) {
    console.error(`❌ Failed to parse day-${dayNumber}.json: ${e.message}`);
    return null;
  }
}

function saveLesson(dayNumber, lesson) {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  fs.writeFileSync(filePath, JSON.stringify(lesson, null, 2) + '\n');
}

async function translateLesson(dayNumber, lang, dryRun = false) {
  const lesson = loadLesson(dayNumber);
  if (!lesson) {
    console.log(`   ⚠️ No lesson file for day ${dayNumber}`);
    return { success: false, error: 'No file' };
  }
  
  const topic = getEnglishValue(lesson.meta?.topic) || 'Unknown Topic';
  console.log(`📚 Day ${dayNumber}: "${topic}"`);
  
  // Translate meta
  const metaTopic = getEnglishValue(lesson.meta?.topic);
  const translatedTopic = await translateText(metaTopic, lang);
  console.log(`   📝 Topic: "${metaTopic}" → "${translatedTopic}"`);
  
  // Translate headline
  const headline = getEnglishValue(lesson.headline);
  const translatedHeadline = await translateText(headline, lang);
  console.log(`   📝 Headline translated`);
  
  // Translate universal_truth
  const truth = getEnglishValue(lesson.universal_truth);
  const translatedTruth = await translateText(truth, lang);
  console.log(`   📝 Universal truth translated`);
  
  // Translate phases
  const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];
  const translatedPhases = {};
  
  for (const phaseName of PHASES) {
    if (!lesson.phases?.[phaseName]) continue;
    process.stdout.write(`   🎬 ${phaseName}...`);
    translatedPhases[phaseName] = await translatePhase(lesson.phases[phaseName], lang, topic);
    console.log(' ✅');
  }
  
  // Translate fun_facts
  let translatedFacts = [];
  if (lesson.fun_facts) {
    process.stdout.write(`   📋 Fun facts...`);
    for (const fact of lesson.fun_facts) {
      const enFact = getEnglishValue(fact);
      if (enFact) {
        translatedFacts.push(await translateText(enFact, lang));
      }
    }
    console.log(' ✅');
  }
  
  // Translate discussion_questions
  let translatedQuestions = [];
  if (lesson.discussion_questions) {
    process.stdout.write(`   ❓ Discussion questions...`);
    for (const q of lesson.discussion_questions) {
      const enQ = getEnglishValue(q);
      if (enQ) {
        translatedQuestions.push(await translateText(enQ, lang));
      }
    }
    console.log(' ✅');
  }
  
  if (dryRun) {
    console.log(`   🔍 DRY RUN - not saving`);
    return { success: true, dryRun: true };
  }
  
  // Merge translations into lesson
  const updatedLesson = { ...lesson };
  
  // Meta topic
  updatedLesson.meta = {
    ...lesson.meta,
    topic: typeof lesson.meta.topic === 'object' 
      ? { ...lesson.meta.topic, [lang]: translatedTopic }
      : { en: lesson.meta.topic, [lang]: translatedTopic }
  };
  
  // Ensure languages array includes this language
  if (!updatedLesson.meta.languages) updatedLesson.meta.languages = ['en'];
  if (!updatedLesson.meta.languages.includes(lang)) {
    updatedLesson.meta.languages.push(lang);
  }
  
  // Headline
  updatedLesson.headline = typeof lesson.headline === 'object'
    ? { ...lesson.headline, [lang]: translatedHeadline }
    : { en: lesson.headline, [lang]: translatedHeadline };
  
  // Universal truth
  updatedLesson.universal_truth = typeof lesson.universal_truth === 'object'
    ? { ...lesson.universal_truth, [lang]: translatedTruth }
    : { en: lesson.universal_truth, [lang]: translatedTruth };
  
  // Phases
  for (const phaseName of PHASES) {
    if (!lesson.phases?.[phaseName] || !translatedPhases[phaseName]) continue;
    updatedLesson.phases[phaseName] = mergeTranslation(
      lesson.phases[phaseName], 
      translatedPhases[phaseName], 
      lang
    );
  }
  
  // Fun facts
  if (lesson.fun_facts && translatedFacts.length > 0) {
    updatedLesson.fun_facts = lesson.fun_facts.map((fact, i) => {
      const enFact = getEnglishValue(fact);
      return typeof fact === 'object'
        ? { ...fact, [lang]: translatedFacts[i] || fact[lang] }
        : { en: enFact, [lang]: translatedFacts[i] || '[NEEDS TRANSLATION]' };
    });
  }
  
  // Discussion questions
  if (lesson.discussion_questions && translatedQuestions.length > 0) {
    updatedLesson.discussion_questions = lesson.discussion_questions.map((q, i) => {
      const enQ = getEnglishValue(q);
      return typeof q === 'object'
        ? { ...q, [lang]: translatedQuestions[i] || q[lang] }
        : { en: enQ, [lang]: translatedQuestions[i] || '[NEEDS TRANSLATION]' };
    });
  }
  
  // Save
  saveLesson(dayNumber, updatedLesson);
  console.log(`   ✅ Saved day-${dayNumber}.json`);
  
  return { success: true };
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  let days = [];
  let lang = null;
  let dryRun = args.includes('--dry-run');
  
  for (const arg of args) {
    if (arg.startsWith('--lang=')) {
      lang = arg.split('=')[1];
    } else if (arg === '--all') {
      for (let d = 1; d <= 365; d++) days.push(d);
    } else if (arg.startsWith('--range=')) {
      const [start, end] = arg.split('=')[1].split('-').map(n => parseInt(n, 10));
      for (let d = start; d <= end; d++) days.push(d);
    } else if (arg.startsWith('--days=')) {
      days = arg.split('=')[1].split(',').map(n => parseInt(n.trim(), 10));
    } else if (arg.startsWith('--day=')) {
      days.push(parseInt(arg.split('=')[1], 10));
    }
  }
  
  if (!lang || days.length === 0) {
    console.log('Usage:');
    console.log('  node scripts/translate-lessons-batch.js --lang=es --day=1');
    console.log('  node scripts/translate-lessons-batch.js --lang=es --range=1-10');
    console.log('  node scripts/translate-lessons-batch.js --lang=pt --all');
    console.log('  node scripts/translate-lessons-batch.js --lang=es --all --dry-run');
    console.log('');
    console.log('Supported languages:');
    for (const [code, config] of Object.entries(LANGUAGES)) {
      console.log(`  ${code}: ${config.name} (${config.nativeName})`);
    }
    process.exit(0);
  }
  
  if (!LANGUAGES[lang]) {
    console.error(`❌ Unsupported language: ${lang}`);
    console.log('Supported:', Object.keys(LANGUAGES).join(', '));
    process.exit(1);
  }
  
  const langConfig = LANGUAGES[lang];
  
  console.log('═'.repeat(60));
  console.log(`🌍 BATCH TRANSLATION: ${langConfig.name} (${langConfig.nativeName})`);
  console.log('═'.repeat(60));
  console.log(`📅 Days to translate: ${days.length}`);
  console.log(`🎯 Target language: ${lang}`);
  console.log(`🔍 Mode: ${dryRun ? 'DRY RUN' : 'LIVE'}`);
  console.log('');
  
  const totals = { success: 0, failed: 0 };
  const startTime = Date.now();
  
  for (const day of days) {
    try {
      const result = await translateLesson(day, lang, dryRun);
      if (result.success) {
        totals.success++;
      } else {
        totals.failed++;
      }
    } catch (error) {
      console.error(`❌ Day ${day} failed: ${error.message}`);
      totals.failed++;
    }
    
    // Rate limiting - avoid hitting API limits
    await new Promise(r => setTimeout(r, 200));
  }
  
  const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 TRANSLATION SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Translated: ${totals.success}`);
  console.log(`❌ Failed: ${totals.failed}`);
  console.log(`📊 Tokens used: ${totalTokens.toLocaleString()}`);
  console.log(`💰 Estimated cost: $${totalCost.toFixed(4)}`);
  console.log(`⏱️ Time elapsed: ${elapsed} minutes`);
}

main().catch(console.error);
