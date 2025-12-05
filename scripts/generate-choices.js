/**
 * Standalone Choice Generator (Backup)
 * Generates interactive choices for lesson atoms
 *
 * Usage:
 *   node scripts/generate-choices.js --start 335 --end 365
 *   node scripts/generate-choices.js --day 333
 *   node scripts/generate-choices.js --all
 */

import { createClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config({ path: resolve(__dirname, '..', '.env') });

// Initialize clients
const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.PUBLIC_SUPABASE_ANON_KEY
);

// Use Claude if available, otherwise could swap to Gemini
const anthropic = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  : null;

// Configuration
const CONFIG = {
  AGE_GROUPS: ['2-5', '6-12', '13-17', '18-35', '36-60', '61+'],
  LANGUAGES: ['en', 'es', 'fr'],
  QUESTION_PHASES: ['teaching', 'practice', 'reflection', 'Fact1', 'Fact2', 'Fact3'],
  RATE_LIMIT_MS: 1000,
  MAX_RETRIES: 3
};

// Prompt template
const CHOICE_PROMPT = `You are Kelly, a warm and intelligent AI teacher. Generate interactive multiple choice options for this learning moment.

## Context
- Topic: {topic}
- Day: {day_number} of 365
- Phase: {phase}
- Universal Truth: {universal_truth}

## Existing Content
{phase_text}

## Requirements
Generate choices for ALL 6 age groups in ALL 3 languages.

### Age Group Guidelines:
- 2-5 years: Simple words, playful, feelings and senses
- 6-12 years: Curious explorer, connections to their world
- 13-17 years: Practical applications, career relevance
- 18-35 years: Professional depth, efficiency
- 36-60 years: Family perspective, community impact
- 61+ years: Reflection, legacy, wisdom

### Choice Structure:
- Choice A: Surface-level understanding
- Choice B: Deeper insight (BEST ANSWER)
- Choice C: Nuanced/challenging perspective

Output ONLY valid JSON in this exact format:
{
  "2-5": {
    "en": [
      {"letter": "A", "text": "...", "response": "..."},
      {"letter": "B", "text": "...", "response": "..."},
      {"letter": "C", "text": "...", "response": "..."}
    ],
    "es": [...],
    "fr": [...]
  },
  "6-12": {"en": [...], "es": [...], "fr": [...]},
  "13-17": {"en": [...], "es": [...], "fr": [...]},
  "18-35": {"en": [...], "es": [...], "fr": [...]},
  "36-60": {"en": [...], "es": [...], "fr": [...]},
  "61+": {"en": [...], "es": [...], "fr": [...]}
}`;

/**
 * Generate choices using Claude API
 */
async function generateChoicesWithClaude(atom, lesson) {
  const prompt = CHOICE_PROMPT.replace('{topic}', lesson.topic)
    .replace('{day_number}', lesson.day_number)
    .replace('{phase}', atom.phase)
    .replace('{universal_truth}', lesson.universal_truth || '')
    .replace(
      '{phase_text}',
      atom.content?.text || atom.content?.script || JSON.stringify(atom.content)
    );

  const response = await anthropic.messages.create({
    model: 'claude-3-5-sonnet-20241022',
    max_tokens: 4000,
    messages: [{ role: 'user', content: prompt }]
  });

  const text = response.content[0].text;

  // Extract JSON from response
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error('No JSON found in response');
  }

  return JSON.parse(jsonMatch[0]);
}

/**
 * Fallback: Generate simple choices without API
 */
function generateFallbackChoices(atom, lesson) {
  const baseText = atom.content?.text || lesson.topic;

  const fallback = {};

  for (const age of CONFIG.AGE_GROUPS) {
    fallback[age] = {};

    for (const lang of CONFIG.LANGUAGES) {
      const langResponses = {
        en: { a: "That's a good start!", b: 'Excellent thinking!', c: 'Very insightful!' },
        es: { a: '¡Buen comienzo!', b: '¡Excelente pensamiento!', c: '¡Muy perspicaz!' },
        fr: { a: "C'est un bon début!", b: 'Excellente réflexion!', c: 'Très perspicace!' }
      };

      fallback[age][lang] = [
        { letter: 'A', text: `Option A about ${lesson.topic}`, response: langResponses[lang].a },
        { letter: 'B', text: `Option B about ${lesson.topic}`, response: langResponses[lang].b },
        { letter: 'C', text: `Option C about ${lesson.topic}`, response: langResponses[lang].c }
      ];
    }
  }

  return fallback;
}

/**
 * Update atom with choices in Supabase
 */
async function updateAtomWithChoices(atomId, choices) {
  // Get current content
  const { data: atom, error: fetchError } = await supabase
    .from('lesson_atoms')
    .select('content')
    .eq('id', atomId)
    .single();

  if (fetchError) throw fetchError;

  // Merge choices into content
  const updatedContent = {
    ...atom.content,
    choices
  };

  // Update
  const { error: updateError } = await supabase
    .from('lesson_atoms')
    .update({ content: updatedContent })
    .eq('id', atomId);

  if (updateError) throw updateError;
}

/**
 * Process a single lesson
 */
async function processLesson(dayNumber) {
  console.log(`\n📚 Processing Day ${dayNumber}...`);

  // Get lesson
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();

  if (lessonError || !lesson) {
    console.log(`  ❌ Lesson not found for day ${dayNumber}`);
    return { success: 0, failed: 0, skipped: 1 };
  }

  console.log(`  Topic: ${lesson.topic}`);

  // Get atoms
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id);

  if (atomsError || !atoms?.length) {
    console.log(`  ⚠️ No atoms found`);
    return { success: 0, failed: 0, skipped: 1 };
  }

  let success = 0,
    failed = 0,
    skipped = 0;

  for (const atom of atoms) {
    // Skip non-question phases
    if (!CONFIG.QUESTION_PHASES.includes(atom.phase)) {
      continue;
    }

    // Skip if already has choices
    if (atom.content?.choices) {
      console.log(`  ⏭️ ${atom.phase} already has choices`);
      skipped++;
      continue;
    }

    console.log(`  🎯 Generating for ${atom.phase}...`);

    try {
      let choices;

      if (anthropic) {
        choices = await generateChoicesWithClaude(atom, lesson);
      } else {
        console.log(`    ⚠️ No API key, using fallback`);
        choices = generateFallbackChoices(atom, lesson);
      }

      await updateAtomWithChoices(atom.id, choices);
      console.log(`  ✅ ${atom.phase} done`);
      success++;

      // Rate limit
      await new Promise((r) => setTimeout(r, CONFIG.RATE_LIMIT_MS));
    } catch (err) {
      console.log(`  ❌ ${atom.phase} failed: ${err.message}`);
      failed++;
    }
  }

  return { success, failed, skipped };
}

/**
 * Main execution
 */
async function main() {
  const args = process.argv.slice(2);

  let startDay = 1,
    endDay = 365;

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--start') startDay = parseInt(args[++i]);
    if (args[i] === '--end') endDay = parseInt(args[++i]);
    if (args[i] === '--day') {
      startDay = endDay = parseInt(args[++i]);
    }
    if (args[i] === '--december') {
      startDay = 335;
      endDay = 365;
    }
  }

  console.log('═'.repeat(60));
  console.log('  CHOICE GENERATOR - Curious Kelly');
  console.log('═'.repeat(60));
  console.log(`  Days: ${startDay} to ${endDay}`);
  console.log(`  API: ${anthropic ? 'Claude' : 'Fallback (no API key)'}`);
  console.log('═'.repeat(60));

  const totals = { success: 0, failed: 0, skipped: 0 };

  // Priority: December first
  const days = [];
  for (let d = startDay; d <= endDay; d++) days.push(d);
  if (startDay <= 334 && endDay >= 335) {
    // Reorder to do December first
    const december = days.filter((d) => d >= 335);
    const rest = days.filter((d) => d < 335);
    days.length = 0;
    days.push(...december, ...rest);
  }

  for (const day of days) {
    const result = await processLesson(day);
    totals.success += result.success;
    totals.failed += result.failed;
    totals.skipped += result.skipped;
  }

  console.log('\n' + '═'.repeat(60));
  console.log('  COMPLETE');
  console.log('═'.repeat(60));
  console.log(`  ✅ Success: ${totals.success}`);
  console.log(`  ❌ Failed: ${totals.failed}`);
  console.log(`  ⏭️ Skipped: ${totals.skipped}`);
  console.log('═'.repeat(60));
}

main().catch(console.error);









