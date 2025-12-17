#!/usr/bin/env npx tsx
/**
 * 🚔 CONVERSATIONAL TONE POLICE
 * 
 * Converts all lesson_atoms from formal/textbook tone to Kelly's
 * conversational voice style (as established in Day 1 "The Scientist" archetype).
 * 
 * Target: Days 2-365, all 10 archetypes, all 5 phases
 * Scope: ~18,200 lesson_atoms
 * 
 * Day 1 "The Scientist" conversational markers:
 * - Casual openers: "Hey!", "Here's what's wild:"
 * - Contractions: "That's", "It's", "don't", "you'll"
 * - Em-dashes for emphasis and casual breaks
 * - Direct "you" address throughout
 * - Short, punchy sentences
 * - Personal feel: "So here's what I want you to take away"
 * 
 * Usage:
 *   npx tsx scripts/conversational-tone-police.ts
 *   npx tsx scripts/conversational-tone-police.ts --day-range 2-10
 *   npx tsx scripts/conversational-tone-police.ts --dry-run
 *   npx tsx scripts/conversational-tone-police.ts --resume-from 50
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
  
  // Processing
  BATCH_SIZE: 50,           // Atoms per batch
  RATE_LIMIT_MS: 100,       // ms between API calls
  MAX_RETRIES: 3,
  
  // Output
  LOG_DIR: path.join(process.cwd(), 'logs', 'tone-police'),
  PROGRESS_FILE: path.join(process.cwd(), 'logs', 'tone-police', 'progress.json'),
};

// =============================================================================
// REFERENCE: DAY 1 "THE SCIENTIST" CONVERSATIONAL EXAMPLES
// =============================================================================

const CONVERSATIONAL_EXAMPLES = {
  Hook: {
    formal: "Today, we embark on an intriguing journey to explore The Three Lives of Water. From vapor in the atmosphere to liquid flowing in rivers...",
    conversational: "Hey! Ever notice how New Year's Day or even just Monday morning makes you feel like anything is possible? That's not just a feeling—it's science. Your brain literally resets at these moments. Today, let's explore how you can use this to your advantage."
  },
  Fact1: {
    formal: "Did you know that water can exist in three distinct states—solid, liquid, and gas—depending on temperature and pressure?",
    conversational: "Here's what's wild: researchers found that people who start a goal on a 'fresh start' day—like New Year's or a birthday—are way more likely to stick with it. It's called the Fresh Start Effect. Your brain treats these moments as a clean slate, like yesterday's failures don't count anymore."
  },
  Fact2: {
    formal: "The concept of starting fresh is closely linked to the water cycle. When water vapor rises from the Earth's surface, it cools at higher altitudes...",
    conversational: "But here's the thing—you don't have to wait for January 1st. You can create your own fresh starts. The first day of the month, a Monday, even just tomorrow morning. The key is making it feel significant to YOU."
  },
  Fact3: {
    formal: "This remarkable ability allows water to travel vast distances, shaping our planet's ecosystems and weather patterns.",
    conversational: "And get this—when you start fresh, your brain's reward system lights up. You literally feel more optimistic. It's like your mind gives you permission to be a different person than you were yesterday."
  },
  Wisdom: {
    formal: "In our exploration of The Three Lives of Water, we observe how water exists as a liquid, vapor, and solid, demonstrating its adaptability and resilience in nature.",
    conversational: "So here's what I want you to take away: you have the power to create a fresh start whenever you need one. Tomorrow could be day one. Next week could be your reset. The calendar doesn't decide—you do."
  }
};

// =============================================================================
// MASTER PROMPT FOR TONE CONVERSION
// =============================================================================

const TONE_CONVERSION_PROMPT = `You are Kelly, an AI teacher who speaks in a warm, conversational style. Your job is to convert formal/textbook educational scripts into conversational scripts that sound like YOU—a friendly, curious teacher talking directly to one learner.

## Your Voice Characteristics (MUST follow):

1. **Casual openers**: Start with "Hey!", "So here's the thing—", "Here's what's wild:", "Okay so—", "Fun fact:", "Get this—"

2. **Contractions ALWAYS**: Use "that's", "it's", "don't", "won't", "you'll", "you're", "there's", "here's"

3. **Em-dashes for energy**: Use em-dashes (—) for emphasis and to create natural pauses. Example: "Your brain literally resets—and that's not just a metaphor."

4. **Direct "you" address**: Always talk TO the learner. "You've probably noticed...", "Here's what you can do..."

5. **Short, punchy sentences**: Mix long and short. Short sentences hit harder. They create rhythm.

6. **Personal feel**: "So here's what I want you to take away:", "I think you'll find this fascinating:", "This is the part that blows my mind:"

7. **Questions that engage**: "Ever noticed how...?", "Know what's interesting?", "But here's the question—"

8. **Avoid at all costs**:
   - "Welcome, learners!" or "Today, we embark on..."
   - "Have you ever considered..." (too formal)
   - "Let us explore..." (too academic)
   - "This process is essential..." (textbook language)
   - Any third-person explanations
   - Overly formal transitions

## Examples of conversions:

FORMAL: "Today, we embark on an intriguing journey to explore The Three Lives of Water."
CONVERSATIONAL: "Hey! Water is wild. It can be a solid, a liquid, AND a gas—sometimes all in the same day."

FORMAL: "Did you know that water can exist in three distinct states?"
CONVERSATIONAL: "Here's what's cool about water—it's like a shapeshifter. Ice, liquid, steam. Same stuff, totally different vibes."

FORMAL: "This process is essential for understanding weather patterns."
CONVERSATIONAL: "And here's why this matters to you—this is literally how rain happens."

FORMAL: "In conclusion, water demonstrates remarkable adaptability."
CONVERSATIONAL: "So next time you see a cloud, remember—that's just water on a journey. Pretty cool, right?"

## Your task:

Convert the following script to Kelly's conversational voice. Keep the SAME core facts and information, but make it sound like Kelly is chatting with a friend, not lecturing from a textbook.

PHASE: {{PHASE}}
ARCHETYPE: {{ARCHETYPE}}
TOPIC: {{TOPIC}}

ORIGINAL SCRIPT:
{{SCRIPT}}

CONVERSATIONAL VERSION (respond with ONLY the converted script, no explanations):`;

// =============================================================================
// TYPES
// =============================================================================

interface LessonAtom {
  id: string;
  core_lesson_id: string;
  archetype: string;
  phase: string;
  content: {
    script: string;
    options?: string[];
    responses?: Record<string, string>;
  };
}

interface CoreLesson {
  id: string;
  day_number: number;
  topic: string;
}

interface ProgressState {
  lastProcessedDay: number;
  lastProcessedAtomId: string | null;
  totalProcessed: number;
  totalSkipped: number;
  totalErrors: number;
  startedAt: string;
  lastUpdated: string;
}

// =============================================================================
// MAIN SCRIPT
// =============================================================================

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
const openai = new OpenAI({ apiKey: CONFIG.OPENAI_API_KEY });

// Parse CLI args
const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const dayRangeArg = args.find(a => a.startsWith('--day-range='));
const resumeFromArg = args.find(a => a.startsWith('--resume-from='));

let startDay = 2;
let endDay = 365;

if (dayRangeArg) {
  const range = dayRangeArg.split('=')[1];
  const [start, end] = range.split('-').map(Number);
  startDay = start;
  endDay = end || start;
}

if (resumeFromArg) {
  startDay = parseInt(resumeFromArg.split('=')[1]);
}

// Ensure log directory exists
if (!fs.existsSync(CONFIG.LOG_DIR)) {
  fs.mkdirSync(CONFIG.LOG_DIR, { recursive: true });
}

// Progress tracking
function loadProgress(): ProgressState | null {
  try {
    if (fs.existsSync(CONFIG.PROGRESS_FILE)) {
      return JSON.parse(fs.readFileSync(CONFIG.PROGRESS_FILE, 'utf-8'));
    }
  } catch (e) {
    console.log('No previous progress found, starting fresh');
  }
  return null;
}

function saveProgress(state: ProgressState): void {
  state.lastUpdated = new Date().toISOString();
  fs.writeFileSync(CONFIG.PROGRESS_FILE, JSON.stringify(state, null, 2));
}

// Rate limiting
async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Convert script to conversational tone
async function convertToConversational(
  script: string,
  phase: string,
  archetype: string,
  topic: string,
  retries = 0
): Promise<string | null> {
  try {
    const prompt = TONE_CONVERSION_PROMPT
      .replace('{{PHASE}}', phase)
      .replace('{{ARCHETYPE}}', archetype)
      .replace('{{TOPIC}}', topic)
      .replace('{{SCRIPT}}', script);

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      max_tokens: 500,
    });

    return response.choices[0]?.message?.content?.trim() || null;
  } catch (error: any) {
    if (retries < CONFIG.MAX_RETRIES) {
      console.log(`  ⚠️ Retry ${retries + 1}/${CONFIG.MAX_RETRIES} after error: ${error.message}`);
      await sleep(1000 * (retries + 1));
      return convertToConversational(script, phase, archetype, topic, retries + 1);
    }
    console.error(`  ❌ Failed after ${CONFIG.MAX_RETRIES} retries: ${error.message}`);
    return null;
  }
}

// Check if script is already conversational
function isAlreadyConversational(script: string): boolean {
  const conversationalMarkers = [
    /^Hey!/i,
    /^Here's what's wild/i,
    /^So here's/i,
    /^Get this—/i,
    /^And get this/i,
    /^Fun fact:/i,
    /^Okay so/i,
    /you'll/i,
    /that's not just/i,
    /—and /,
    /—it's /,
    /—you /,
  ];
  
  const formalMarkers = [
    /^Today, we embark/i,
    /^Welcome, learners/i,
    /^Have you ever considered/i,
    /^Let us explore/i,
    /^In this lesson/i,
    /This process is essential/i,
    /It is important to note/i,
    /One must understand/i,
  ];

  const hasConversationalMarkers = conversationalMarkers.some(m => m.test(script));
  const hasFormalMarkers = formalMarkers.some(m => m.test(script));
  
  // If it has conversational markers and no formal markers, it's good
  return hasConversationalMarkers && !hasFormalMarkers;
}

// Main processing function
async function processDays(): Promise<void> {
  console.log('\n╔══════════════════════════════════════════════════════════════════╗');
  console.log('║  🚔 CONVERSATIONAL TONE POLICE - FULL AUDIT & CONVERSION         ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║  Mode: ${dryRun ? 'DRY RUN (no changes)' : '🔥 LIVE - Updating database'}`.padEnd(67) + '║');
  console.log(`║  Day Range: ${startDay} - ${endDay}`.padEnd(67) + '║');
  console.log(`║  Estimated Atoms: ${(endDay - startDay + 1) * 50}`.padEnd(67) + '║');
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  const state: ProgressState = {
    lastProcessedDay: startDay - 1,
    lastProcessedAtomId: null,
    totalProcessed: 0,
    totalSkipped: 0,
    totalErrors: 0,
    startedAt: new Date().toISOString(),
    lastUpdated: new Date().toISOString(),
  };

  // Load existing progress if resuming
  const existingProgress = loadProgress();
  if (existingProgress && resumeFromArg) {
    state.totalProcessed = existingProgress.totalProcessed;
    state.totalSkipped = existingProgress.totalSkipped;
    state.totalErrors = existingProgress.totalErrors;
    console.log(`📂 Resuming from day ${startDay}, ${state.totalProcessed} already processed\n`);
  }

  // Open log file
  const logPath = path.join(CONFIG.LOG_DIR, `conversion-${new Date().toISOString().split('T')[0]}.log`);
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });
  
  const log = (msg: string) => {
    console.log(msg);
    logStream.write(msg + '\n');
  };

  log(`\n${'═'.repeat(70)}`);
  log(`TONE POLICE RUN: ${new Date().toISOString()}`);
  log(`${'═'.repeat(70)}\n`);

  // Process day by day
  for (let day = startDay; day <= endDay; day++) {
    log(`\n📅 DAY ${day}/${endDay}`);
    log(`${'─'.repeat(50)}`);

    // Get core lesson for this day
    const { data: lesson, error: lessonError } = await supabase
      .from('core_lessons')
      .select('id, day_number, topic')
      .eq('day_number', day)
      .single();

    if (lessonError || !lesson) {
      log(`  ⚠️ Day ${day}: No lesson found, skipping`);
      continue;
    }

    // Get all atoms for this day
    const { data: atoms, error: atomsError } = await supabase
      .from('lesson_atoms')
      .select('id, core_lesson_id, archetype, phase, content')
      .eq('core_lesson_id', lesson.id);

    if (atomsError || !atoms) {
      log(`  ❌ Day ${day}: Error fetching atoms - ${atomsError?.message}`);
      state.totalErrors++;
      continue;
    }

    log(`  📚 Topic: ${lesson.topic}`);
    log(`  🔢 Atoms to process: ${atoms.length}`);

    let dayProcessed = 0;
    let daySkipped = 0;
    let dayErrors = 0;

    // Process each atom
    for (const atom of atoms) {
      const script = atom.content?.script;
      
      if (!script) {
        log(`  ⚠️ ${atom.archetype}/${atom.phase}: No script, skipping`);
        daySkipped++;
        continue;
      }

      // Check if already conversational
      if (isAlreadyConversational(script)) {
        daySkipped++;
        continue;
      }

      // Convert to conversational
      const newScript = await convertToConversational(
        script,
        atom.phase,
        atom.archetype,
        lesson.topic
      );

      if (!newScript) {
        log(`  ❌ ${atom.archetype}/${atom.phase}: Conversion failed`);
        dayErrors++;
        continue;
      }

      // Update database (unless dry run)
      if (!dryRun) {
        const newContent = {
          ...atom.content,
          script: newScript,
        };

        const { error: updateError } = await supabase
          .from('lesson_atoms')
          .update({ content: newContent })
          .eq('id', atom.id);

        if (updateError) {
          log(`  ❌ ${atom.archetype}/${atom.phase}: Update failed - ${updateError.message}`);
          dayErrors++;
          continue;
        }
      }

      dayProcessed++;
      state.totalProcessed++;

      // Log sample conversions (first 2 per day)
      if (dayProcessed <= 2) {
        log(`  ✅ ${atom.archetype}/${atom.phase}:`);
        log(`     BEFORE: "${script.substring(0, 80)}..."`);
        log(`     AFTER:  "${newScript.substring(0, 80)}..."`);
      }

      // Rate limiting
      await sleep(CONFIG.RATE_LIMIT_MS);
    }

    state.totalSkipped += daySkipped;
    state.totalErrors += dayErrors;
    state.lastProcessedDay = day;

    log(`  📊 Day ${day} Complete: ${dayProcessed} converted, ${daySkipped} skipped, ${dayErrors} errors`);

    // Save progress after each day
    saveProgress(state);
  }

  // Final summary
  log(`\n${'═'.repeat(70)}`);
  log('🏁 TONE POLICE AUDIT COMPLETE');
  log(`${'═'.repeat(70)}`);
  log(`✅ Converted:  ${state.totalProcessed}`);
  log(`⏭️  Skipped:    ${state.totalSkipped}`);
  log(`❌ Errors:     ${state.totalErrors}`);
  log(`⏱️  Duration:   ${Math.round((Date.now() - new Date(state.startedAt).getTime()) / 1000 / 60)} minutes`);
  log(`📁 Log file:   ${logPath}`);
  
  if (dryRun) {
    log(`\n⚠️  DRY RUN - No changes were made to the database`);
    log(`   Run without --dry-run to apply changes`);
  }

  log(`\n${'═'.repeat(70)}\n`);

  logStream.close();
  
  // Record in audit trail
  if (!dryRun && state.totalProcessed > 0) {
    await supabase.from('lesson_audits').insert({
      day_number: 0, // 0 = batch operation
      audit_type: 'full_audit',
      status: 'fixed',
      field_name: 'lesson_atoms.content.script',
      original_value: `Days ${startDay}-${endDay}: Formal textbook tone`,
      fixed_value: `Converted to Kelly conversational voice`,
      fix_method: 'GPT-4o-mini tone conversion',
      fix_rationale: 'Day 1 was updated to conversational tone but Days 2-365 remained formal. This brings all days to consistent conversational voice.',
      fixed_by: 'tone_police_v1',
      fixed_at: new Date().toISOString(),
      confidence_score: 0.95,
    });
    
    console.log('📝 Audit recorded in lesson_audits table\n');
  }
}

// Run
processDays().catch(console.error);












