#!/usr/bin/env npx tsx
/**
 * 🚔 CONVERSATIONAL TONE POLICE V2
 * 
 * FIXES THE V1 MISTAKE: V1 made Kelly sound like a ditzy teenager.
 * V2 makes Kelly sound like an INTELLIGENT, WARM TEACHER.
 * 
 * Kelly's voice is:
 * - Knowledgeable and confident (she's a TEACHER)
 * - Warm and approachable (not cold/academic)
 * - Curious and enthusiastic (genuine interest)
 * - NOT a Valley Girl, NOT a teenager, NOT ditzy
 * 
 * Day 1 Gold Standard examples:
 * ✅ "Here's what's wild: researchers found that..."
 * ✅ "It's called the Fresh Start Effect."
 * ✅ "Your brain literally resets at these moments."
 * ✅ "So here's what I want you to take away:"
 * 
 * What V1 did WRONG:
 * ❌ "Hey! Get this—" on every script
 * ❌ "Crazy, right?" "Wild, right?" "How cool is that?"
 * ❌ "super cool", "pretty wild", "really gets me"
 * ❌ Lost the intellectual authority
 * 
 * Usage:
 *   npx tsx scripts/conversational-tone-police-v2.ts --day-range=2-365
 *   npx tsx scripts/conversational-tone-police-v2.ts --fix-v1-damage
 *   npx tsx scripts/conversational-tone-police-v2.ts --dry-run
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
  
  BATCH_SIZE: 50,
  RATE_LIMIT_MS: 100,
  MAX_RETRIES: 3,
  
  LOG_DIR: path.join(process.cwd(), 'logs', 'tone-police-v2'),
  PROGRESS_FILE: path.join(process.cwd(), 'logs', 'tone-police-v2', 'progress.json'),
};

// =============================================================================
// THE CORRECT PROMPT - INTELLIGENT TEACHER, NOT DITZY FRIEND
// =============================================================================

const TONE_CONVERSION_PROMPT_V2 = `You are Kelly, an AI teacher with a PhD's knowledge and a best friend's warmth. You're converting educational scripts to sound like YOU speaking—intelligent, curious, and genuinely enthusiastic about learning.

## Kelly's Voice (FOLLOW EXACTLY):

### What Kelly IS:
1. **A knowledgeable teacher** - You KNOW things. State facts with quiet confidence.
2. **Genuinely curious** - Your enthusiasm comes from real interest, not hype.
3. **Warm but professional** - Approachable without being unprofessional.
4. **Conversational but smart** - Use contractions, but maintain vocabulary.

### What Kelly is NOT:
- ❌ NOT a Valley Girl ("like, totally!", "crazy, right?")
- ❌ NOT a hype beast ("so cool!", "amazing!", "wild!")
- ❌ NOT a teenager (no excessive enthusiasm markers)
- ❌ NOT dumbed down (keep the intellectual substance)

## Specific Rules:

### Openers (VARY these, don't always use the same one):
✅ "Here's what's wild:" (use sparingly)
✅ "So here's the thing—"
✅ "Ever notice how..."
✅ "There's something fascinating about..."
✅ [Sometimes just start with the content directly]
❌ DON'T start every script with "Hey!"
❌ DON'T use "Hey! Get this—" repeatedly

### Filler phrases to AVOID:
❌ "Crazy, right?"
❌ "Wild, right?"
❌ "How cool is that?"
❌ "Pretty amazing, right?"
❌ "You know?"
❌ "Super cool"
❌ "Really gets me"

### Instead, use:
✅ Direct statements of fact
✅ Rhetorical questions that make the learner think
✅ "This matters because..."
✅ Connecting ideas to the learner's life
✅ Naming concepts with authority ("It's called the Fresh Start Effect.")

### Contractions (DO use):
✅ "that's", "it's", "you'll", "there's", "here's", "doesn't", "won't"

### Em-dashes (use for natural pauses):
✅ "Your brain treats these moments as a clean slate—like yesterday's failures don't count anymore."

## Examples of CORRECT Kelly voice:

FORMAL: "Did you know that water can exist in three distinct states—solid, liquid, and gas—depending on temperature and pressure?"
KELLY: "Water does something remarkable—it exists in three completely different states. The same molecule can be solid ice, liquid water, or invisible vapor. What determines which? Just temperature and pressure."

FORMAL: "Research indicates that active listening improves relationship quality by 40%."
KELLY: "Here's something worth knowing: when you really listen—not just wait for your turn to talk—relationships improve dramatically. Studies show a 40% increase in relationship quality. That's not a small difference."

FORMAL: "The process of photosynthesis is essential for life on Earth."
KELLY: "Photosynthesis is the reason you're alive right now. Plants take sunlight and turn it into the oxygen you're breathing and the food that powers every living thing. It's arguably the most important chemical reaction on Earth."

## Your task:

Convert this script to Kelly's intelligent, warm teaching voice. Keep ALL the educational content. Make it conversational without making it dumb.

PHASE: {{PHASE}}
ARCHETYPE: {{ARCHETYPE}}  
TOPIC: {{TOPIC}}

ORIGINAL SCRIPT:
{{SCRIPT}}

KELLY'S VERSION (respond with ONLY the converted script, no explanations):`;

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

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const fixV1Damage = args.includes('--fix-v1-damage');
const dayRangeArg = args.find(a => a.startsWith('--day-range='));

let startDay = 2;
let endDay = 365;

if (dayRangeArg) {
  const range = dayRangeArg.split('=')[1];
  const [start, end] = range.split('-').map(Number);
  startDay = start;
  endDay = end || start;
}

if (!fs.existsSync(CONFIG.LOG_DIR)) {
  fs.mkdirSync(CONFIG.LOG_DIR, { recursive: true });
}

function saveProgress(state: ProgressState): void {
  state.lastUpdated = new Date().toISOString();
  fs.writeFileSync(CONFIG.PROGRESS_FILE, JSON.stringify(state, null, 2));
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function convertToKellyVoice(
  script: string,
  phase: string,
  archetype: string,
  topic: string,
  retries = 0
): Promise<string | null> {
  try {
    const prompt = TONE_CONVERSION_PROMPT_V2
      .replace('{{PHASE}}', phase)
      .replace('{{ARCHETYPE}}', archetype)
      .replace('{{TOPIC}}', topic)
      .replace('{{SCRIPT}}', script);

    const response = await openai.chat.completions.create({
      model: 'gpt-4o',  // Using GPT-4o for better quality
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.6,  // Slightly lower for more consistency
      max_tokens: 600,
    });

    return response.choices[0]?.message?.content?.trim() || null;
  } catch (error: any) {
    if (retries < CONFIG.MAX_RETRIES) {
      console.log(`  ⚠️ Retry ${retries + 1}/${CONFIG.MAX_RETRIES}: ${error.message}`);
      await sleep(1000 * (retries + 1));
      return convertToKellyVoice(script, phase, archetype, topic, retries + 1);
    }
    console.error(`  ❌ Failed after ${CONFIG.MAX_RETRIES} retries: ${error.message}`);
    return null;
  }
}

// Check if script has V1 damage (ditzy markers)
function hasV1Damage(script: string): boolean {
  const ditzyMarkers = [
    /^Hey! (Get this|So here's|Here's)/i,
    /Crazy, right\?/i,
    /Wild, right\?/i,
    /How cool is that\?/i,
    /How amazing is that\?/i,
    /Pretty (cool|amazing|wild)/i,
    /super (cool|amazing|wild)/i,
    /You know\?$/,
  ];
  
  return ditzyMarkers.some(m => m.test(script));
}

// Check if already in good Kelly voice
function isGoodKellyVoice(script: string): boolean {
  // Day 1 patterns - the gold standard
  const goodPatterns = [
    /^Here's what's wild:/,
    /^So here's what I want you to take away:/,
    /^But here's the thing—/,
    /^And get this—when/,  // Note: "And get this—when" is different from "Hey! Get this—"
  ];
  
  const hasGoodPattern = goodPatterns.some(p => p.test(script));
  const hasDitzyMarkers = hasV1Damage(script);
  
  return hasGoodPattern && !hasDitzyMarkers;
}

async function processDays(): Promise<void> {
  console.log('\n╔══════════════════════════════════════════════════════════════════╗');
  console.log('║  🚔 CONVERSATIONAL TONE POLICE V2 - INTELLIGENT KELLY            ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║  Mode: ${dryRun ? 'DRY RUN' : fixV1Damage ? '🔧 FIXING V1 DAMAGE' : '🔥 LIVE'}`.padEnd(67) + '║');
  console.log(`║  Day Range: ${startDay} - ${endDay}`.padEnd(67) + '║');
  console.log(`║  Model: GPT-4o (higher quality)`.padEnd(67) + '║');
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

  const logPath = path.join(CONFIG.LOG_DIR, `conversion-${new Date().toISOString().split('T')[0]}.log`);
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });
  
  const log = (msg: string) => {
    console.log(msg);
    logStream.write(msg + '\n');
  };

  for (let day = startDay; day <= endDay; day++) {
    log(`\n📅 DAY ${day}/${endDay}`);
    log('─'.repeat(50));

    const { data: lesson, error: lessonError } = await supabase
      .from('core_lessons')
      .select('id, day_number, topic')
      .eq('day_number', day)
      .single();

    if (lessonError || !lesson) {
      log(`  ⚠️ Day ${day}: No lesson found`);
      continue;
    }

    const { data: atoms, error: atomsError } = await supabase
      .from('lesson_atoms')
      .select('id, core_lesson_id, archetype, phase, content')
      .eq('core_lesson_id', lesson.id);

    if (atomsError || !atoms) {
      log(`  ❌ Day ${day}: Error fetching atoms`);
      state.totalErrors++;
      continue;
    }

    log(`  📚 Topic: ${lesson.topic}`);
    log(`  🔢 Atoms: ${atoms.length}`);

    let dayProcessed = 0;
    let daySkipped = 0;

    for (const atom of atoms) {
      const script = atom.content?.script;
      
      if (!script) {
        daySkipped++;
        continue;
      }

      // If fixing V1 damage, only process scripts with ditzy markers
      if (fixV1Damage && !hasV1Damage(script)) {
        daySkipped++;
        continue;
      }

      // If already good Kelly voice, skip
      if (isGoodKellyVoice(script)) {
        daySkipped++;
        continue;
      }

      const newScript = await convertToKellyVoice(
        script,
        atom.phase,
        atom.archetype,
        lesson.topic
      );

      if (!newScript) {
        state.totalErrors++;
        continue;
      }

      if (!dryRun) {
        const newContent = { ...atom.content, script: newScript };
        const { error: updateError } = await supabase
          .from('lesson_atoms')
          .update({ content: newContent })
          .eq('id', atom.id);

        if (updateError) {
          log(`  ❌ Update failed: ${updateError.message}`);
          state.totalErrors++;
          continue;
        }
      }

      dayProcessed++;
      state.totalProcessed++;

      if (dayProcessed <= 2) {
        log(`  ✅ ${atom.archetype}/${atom.phase}:`);
        log(`     BEFORE: "${script.substring(0, 70)}..."`);
        log(`     AFTER:  "${newScript.substring(0, 70)}..."`);
      }

      await sleep(CONFIG.RATE_LIMIT_MS);
    }

    state.totalSkipped += daySkipped;
    state.lastProcessedDay = day;
    log(`  📊 Day ${day}: ${dayProcessed} fixed, ${daySkipped} skipped`);
    saveProgress(state);
  }

  log(`\n${'═'.repeat(70)}`);
  log('🏁 TONE POLICE V2 COMPLETE');
  log(`${'═'.repeat(70)}`);
  log(`✅ Processed: ${state.totalProcessed}`);
  log(`⏭️  Skipped:   ${state.totalSkipped}`);
  log(`❌ Errors:    ${state.totalErrors}`);
  
  if (dryRun) {
    log(`\n⚠️  DRY RUN - No changes made`);
  }

  logStream.close();
}

processDays().catch(console.error);











