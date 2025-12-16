#!/usr/bin/env npx tsx
/**
 * 🔧 REGENERATE DAMAGED ARCHETYPES
 * 
 * Regenerates the 9 damaged archetypes (NOT The Scientist) for Days 2-150
 * using the ORIGINAL archetype descriptions from prompts.py.
 * 
 * This is the proper fix - regenerating content that matches each archetype's
 * intended voice and personality.
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';
import * as fs from 'fs';
import * as path from 'path';

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
  
  DAMAGED_DAY_START: 2,
  DAMAGED_DAY_END: 150,
  
  RATE_LIMIT_MS: 150,
  LOG_DIR: path.join(process.cwd(), 'logs', 'archetype-regen'),
};

// Original archetype descriptions from prompts.py
const ARCHETYPE_DESCRIPTIONS: Record<string, string> = {
  "The Architect": "systems thinker, structured, big-picture. Sees connections between parts, loves frameworks, explains how things fit together.",
  "The Diplomat": "balanced, multiple perspectives, bridge-builder. Presents various viewpoints fairly, finds common ground, values nuance.",
  "The Empath": "emotionally intelligent, relational, compassionate. Focuses on feelings, relationships, and how topics affect people and communities.",
  "The Explorer": "curious, adventurous, discovery-focused. Frames learning as an expedition, celebrates unknowns, encourages wonder and exploration.",
  "The MacGyver": "creative problem-solver, hands-on, innovative. Loves DIY approaches, asks 'how can we use this?', celebrates ingenuity.",
  "The Mystic": "philosophical, meaning-seeking, contemplative. Explores deeper significance, asks 'why does this matter?', seeks transcendent insights.",
  "The Rebel": "questioning, challenging assumptions, unconventional. Pushes back on common beliefs, celebrates disruption, encourages critical thinking.",
  "The Storyteller": "narrative-driven, emotional, relatable. Weaves information into stories, uses metaphors, connects facts to human experiences.",
  "The Survivor": "practical, resilient, real-world focused. Emphasizes survival value, practical applications, preparedness. Direct and no-nonsense.",
  // NOT regenerating The Scientist - that one needs the conversational treatment
};

const PHASE_INSTRUCTIONS: Record<string, string> = {
  "Hook": "Create an attention-grabbing opening that draws the learner in. Make them curious to learn more. 2-3 sentences max.",
  "Fact1": "Present the first fascinating fact about this topic. Make it surprising or counter-intuitive if possible. Include a clear explanation.",
  "Fact2": "Build on Fact1 with a second insight that adds depth. Show a different angle or application of the concept.",
  "Fact3": "Deliver a 'wow' moment - the most surprising or delightful fact. This should be memorable and shareable.",
  "Wisdom": "Close with a reflective insight that connects this knowledge to the learner's life. What does this mean for them? How can they apply it?",
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
const openai = new OpenAI({ apiKey: CONFIG.OPENAI_API_KEY });

// Parse args
const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const dayRangeArg = args.find(a => a.startsWith('--day-range='));

let startDay = CONFIG.DAMAGED_DAY_START;
let endDay = CONFIG.DAMAGED_DAY_END;

if (dayRangeArg) {
  const [s, e] = dayRangeArg.split('=')[1].split('-').map(Number);
  startDay = s;
  endDay = e || s;
}

if (!fs.existsSync(CONFIG.LOG_DIR)) {
  fs.mkdirSync(CONFIG.LOG_DIR, { recursive: true });
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function regenerateScript(
  topic: string,
  universalTruth: string,
  archetype: string,
  phase: string
): Promise<string | null> {
  const archetypeDesc = ARCHETYPE_DESCRIPTIONS[archetype];
  const phaseInst = PHASE_INSTRUCTIONS[phase];
  
  if (!archetypeDesc) return null;
  
  const prompt = `Generate educational content for Kelly, an AI teacher avatar.

TOPIC: ${topic}
UNIVERSAL TRUTH: ${universalTruth}

ARCHETYPE: ${archetype}
Archetype personality: ${archetypeDesc}

PHASE: ${phase}
Phase goal: ${phaseInst}

Generate ONLY the script text (what Kelly says to the learner). 
- 2-4 sentences
- Written in the ${archetype} voice/personality
- Conversational but intelligent
- No "Hey!" or excessive casual markers
- Match the archetype's unique perspective and metaphor style

OUTPUT: Just the script text, nothing else.`;

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      max_tokens: 400,
    });
    
    return response.choices[0]?.message?.content?.trim() || null;
  } catch (error: any) {
    console.error(`  ❌ Generation error: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('\n╔══════════════════════════════════════════════════════════════════╗');
  console.log('║  🔧 REGENERATE DAMAGED ARCHETYPES                                ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║  Mode: ${dryRun ? 'DRY RUN' : '🔥 LIVE - Regenerating content'}`.padEnd(67) + '║');
  console.log(`║  Days: ${startDay}-${endDay}`.padEnd(67) + '║');
  console.log(`║  Archetypes: 9 (excluding The Scientist)`.padEnd(67) + '║');
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  const logPath = path.join(CONFIG.LOG_DIR, `regen-${new Date().toISOString().split('T')[0]}.log`);
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });
  const log = (msg: string) => { console.log(msg); logStream.write(msg + '\n'); };

  let totalRegenerated = 0;
  let totalErrors = 0;

  for (let day = startDay; day <= endDay; day++) {
    log(`\n📅 DAY ${day}/${endDay}`);
    log('─'.repeat(50));

    // Get lesson
    const { data: lesson, error: lessonError } = await supabase
      .from('core_lessons')
      .select('id, topic, universal_truth')
      .eq('day_number', day)
      .single();

    if (lessonError || !lesson) {
      log(`  ⚠️ No lesson found`);
      continue;
    }

    log(`  📚 Topic: ${lesson.topic}`);

    // Get damaged atoms (all archetypes except The Scientist)
    const { data: atoms, error: atomsError } = await supabase
      .from('lesson_atoms')
      .select('id, archetype, phase, content')
      .eq('core_lesson_id', lesson.id)
      .neq('archetype', 'The Scientist');

    if (atomsError || !atoms) {
      log(`  ❌ Error fetching atoms`);
      totalErrors++;
      continue;
    }

    let dayRegen = 0;

    for (const atom of atoms) {
      // Skip if not a known archetype
      if (!ARCHETYPE_DESCRIPTIONS[atom.archetype]) {
        continue;
      }

      const newScript = await regenerateScript(
        lesson.topic,
        lesson.universal_truth || '',
        atom.archetype,
        atom.phase
      );

      if (!newScript) {
        totalErrors++;
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
          totalErrors++;
          continue;
        }
      }

      dayRegen++;
      totalRegenerated++;

      // Show samples
      if (dayRegen <= 2) {
        log(`  ✅ ${atom.archetype}/${atom.phase}:`);
        log(`     "${newScript.substring(0, 80)}..."`);
      }

      await sleep(CONFIG.RATE_LIMIT_MS);
    }

    log(`  📊 Day ${day}: ${dayRegen} regenerated`);
  }

  log('\n' + '═'.repeat(70));
  log('🏁 REGENERATION COMPLETE');
  log('═'.repeat(70));
  log(`✅ Regenerated: ${totalRegenerated}`);
  log(`❌ Errors: ${totalErrors}`);
  if (dryRun) log('\n⚠️ DRY RUN - No changes made');
  log('═'.repeat(70));

  logStream.close();
}

main().catch(console.error);








