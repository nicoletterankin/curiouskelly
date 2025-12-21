#!/usr/bin/env npx tsx
/**
 * 🔧 REWRITE ALL LESSONS
 * 
 * Cleans up ALL 365 days of content:
 * 1. Rewrites scripts in proper Kelly voice (removes V1 ditzy damage)
 * 2. Adds meaningful phase titles based on content
 * 3. Ensures response scripts exist for all options
 * 
 * Usage:
 *   npx tsx scripts/rewrite-all-lessons.ts --day-range=1-365
 *   npx tsx scripts/rewrite-all-lessons.ts --day=355
 *   npx tsx scripts/rewrite-all-lessons.ts --dry-run --day-range=1-10
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
  
  RATE_LIMIT_MS: 200,
  MAX_RETRIES: 3,
  BATCH_SIZE: 10, // Process 10 atoms at a time
  
  LOG_DIR: path.join(process.cwd(), 'logs', 'rewrite-lessons'),
};

// =============================================================================
// THE MASTER PROMPT
// =============================================================================

const REWRITE_PROMPT = `You are Kelly, an AI teacher with a PhD's knowledge and a best friend's warmth. 

Your task is to:
1. REWRITE the script in Kelly's authentic voice
2. Generate a SHORT, MEANINGFUL phase title (2-5 words) that captures what this phase is about

## Kelly's Voice Rules:

### ALWAYS:
- Use contractions (that's, it's, you're, don't)
- Use em-dashes for natural pauses (—)
- State facts with quiet confidence
- Be warm but professional
- Maintain intellectual authority

### NEVER:
- "Hey!" as opener
- "Get this—" repeatedly  
- "Crazy, right?" or "Wild, right?"
- "How cool is that?" or "Super cool"
- Valley Girl tone or teenage enthusiasm
- Dumbed-down vocabulary

### Good Kelly openers (VARY these):
- "There's something fascinating about..."
- "Here's what's interesting—"
- "Ever notice how..."
- [Sometimes just start with the content directly]
- "So here's the thing—"

## Your response format (JSON only):

{
  "phaseTitle": "Short Meaningful Title",
  "script": "The rewritten script in Kelly's voice..."
}

## Input:
PHASE: {{PHASE}}
TOPIC: {{TOPIC}}
ORIGINAL SCRIPT: {{SCRIPT}}

Respond with ONLY the JSON object, no other text.`;

// =============================================================================
// PHASE TITLE TEMPLATES (for phases without scripts)
// =============================================================================

const DEFAULT_PHASE_TITLES: Record<string, string> = {
  'Hook': 'The Opening Question',
  'Cliff': 'Choose Your Path',
  'Fact1': 'Discovery One',
  'Fact2': 'Discovery Two',
  'Fact3': 'Discovery Three',
  'Wisdom': 'The Takeaway',
  'Outro': 'Until Tomorrow',
};

// =============================================================================
// TYPES
// =============================================================================

interface LessonAtom {
  id: string;
  core_lesson_id: string;
  archetype: string;
  phase: string;
  content: any;
}

interface RewriteResult {
  phaseTitle: string;
  script: string;
}

// =============================================================================
// MAIN
// =============================================================================

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
const openai = new OpenAI({ apiKey: CONFIG.OPENAI_API_KEY });

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const dayArg = args.find(a => a.startsWith('--day='));
const dayRangeArg = args.find(a => a.startsWith('--day-range='));
const titlesOnly = args.includes('--titles-only');

let startDay = 1;
let endDay = 365;

if (dayArg) {
  startDay = endDay = parseInt(dayArg.split('=')[1]);
} else if (dayRangeArg) {
  const range = dayRangeArg.split('=')[1];
  const [start, end] = range.split('-').map(Number);
  startDay = start;
  endDay = end || start;
}

// Ensure log directory exists
if (!fs.existsSync(CONFIG.LOG_DIR)) {
  fs.mkdirSync(CONFIG.LOG_DIR, { recursive: true });
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function rewriteWithGPT(
  script: string,
  phase: string,
  topic: string,
  retries = 0
): Promise<RewriteResult | null> {
  try {
    const prompt = REWRITE_PROMPT
      .replace('{{PHASE}}', phase)
      .replace('{{TOPIC}}', topic)
      .replace('{{SCRIPT}}', script);

    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.6,
      max_tokens: 800,
      response_format: { type: 'json_object' },
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return null;

    return JSON.parse(content) as RewriteResult;
  } catch (error: any) {
    if (retries < CONFIG.MAX_RETRIES) {
      console.log(`    ⚠️ Retry ${retries + 1}: ${error.message}`);
      await sleep(1000 * (retries + 1));
      return rewriteWithGPT(script, phase, topic, retries + 1);
    }
    console.error(`    ❌ Failed: ${error.message}`);
    return null;
  }
}

// Generate a phase title from the script content (without API call)
function generateQuickTitle(script: string, phase: string, topic: string): string {
  // Extract key concepts from the script
  const words = script.split(/\s+/).slice(0, 30).join(' ');
  
  // Phase-specific title generation
  switch (phase) {
    case 'Hook':
      if (topic.length < 30) return topic;
      return 'The Opening';
    case 'Cliff':
      return 'Your Choice';
    case 'Fact1':
      // Try to extract a key concept
      const match1 = script.match(/called (?:the |a )?["']?([^"'.]+)["']?/i);
      if (match1) return match1[1].substring(0, 25);
      return 'The First Discovery';
    case 'Fact2':
      const match2 = script.match(/called (?:the |a )?["']?([^"'.]+)["']?/i);
      if (match2) return match2[1].substring(0, 25);
      return 'Going Deeper';
    case 'Fact3':
      const match3 = script.match(/called (?:the |a )?["']?([^"'.]+)["']?/i);
      if (match3) return match3[1].substring(0, 25);
      return 'The Connection';
    case 'Wisdom':
      return 'What This Means';
    case 'Outro':
      return 'Until Tomorrow';
    default:
      return DEFAULT_PHASE_TITLES[phase] || phase;
  }
}

async function processDay(day: number, log: (msg: string) => void): Promise<{ processed: number; skipped: number; errors: number }> {
  const stats = { processed: 0, skipped: 0, errors: 0 };
  
  // Get all lessons for this day
  const { data: lessons, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id, topic')
    .eq('day_number', day);

  if (lessonError || !lessons?.length) {
    log(`  ⚠️ No lessons found`);
    return stats;
  }

  for (const lesson of lessons) {
    log(`  📚 Topic: ${lesson.topic}`);
    
    // Get all atoms for this lesson
    const { data: atoms, error: atomsError } = await supabase
      .from('lesson_atoms')
      .select('*')
      .eq('core_lesson_id', lesson.id);

    if (atomsError || !atoms?.length) {
      log(`    ⚠️ No atoms found`);
      continue;
    }

    for (const atom of atoms as LessonAtom[]) {
      const script = atom.content?.script;
      
      if (!script) {
        stats.skipped++;
        continue;
      }

      // Check if already has a phase title
      const hasTitle = atom.content?.phaseTitle;
      
      if (titlesOnly) {
        // Just add titles, don't rewrite
        if (hasTitle) {
          stats.skipped++;
          continue;
        }
        
        const title = generateQuickTitle(script, atom.phase, lesson.topic);
        
        if (!dryRun) {
          const newContent = { ...atom.content, phaseTitle: title };
          await supabase
            .from('lesson_atoms')
            .update({ content: newContent })
            .eq('id', atom.id);
        }
        
        stats.processed++;
        continue;
      }

      // Full rewrite with GPT
      const result = await rewriteWithGPT(script, atom.phase, lesson.topic);
      
      if (!result) {
        stats.errors++;
        continue;
      }

      if (!dryRun) {
        const newContent = {
          ...atom.content,
          script: result.script,
          phaseTitle: result.phaseTitle,
          originalScript: script, // Keep backup
        };
        
        const { error: updateError } = await supabase
          .from('lesson_atoms')
          .update({ content: newContent })
          .eq('id', atom.id);

        if (updateError) {
          log(`    ❌ Update failed: ${updateError.message}`);
          stats.errors++;
          continue;
        }
      }

      stats.processed++;
      
      if (stats.processed <= 2) {
        log(`    ✅ ${atom.archetype}/${atom.phase}: "${result.phaseTitle}"`);
      }

      await sleep(CONFIG.RATE_LIMIT_MS);
    }
  }

  return stats;
}

async function main(): Promise<void> {
  console.log('\n╔══════════════════════════════════════════════════════════════════╗');
  console.log('║  🔧 REWRITE ALL LESSONS                                          ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║  Mode: ${dryRun ? 'DRY RUN' : titlesOnly ? 'TITLES ONLY' : '🔥 FULL REWRITE'}`.padEnd(67) + '║');
  console.log(`║  Day Range: ${startDay} - ${endDay}`.padEnd(67) + '║');
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  const logPath = path.join(CONFIG.LOG_DIR, `rewrite-${new Date().toISOString().split('T')[0]}.log`);
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });
  
  const log = (msg: string) => {
    console.log(msg);
    logStream.write(msg + '\n');
  };

  let totalProcessed = 0;
  let totalSkipped = 0;
  let totalErrors = 0;

  for (let day = startDay; day <= endDay; day++) {
    log(`\n📅 DAY ${day}/${endDay}`);
    log('─'.repeat(50));

    const stats = await processDay(day, log);
    
    totalProcessed += stats.processed;
    totalSkipped += stats.skipped;
    totalErrors += stats.errors;

    log(`  📊 Processed: ${stats.processed}, Skipped: ${stats.skipped}, Errors: ${stats.errors}`);
  }

  log(`\n${'═'.repeat(70)}`);
  log('🏁 COMPLETE');
  log(`${'═'.repeat(70)}`);
  log(`✅ Processed: ${totalProcessed}`);
  log(`⏭️  Skipped:   ${totalSkipped}`);
  log(`❌ Errors:    ${totalErrors}`);
  
  if (dryRun) {
    log(`\n⚠️  DRY RUN - No changes made`);
  }

  logStream.close();
}

main().catch(console.error);
