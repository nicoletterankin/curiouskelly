#!/usr/bin/env npx tsx
/**
 * PHASE-ALIGNED VISUAL GENERATOR
 * 
 * Generates visuals that are deeply integrated with lesson content:
 * - Uses quiz questions and answers for teaching visuals
 * - Incorporates misconceptions for contrast visuals
 * - Aligns with learning objectives
 * - Creates answer-illustration images
 * 
 * Usage:
 *   npx tsx scripts/seed-phase-aligned-visuals.ts --day=1
 *   npx tsx scripts/seed-phase-aligned-visuals.ts --day=1 --all-phases --all-styles
 *   npx tsx scripts/seed-phase-aligned-visuals.ts --range=1-7
 * 
 * @created December 17, 2025
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { createClient } from '@supabase/supabase-js';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  MODEL: 'imagen-4.0-ultra-generate-001',
  COST_PER_IMAGE: 0.06,
  ASPECT_RATIO: '16:9',
  DELAY_BETWEEN_IMAGES_MS: 2500,
  
  ALL_PHASES: ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro', 'complete'] as const,
  KEY_PHASES: ['hook', 'fact1', 'fact3', 'wisdom', 'complete'] as const,
  
  // Phase-optimized styles
  PHASE_STYLES: {
    hook: ['artistic', 'minimal'],
    cliff: ['artistic', 'comparison'],
    fact1: ['textbook', 'diagram'],
    fact2: ['diagram', 'textbook'],
    fact3: ['artistic', 'infographic'],
    wisdom: ['artistic', 'minimal'],
    outro: ['artistic'],
    complete: ['infographic', 'textbook']
  } as Record<string, string[]>,
  
  ALL_STYLES: ['artistic', 'textbook', 'diagram', 'minimal', 'infographic', 'comparison'] as const,
};

type Phase = typeof CONFIG.ALL_PHASES[number];
type Style = typeof CONFIG.ALL_STYLES[number];

// =============================================================================
// ENVIRONMENT
// =============================================================================

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';
const GOOGLE_API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_API_KEY || '';

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ Missing Supabase credentials');
  process.exit(1);
}

if (!GOOGLE_API_KEY) {
  console.error('❌ Missing Google API key');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// =============================================================================
// TYPES
// =============================================================================

interface QuizQuestion {
  question: string;
  options: string[];
  correct: string;
}

interface Misconception {
  misconception: string;
  correction: string;
}

interface FullLesson {
  day_number: number;
  topic: string;
  universal_truth: string;
  wow_moment: string;
  fun_facts: string[];
  extended_explanation: string;
  learning_objectives: string[];
  quick_quiz_questions: QuizQuestion[];
  common_misconceptions: Misconception[];
  real_world_applications: string[];
  marketing_headline: string;
  discussion_questions: string[];
}

// =============================================================================
// STYLE FOUNDATIONS
// =============================================================================

const STYLE_FOUNDATIONS: Record<Style, { prompt: string; includesText: boolean }> = {
  artistic: {
    prompt: `
VISUAL STYLE: Ultra Photorealistic Cinematic
- Professional photography, dramatic lighting
- Emotional resonance and visual storytelling
- Warm, inviting color palette
- 16:9 aspect ratio, 4K quality
- Leave right 30% simpler for text overlay
DO NOT include any text, logos, or watermarks.`,
    includesText: false
  },
  
  textbook: {
    prompt: `
VISUAL STYLE: Educational Textbook Illustration
- Professional educational illustration quality
- Clean, organized visual hierarchy
- Light background (white or cream)
- Include clear text labels and annotations
- 16:9 aspect ratio, print-ready quality

TEXT ELEMENTS:
- Title at top with topic name
- 3-5 clear labels pointing to key elements
- Brief explanatory caption
- Clear sans-serif fonts, high contrast`,
    includesText: true
  },
  
  diagram: {
    prompt: `
VISUAL STYLE: Technical Educational Diagram
- Clean, precise technical drawing
- Blueprint or schematic aesthetic
- Numbered/lettered components with legend
- Arrows showing flow, relationships, causation
- 16:9 aspect ratio

TEXT ELEMENTS:
- Component labels (1, 2, 3 or A, B, C)
- Key term labels with leader lines
- Directional arrows with brief annotations`,
    includesText: true
  },
  
  minimal: {
    prompt: `
VISUAL STYLE: Minimalist Concept
- Ultra-clean modern design
- Maximum 3 colors
- Single central concept
- Generous negative space
- 16:9 aspect ratio
DO NOT include any text.`,
    includesText: false
  },
  
  infographic: {
    prompt: `
VISUAL STYLE: Bold Infographic Design
- Eye-catching data visualization
- Clear visual hierarchy with icons
- Statistics displayed prominently
- Modern color scheme
- 16:9 aspect ratio

TEXT TO INCLUDE:
- Headline with topic
- 2-3 key statistics or facts
- Icons representing key concepts`,
    includesText: true
  },
  
  comparison: {
    prompt: `
VISUAL STYLE: Split Comparison Visual
- Side-by-side composition
- Clear visual contrast between two states
- Labels identifying each side
- 16:9 aspect ratio

TEXT ELEMENTS:
- Labels for each side (e.g., "MYTH" vs "REALITY")
- Key difference callouts`,
    includesText: true
  }
};

// =============================================================================
// PHASE-ALIGNED PROMPT BUILDERS
// =============================================================================

function buildHookPrompt(lesson: FullLesson, style: Style): string {
  const misconception = lesson.common_misconceptions?.[0];
  const styleConfig = STYLE_FOUNDATIONS[style];
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.prompt}

═══════════════════════════════════════════════════════════
PHASE: HOOK - Create Curiosity
═══════════════════════════════════════════════════════════

PURPOSE:
Spark curiosity and cognitive tension.
This is the OPENING - viewers should think "Wait, what?!"

ATTENTION GRABBER:
"${lesson.marketing_headline}"

COMMON MISCONCEPTION (hint that this might be wrong):
"${misconception?.misconception || 'Common assumptions about ' + lesson.topic}"

THE SURPRISE TO TEASE (don't reveal):
"${lesson.wow_moment}"

VISUAL DIRECTIVE:
Create visual MYSTERY. Show what people commonly believe,
but hint that something unexpected is about to be revealed.

CRITICAL: Create curiosity WITHOUT giving away the answer.`;
}

function buildCliffPrompt(lesson: FullLesson, style: Style): string {
  const misconception = lesson.common_misconceptions?.[0];
  const styleConfig = STYLE_FOUNDATIONS[style];
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.prompt}

═══════════════════════════════════════════════════════════
PHASE: CLIFF - The Plot Twist
═══════════════════════════════════════════════════════════

PURPOSE:
Deepen the mystery. Show CONTRAST between belief and reality.

MISCONCEPTION (what people wrongly believe):
"${misconception?.misconception}"

CORRECTION (the surprising truth):
"${misconception?.correction}"

VISUAL DIRECTIVE:
${style === 'comparison' ? 
`Create a SPLIT composition:
LEFT SIDE: "${misconception?.misconception}" (the wrong belief)
RIGHT SIDE: Hints at "${misconception?.correction}" (the truth)` :
`Show the tension between what people think and what's actually true.
Create visual anticipation for the reveal.`}

CRITICAL: Show the contrast that makes viewers lean in.`;
}

function buildFact1Prompt(lesson: FullLesson, style: Style): string {
  const q1 = lesson.quick_quiz_questions?.[0];
  const fact1 = lesson.fun_facts?.[0];
  const obj1 = lesson.learning_objectives?.[0];
  const styleConfig = STYLE_FOUNDATIONS[style];
  
  let questionBlock = '';
  if (q1) {
    const wrongOptions = q1.options.filter(o => o !== q1.correct);
    questionBlock = `
THIS VISUAL ANSWERS THE QUESTION:
"${q1.question}"

THE CORRECT ANSWER IS:
"${q1.correct}"

WRONG ANSWERS (do NOT illustrate):
${wrongOptions.map(o => `✗ "${o}"`).join('\n')}`;
  }
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.prompt}

═══════════════════════════════════════════════════════════
PHASE: FACT1 - First Key Concept
═══════════════════════════════════════════════════════════

PURPOSE:
TEACH the foundational concept with crystal clarity.

KEY FACT TO ILLUSTRATE:
"${fact1}"
${questionBlock}

LEARNING OBJECTIVE:
"${obj1}"

VISUAL DIRECTIVE:
${style === 'diagram' ? 
`Create a clear DIAGRAM showing:
- The concept visually represented
- Labeled components explaining the process
- Arrows or flow showing relationships
- Legend if needed` :
style === 'textbook' ?
`Create a TEXTBOOK-STYLE illustration:
- Title: "${lesson.topic}"
- Clear central illustration of the concept
- Labels pointing to key elements
- Caption explaining "${q1?.correct || fact1}"` :
`Illustrate the core concept so clearly that
a viewer can understand "${q1?.correct || fact1}"
just by looking at this image.`}

CRITICAL: The correct answer "${q1?.correct}" MUST be clearly illustrated.`;
}

function buildFact2Prompt(lesson: FullLesson, style: Style): string {
  const q2 = lesson.quick_quiz_questions?.[1];
  const fact2 = lesson.fun_facts?.[1];
  const styleConfig = STYLE_FOUNDATIONS[style];
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.prompt}

═══════════════════════════════════════════════════════════
PHASE: FACT2 - Deeper Understanding
═══════════════════════════════════════════════════════════

PURPOSE:
Go deeper. Show RELATIONSHIPS and CONNECTIONS.

DEEPER FACT:
"${fact2}"

${q2 ? `THIS ANSWERS:
"${q2.question}"
CORRECT: "${q2.correct}"` : ''}

EXTENDED CONTEXT:
"${lesson.extended_explanation?.substring(0, 400)}..."

VISUAL DIRECTIVE:
Show how concepts CONNECT. Use visual hierarchy
to show cause → effect relationships.
More detail than Fact1, revealing the "why".

CRITICAL: Must show relationships between concepts.`;
}

function buildFact3Prompt(lesson: FullLesson, style: Style): string {
  const fact3 = lesson.fun_facts?.[2] || lesson.fun_facts?.[3];
  const styleConfig = STYLE_FOUNDATIONS[style];
  
  // Extract statistics if present
  const numbers = lesson.wow_moment?.match(/\d+%?/g) || fact3?.match(/\d+%?/g) || [];
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.prompt}

═══════════════════════════════════════════════════════════
PHASE: FACT3 - The WOW Moment
═══════════════════════════════════════════════════════════

PURPOSE:
Create the MEMORABLE revelation. The "I had no idea!" moment.

THE WOW:
"${lesson.wow_moment}"

SUPPORTING DETAIL:
"${fact3}"

${numbers.length > 0 ? `KEY STATISTICS:
${numbers.map(n => `• ${n}`).join('\n')}` : ''}

VISUAL DIRECTIVE:
${style === 'infographic' ?
`Create a DATA VISUALIZATION that makes these statistics IMPACTFUL:
- Bold numbers that catch the eye
- Comparison or scale to show significance
- Visual metaphor for the concept` :
`Maximum visual IMPACT. This is the shareable moment.
Create something that makes viewers say "Wow, really?!"
Dramatic, memorable, share-worthy.`}

CRITICAL: Must create a "wow" reaction.`;
}

function buildWisdomPrompt(lesson: FullLesson, style: Style): string {
  const application = lesson.real_world_applications?.[0];
  const discussion = lesson.discussion_questions?.[0];
  const styleConfig = STYLE_FOUNDATIONS[style];
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.prompt}

═══════════════════════════════════════════════════════════
PHASE: WISDOM - Life Application
═══════════════════════════════════════════════════════════

PURPOSE:
Connect knowledge to REAL LIFE. Poster-worthy wisdom.

UNIVERSAL TRUTH:
"${lesson.universal_truth}"

REAL-WORLD APPLICATION:
"${application}"

REFLECTION:
"${discussion}"

VISUAL DIRECTIVE:
Create something INSPIRATIONAL and TIMELESS.
${style === 'minimal' ?
`Single powerful visual metaphor.
The simplest representation of the universal truth.
Elegant. Memorable. Iconic.` :
`Connect the concept to everyday human experience.
Show how this wisdom applies to real life.
Poster-worthy. Worth remembering.`}

CRITICAL: Must connect learning to real life.`;
}

function buildCompletePrompt(lesson: FullLesson, style: Style): string {
  const styleConfig = STYLE_FOUNDATIONS[style];
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.prompt}

═══════════════════════════════════════════════════════════
PHASE: COMPLETE - Comprehensive Summary
═══════════════════════════════════════════════════════════

PURPOSE:
ONE image that captures EVERYTHING. Reference-quality summary.

UNIVERSAL TRUTH:
"${lesson.universal_truth}"

LEARNING OBJECTIVES:
${lesson.learning_objectives?.slice(0, 3).map((o, i) => `${i + 1}. ${o}`).join('\n')}

KEY FACTS:
${lesson.fun_facts?.slice(0, 3).map(f => `• ${f}`).join('\n')}

WOW MOMENT:
"${lesson.wow_moment}"

VISUAL DIRECTIVE:
${style === 'infographic' ?
`Create a COMPREHENSIVE INFOGRAPHIC:
- Title: "${lesson.topic}"
- Visual representation of key concepts
- Statistics and key facts highlighted
- Learning objectives as visual checkpoints
- Universal truth as the anchor message` :
`Reference-quality comprehensive visual.
Include visual references to multiple key concepts.
The entire lesson captured in one shareable image.`}

CRITICAL: Must comprehensively represent the full lesson.`;
}

function buildOutroPrompt(lesson: FullLesson, style: Style): string {
  const styleConfig = STYLE_FOUNDATIONS[style];
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.prompt}

═══════════════════════════════════════════════════════════
PHASE: OUTRO - Celebration
═══════════════════════════════════════════════════════════

PURPOSE:
Celebrate completion. Mark achievement with forward energy.

ACHIEVEMENT:
${lesson.learning_objectives?.slice(0, 2).map(o => `✓ ${o}`).join('\n')}

VISUAL DIRECTIVE:
Create a sense of ACHIEVEMENT and FORWARD MOMENTUM.
The learner has grown. What's next?
Celebratory but not cheesy. Energizing.

CRITICAL: Must feel celebratory and forward-looking.`;
}

// =============================================================================
// PROMPT DISPATCHER
// =============================================================================

function buildPromptForPhase(lesson: FullLesson, phase: Phase, style: Style): string {
  switch (phase) {
    case 'hook': return buildHookPrompt(lesson, style);
    case 'cliff': return buildCliffPrompt(lesson, style);
    case 'fact1': return buildFact1Prompt(lesson, style);
    case 'fact2': return buildFact2Prompt(lesson, style);
    case 'fact3': return buildFact3Prompt(lesson, style);
    case 'wisdom': return buildWisdomPrompt(lesson, style);
    case 'complete': return buildCompletePrompt(lesson, style);
    case 'outro': return buildOutroPrompt(lesson, style);
  }
}

// =============================================================================
// HASH GENERATION
// =============================================================================

function generateHash(dayNumber: number, phase: string, style: string): string {
  const normalized = {
    d: dayNumber,
    p: phase.toLowerCase(),
    s: style,
    ver: '3' // v3 for phase-aligned prompts
  };
  
  const canonical = JSON.stringify(normalized, Object.keys(normalized).sort());
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

// =============================================================================
// IMAGEN GENERATION
// =============================================================================

async function generateWithImagen(prompt: string): Promise<Buffer | null> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${CONFIG.MODEL}:predict?key=${GOOGLE_API_KEY}`;
  
  const requestBody = {
    instances: [{ prompt }],
    parameters: {
      sampleCount: 1,
      aspectRatio: CONFIG.ASPECT_RATIO,
      personGeneration: 'dont_allow',
      safetySetting: 'block_low_and_above'
    }
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API Error: ${response.status}`);
      return null;
    }

    const data = await response.json() as any;
    
    if (data.predictions?.[0]?.bytesBase64Encoded) {
      return Buffer.from(data.predictions[0].bytesBase64Encoded, 'base64');
    }
    
    console.error('No image in response (content may have been filtered)');
    return null;
  } catch (error: any) {
    console.error('Generation error:', error.message);
    return null;
  }
}

// =============================================================================
// DATABASE OPERATIONS
// =============================================================================

async function getFullLesson(dayNumber: number): Promise<FullLesson | null> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select(`
      day_number, topic, universal_truth, wow_moment,
      fun_facts, extended_explanation, learning_objectives,
      quick_quiz_questions, common_misconceptions,
      real_world_applications, marketing_headline,
      discussion_questions
    `)
    .eq('day_number', dayNumber)
    .maybeSingle();
  
  if (error || !data) {
    console.error(`Failed to get lesson ${dayNumber}:`, error?.message);
    return null;
  }
  
  return data as FullLesson;
}

async function checkExisting(contentHash: string): Promise<boolean> {
  const { data } = await supabase
    .from('visual_commons')
    .select('id')
    .eq('content_hash', contentHash)
    .maybeSingle();
  
  return !!data;
}

async function uploadAndRegister(
  imageBuffer: Buffer,
  lesson: FullLesson,
  phase: Phase,
  style: Style,
  prompt: string,
  contentHash: string
): Promise<string | null> {
  const storagePath = `phase-aligned/${style}/${contentHash}.png`;
  
  const { error: uploadError } = await supabase.storage
    .from('visuals')
    .upload(storagePath, imageBuffer, {
      contentType: 'image/png',
      upsert: true
    });

  if (uploadError) {
    console.error('Upload error:', uploadError.message);
    return null;
  }

  const { data: urlData } = supabase.storage
    .from('visuals')
    .getPublicUrl(storagePath);

  const publicUrl = urlData.publicUrl;

  // Get aligned question if applicable
  let alignedQuestion = null;
  if (phase === 'fact1') alignedQuestion = lesson.quick_quiz_questions?.[0];
  if (phase === 'fact2') alignedQuestion = lesson.quick_quiz_questions?.[1];
  if (phase === 'fact3') alignedQuestion = lesson.quick_quiz_questions?.[2];

  const insertData: Record<string, any> = {
    content_hash: contentHash,
    day_number: lesson.day_number,
    phase,
    topic: lesson.topic,
    visual_type: 'scene',
    age_group: 'all',
    style: style,
    storage_path: storagePath,
    public_url: publicUrl,
    format: 'png',
    prompt_used: prompt.substring(0, 5000),
    model_used: CONFIG.MODEL,
    generation_params: { 
      aspectRatio: CONFIG.ASPECT_RATIO,
      style,
      phaseAligned: true,
      promptVersion: '3',
      alignedQuestion: alignedQuestion?.question || null,
      alignedAnswer: alignedQuestion?.correct || null
    },
    estimated_cost: CONFIG.COST_PER_IMAGE,
    generated_by: null,
    generated_by_display_name: 'Curious Kelly Team',
    generation_source: 'seed',
    status: 'active'
  };

  const { error: insertError } = await supabase
    .from('visual_commons')
    .upsert(insertData, { onConflict: 'content_hash' });

  if (insertError) {
    console.error('Insert error:', insertError.message);
    return null;
  }

  return publicUrl;
}

// =============================================================================
// MAIN LOGIC
// =============================================================================

async function generateVariant(
  lesson: FullLesson,
  phase: Phase,
  style: Style,
  dryRun: boolean
): Promise<{ success: boolean; cost: number; skipped?: boolean }> {
  const contentHash = generateHash(lesson.day_number, phase, style);

  if (await checkExisting(contentHash)) {
    console.log(`      ⏭️  ${style}: Already exists`);
    return { success: true, cost: 0, skipped: true };
  }

  console.log(`      🎨 ${style}: Generating...`);

  if (dryRun) {
    return { success: true, cost: CONFIG.COST_PER_IMAGE };
  }

  const prompt = buildPromptForPhase(lesson, phase, style);
  const imageBuffer = await generateWithImagen(prompt);
  
  if (!imageBuffer) {
    console.log(`         ❌ Failed`);
    return { success: false, cost: CONFIG.COST_PER_IMAGE };
  }

  const publicUrl = await uploadAndRegister(
    imageBuffer,
    lesson,
    phase,
    style,
    prompt,
    contentHash
  );

  if (!publicUrl) {
    return { success: false, cost: CONFIG.COST_PER_IMAGE };
  }

  console.log(`         ✅ Saved`);
  return { success: true, cost: CONFIG.COST_PER_IMAGE };
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// CLI
// =============================================================================

function parseArgs() {
  const args = process.argv.slice(2);
  let days: number[] = [];
  let phases: Phase[] = [...CONFIG.KEY_PHASES];
  let useOptimalStyles = true;
  let allStyles = false;
  let dryRun = false;

  for (const arg of args) {
    if (arg === '--dry-run') dryRun = true;
    else if (arg === '--all-phases') phases = [...CONFIG.ALL_PHASES];
    else if (arg === '--all-styles') { allStyles = true; useOptimalStyles = false; }
    else if (arg.startsWith('--day=')) {
      const day = parseInt(arg.split('=')[1]);
      if (day >= 1 && day <= 365) days.push(day);
    }
    else if (arg.startsWith('--range=')) {
      const [start, end] = arg.split('=')[1].split('-').map(Number);
      for (let d = start; d <= end && d <= 365; d++) {
        if (d >= 1) days.push(d);
      }
    }
  }

  if (days.length === 0) days = [1];
  
  return { days, phases, useOptimalStyles, allStyles, dryRun };
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('🎨 PHASE-ALIGNED VISUAL GENERATOR');
  console.log('━'.repeat(60));
  console.log(`💎 Model: ${CONFIG.MODEL}`);
  console.log(`💰 Cost: $${CONFIG.COST_PER_IMAGE.toFixed(2)} per image`);
  console.log('');

  const { days, phases, useOptimalStyles, allStyles, dryRun } = parseArgs();

  if (dryRun) console.log('🔍 DRY RUN MODE');
  if (useOptimalStyles) console.log('🎯 Using phase-optimal styles');

  console.log(`📅 Days: ${days.join(', ')}`);
  console.log(`📝 Phases: ${phases.join(', ')}`);
  console.log('');

  if (!dryRun) {
    console.log('⏳ Starting in 3 seconds...');
    await sleep(3000);
  }

  let totalSuccess = 0, totalFailed = 0, totalSkipped = 0, totalCost = 0;

  for (const day of days) {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`📅 DAY ${day}`);
    
    const lesson = await getFullLesson(day);
    if (!lesson) {
      console.log(`❌ Lesson not found`);
      continue;
    }
    
    console.log(`📚 Topic: "${lesson.topic}"`);
    console.log(`💡 Truth: "${lesson.universal_truth?.substring(0, 60)}..."`);
    console.log(`❓ Quiz Questions: ${lesson.quick_quiz_questions?.length || 0}`);
    console.log('');

    for (const phase of phases) {
      console.log(`  📝 Phase: ${phase.toUpperCase()}`);
      
      // Determine styles for this phase
      const styles: Style[] = useOptimalStyles 
        ? (CONFIG.PHASE_STYLES[phase] as Style[]) || ['artistic']
        : allStyles 
          ? [...CONFIG.ALL_STYLES]
          : ['artistic', 'textbook'];

      for (const style of styles) {
        const result = await generateVariant(lesson, phase, style, dryRun);
        totalCost += result.cost;
        if (result.skipped) totalSkipped++;
        else if (result.success) totalSuccess++;
        else totalFailed++;
        
        if (!dryRun && result.success && !result.skipped) {
          await sleep(CONFIG.DELAY_BETWEEN_IMAGES_MS);
        }
      }
    }
  }

  console.log('\n' + '═'.repeat(60));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${totalSuccess}`);
  console.log(`⏭️  Skipped: ${totalSkipped}`);
  console.log(`❌ Failed: ${totalFailed}`);
  console.log(`💰 Cost: $${totalCost.toFixed(2)}`);

  const manifestPath = path.join(process.cwd(), 'generated-visuals', 'phase-aligned-manifest.json');
  fs.mkdirSync(path.dirname(manifestPath), { recursive: true });
  fs.writeFileSync(manifestPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    model: CONFIG.MODEL,
    totalSuccess,
    totalFailed,
    totalSkipped,
    totalCost,
    days,
    phases,
    promptVersion: '3-phase-aligned'
  }, null, 2));
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
