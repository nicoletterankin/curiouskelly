#!/usr/bin/env npx tsx
/**
 * VISUAL VARIANTS SEED GENERATOR
 * 
 * Generates multiple visual styles for each lesson phase:
 * - ARTISTIC: Photorealistic, cinematic (no text)
 * - TEXTBOOK: Educational illustration with labels
 * - DIAGRAM: Technical diagram with annotations
 * - MINIMAL: Simple, clean, one concept
 * 
 * Usage:
 *   npx tsx scripts/seed-visual-variants.ts --day=1 --phase=hook
 *   npx tsx scripts/seed-visual-variants.ts --day=1 --all-phases --all-styles
 *   npx tsx scripts/seed-visual-variants.ts --range=1-7 --styles=artistic,textbook
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
  DELAY_BETWEEN_IMAGES_MS: 2000,
  
  ALL_PHASES: ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro', 'complete'] as const,
  DEFAULT_PHASES: ['hook', 'fact1', 'wisdom', 'complete'] as const,
  
  ALL_STYLES: ['artistic', 'textbook', 'diagram', 'minimal'] as const,
  DEFAULT_STYLES: ['artistic', 'textbook'] as const,
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

interface Lesson {
  day_number: number;
  topic: string;
  universal_truth: string;
  fun_facts?: string[];
  wow_moment?: string;
}

// =============================================================================
// STYLE-SPECIFIC PROMPT TEMPLATES
// =============================================================================

const STYLE_PROMPTS: Record<Style, {
  foundation: string;
  includesText: 'none' | 'labels' | 'full';
  complexity: 'standard' | 'detailed';
}> = {
  artistic: {
    foundation: `
VISUAL STYLE: Ultra Photorealistic Artistic
- Professional photography aesthetic, cinematic quality
- Dramatic lighting with natural feel
- Warm, inviting color palette
- Emotional resonance and wonder
- 16:9 aspect ratio, 4K resolution
- Leave right 35% simpler for overlay content

DO NOT include any text, logos, or watermarks.`,
    includesText: 'none',
    complexity: 'standard'
  },
  
  textbook: {
    foundation: `
VISUAL STYLE: Educational Textbook Page
- Professional educational illustration quality
- Clear, organized visual hierarchy
- Clean light background (white or cream)
- Suitable for printing in a textbook
- 16:9 aspect ratio, high resolution

TEXT ELEMENTS TO INCLUDE:
- Title at top: The topic name
- 2-4 clear labels pointing to key elements
- Optional brief caption at bottom
- Use clear, legible sans-serif fonts
- High contrast text on light background`,
    includesText: 'labels',
    complexity: 'detailed'
  },
  
  diagram: {
    foundation: `
VISUAL STYLE: Technical Educational Diagram
- Clean, precise technical drawing style
- Blueprint or schematic aesthetic with educational warmth
- Clearly numbered or lettered components
- Arrows showing relationships, flow, or processes
- Include a small legend/key if helpful
- 16:9 aspect ratio

TEXT ELEMENTS:
- Component labels (A, B, C or 1, 2, 3)
- Key term labels
- Directional arrows with brief annotations
- Use clean, technical typography`,
    includesText: 'labels',
    complexity: 'detailed'
  },
  
  minimal: {
    foundation: `
VISUAL STYLE: Minimalist Concept
- Ultra-clean, modern minimalist design
- Maximum 3 colors
- Single central concept, no clutter
- Generous negative space
- Elegant simplicity
- 16:9 aspect ratio

DO NOT include any text.
Think Apple design principles meets educational clarity.`,
    includesText: 'none',
    complexity: 'standard'
  }
};

// =============================================================================
// PHASE-SPECIFIC CONTENT
// =============================================================================

function getPhaseContent(phase: Phase, lesson: Lesson): string {
  const phaseMap: Record<Phase, string> = {
    hook: `
PURPOSE: Opening Hook - Spark curiosity
Create a visual that makes viewers say "Wait, what?!"
Topic: "${lesson.topic}"
Hint at: ${lesson.universal_truth.substring(0, 100)}
Make it mysterious and intriguing.`,

    cliff: `
PURPOSE: Cliffhanger - Deepen mystery
Show the tension between expectation and reality.
Topic: "${lesson.topic}"
Truth to hint at: ${lesson.universal_truth}`,

    fact1: `
PURPOSE: First Key Concept - Clear teaching
Illustrate the foundational idea clearly.
Topic: "${lesson.topic}"
Core concept: ${lesson.fun_facts?.[0] || lesson.universal_truth}
Maximum clarity - understand at a glance.`,

    fact2: `
PURPOSE: Deeper Understanding
Build on the foundation with more detail.
Topic: "${lesson.topic}"
Deeper concept: ${lesson.fun_facts?.[1] || 'Show relationships and connections'}`,

    fact3: `
PURPOSE: Wow Moment - The surprising detail
The memorable, shareable revelation.
Topic: "${lesson.topic}"
Wow factor: ${lesson.wow_moment || lesson.fun_facts?.[2] || 'Most surprising aspect'}`,

    wisdom: `
PURPOSE: Life Application - Universal wisdom
Timeless truth worth remembering.
Topic: "${lesson.topic}"
Universal truth: ${lesson.universal_truth}
Create something poster-worthy.`,

    outro: `
PURPOSE: Celebration & Closure
Mark completion with forward energy.
Topic: "${lesson.topic}"
Sense of achievement and "what's next" feeling.`,

    complete: `
PURPOSE: Complete Summary - One comprehensive image
Captures the entire lesson at a glance.
Topic: "${lesson.topic}"
Universal truth: ${lesson.universal_truth}
Reference multiple key concepts. Shareable and memorable.`
  };
  
  return phaseMap[phase];
}

// =============================================================================
// BUILD VARIANT PROMPT
// =============================================================================

function buildPrompt(lesson: Lesson, phase: Phase, style: Style): string {
  const styleConfig = STYLE_PROMPTS[style];
  const phaseContent = getPhaseContent(phase, lesson);
  
  return `Create an educational visual for: "${lesson.topic}"

${styleConfig.foundation}

${phaseContent}

IMPORTANT GUIDELINES:
- Educational accuracy is paramount
- No copyrighted characters or logos
- Safe for all ages
- Culturally inclusive`;
}

// =============================================================================
// CONTENT HASH FOR VARIANTS
// =============================================================================

function generateVariantHash(context: {
  dayNumber: number;
  phase: string;
  style: string;
  complexity: string;
  includesText: string;
}): string {
  const normalized = {
    d: context.dayNumber,
    p: context.phase.toLowerCase(),
    s: context.style,
    c: context.complexity,
    t: context.includesText,
    a: 'all',
    ver: '2'
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
      console.error(`API Error: ${response.status} - ${errorText.substring(0, 200)}`);
      return null;
    }

    const data = await response.json() as any;
    
    if (data.predictions?.[0]?.bytesBase64Encoded) {
      return Buffer.from(data.predictions[0].bytesBase64Encoded, 'base64');
    }
    
    console.error('No image in response');
    return null;
  } catch (error: any) {
    console.error('Generation error:', error.message);
    return null;
  }
}

// =============================================================================
// DATABASE OPERATIONS
// =============================================================================

async function getLessonDetails(dayNumber: number): Promise<Lesson | null> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('day_number, topic, universal_truth, fun_facts, wow_moment')
    .eq('day_number', dayNumber)
    .maybeSingle();
  
  if (error || !data) {
    console.error(`Failed to get lesson ${dayNumber}:`, error?.message);
    return null;
  }
  
  return data as Lesson;
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
  dayNumber: number,
  phase: string,
  topic: string,
  prompt: string,
  contentHash: string,
  style: Style
): Promise<string | null> {
  const styleConfig = STYLE_PROMPTS[style];
  const storagePath = `variants/${style}/${contentHash}.png`;
  
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

  // Build insert object - only include columns that exist
  const insertData: Record<string, any> = {
    content_hash: contentHash,
    day_number: dayNumber,
    phase,
    topic,
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
      complexity: styleConfig.complexity,
      includesText: styleConfig.includesText 
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
// MAIN GENERATION LOGIC
// =============================================================================

async function generateVariant(
  dayNumber: number,
  phase: Phase,
  style: Style,
  lesson: Lesson,
  dryRun: boolean
): Promise<{ success: boolean; cost: number }> {
  const styleConfig = STYLE_PROMPTS[style];
  const contentHash = generateVariantHash({
    dayNumber,
    phase,
    style,
    complexity: styleConfig.complexity,
    includesText: styleConfig.includesText
  });

  if (await checkExisting(contentHash)) {
    console.log(`    ⏭️  ${style}: Already exists`);
    return { success: true, cost: 0 };
  }

  console.log(`    🎨 ${style}: Generating...`);

  if (dryRun) {
    console.log(`       [DRY RUN] Would generate ${style} variant`);
    return { success: true, cost: CONFIG.COST_PER_IMAGE };
  }

  const prompt = buildPrompt(lesson, phase, style);
  const imageBuffer = await generateWithImagen(prompt);
  
  if (!imageBuffer) {
    console.log(`       ❌ Generation failed`);
    return { success: false, cost: CONFIG.COST_PER_IMAGE };
  }

  const publicUrl = await uploadAndRegister(
    imageBuffer,
    dayNumber,
    phase,
    lesson.topic,
    prompt,
    contentHash,
    style
  );

  if (!publicUrl) {
    console.log(`       ❌ Upload failed`);
    return { success: false, cost: CONFIG.COST_PER_IMAGE };
  }

  console.log(`       ✅ Saved: ${style}/${contentHash.substring(0, 8)}...`);
  return { success: true, cost: CONFIG.COST_PER_IMAGE };
}

async function generateForPhase(
  dayNumber: number,
  phase: Phase,
  styles: Style[],
  lesson: Lesson,
  dryRun: boolean
): Promise<{ success: number; failed: number; cost: number }> {
  console.log(`  📝 Phase: ${phase}`);
  
  let success = 0, failed = 0, cost = 0;
  
  for (const style of styles) {
    const result = await generateVariant(dayNumber, phase, style, lesson, dryRun);
    cost += result.cost;
    if (result.success) success++; else failed++;
    
    if (!dryRun && result.success) {
      await sleep(CONFIG.DELAY_BETWEEN_IMAGES_MS);
    }
  }
  
  return { success, failed, cost };
}

// =============================================================================
// CLI & MAIN
// =============================================================================

function parseArgs(): {
  days: number[];
  phases: Phase[];
  styles: Style[];
  dryRun: boolean;
} {
  const args = process.argv.slice(2);
  let days: number[] = [];
  let phases: Phase[] = [...CONFIG.DEFAULT_PHASES];
  let styles: Style[] = [...CONFIG.DEFAULT_STYLES];
  let dryRun = false;

  for (const arg of args) {
    if (arg === '--dry-run') dryRun = true;
    else if (arg === '--all-phases') phases = [...CONFIG.ALL_PHASES];
    else if (arg === '--all-styles') styles = [...CONFIG.ALL_STYLES];
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
    else if (arg.startsWith('--phases=')) {
      phases = arg.split('=')[1].split(',') as Phase[];
    }
    else if (arg.startsWith('--styles=')) {
      styles = arg.split('=')[1].split(',') as Style[];
    }
  }

  if (days.length === 0) days = [1];
  return { days, phases, styles, dryRun };
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  console.log('🎨 VISUAL VARIANTS SEED GENERATOR');
  console.log('━'.repeat(60));
  console.log(`💎 Model: ${CONFIG.MODEL}`);
  console.log(`💰 Cost: $${CONFIG.COST_PER_IMAGE.toFixed(2)} per image`);
  console.log('');

  const { days, phases, styles, dryRun } = parseArgs();

  if (dryRun) console.log('🔍 DRY RUN MODE');

  console.log(`📅 Days: ${days.length}`);
  console.log(`🎯 Phases: ${phases.join(', ')}`);
  console.log(`🎨 Styles: ${styles.join(', ')}`);
  
  const estimatedImages = days.length * phases.length * styles.length;
  const estimatedCost = estimatedImages * CONFIG.COST_PER_IMAGE;
  console.log(`📊 Estimated: ${estimatedImages} images = $${estimatedCost.toFixed(2)}`);
  console.log('');

  if (!dryRun) {
    console.log('⏳ Starting in 3 seconds...');
    await sleep(3000);
  }

  let totalSuccess = 0, totalFailed = 0, totalCost = 0;

  for (const day of days) {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`📅 DAY ${day}`);
    
    const lesson = await getLessonDetails(day);
    if (!lesson) {
      console.log(`❌ Lesson not found`);
      continue;
    }
    
    console.log(`📚 Topic: ${lesson.topic}`);
    console.log('');

    for (const phase of phases) {
      const result = await generateForPhase(day, phase, styles, lesson, dryRun);
      totalSuccess += result.success;
      totalFailed += result.failed;
      totalCost += result.cost;
    }
  }

  console.log('\n' + '═'.repeat(60));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${totalSuccess}`);
  console.log(`❌ Failed: ${totalFailed}`);
  console.log(`💰 Cost: $${totalCost.toFixed(2)}`);

  const manifestPath = path.join(process.cwd(), 'generated-visuals', 'variants-manifest.json');
  fs.mkdirSync(path.dirname(manifestPath), { recursive: true });
  fs.writeFileSync(manifestPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    totalSuccess,
    totalFailed,
    totalCost,
    days,
    phases,
    styles
  }, null, 2));
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});

