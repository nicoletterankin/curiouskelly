#!/usr/bin/env npx tsx
/**
 * VISUAL COMMONS SEED GENERATOR
 * 
 * Generates high-quality educational visuals using Imagen 4 Ultra
 * to bootstrap the Visual Commons with premium content.
 * 
 * COST: $0.06 per image (Imagen 4 Ultra)
 * 
 * Usage:
 *   npx tsx scripts/seed-visual-commons.ts --day=1
 *   npx tsx scripts/seed-visual-commons.ts --range=1-30
 *   npx tsx scripts/seed-visual-commons.ts --range=1-365 --phases=hook,wisdom
 *   npx tsx scripts/seed-visual-commons.ts --priority  # High-traffic days only
 * 
 * @created December 17, 2025
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // Imagen 4 Ultra - Highest quality
  MODEL: 'imagen-3.0-generate-002', // Latest Imagen model
  COST_PER_IMAGE: 0.06,
  
  // Output settings
  ASPECT_RATIO: '16:9',
  
  // Rate limiting
  DELAY_BETWEEN_IMAGES_MS: 2000,
  
  // Phases to generate (in priority order)
  DEFAULT_PHASES: ['hook', 'fact1', 'wisdom', 'complete'] as const,
  ALL_PHASES: ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro', 'complete'] as const,
  
  // High-priority days (first week, popular topics)
  PRIORITY_DAYS: [1, 2, 3, 4, 5, 6, 7, 17, 42, 100, 200, 300, 365],
};

type Phase = typeof CONFIG.ALL_PHASES[number];

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
  console.error('❌ Missing GEMINI_API_KEY or GOOGLE_AI_API_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);

// =============================================================================
// TYPES
// =============================================================================

interface Lesson {
  day_number: number;
  topic: string;
  universal_truth: string;
  facts?: string[];
}

interface GenerationResult {
  success: boolean;
  dayNumber: number;
  phase: string;
  publicUrl?: string;
  error?: string;
  cost: number;
}

// =============================================================================
// CONTENT HASH
// =============================================================================

function generateVisualHash(context: {
  dayNumber: number;
  phase: string;
  ageGroup?: string;
  visualType?: string;
  style?: string;
}): string {
  const normalized = {
    d: context.dayNumber,
    p: context.phase.toLowerCase(),
    a: context.ageGroup || 'all',
    t: context.visualType || 'scene',
    s: context.style || 'default',
    ver: '1'
  };
  
  const canonical = JSON.stringify(normalized, Object.keys(normalized).sort());
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

// =============================================================================
// PROMPT GENERATION
// =============================================================================

const PHASE_PROMPTS: Record<Phase, (lesson: Lesson) => string> = {
  hook: (lesson) => `
Create a stunning, curiosity-sparking educational scene for: "${lesson.topic}"

This is the OPENING moment of a lesson. The goal is to make viewers say "Wait, what?!"

Key insight to hint at: ${lesson.universal_truth}

STYLE:
- Ultra photorealistic, professional photography aesthetic
- Dramatic lighting, cinematic composition
- 16:9 aspect ratio, 4K quality
- Warm, inviting color palette with a sense of wonder
- Leave right 35% of frame with simpler background (for overlay content)

COMPOSITION:
- Subject prominently displayed but mysterious
- Capture the moment of discovery or question
- Include visual details that spark curiosity without revealing answers
- Think "National Geographic meets Pixar magic"

DO NOT include any text, logos, or watermarks.
Create an image that makes someone stop scrolling and want to learn more.
`,

  cliff: (lesson) => `
Create an intriguing educational scene that deepens mystery for: "${lesson.topic}"

This shows the TENSION between what we think we know and surprising reality.

Universal truth: ${lesson.universal_truth}

STYLE:
- Ultra photorealistic, slightly dramatic lighting
- Sense of revelation or "plot twist" moment
- 16:9 aspect ratio, 4K quality
- Visual contrast between expectation and reality

COMPOSITION:
- Show duality or transformation
- Hint at hidden complexity
- Create visual tension that needs resolution

DO NOT include any text. Create pure visual storytelling.
`,

  fact1: (lesson) => `
Create a clear, educational scene illustrating the FIRST key concept of: "${lesson.topic}"

This is TEACHING content - clarity is everything.

Core concept: ${lesson.facts?.[0] || lesson.universal_truth}

STYLE:
- Ultra photorealistic, bright and clear lighting
- Educational but beautiful - think museum exhibit quality
- 16:9 aspect ratio, 4K quality
- Clean, organized composition

COMPOSITION:
- Main concept clearly visible and understandable
- Include helpful visual details that teach
- A viewer should grasp the concept at a glance
- Leave right 35% for overlay content

DO NOT include any text. The image should teach through visuals alone.
`,

  fact2: (lesson) => `
Create an educational scene showing DEEPER insight into: "${lesson.topic}"

Building on foundational knowledge with: ${lesson.facts?.[1] || 'deeper understanding'}

STYLE:
- Ultra photorealistic, layered lighting showing depth
- More detailed than fact1 - reveals complexity
- 16:9 aspect ratio, 4K quality

COMPOSITION:
- Show relationships and connections
- Zoom into detail or expand to show context
- Visual progression from simple to complex

DO NOT include any text.
`,

  fact3: (lesson) => `
Create a WOW MOMENT scene for: "${lesson.topic}"

This is the SURPRISING detail that makes the lesson memorable.

Wow factor: ${lesson.facts?.[2] || lesson.universal_truth}

STYLE:
- Ultra photorealistic, dramatic "reveal" lighting
- Maximum visual impact
- 16:9 aspect ratio, 4K quality
- Bold, memorable composition

COMPOSITION:
- Capture the "mind-blown" moment
- Show the unexpected truth
- Create an image worth sharing

DO NOT include any text. Pure visual impact.
`,

  wisdom: (lesson) => `
Create an inspiring, LIFE APPLICATION scene for: "${lesson.topic}"

Universal truth: ${lesson.universal_truth}

This visual should feel like a POSTER ON THE WALL - wisdom worth remembering.

STYLE:
- Ultra photorealistic, warm golden hour or inspirational lighting
- Timeless, universal appeal
- 16:9 aspect ratio, 4K quality
- Emotionally resonant

COMPOSITION:
- Connect the concept to everyday life
- Show human element or relatable context
- Create a feeling of insight and growth
- Suitable for sharing on social media

DO NOT include any text. Let the image speak wisdom.
`,

  outro: (lesson) => `
Create a CELEBRATORY scene marking completion of learning about: "${lesson.topic}"

STYLE:
- Ultra photorealistic, bright and uplifting
- Sense of achievement and forward momentum
- 16:9 aspect ratio, 4K quality
- Energetic, positive vibes

COMPOSITION:
- Visual closure with hint of "what's next"
- Celebratory without being cheesy
- Forward-looking energy

DO NOT include any text.
`,

  complete: (lesson) => `
Create a COMPREHENSIVE SUMMARY scene for the entire lesson on: "${lesson.topic}"

Universal truth: ${lesson.universal_truth}

This is the ONE IMAGE that captures everything - shareable, memorable, complete.

STYLE:
- Ultra photorealistic, rich and detailed
- Museum-quality educational art
- 16:9 aspect ratio, 4K quality
- Multiple layers of detail that reward closer inspection

COMPOSITION:
- Capture the full scope of the topic
- Include visual references to key concepts
- Create an image that teaches the whole lesson at a glance
- Suitable for printing or sharing

DO NOT include any text. The ultimate visual summary.
`
};

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
      personGeneration: 'dont_allow', // Educational content, avoid people issues
      safetySetting: 'block_only_high'
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
      console.error(`API Error: ${response.status} - ${errorText}`);
      return null;
    }

    const data = await response.json() as any;
    
    if (data.predictions?.[0]?.bytesBase64Encoded) {
      return Buffer.from(data.predictions[0].bytesBase64Encoded, 'base64');
    }
    
    console.error('No image in response:', JSON.stringify(data, null, 2));
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
    .select('day_number, topic, universal_truth, facts')
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
  contentHash: string
): Promise<string | null> {
  // Upload to storage
  const storagePath = `seed/${contentHash}.png`;
  
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

  // Get public URL
  const { data: urlData } = supabase.storage
    .from('visuals')
    .getPublicUrl(storagePath);

  const publicUrl = urlData.publicUrl;

  // Register in database
  const { error: insertError } = await supabase
    .from('visual_commons')
    .upsert({
      content_hash: contentHash,
      day_number: dayNumber,
      phase,
      topic,
      visual_type: 'scene',
      age_group: 'all',
      style: 'default',
      storage_path: storagePath,
      public_url: publicUrl,
      format: 'png',
      prompt_used: prompt.substring(0, 5000),
      model_used: CONFIG.MODEL,
      generation_params: { aspectRatio: CONFIG.ASPECT_RATIO },
      estimated_cost: CONFIG.COST_PER_IMAGE,
      generated_by: null,
      generated_by_display_name: 'Curious Kelly Team',
      generation_source: 'seed',
      status: 'active'
    }, { onConflict: 'content_hash' });

  if (insertError) {
    console.error('Insert error:', insertError.message);
    return null;
  }

  return publicUrl;
}

// =============================================================================
// MAIN GENERATION LOGIC
// =============================================================================

async function generateForDayPhase(
  dayNumber: number,
  phase: Phase,
  lesson: Lesson,
  dryRun: boolean
): Promise<GenerationResult> {
  const contentHash = generateVisualHash({
    dayNumber,
    phase,
    ageGroup: 'all',
    visualType: 'scene',
    style: 'default'
  });

  // Check if already exists
  if (await checkExisting(contentHash)) {
    console.log(`  ⏭️  ${phase}: Already exists`);
    return { success: true, dayNumber, phase, cost: 0 };
  }

  console.log(`  🎨 ${phase}: Generating...`);

  if (dryRun) {
    console.log(`     [DRY RUN] Would generate with Imagen 4 Ultra`);
    return { success: true, dayNumber, phase, cost: CONFIG.COST_PER_IMAGE };
  }

  // Generate prompt
  const promptFn = PHASE_PROMPTS[phase];
  const prompt = promptFn(lesson);

  // Generate image
  const imageBuffer = await generateWithImagen(prompt);
  
  if (!imageBuffer) {
    console.log(`     ❌ Generation failed`);
    return { success: false, dayNumber, phase, error: 'Generation failed', cost: CONFIG.COST_PER_IMAGE };
  }

  // Upload and register
  const publicUrl = await uploadAndRegister(
    imageBuffer,
    dayNumber,
    phase,
    lesson.topic,
    prompt,
    contentHash
  );

  if (!publicUrl) {
    console.log(`     ❌ Upload failed`);
    return { success: false, dayNumber, phase, error: 'Upload failed', cost: CONFIG.COST_PER_IMAGE };
  }

  console.log(`     ✅ Saved: ${publicUrl.split('/').pop()}`);
  return { success: true, dayNumber, phase, publicUrl, cost: CONFIG.COST_PER_IMAGE };
}

async function generateForDay(
  dayNumber: number,
  phases: Phase[],
  dryRun: boolean
): Promise<GenerationResult[]> {
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📅 DAY ${dayNumber}`);
  
  const lesson = await getLessonDetails(dayNumber);
  if (!lesson) {
    console.log(`❌ Lesson not found`);
    return [];
  }
  
  console.log(`📚 Topic: ${lesson.topic}`);
  console.log(`💡 Truth: ${lesson.universal_truth?.substring(0, 80)}...`);
  console.log('');

  const results: GenerationResult[] = [];

  for (const phase of phases) {
    const result = await generateForDayPhase(dayNumber, phase, lesson, dryRun);
    results.push(result);
    
    if (!dryRun && result.success) {
      await sleep(CONFIG.DELAY_BETWEEN_IMAGES_MS);
    }
  }

  return results;
}

// =============================================================================
// CLI PARSING
// =============================================================================

function parseArgs(): {
  days: number[];
  phases: Phase[];
  dryRun: boolean;
} {
  const args = process.argv.slice(2);
  let days: number[] = [];
  let phases: Phase[] = [...CONFIG.DEFAULT_PHASES];
  let dryRun = false;

  for (const arg of args) {
    if (arg === '--dry-run') {
      dryRun = true;
    } else if (arg === '--priority') {
      days = [...CONFIG.PRIORITY_DAYS];
    } else if (arg === '--all-phases') {
      phases = [...CONFIG.ALL_PHASES];
    } else if (arg.startsWith('--day=')) {
      const day = parseInt(arg.split('=')[1]);
      if (day >= 1 && day <= 365) days.push(day);
    } else if (arg.startsWith('--range=')) {
      const [start, end] = arg.split('=')[1].split('-').map(Number);
      for (let d = start; d <= end && d <= 365; d++) {
        if (d >= 1) days.push(d);
      }
    } else if (arg.startsWith('--phases=')) {
      const phaseList = arg.split('=')[1].split(',') as Phase[];
      phases = phaseList.filter(p => CONFIG.ALL_PHASES.includes(p));
    }
  }

  // Default to day 1 if nothing specified
  if (days.length === 0) {
    days = [1];
  }

  return { days, phases, dryRun };
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('🎨 VISUAL COMMONS SEED GENERATOR');
  console.log('━'.repeat(60));
  console.log(`💎 Model: Imagen 4 Ultra (${CONFIG.MODEL})`);
  console.log(`💰 Cost: $${CONFIG.COST_PER_IMAGE.toFixed(2)} per image`);
  console.log('');

  const { days, phases, dryRun } = parseArgs();

  if (dryRun) {
    console.log('🔍 DRY RUN MODE - No actual generation');
  }

  console.log(`📅 Days: ${days.length} (${days.slice(0, 5).join(', ')}${days.length > 5 ? '...' : ''})`);
  console.log(`🎯 Phases: ${phases.join(', ')}`);
  
  const estimatedImages = days.length * phases.length;
  const estimatedCost = estimatedImages * CONFIG.COST_PER_IMAGE;
  console.log(`📊 Estimated: ${estimatedImages} images = $${estimatedCost.toFixed(2)}`);
  console.log('');

  if (!dryRun) {
    console.log('⏳ Starting in 3 seconds... (Ctrl+C to cancel)');
    await sleep(3000);
  }

  const allResults: GenerationResult[] = [];
  let totalCost = 0;
  let successCount = 0;
  let failCount = 0;
  let skipCount = 0;

  for (const day of days) {
    const results = await generateForDay(day, phases, dryRun);
    allResults.push(...results);
    
    for (const result of results) {
      totalCost += result.cost;
      if (result.success && result.publicUrl) successCount++;
      else if (result.success) skipCount++;
      else failCount++;
    }
  }

  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log('📊 GENERATION SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${successCount}`);
  console.log(`⏭️  Skipped (existing): ${skipCount}`);
  console.log(`❌ Failed: ${failCount}`);
  console.log(`💰 Total cost: $${totalCost.toFixed(2)}`);
  
  if (dryRun) {
    console.log('\n🔍 This was a DRY RUN. Run without --dry-run to generate.');
  }

  // Save manifest
  const manifestPath = path.join(process.cwd(), 'generated-visuals', 'seed-manifest.json');
  fs.mkdirSync(path.dirname(manifestPath), { recursive: true });
  fs.writeFileSync(manifestPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    model: CONFIG.MODEL,
    totalCost,
    results: allResults
  }, null, 2));
  console.log(`\n📄 Manifest saved: ${manifestPath}`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
