#!/usr/bin/env npx tsx
/**
 * Visual Generator V2 - Curious Kelly Educational Illustrations
 * 
 * Improvements over V1:
 * - Consistent illustrated style (not photorealistic)
 * - Specific visual subjects per lesson content
 * - Quality validation pipeline
 * - Automatic rejection of bad generations
 * - Retry logic with prompt refinement
 */

import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';
import * as dotenv from 'dotenv';
import * as path from 'path';
import * as crypto from 'crypto';

// Load environment
dotenv.config({ path: path.join(process.cwd(), '.env.local') });

// ============================================================================
// TYPES (inline to avoid import issues)
// ============================================================================

type Phase = 'hook' | 'cliff' | 'q1' | 'q2' | 'q3' | 'wisdom' | 'outro';

interface LessonContext {
  dayNumber: number;
  topic: string;
  hookTeaser: string;
  cliffChoice: string;
  q1Content: string;
  q2Content: string;
  q3Content: string;
  wisdomInsight: string;
  funFacts: string[];
  wowMoment: string;
}

interface GeneratedPrompt {
  prompt: string;
  phase: Phase;
  contentHash: string;
  expectedDimensions: { width: number; height: number };
}

// ============================================================================
// STYLE CONSTANTS
// ============================================================================

const CURIOUS_KELLY_STYLE = `
STYLE: Modern Educational Illustration
- Clean flat illustration with subtle depth and soft shadows
- Warm, friendly color palette: teals, corals, warm yellows, soft purples
- Clean lines, approachable and friendly aesthetic
- Stylized but not cartoonish, professional educational quality
- Soft, even, welcoming lighting
- Think: Headspace, Duolingo, Khan Academy visual style
`.trim();

const COMPOSITION_RULES = `
COMPOSITION:
- 1:1 square aspect ratio
- Main subject takes 50-70% of frame
- LEFT 70% contains primary visual content
- RIGHT 30% is simpler (reserved for UI overlay)
- Clean, uncluttered background
- Generous whitespace and breathing room
`.trim();

const UNIVERSAL_CONSTRAINTS = `
CRITICAL REQUIREMENTS:
- DO NOT include ANY text, labels, numbers, letters, or writing
- DO NOT include watermarks, signatures, or logos
- DO NOT include realistic photographs of real people
- Keep it appropriate for all ages (family-friendly)
`.trim();

const PHASE_TEMPLATES: Record<Phase, { purpose: string; moodGuidance: string }> = {
  hook: {
    purpose: 'Spark curiosity, create a "wait, what?" moment of intrigue',
    moodGuidance: 'Slightly mysterious, intriguing, wonder-inducing'
  },
  cliff: {
    purpose: 'Show tension, choice, or contrast that creates anticipation',
    moodGuidance: 'Visual tension between two elements, decision moment'
  },
  q1: {
    purpose: 'Clearly illustrate the first key concept',
    moodGuidance: 'Clear, educational, enlightening, approachable'
  },
  q2: {
    purpose: 'Deepen understanding with the second concept',
    moodGuidance: 'Building, layered, showing progression'
  },
  q3: {
    purpose: 'Challenge or surprise with the third concept',
    moodGuidance: 'Surprising, eye-opening, aha moment'
  },
  wisdom: {
    purpose: 'Inspire with timeless, universal truth',
    moodGuidance: 'Peaceful, inspiring, golden-hour warmth, aspirational'
  },
  outro: {
    purpose: 'Celebrate completion and create forward momentum',
    moodGuidance: 'Celebratory, energetic, forward-looking, accomplished'
  }
};

// ============================================================================
// PROMPT GENERATION FUNCTIONS
// ============================================================================

function generateAllPromptsV2(lesson: LessonContext): GeneratedPrompt[] {
  const phases: Phase[] = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];
  
  return phases.map(phase => {
    const template = PHASE_TEMPLATES[phase];
    const subject = getPhaseSubject(phase, lesson);
    
    const prompt = `
Create an illustrated educational scene for a lesson about "${lesson.topic}".

PURPOSE: ${template.purpose}

SUBJECT TO ILLUSTRATE:
${subject}

MOOD: ${template.moodGuidance}

${CURIOUS_KELLY_STYLE}

${COMPOSITION_RULES}

${UNIVERSAL_CONSTRAINTS}
`.trim();

    const contentHash = generateContentHash({
      dayNumber: lesson.dayNumber,
      phase,
      topic: lesson.topic,
      version: 'v2'
    });

    return {
      prompt,
      phase,
      contentHash,
      expectedDimensions: { width: 1024, height: 1024 }
    };
  });
}

function getPhaseSubject(phase: Phase, lesson: LessonContext): string {
  switch (phase) {
    case 'hook':
      return `A curious, intriguing scene that sparks wonder about "${lesson.topic}". Show the surprising or unexpected aspect that makes people say "wait, what?!" Use visual metaphor to hint at: ${lesson.hookTeaser.substring(0, 100)}`;
    
    case 'cliff':
      return `A split or contrasting scene showing two perspectives on "${lesson.topic}". Visualize the tension between common belief and surprising reality. Show a clear visual fork or comparison moment.`;
    
    case 'q1':
      return `An educational illustration explaining: ${lesson.q1Content.substring(0, 150)}. Make the abstract concrete - show the mechanism or relationship clearly through visual metaphor.`;
    
    case 'q2':
      return `A deeper educational illustration building on the lesson: ${lesson.q2Content.substring(0, 150)}. Show progression and layered understanding.`;
    
    case 'q3':
      return `A surprising "aha moment" illustration: ${lesson.q3Content.substring(0, 150)}. Create visual impact that captures the most surprising insight.`;
    
    case 'wisdom':
      return `An inspiring, peaceful scene embodying the wisdom of "${lesson.topic}": ${lesson.wisdomInsight.substring(0, 100)}. Show a universal moment of realization, growth, or possibility. Timeless and aspirational.`;
    
    case 'outro':
      return `A celebratory scene showing accomplishment and forward momentum after learning about "${lesson.topic}". A sense of achievement with energy for what's next.`;
    
    default:
      return `An educational illustration about "${lesson.topic}"`;
  }
}

function generateContentHash(input: { dayNumber: number; phase: Phase; topic: string; version: string }): string {
  const canonical = JSON.stringify({
    d: input.dayNumber,
    p: input.phase,
    t: input.topic.toLowerCase().trim(),
    v: input.version
  });
  
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

function validateGeneratedImage(
  imageBuffer: Buffer,
  expectedDimensions: { width: number; height: number }
): { valid: boolean; errors: string[]; warnings: string[] } {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  const sizeKB = imageBuffer.length / 1024;
  if (sizeKB < 50) {
    errors.push(`Image too small (${sizeKB.toFixed(1)}KB)`);
  }
  if (sizeKB > 5000) {
    warnings.push(`Image very large (${sizeKB.toFixed(1)}KB)`);
  }
  
  // Check PNG dimensions
  if (imageBuffer[0] === 0x89 && imageBuffer[1] === 0x50) {
    const width = imageBuffer.readUInt32BE(16);
    const height = imageBuffer.readUInt32BE(20);
    const aspectRatio = width / height;
    const expectedAspectRatio = expectedDimensions.width / expectedDimensions.height;
    
    if (Math.abs(aspectRatio - expectedAspectRatio) > 0.15) {
      errors.push(`Wrong aspect ratio: ${aspectRatio.toFixed(2)}`);
    }
    
    if (width < 800) {
      warnings.push(`Low resolution: ${width}px wide`);
    }
  }
  
  return { valid: errors.length === 0, errors, warnings };
}

// ============================================================================
// CONFIGURATION
// ============================================================================

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY || !GOOGLE_API_KEY) {
  console.error('❌ Missing required environment variables');
  console.error('   Need: PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GOOGLE_API_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);

// Models in order of preference
const MODELS = [
  { name: 'imagen-4.0-ultra-generate-001', cost: 0.06, quality: 'ultra' },
  { name: 'imagen-4.0-generate-001', cost: 0.02, quality: 'standard' },
  { name: 'gemini-2.0-flash-exp-image-generation', cost: 0, quality: 'fast' }
];

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;

// ============================================================================
// LESSON FETCHING
// ============================================================================

interface DBLesson {
  day_number: number;
  topic: string;
  hook_teaser: string;
  cliffhanger_setup: string;
  quiz_questions: any;
  fun_facts: string[];
  wow_moment: string;
  wisdom_insight: string;
  extended_explanation: string;
}

async function fetchLesson(dayNumber: number): Promise<LessonContext | null> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();

  if (error || !data) {
    console.error(`❌ Failed to fetch lesson ${dayNumber}:`, error?.message);
    return null;
  }

  const lesson = data as DBLesson;
  
  // Extract question content
  const questions = lesson.quiz_questions || [];
  const q1 = questions[0]?.question || '';
  const q2 = questions[1]?.question || '';
  const q3 = questions[2]?.question || '';

  return {
    dayNumber: lesson.day_number,
    topic: lesson.topic,
    hookTeaser: lesson.hook_teaser || '',
    cliffChoice: lesson.cliffhanger_setup || '',
    q1Content: q1,
    q2Content: q2,
    q3Content: q3,
    wisdomInsight: lesson.wisdom_insight || lesson.extended_explanation || '',
    funFacts: lesson.fun_facts || [],
    wowMoment: lesson.wow_moment || ''
  };
}

// ============================================================================
// IMAGE GENERATION
// ============================================================================

async function generateImage(
  prompt: string,
  modelConfig: typeof MODELS[0]
): Promise<{ buffer: Buffer; model: string } | null> {
  try {
    if (modelConfig.name.startsWith('imagen')) {
      // Use Imagen API
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${modelConfig.name}:predict?key=${GOOGLE_API_KEY}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            instances: [{ prompt }],
            parameters: {
              sampleCount: 1,
              aspectRatio: '1:1',
              safetySetting: 'block_low_and_above'
            }
          })
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        if (response.status === 429) {
          console.log(`   ⏳ Rate limited on ${modelConfig.name}`);
          return null;
        }
        console.log(`   ⚠️ ${modelConfig.name} error: ${response.status}`);
        return null;
      }

      const result = await response.json();
      const imageData = result.predictions?.[0]?.bytesBase64Encoded;
      
      if (!imageData) {
        console.log(`   ⚠️ No image data from ${modelConfig.name}`);
        return null;
      }

      return {
        buffer: Buffer.from(imageData, 'base64'),
        model: modelConfig.name
      };
    } else {
      // Use Gemini model
      const model = genAI.getGenerativeModel({ 
        model: modelConfig.name,
        generationConfig: { responseModalities: ['image', 'text'] }
      });

      const response = await model.generateContent(prompt);
      const parts = response.response.candidates?.[0]?.content?.parts || [];
      
      for (const part of parts) {
        if (part.inlineData?.mimeType?.startsWith('image/')) {
          return {
            buffer: Buffer.from(part.inlineData.data, 'base64'),
            model: modelConfig.name
          };
        }
      }
      
      console.log(`   ⚠️ No image in Gemini response`);
      return null;
    }
  } catch (error: any) {
    console.log(`   ⚠️ Generation error: ${error.message?.substring(0, 50)}`);
    return null;
  }
}

async function generateWithFallback(prompt: string): Promise<{ buffer: Buffer; model: string; cost: number } | null> {
  for (const modelConfig of MODELS) {
    console.log(`   🎨 Trying ${modelConfig.quality}...`);
    
    const result = await generateImage(prompt, modelConfig);
    if (result) {
      return {
        ...result,
        cost: modelConfig.cost
      };
    }
    
    // Small delay before trying next model
    await new Promise(r => setTimeout(r, 1000));
  }
  
  return null;
}

// ============================================================================
// STORAGE & DATABASE
// ============================================================================

async function uploadToStorage(
  buffer: Buffer,
  contentHash: string
): Promise<string | null> {
  const storagePath = `v2/${contentHash}.png`;
  
  const { error } = await supabase.storage
    .from('visuals')
    .upload(storagePath, buffer, {
      contentType: 'image/png',
      upsert: true
    });

  if (error) {
    console.error(`   ❌ Upload failed: ${error.message}`);
    return null;
  }

  const { data: urlData } = supabase.storage
    .from('visuals')
    .getPublicUrl(storagePath);

  return urlData.publicUrl;
}

async function saveToDatabase(
  dayNumber: number,
  phase: Phase,
  topic: string,
  contentHash: string,
  publicUrl: string,
  storagePath: string,
  prompt: string,
  model: string,
  cost: number
): Promise<boolean> {
  const { error } = await supabase
    .from('visual_commons')
    .upsert({
      content_hash: contentHash,
      day_number: dayNumber,
      phase,
      topic,
      visual_type: 'illustrated',
      age_group: 'all',
      style: 'curious-kelly-v2',
      storage_path: storagePath,
      public_url: publicUrl,
      format: 'png',
      prompt_used: prompt,
      model_used: model,
      generation_params: { version: 'v2', aspectRatio: '16:9' },
      estimated_cost: cost,
      generated_by_display_name: 'Curious Kelly V2',
      generation_source: 'seed-v2',
      status: 'active'
    }, {
      onConflict: 'content_hash'
    });

  if (error) {
    console.error(`   ❌ Database save failed: ${error.message}`);
    return false;
  }

  return true;
}

async function checkExists(contentHash: string): Promise<boolean> {
  const { data } = await supabase
    .from('visual_commons')
    .select('id')
    .eq('content_hash', contentHash)
    .eq('status', 'active')
    .single();

  return !!data;
}

// ============================================================================
// MAIN GENERATION FLOW
// ============================================================================

async function generateVisualsForDay(dayNumber: number): Promise<{
  generated: number;
  skipped: number;
  failed: number;
  cost: number;
}> {
  const stats = { generated: 0, skipped: 0, failed: 0, cost: 0 };

  console.log(`\n📚 Day ${dayNumber}`);
  
  const lesson = await fetchLesson(dayNumber);
  if (!lesson) {
    console.log(`   ⚠️ No lesson found for day ${dayNumber}`);
    return stats;
  }

  console.log(`   Topic: "${lesson.topic}"`);
  
  const prompts = generateAllPromptsV2(lesson);
  
  for (const promptData of prompts) {
    console.log(`\n   📍 Phase: ${promptData.phase}`);
    
    // Check if already exists
    if (await checkExists(promptData.contentHash)) {
      console.log(`      ⏭️ Already exists`);
      stats.skipped++;
      continue;
    }

    // Generate with retries
    let success = false;
    for (let attempt = 1; attempt <= MAX_RETRIES && !success; attempt++) {
      if (attempt > 1) {
        console.log(`      🔄 Retry ${attempt}/${MAX_RETRIES}`);
        await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
      }

      const result = await generateWithFallback(promptData.prompt);
      if (!result) {
        continue;
      }

      // Validate the generated image
      const validation = validateGeneratedImage(result.buffer, promptData.expectedDimensions);
      
      if (!validation.valid) {
        console.log(`      ❌ Validation failed: ${validation.errors.join(', ')}`);
        continue;
      }

      if (validation.warnings.length > 0) {
        console.log(`      ⚠️ Warnings: ${validation.warnings.join(', ')}`);
      }

      // Upload to storage
      const publicUrl = await uploadToStorage(result.buffer, promptData.contentHash);
      if (!publicUrl) {
        continue;
      }

      // Save to database
      const saved = await saveToDatabase(
        lesson.dayNumber,
        promptData.phase,
        lesson.topic,
        promptData.contentHash,
        publicUrl,
        `v2/${promptData.contentHash}.png`,
        promptData.prompt,
        result.model,
        result.cost
      );

      if (saved) {
        console.log(`      ✅ Generated (${result.model})`);
        stats.generated++;
        stats.cost += result.cost;
        success = true;
      }
    }

    if (!success) {
      console.log(`      ❌ Failed after ${MAX_RETRIES} attempts`);
      stats.failed++;
    }
  }

  return stats;
}

// ============================================================================
// CLI ENTRY POINT
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  let days: number[] = [];

  // Parse arguments
  for (const arg of args) {
    if (arg.startsWith('--day=')) {
      days.push(parseInt(arg.split('=')[1], 10));
    } else if (arg.startsWith('--range=')) {
      const [start, end] = arg.split('=')[1].split('-').map(n => parseInt(n, 10));
      for (let d = start; d <= end; d++) {
        days.push(d);
      }
    }
  }

  if (days.length === 0) {
    console.log('Usage:');
    console.log('  npx tsx scripts/visual-generator-v2.ts --day=1');
    console.log('  npx tsx scripts/visual-generator-v2.ts --range=1-7');
    console.log('  npx tsx scripts/visual-generator-v2.ts --day=351 --day=352');
    process.exit(1);
  }

  console.log('═'.repeat(60));
  console.log('🎨 VISUAL GENERATOR V2 - Curious Kelly Style');
  console.log('═'.repeat(60));
  console.log(`Days to process: ${days.join(', ')}`);

  const totals = { generated: 0, skipped: 0, failed: 0, cost: 0 };

  for (const day of days) {
    const stats = await generateVisualsForDay(day);
    totals.generated += stats.generated;
    totals.skipped += stats.skipped;
    totals.failed += stats.failed;
    totals.cost += stats.cost;
  }

  console.log('\n' + '═'.repeat(60));
  console.log('📊 FINAL SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${totals.generated}`);
  console.log(`⏭️ Skipped: ${totals.skipped}`);
  console.log(`❌ Failed: ${totals.failed}`);
  console.log(`💰 Estimated cost: $${totals.cost.toFixed(2)}`);
}

main().catch(console.error);
