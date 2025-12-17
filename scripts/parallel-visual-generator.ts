#!/usr/bin/env npx tsx
/**
 * PARALLEL VISUAL GENERATOR
 * 
 * Uses multiple image models simultaneously to bypass rate limits:
 * - Imagen 4 Ultra (highest quality, rate limited)
 * - Imagen 4 Standard (high quality)
 * - Imagen 4 Fast (good quality, fastest)
 * - Gemini 2.0 Flash Image (good quality)
 * 
 * Cycles through models to maximize throughput.
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

const MODELS = {
  'imagen-ultra': {
    name: 'imagen-4.0-ultra-generate-001',
    type: 'predict',
    cost: 0.06,
    quality: 'ultra'
  },
  'imagen-standard': {
    name: 'imagen-4.0-generate-001',
    type: 'predict',
    cost: 0.04,
    quality: 'high'
  },
  'imagen-fast': {
    name: 'imagen-4.0-fast-generate-001',
    type: 'predict',
    cost: 0.02,
    quality: 'good'
  },
  'gemini-flash': {
    name: 'gemini-2.0-flash-exp-image-generation',
    type: 'generate',
    cost: 0.00, // Free tier
    quality: 'good'
  }
} as const;

type ModelKey = keyof typeof MODELS;

const CONFIG = {
  ASPECT_RATIO: '16:9',
  DELAY_BETWEEN_CALLS_MS: 1000,
  
  ALL_PHASES: ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro', 'complete'] as const,
  KEY_PHASES: ['hook', 'fact1', 'fact3', 'wisdom', 'complete'] as const,
  
  ALL_STYLES: ['artistic', 'textbook', 'diagram', 'minimal'] as const,
};

type Phase = typeof CONFIG.ALL_PHASES[number];
type Style = typeof CONFIG.ALL_STYLES[number];

// =============================================================================
// ENVIRONMENT
// =============================================================================

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';
const GOOGLE_API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_API_KEY || '';

if (!SUPABASE_URL || !SUPABASE_KEY || !GOOGLE_API_KEY) {
  console.error('❌ Missing credentials');
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
  wow_moment?: string;
  fun_facts?: string[];
  marketing_headline?: string;
  learning_objectives?: string[];
  quick_quiz_questions?: { question: string; options: string[]; correct: string }[];
  common_misconceptions?: { misconception: string; correction: string }[];
}

interface GenerationTask {
  lesson: Lesson;
  phase: Phase;
  style: Style;
  model: ModelKey;
}

// =============================================================================
// PROMPT BUILDING
// =============================================================================

const STYLE_PROMPTS: Record<Style, string> = {
  artistic: `Ultra photorealistic, cinematic lighting, dramatic, emotional, warm color palette. 16:9, 4K. Leave right 30% simpler for overlay. NO text.`,
  textbook: `Educational textbook illustration, clean layout, labeled diagram, light background, clear annotations, print quality. Include title and 3-5 labels.`,
  diagram: `Technical educational diagram, precise lines, numbered components, arrows showing flow, legend. Include component labels.`,
  minimal: `Minimalist design, max 3 colors, single concept, generous negative space, elegant. NO text.`
};

function buildPrompt(lesson: Lesson, phase: Phase, style: Style): string {
  const styleGuide = STYLE_PROMPTS[style];
  
  let content = '';
  switch (phase) {
    case 'hook':
      content = `Create curiosity about "${lesson.topic}". Hint at: "${lesson.marketing_headline || lesson.universal_truth}". Make viewers say "Wait, what?!"`;
      break;
    case 'cliff':
      const misc = lesson.common_misconceptions?.[0];
      content = `Show contrast: What people think "${misc?.misconception || 'common belief'}" vs reality "${misc?.correction || lesson.universal_truth}"`;
      break;
    case 'fact1':
      const q1 = lesson.quick_quiz_questions?.[0];
      content = `Teach: "${lesson.fun_facts?.[0]}". ${q1 ? `Illustrate the answer: "${q1.correct}"` : ''}`;
      break;
    case 'fact2':
      content = `Show deeper understanding: "${lesson.fun_facts?.[1] || lesson.universal_truth}". Show relationships and connections.`;
      break;
    case 'fact3':
      content = `WOW moment: "${lesson.wow_moment || lesson.fun_facts?.[2]}". Maximum visual impact, shareable.`;
      break;
    case 'wisdom':
      content = `Life application: "${lesson.universal_truth}". Inspirational, poster-worthy, timeless wisdom.`;
      break;
    case 'outro':
      content = `Celebration of learning "${lesson.topic}". Achievement, forward momentum, what's next.`;
      break;
    case 'complete':
      content = `Complete summary of "${lesson.topic}": ${lesson.learning_objectives?.slice(0, 2).join('; ')}. Comprehensive infographic style.`;
      break;
  }
  
  return `Educational visual for "${lesson.topic}"

${styleGuide}

${content}

Safe for all ages. No copyrighted content.`;
}

// =============================================================================
// IMAGE GENERATION
// =============================================================================

async function generateWithModel(prompt: string, modelKey: ModelKey): Promise<{ buffer: Buffer | null; model: string }> {
  const model = MODELS[modelKey];
  
  if (model.type === 'predict') {
    // Imagen API
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model.name}:predict?key=${GOOGLE_API_KEY}`;
    const body = {
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
        body: JSON.stringify(body)
      });
      
      if (!response.ok) {
        return { buffer: null, model: model.name };
      }
      
      const data = await response.json() as any;
      if (data.predictions?.[0]?.bytesBase64Encoded) {
        return { 
          buffer: Buffer.from(data.predictions[0].bytesBase64Encoded, 'base64'),
          model: model.name 
        };
      }
      return { buffer: null, model: model.name };
    } catch {
      return { buffer: null, model: model.name };
    }
  } else {
    // Gemini generateContent API
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model.name}:generateContent?key=${GOOGLE_API_KEY}`;
    const body = {
      contents: [{ parts: [{ text: `Generate an image: ${prompt}` }] }],
      generationConfig: { responseModalities: ['IMAGE', 'TEXT'] }
    };
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      if (!response.ok) {
        return { buffer: null, model: model.name };
      }
      
      const data = await response.json() as any;
      const parts = data.candidates?.[0]?.content?.parts || [];
      const imagePart = parts.find((p: any) => p.inlineData?.mimeType?.startsWith('image/'));
      
      if (imagePart?.inlineData?.data) {
        return {
          buffer: Buffer.from(imagePart.inlineData.data, 'base64'),
          model: model.name
        };
      }
      return { buffer: null, model: model.name };
    } catch {
      return { buffer: null, model: model.name };
    }
  }
}

// Try models in order until one succeeds
async function generateWithFallback(prompt: string, preferredModels: ModelKey[]): Promise<{ buffer: Buffer | null; model: string }> {
  for (const modelKey of preferredModels) {
    const result = await generateWithModel(prompt, modelKey);
    if (result.buffer) {
      return result;
    }
    console.log(`      ⚠️  ${modelKey} failed, trying next...`);
  }
  return { buffer: null, model: 'none' };
}

// =============================================================================
// DATABASE
// =============================================================================

async function getLesson(dayNumber: number): Promise<Lesson | null> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .maybeSingle();
  
  if (error || !data) return null;
  return data as Lesson;
}

function generateHash(dayNumber: number, phase: string, style: string, modelQuality: string): string {
  const normalized = { d: dayNumber, p: phase, s: style, q: modelQuality, ver: '4' };
  return crypto.createHash('sha256').update(JSON.stringify(normalized)).digest('hex');
}

async function checkExists(hash: string): Promise<boolean> {
  const { data } = await supabase.from('visual_commons').select('id').eq('content_hash', hash).maybeSingle();
  return !!data;
}

async function saveVisual(
  buffer: Buffer,
  lesson: Lesson,
  phase: Phase,
  style: Style,
  modelUsed: string,
  prompt: string,
  hash: string
): Promise<string | null> {
  const storagePath = `multi-model/${style}/${hash}.png`;
  
  const { error: uploadError } = await supabase.storage
    .from('visuals')
    .upload(storagePath, buffer, { contentType: 'image/png', upsert: true });
  
  if (uploadError) return null;
  
  const { data: urlData } = supabase.storage.from('visuals').getPublicUrl(storagePath);
  
  const { error: insertError } = await supabase.from('visual_commons').upsert({
    content_hash: hash,
    day_number: lesson.day_number,
    phase,
    topic: lesson.topic,
    visual_type: 'scene',
    age_group: 'all',
    style,
    storage_path: storagePath,
    public_url: urlData.publicUrl,
    format: 'png',
    prompt_used: prompt.substring(0, 5000),
    model_used: modelUsed,
    generation_params: { multiModel: true, style },
    estimated_cost: MODELS[Object.keys(MODELS).find(k => MODELS[k as ModelKey].name === modelUsed) as ModelKey]?.cost || 0,
    generated_by_display_name: 'Curious Kelly Team',
    generation_source: 'seed',
    status: 'active'
  }, { onConflict: 'content_hash' });
  
  return insertError ? null : urlData.publicUrl;
}

// =============================================================================
// MAIN
// =============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  console.log('🚀 PARALLEL MULTI-MODEL VISUAL GENERATOR');
  console.log('━'.repeat(60));
  console.log('📊 Available Models:');
  Object.entries(MODELS).forEach(([key, m]) => {
    console.log(`   ${key}: $${m.cost.toFixed(2)}/image (${m.quality})`);
  });
  console.log('');
  
  const args = process.argv.slice(2);
  let days: number[] = [1];
  let phases: Phase[] = [...CONFIG.KEY_PHASES];
  let styles: Style[] = ['artistic', 'textbook'];
  
  for (const arg of args) {
    if (arg.startsWith('--day=')) days = [parseInt(arg.split('=')[1])];
    if (arg.startsWith('--range=')) {
      const [s, e] = arg.split('=')[1].split('-').map(Number);
      days = Array.from({ length: e - s + 1 }, (_, i) => s + i);
    }
    if (arg === '--all-phases') phases = [...CONFIG.ALL_PHASES];
    if (arg === '--all-styles') styles = [...CONFIG.ALL_STYLES];
  }
  
  console.log(`📅 Days: ${days.join(', ')}`);
  console.log(`📝 Phases: ${phases.join(', ')}`);
  console.log(`🎨 Styles: ${styles.join(', ')}`);
  console.log('');
  console.log('⏳ Starting in 3 seconds...');
  await sleep(3000);
  
  let success = 0, failed = 0, skipped = 0;
  
  // Model rotation for parallel usage
  const modelRotation: ModelKey[] = ['imagen-standard', 'imagen-fast', 'gemini-flash', 'imagen-ultra'];
  let modelIndex = 0;
  
  for (const day of days) {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`📅 DAY ${day}`);
    
    const lesson = await getLesson(day);
    if (!lesson) {
      console.log('❌ Lesson not found');
      continue;
    }
    console.log(`📚 "${lesson.topic}"`);
    
    for (const phase of phases) {
      console.log(`\n  📝 ${phase.toUpperCase()}`);
      
      for (const style of styles) {
        // Rotate through models
        const preferredModel = modelRotation[modelIndex % modelRotation.length];
        modelIndex++;
        
        // Create hash based on quality level
        const modelQuality = MODELS[preferredModel].quality;
        const hash = generateHash(day, phase, style, modelQuality);
        
        if (await checkExists(hash)) {
          console.log(`      ⏭️  ${style} (${modelQuality}): exists`);
          skipped++;
          continue;
        }
        
        console.log(`      🎨 ${style}: trying ${preferredModel}...`);
        
        const prompt = buildPrompt(lesson, phase, style);
        
        // Try preferred model first, then fallback to others
        const fallbackOrder: ModelKey[] = [
          preferredModel,
          ...modelRotation.filter(m => m !== preferredModel)
        ];
        
        const result = await generateWithFallback(prompt, fallbackOrder);
        
        if (result.buffer) {
          const url = await saveVisual(result.buffer, lesson, phase, style, result.model, prompt, hash);
          if (url) {
            console.log(`         ✅ Saved (${result.model.split('-').slice(0,2).join('-')})`);
            success++;
          } else {
            console.log(`         ❌ Save failed`);
            failed++;
          }
        } else {
          console.log(`         ❌ All models failed`);
          failed++;
        }
        
        await sleep(CONFIG.DELAY_BETWEEN_CALLS_MS);
      }
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${success}`);
  console.log(`⏭️  Skipped: ${skipped}`);
  console.log(`❌ Failed: ${failed}`);
  
  const manifestPath = path.join(process.cwd(), 'generated-visuals', 'parallel-manifest.json');
  fs.mkdirSync(path.dirname(manifestPath), { recursive: true });
  fs.writeFileSync(manifestPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    success, skipped, failed,
    days, phases, styles
  }, null, 2));
}

main().catch(console.error);
