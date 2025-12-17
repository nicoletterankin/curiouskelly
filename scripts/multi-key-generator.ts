#!/usr/bin/env npx tsx
/**
 * Multi-Key Imagen Generator
 * 
 * Uses multiple Google API keys to bypass per-project rate limits.
 * Each project gets 30 Ultra + 30 Standard + 30 Fast = 90 images/day.
 * 
 * Usage:
 *   npx tsx scripts/multi-key-generator.ts --range=1-30
 *   npx tsx scripts/multi-key-generator.ts --days=351,352,353
 */

import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';
import * as path from 'path';
import * as crypto from 'crypto';

dotenv.config({ path: path.join(process.cwd(), '.env.local') });

// ============================================================================
// CONFIGURATION
// ============================================================================

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Collect all API keys from environment
function getApiKeys(): string[] {
  const keys: string[] = [];
  
  // Check for numbered keys: GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.
  for (let i = 1; i <= 20; i++) {
    const key = process.env[`GOOGLE_API_KEY_${i}`];
    if (key) keys.push(key);
  }
  
  // Also include the main key
  const mainKey = process.env.GOOGLE_API_KEY;
  if (mainKey && !keys.includes(mainKey)) {
    keys.unshift(mainKey);
  }
  
  return keys;
}

const API_KEYS = getApiKeys();
console.log(`🔑 Found ${API_KEYS.length} API key(s)`);

if (API_KEYS.length === 0) {
  console.error('❌ No API keys found. Add GOOGLE_API_KEY or GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.');
  process.exit(1);
}

// Models to use (in order of quality)
const MODELS = [
  { name: 'imagen-4.0-ultra-generate-001', quality: 'ultra', cost: 0.06 },
  { name: 'imagen-4.0-generate-001', quality: 'standard', cost: 0.04 },
  { name: 'imagen-4.0-fast-generate-001', quality: 'fast', cost: 0.02 },
];

const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'] as const;
type Phase = typeof PHASES[number];

// Track usage per key per model
const keyUsage: Map<string, Map<string, number>> = new Map();
const KEY_LIMIT_PER_MODEL = 30;

// ============================================================================
// API KEY ROTATION
// ============================================================================

function getAvailableKey(model: string): string | null {
  for (const key of API_KEYS) {
    if (!keyUsage.has(key)) {
      keyUsage.set(key, new Map());
    }
    const modelUsage = keyUsage.get(key)!.get(model) || 0;
    if (modelUsage < KEY_LIMIT_PER_MODEL) {
      return key;
    }
  }
  return null;
}

function recordUsage(key: string, model: string) {
  if (!keyUsage.has(key)) {
    keyUsage.set(key, new Map());
  }
  const current = keyUsage.get(key)!.get(model) || 0;
  keyUsage.get(key)!.set(model, current + 1);
}

function getTotalCapacity(): number {
  return API_KEYS.length * MODELS.length * KEY_LIMIT_PER_MODEL;
}

function getRemainingCapacity(): number {
  let used = 0;
  for (const [_, modelMap] of keyUsage) {
    for (const [_, count] of modelMap) {
      used += count;
    }
  }
  return getTotalCapacity() - used;
}

// ============================================================================
// IMAGE GENERATION
// ============================================================================

async function generateImage(
  prompt: string,
  apiKey: string,
  model: string
): Promise<Buffer | null> {
  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:predict?key=${apiKey}`,
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
      const error = await response.json();
      if (response.status === 429 || error.error?.status === 'RESOURCE_EXHAUSTED') {
        // Mark this key/model as exhausted
        recordUsage(apiKey, model);
        for (let i = 0; i < KEY_LIMIT_PER_MODEL; i++) {
          recordUsage(apiKey, model); // Fill it up
        }
        return null;
      }
      return null;
    }

    const result = await response.json();
    const imageData = result.predictions?.[0]?.bytesBase64Encoded;
    
    if (!imageData) return null;
    
    recordUsage(apiKey, model);
    return Buffer.from(imageData, 'base64');
  } catch (error) {
    return null;
  }
}

async function generateWithRotation(prompt: string): Promise<{ buffer: Buffer; model: string; cost: number } | null> {
  for (const modelConfig of MODELS) {
    const apiKey = getAvailableKey(modelConfig.name);
    if (!apiKey) {
      console.log(`      ⚠️  All keys exhausted for ${modelConfig.quality}`);
      continue;
    }
    
    const keyIndex = API_KEYS.indexOf(apiKey) + 1;
    process.stdout.write(`      🎨 Key${keyIndex}/${modelConfig.quality}...`);
    
    const buffer = await generateImage(prompt, apiKey, modelConfig.name);
    if (buffer) {
      console.log(' ✅');
      return { buffer, model: modelConfig.name, cost: modelConfig.cost };
    }
    console.log(' ❌');
  }
  
  return null;
}

// ============================================================================
// LESSON FETCHING
// ============================================================================

async function fetchLesson(dayNumber: number): Promise<any> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();
    
  return data;
}

// ============================================================================
// PROMPT BUILDING
// ============================================================================

function buildPrompt(phase: Phase, lesson: any): string {
  const topic = lesson.topic;
  
  const phasePrompts: Record<Phase, string> = {
    hook: `Create a stunning, curiosity-sparking photorealistic scene for: "${topic}"

This is the OPENING moment of a lesson. The goal is to make viewers say "Wait, what?!"

STYLE:
- Ultra photorealistic, professional photography aesthetic
- Dramatic lighting, cinematic composition
- 1:1 square aspect ratio, high quality
- Warm, inviting color palette with a sense of wonder

DO NOT include any text, logos, or watermarks.`,

    cliff: `Create an intriguing photorealistic scene that deepens mystery for: "${topic}"

This shows the TENSION between what we think we know and surprising reality.

STYLE:
- Ultra photorealistic, slightly dramatic lighting
- Sense of revelation or "plot twist" moment
- 1:1 square, high quality
- Visual contrast between expectation and reality

DO NOT include any text.`,

    q1: `Create a clear, photorealistic educational scene for: "${topic}"

This illustrates the FIRST key concept - clarity is everything.

STYLE:
- Ultra photorealistic, bright educational lighting
- Clear, organized composition
- 1:1 square, high quality

DO NOT include any text.`,

    q2: `Create a photorealistic educational scene showing DEEPER insight into: "${topic}"

Building on foundational knowledge with more detail.

STYLE:
- Ultra photorealistic, layered lighting showing depth
- 1:1 square, high quality

DO NOT include any text.`,

    q3: `Create a stunning photorealistic scene about: "${topic}"

This captures the most surprising aspect of the topic.

STYLE:
- Ultra photorealistic, dramatic lighting
- Maximum visual impact and wonder
- 1:1 square, high quality

DO NOT include any text.`,

    wisdom: `Create an inspiring photorealistic scene about: "${topic}"

This visual captures timeless wisdom and new possibilities.

STYLE:
- Ultra photorealistic, warm golden hour lighting
- Timeless, universal appeal
- 1:1 square, high quality

DO NOT include any text.`,

    outro: `Create a CELEBRATORY photorealistic scene marking completion of learning about: "${topic}"

STYLE:
- Ultra photorealistic, bright and uplifting
- Sense of achievement and forward momentum
- 1:1 square, high quality

DO NOT include any text.`
  };

  return phasePrompts[phase];
}

// ============================================================================
// STORAGE
// ============================================================================

function generateHash(dayNumber: number, phase: Phase): string {
  return crypto.createHash('sha256')
    .update(JSON.stringify({ d: dayNumber, p: phase, v: 'multi-key-v1' }))
    .digest('hex');
}

async function checkExists(hash: string): Promise<boolean> {
  const { data } = await supabase
    .from('visual_commons')
    .select('id')
    .eq('content_hash', hash)
    .eq('status', 'active')
    .single();
  return !!data;
}

async function saveVisual(
  dayNumber: number,
  phase: Phase,
  topic: string,
  buffer: Buffer,
  model: string,
  cost: number,
  prompt: string
): Promise<boolean> {
  const hash = generateHash(dayNumber, phase);
  const storagePath = `multi-key/${hash}.png`;
  
  // Upload to storage
  const { error: uploadError } = await supabase.storage
    .from('visuals')
    .upload(storagePath, buffer, { contentType: 'image/png', upsert: true });
    
  if (uploadError) {
    console.error(`      ❌ Upload failed: ${uploadError.message}`);
    return false;
  }
  
  const { data: urlData } = supabase.storage.from('visuals').getPublicUrl(storagePath);
  
  // Save to database
  const { error: dbError } = await supabase
    .from('visual_commons')
    .upsert({
      content_hash: hash,
      day_number: dayNumber,
      phase,
      topic,
      visual_type: 'scene',
      age_group: 'all',
      style: 'photorealistic-square',
      storage_path: storagePath,
      public_url: urlData.publicUrl,
      format: 'png',
      prompt_used: prompt,
      model_used: model,
      generation_params: { aspectRatio: '1:1', version: 'multi-key-v1' },
      estimated_cost: cost,
      generated_by_display_name: 'Multi-Key Generator',
      generation_source: 'multi-key',
      status: 'active'
    }, { onConflict: 'content_hash' });
    
  if (dbError) {
    console.error(`      ❌ DB save failed: ${dbError.message}`);
    return false;
  }
  
  return true;
}

// ============================================================================
// MAIN
// ============================================================================

async function generateForDay(dayNumber: number): Promise<{ generated: number; skipped: number; failed: number; cost: number }> {
  const stats = { generated: 0, skipped: 0, failed: 0, cost: 0 };
  
  const lesson = await fetchLesson(dayNumber);
  if (!lesson) {
    console.log(`   ⚠️  No lesson found for day ${dayNumber}`);
    return stats;
  }
  
  console.log(`\n📚 Day ${dayNumber}: "${lesson.topic}"`);
  
  for (const phase of PHASES) {
    const hash = generateHash(dayNumber, phase);
    
    // Check if exists
    if (await checkExists(hash)) {
      console.log(`   ⏭️  ${phase}: exists`);
      stats.skipped++;
      continue;
    }
    
    console.log(`   📍 ${phase}:`);
    
    const prompt = buildPrompt(phase, lesson);
    const result = await generateWithRotation(prompt);
    
    if (!result) {
      console.log(`      ❌ Failed - all keys exhausted`);
      stats.failed++;
      continue;
    }
    
    const saved = await saveVisual(dayNumber, phase, lesson.topic, result.buffer, result.model, result.cost, prompt);
    if (saved) {
      stats.generated++;
      stats.cost += result.cost;
    } else {
      stats.failed++;
    }
  }
  
  return stats;
}

async function main() {
  const args = process.argv.slice(2);
  let days: number[] = [];
  
  for (const arg of args) {
    if (arg.startsWith('--range=')) {
      const [start, end] = arg.split('=')[1].split('-').map(n => parseInt(n, 10));
      for (let d = start; d <= end; d++) days.push(d);
    } else if (arg.startsWith('--days=')) {
      days = arg.split('=')[1].split(',').map(n => parseInt(n.trim(), 10));
    } else if (arg.startsWith('--day=')) {
      days.push(parseInt(arg.split('=')[1], 10));
    }
  }
  
  if (days.length === 0) {
    console.log('Usage:');
    console.log('  npx tsx scripts/multi-key-generator.ts --range=1-30');
    console.log('  npx tsx scripts/multi-key-generator.ts --days=351,352,353');
    console.log('  npx tsx scripts/multi-key-generator.ts --day=1');
    console.log('');
    console.log(`🔑 API Keys loaded: ${API_KEYS.length}`);
    console.log(`📊 Total capacity: ${getTotalCapacity()} images`);
    console.log(`📊 Images needed for 365 days: ${365 * 7} = 2,555`);
    process.exit(0);
  }
  
  console.log('═'.repeat(60));
  console.log('🚀 MULTI-KEY IMAGEN GENERATOR');
  console.log('═'.repeat(60));
  console.log(`🔑 API Keys: ${API_KEYS.length}`);
  console.log(`📊 Total capacity: ${getTotalCapacity()} images`);
  console.log(`📅 Days to process: ${days.length}`);
  console.log(`🎯 Images needed: ${days.length * 7}`);
  
  const totals = { generated: 0, skipped: 0, failed: 0, cost: 0 };
  
  for (const day of days) {
    if (getRemainingCapacity() === 0) {
      console.log('\n❌ ALL KEYS EXHAUSTED - stopping');
      break;
    }
    
    const stats = await generateForDay(day);
    totals.generated += stats.generated;
    totals.skipped += stats.skipped;
    totals.failed += stats.failed;
    totals.cost += stats.cost;
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 FINAL SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${totals.generated}`);
  console.log(`⏭️  Skipped: ${totals.skipped}`);
  console.log(`❌ Failed: ${totals.failed}`);
  console.log(`💰 Cost: $${totals.cost.toFixed(2)}`);
  console.log(`📊 Remaining capacity: ${getRemainingCapacity()}`);
}

main().catch(console.error);
