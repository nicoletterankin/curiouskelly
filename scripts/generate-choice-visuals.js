#!/usr/bin/env node
/**
 * Generate Choice A/B Visuals for All Phases
 * 
 * v5.0-full-choices requires:
 * - 7 scene visuals per lesson (one per phase)
 * - 14 choice visuals per lesson (A + B for each phase)
 * - Total: 21 visuals per lesson × 365 = 7,665 visuals
 * 
 * This script generates choice A and choice B visuals based on:
 * - The option text from the lesson JSON
 * - The phase context and topic
 * 
 * Usage:
 *   node scripts/generate-choice-visuals.js --day=1
 *   node scripts/generate-choice-visuals.js --range=1-10
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '..', '.env.local') });

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
function getApiKeys() {
  const keys = [];
  const mainKey = process.env.GOOGLE_API_KEY;
  if (mainKey) keys.push(mainKey);
  
  for (let i = 1; i <= 10; i++) {
    const key = process.env[`GOOGLE_API_KEY_${i}`];
    if (key && !keys.includes(key)) keys.push(key);
  }
  return keys;
}

const API_KEYS = getApiKeys();
console.log(`🔑 Found ${API_KEYS.length} API keys`);

if (API_KEYS.length === 0) {
  console.error('❌ No API keys found. Add GOOGLE_API_KEY or GOOGLE_API_KEY_1, etc.');
  process.exit(1);
}

const LESSONS_DIR = path.join(__dirname, '..', 'public', 'lessons');
const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];

// Track key usage
const keyUsage = new Map();
const KEY_LIMIT = 30;

// ============================================================================
// API KEY ROTATION
// ============================================================================

function getAvailableKey() {
  for (const key of API_KEYS) {
    const usage = keyUsage.get(key) || 0;
    if (usage < KEY_LIMIT) {
      return key;
    }
  }
  return null;
}

function recordUsage(key) {
  const current = keyUsage.get(key) || 0;
  keyUsage.set(key, current + 1);
}

// ============================================================================
// PROMPT GENERATION
// ============================================================================

function buildChoicePrompt(topic, phase, optionText, choiceLetter) {
  const phaseEmotions = {
    hook: 'curious, inviting',
    cliff: 'thoughtful, diverging paths',
    q1: 'clear, foundational',
    q2: 'connecting, building',
    q3: 'surprising, insightful',
    wisdom: 'warm, contemplative',
    outro: 'celebratory, forward-looking'
  };
  
  const emotion = phaseEmotions[phase] || 'engaging';
  
  return `Create a photorealistic educational visual representing this learning choice:

TOPIC: "${topic}"
PHASE: ${phase}
CHOICE ${choiceLetter}: "${optionText}"

STYLE:
- Ultra photorealistic, cinematic quality
- 1:1 square aspect ratio, high quality
- ${emotion} mood and lighting
- Abstract conceptual representation of the choice
- No text, no labels, no words
- Suitable for an educational platform

The visual should feel like one of two distinct paths a learner could take.`;
}

// ============================================================================
// IMAGE GENERATION
// ============================================================================

async function generateImage(prompt, apiKey) {
  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-ultra-generate-001:predict?key=${apiKey}`,
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
        return { error: 'rate_limited' };
      }
      return { error: error.error?.message || 'Unknown error' };
    }

    const result = await response.json();
    const imageData = result.predictions?.[0]?.bytesBase64Encoded;
    
    if (!imageData) return { error: 'No image in response' };
    
    return { buffer: Buffer.from(imageData, 'base64') };
  } catch (error) {
    return { error: error.message };
  }
}

// ============================================================================
// STORAGE
// ============================================================================

import * as crypto from 'crypto';

function generateHash(dayNumber, phase, choice) {
  return crypto.createHash('sha256')
    .update(JSON.stringify({ d: dayNumber, p: phase, c: choice, v: 'choice-v1' }))
    .digest('hex');
}

async function checkExists(hash) {
  const { data } = await supabase
    .from('visual_commons')
    .select('id')
    .eq('content_hash', hash)
    .eq('status', 'active')
    .single();
  return !!data;
}

async function saveVisual(dayNumber, phase, choice, topic, optionText, buffer, prompt) {
  const hash = generateHash(dayNumber, phase, choice);
  const storagePath = `choice-visuals/${hash}.png`;
  
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
      visual_type: `choice_${choice.toLowerCase()}`,
      age_group: 'all',
      style: 'photorealistic-square',
      storage_path: storagePath,
      public_url: urlData.publicUrl,
      format: 'png',
      prompt_used: prompt,
      model_used: 'imagen-4.0-ultra-generate-001',
      generation_params: { 
        aspectRatio: '1:1', 
        version: 'choice-v1',
        option_text: optionText
      },
      estimated_cost: 0.06,
      generated_by_display_name: 'Choice Visual Generator',
      generation_source: 'choice-generator',
      status: 'active'
    }, { onConflict: 'content_hash' });
    
  if (dbError) {
    console.error(`      ❌ DB save failed: ${dbError.message}`);
    return false;
  }
  
  return true;
}

// ============================================================================
// LESSON PROCESSING
// ============================================================================

function loadLesson(dayNumber) {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  if (!fs.existsSync(filePath)) return null;
  
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  } catch (e) {
    console.error(`❌ Failed to parse day-${dayNumber}.json: ${e.message}`);
    return null;
  }
}

async function generateChoicesForPhase(dayNumber, topic, phase, phaseData) {
  const stats = { generated: 0, skipped: 0, failed: 0 };
  
  const options = phaseData.options || [];
  if (options.length < 2) {
    console.log(`      ⚠️ Phase ${phase} has < 2 options, skipping`);
    return stats;
  }
  
  for (let i = 0; i < 2; i++) {
    const choice = i === 0 ? 'A' : 'B';
    const option = options[i];
    const optionText = option.text || `Option ${choice}`;
    const hash = generateHash(dayNumber, phase, choice);
    
    // Check if exists
    if (await checkExists(hash)) {
      console.log(`      ⏭️ ${phase}_${choice}: exists`);
      stats.skipped++;
      continue;
    }
    
    // Get API key
    const apiKey = getAvailableKey();
    if (!apiKey) {
      console.log(`      ❌ ${phase}_${choice}: all keys exhausted`);
      stats.failed++;
      continue;
    }
    
    const keyIndex = API_KEYS.indexOf(apiKey) + 1;
    process.stdout.write(`      🎨 ${phase}_${choice} (key${keyIndex})...`);
    
    const prompt = buildChoicePrompt(topic, phase, optionText, choice);
    const result = await generateImage(prompt, apiKey);
    
    if (result.error === 'rate_limited') {
      // Mark key as exhausted
      for (let j = 0; j < KEY_LIMIT; j++) recordUsage(apiKey);
      console.log(' ❌ rate limited');
      stats.failed++;
      continue;
    }
    
    if (result.error) {
      console.log(` ❌ ${result.error}`);
      stats.failed++;
      continue;
    }
    
    recordUsage(apiKey);
    
    // Save
    const saved = await saveVisual(dayNumber, phase, choice, topic, optionText, result.buffer, prompt);
    if (saved) {
      console.log(' ✅');
      stats.generated++;
    } else {
      console.log(' ❌ save failed');
      stats.failed++;
    }
  }
  
  return stats;
}

async function generateForDay(dayNumber) {
  const totals = { generated: 0, skipped: 0, failed: 0 };
  
  const lesson = loadLesson(dayNumber);
  if (!lesson) {
    console.log(`   ⚠️ No lesson file for day ${dayNumber}`);
    return totals;
  }
  
  const topic = lesson.meta?.topic || 'Unknown Topic';
  console.log(`\n📚 Day ${dayNumber}: "${topic}"`);
  
  const phases = lesson.phases || {};
  
  for (const phase of PHASES) {
    const phaseData = phases[phase];
    if (!phaseData) {
      console.log(`   ⚠️ No phase data for ${phase}`);
      continue;
    }
    
    console.log(`   📍 ${phase}:`);
    const stats = await generateChoicesForPhase(dayNumber, topic, phase, phaseData);
    totals.generated += stats.generated;
    totals.skipped += stats.skipped;
    totals.failed += stats.failed;
  }
  
  return totals;
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  let days = [];
  
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
    console.log('  node scripts/generate-choice-visuals.js --day=1');
    console.log('  node scripts/generate-choice-visuals.js --range=1-10');
    console.log('  node scripts/generate-choice-visuals.js --days=352,353,354');
    console.log('');
    console.log(`🔑 API Keys: ${API_KEYS.length}`);
    console.log(`📊 Capacity: ${API_KEYS.length * KEY_LIMIT} Ultra images`);
    console.log(`📊 Per day: 14 choice visuals (7 phases × 2 options)`);
    process.exit(0);
  }
  
  console.log('═'.repeat(60));
  console.log('🎨 CHOICE VISUAL GENERATOR');
  console.log('═'.repeat(60));
  console.log(`🔑 API Keys: ${API_KEYS.length}`);
  console.log(`📅 Days to process: ${days.length}`);
  console.log(`🎯 Visuals per day: 14`);
  console.log(`📊 Total needed: ${days.length * 14}`);
  
  const totals = { generated: 0, skipped: 0, failed: 0 };
  
  for (const day of days) {
    const stats = await generateForDay(day);
    totals.generated += stats.generated;
    totals.skipped += stats.skipped;
    totals.failed += stats.failed;
    
    // Check if we're out of capacity
    let remaining = 0;
    for (const key of API_KEYS) {
      remaining += KEY_LIMIT - (keyUsage.get(key) || 0);
    }
    if (remaining === 0) {
      console.log('\n❌ ALL KEYS EXHAUSTED - stopping');
      break;
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${totals.generated}`);
  console.log(`⏭️  Skipped: ${totals.skipped}`);
  console.log(`❌ Failed: ${totals.failed}`);
  console.log(`💰 Cost: $${(totals.generated * 0.06).toFixed(2)}`);
}

main().catch(console.error);
