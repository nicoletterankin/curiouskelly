/**
 * ROBUST KELLY VISUAL GENERATOR
 * Handles NSFW false flags with retry logic and prompt variations
 */

import Replicate from 'replicate';
import { createClient } from '@supabase/supabase-js';
import fs from 'fs';
import https from 'https';
import path from 'path';

const replicate = new Replicate();
const supabase = createClient(
  'https://tvjalxxsyryjphkforjv.supabase.co',
  process.env.SUPABASE_SERVICE_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI'
);

const KELLY_LORA = 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors';
const OUTPUT_BASE = 'public/kelly/phases';

// ═══════════════════════════════════════════════════════════════
// SAFETY-ENHANCED PROMPTS
// Add explicit safe-for-work language to prevent false flags
// ═══════════════════════════════════════════════════════════════

const SAFETY_PREFIX = 'safe for work, family friendly, educational content, fully clothed, professional photography, ';

const SAFETY_SUFFIX = ', wholesome, appropriate for children, clean professional image, no suggestive content';

// Conservative pose descriptions (less likely to trigger filters)
const SAFE_POSES = {
  hook: [
    'standing with welcoming gesture, warm friendly smile',
    'professional stance, arms relaxed at sides, genuine smile',
    'educator pose, hands clasped in front, inviting expression'
  ],
  q1: [
    'holding a book, curious engaged expression',
    'pointing to educational chart, interested look',
    'gesturing while explaining, teacher pose'
  ],
  q2: [
    'thoughtful expression, hand on chin, contemplating',
    'listening pose, slight head tilt, attentive',
    'pondering with finger to temple, curious'
  ],
  q3: [
    'encouraging gesture, supportive smile',
    'open palm gesture while explaining',
    'nodding with approval, warm expression'
  ],
  wisdom: [
    'hand over heart, proud satisfied smile',
    'arms crossed contentedly, accomplished look',
    'hands together, grateful peaceful expression'
  ]
};

// Visual contexts (kept family-friendly)
const VISUAL_CONTEXTS = {
  friend: { env: 'bright living room with comfortable furniture', mood: 'warm friendly' },
  kindness: { env: 'sunny park with trees and flowers', mood: 'cheerful hopeful' },
  listen: { env: 'cozy library with bookshelves', mood: 'quiet attentive' },
  patience: { env: 'peaceful garden with plants', mood: 'calm serene' },
  gratitude: { env: 'sunny field at golden hour', mood: 'thankful joyful' },
  courage: { env: 'mountain vista at sunrise', mood: 'brave determined' },
  curious: { env: 'study room with maps and globe', mood: 'inquisitive excited' },
  body: { env: 'bright wellness studio', mood: 'healthy balanced' },
  breath: { env: 'fresh outdoor setting with trees', mood: 'refreshing calm' },
  move: { env: 'outdoor playground or park', mood: 'active energetic' },
  rest: { env: 'cozy bedroom with soft lighting', mood: 'peaceful restful' },
  energy: { env: 'science classroom with equipment', mood: 'exciting educational' },
  water: { env: 'lakeside with mountains', mood: 'serene wonder' },
  cloud: { env: 'hilltop with blue sky and clouds', mood: 'airy magical' },
  light: { env: 'bright science lab with prisms', mood: 'illuminating discovery' },
  sound: { env: 'music classroom with instruments', mood: 'harmonious joyful' },
  seed: { env: 'garden with plants and sunshine', mood: 'growing nurturing' },
  star: { env: 'observatory with telescope', mood: 'cosmic wonder' },
  default: { env: 'bright modern classroom', mood: 'educational engaging' }
};

function getContextForTopic(topic) {
  const topicLower = topic.toLowerCase();
  for (const [keyword, context] of Object.entries(VISUAL_CONTEXTS)) {
    if (keyword !== 'default' && topicLower.includes(keyword)) {
      return context;
    }
  }
  return VISUAL_CONTEXTS.default;
}

// ═══════════════════════════════════════════════════════════════
// RETRY LOGIC WITH PROMPT VARIATIONS
// ═══════════════════════════════════════════════════════════════

async function downloadImage(url, outputPath) {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    const file = fs.createWriteStream(outputPath);
    const handleResponse = (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        https.get(response.headers.location, handleResponse).on('error', reject);
      } else {
        response.pipe(file);
        file.on('finish', () => { file.close(); resolve(); });
      }
    };
    https.get(url, handleResponse).on('error', reject);
  });
}

async function generateWithRetry(prompt, outputPath, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const prediction = await replicate.predictions.create({
        model: 'lucataco/flux-dev-lora',
        version: 'a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
        input: {
          prompt: prompt,
          hf_lora: KELLY_LORA,
          lora_scale: 0.85,
          num_outputs: 1,
          aspect_ratio: '16:9',
          output_format: 'png',
          guidance_scale: 3.5,
          num_inference_steps: 28,
          seed: attempt === 1 ? undefined : Math.floor(Math.random() * 1000000) // Different seed on retry
        }
      });
      
      let result = prediction;
      while (result.status !== 'succeeded' && result.status !== 'failed') {
        await new Promise(r => setTimeout(r, 2000));
        result = await replicate.predictions.get(prediction.id);
      }
      
      if (result.status === 'failed') {
        const errorMsg = result.error || '';
        
        // If NSFW flag, retry with different seed
        if (errorMsg.includes('NSFW') && attempt < maxRetries) {
          console.log(` ⚠️ NSFW flag (attempt ${attempt}/${maxRetries}), retrying...`);
          await new Promise(r => setTimeout(r, 1000));
          continue;
        }
        
        return { success: false, error: errorMsg, attempts: attempt };
      }
      
      const imageUrl = result.output?.[0];
      if (imageUrl) {
        await downloadImage(imageUrl, outputPath);
        return { success: true, cost: 0.04 * attempt, attempts: attempt };
      }
      
      return { success: false, error: 'No URL', attempts: attempt };
      
    } catch (error) {
      if (attempt === maxRetries) {
        return { success: false, error: error.message, attempts: attempt };
      }
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  
  return { success: false, error: 'Max retries exceeded', attempts: maxRetries };
}

async function generateLessonPhases(dayNumber, topic) {
  const paddedDay = String(dayNumber).padStart(3, '0');
  const dayDir = path.join(OUTPUT_BASE, paddedDay);
  const context = getContextForTopic(topic);
  
  console.log(`\n📚 Day ${dayNumber}: ${topic}`);
  
  let success = 0, cost = 0;
  const failedPhases = [];
  
  const phases = ['hook', 'q1', 'q2', 'q3', 'wisdom'];
  
  for (const phaseName of phases) {
    const outputPath = path.join(dayDir, `${phaseName}.png`);
    
    if (fs.existsSync(outputPath)) {
      console.log(`   ⏭️ ${phaseName}: exists`);
      success++;
      continue;
    }
    
    // Get a random safe pose variation
    const poseOptions = SAFE_POSES[phaseName];
    const pose = poseOptions[Math.floor(Math.random() * poseOptions.length)];
    
    // Build safe prompt
    const prompt = `${SAFETY_PREFIX}kelly, photorealistic professional woman educator, late 20s, brown wavy hair with highlights, hazel eyes, wearing modest powder blue sweater, in ${context.env}, ${pose}, mood: ${context.mood}${SAFETY_SUFFIX}, cinematic lighting, 8K`;
    
    process.stdout.write(`   🎨 ${phaseName}...`);
    const result = await generateWithRetry(prompt, outputPath);
    
    if (result.success) {
      if (result.attempts > 1) {
        console.log(` ✅ (retry ${result.attempts})`);
      } else {
        console.log(' ✅');
      }
      success++;
      cost += result.cost;
    } else {
      console.log(` ❌ ${result.error}`);
      failedPhases.push({ day: dayNumber, phase: phaseName, error: result.error });
    }
    
    await new Promise(r => setTimeout(r, 500));
  }
  
  return { success, cost, failedPhases };
}

// ═══════════════════════════════════════════════════════════════
// FAILED IMAGES LOG
// Save failed images for manual review/regeneration
// ═══════════════════════════════════════════════════════════════

async function saveFailed(failedList) {
  const logPath = 'logs/failed-generations.json';
  fs.mkdirSync('logs', { recursive: true });
  
  let existing = [];
  if (fs.existsSync(logPath)) {
    existing = JSON.parse(fs.readFileSync(logPath, 'utf8'));
  }
  
  const updated = [...existing, ...failedList];
  fs.writeFileSync(logPath, JSON.stringify(updated, null, 2));
  
  console.log(`\n📝 Logged ${failedList.length} failed images to ${logPath}`);
}

async function main() {
  const startDay = parseInt(process.argv[2]) || 8;
  const endDay = parseInt(process.argv[3]) || 365;
  
  console.log('🛡️ ROBUST KELLY VISUAL GENERATOR');
  console.log('=================================');
  console.log('Features: NSFW retry, safe prompts, failure logging\n');
  
  // Fetch lessons
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('day_number, topic')
    .gte('day_number', startDay)
    .lte('day_number', endDay)
    .order('day_number');
  
  if (error) {
    console.error('❌ Database error:', error.message);
    process.exit(1);
  }
  
  console.log(`📥 Found ${lessons.length} lessons`);
  console.log(`📊 Estimated: ${lessons.length * 5} images, ~$${(lessons.length * 5 * 0.04).toFixed(2)}\n`);
  
  let totalSuccess = 0, totalCost = 0;
  const allFailed = [];
  const startTime = Date.now();
  
  for (const lesson of lessons) {
    const result = await generateLessonPhases(lesson.day_number, lesson.topic);
    totalSuccess += result.success;
    totalCost += result.cost;
    allFailed.push(...result.failedPhases);
    
    const progress = ((lesson.day_number - startDay + 1) / lessons.length * 100).toFixed(1);
    const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
    
    console.log(`   📊 ${progress}% | $${totalCost.toFixed(2)} | ${elapsed}min | Failed: ${allFailed.length}`);
  }
  
  // Save failed for manual review
  if (allFailed.length > 0) {
    await saveFailed(allFailed);
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('🎉 GENERATION COMPLETE!');
  console.log(`✅ Success: ${totalSuccess} images`);
  console.log(`❌ Failed: ${allFailed.length} images (logged for retry)`);
  console.log(`💰 Cost: $${totalCost.toFixed(2)}`);
  console.log(`⏱️ Time: ${((Date.now() - startTime) / 1000 / 60).toFixed(1)} minutes`);
}

main().catch(console.error);


