/**
 * FULL 365 LESSON PHASE VISUAL GENERATOR
 * Pulls topics from Supabase, generates with Kelly LoRA
 */

import Replicate from 'replicate';
import { createClient } from '@supabase/supabase-js';
import fs from 'fs';
import https from 'https';
import path from 'path';

const replicate = new Replicate();
const supabase = createClient(
  'https://tvjalxxsyryjphkforjv.supabase.co',
  process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI'
);

const KELLY_LORA = 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors';
const OUTPUT_BASE = 'public/kelly/phases';

// Visual contexts for different topics
const VISUAL_CONTEXTS = {
  friend: { env: 'cozy living room with warm lighting, comfortable atmosphere', mood: 'warm, connected, trusting' },
  kindness: { env: 'sunny community park, people helping each other', mood: 'generous, hopeful' },
  listen: { env: 'quiet library with comfortable reading chairs, warm wood', mood: 'attentive, understanding' },
  patience: { env: 'zen garden with raked sand, peaceful space', mood: 'calm, enduring' },
  gratitude: { env: 'golden hour field, abundance, harvest setting', mood: 'thankful, glowing' },
  courage: { env: 'mountain peak at sunrise, triumphant setting', mood: 'brave, victorious' },
  curious: { env: 'explorer study room with maps, discoveries', mood: 'inquisitive, discovering' },
  body: { env: 'clean wellness space, healthy lifestyle imagery', mood: 'balanced, aware' },
  breath: { env: 'mountain air setting, fresh morning', mood: 'refreshing, vital' },
  move: { env: 'active outdoors, playground, sports field', mood: 'energetic, joyful' },
  rest: { env: 'cozy bedroom, peaceful night sky', mood: 'peaceful, restorative' },
  energy: { env: 'energy transformation lab, power visualization', mood: 'powerful, electric' },
  sense: { env: 'sensory garden with diverse textures', mood: 'aware, discovering' },
  water: { env: 'lakeside with mountains, misty morning', mood: 'fluid, wonder' },
  cloud: { env: 'open sky with dramatic clouds, sun rays', mood: 'airy, magical' },
  light: { env: 'bright laboratory with prisms, rainbow', mood: 'illuminating, discovery' },
  sound: { env: 'music room with instruments', mood: 'dynamic, vibrant' },
  music: { env: 'concert hall with instruments, warm stage', mood: 'harmonious, moving' },
  seed: { env: 'garden setting with rich soil, spring sunshine', mood: 'nurturing, growth' },
  plant: { env: 'garden setting with rich soil, spring sunshine', mood: 'nurturing, growth' },
  grow: { env: 'garden setting with rich soil, spring sunshine', mood: 'nurturing, growth' },
  star: { env: 'night sky observatory, constellation backdrop', mood: 'cosmic, vast' },
  space: { env: 'night sky observatory, stars visible', mood: 'cosmic, wonder' },
  moon: { env: 'night beach with moonlight on water', mood: 'serene, mysterious' },
  sun: { env: 'golden sunrise over landscape', mood: 'warm, powerful' },
  rain: { env: 'rainy day with umbrella, puddles reflecting', mood: 'refreshing, cozy' },
  thunder: { env: 'dramatic storm clouds, safe observation deck', mood: 'powerful, exciting' },
  rainbow: { env: 'after rain with rainbow visible', mood: 'hopeful, magical' },
  wind: { env: 'windy hilltop with grass swaying', mood: 'free, dynamic' },
  fire: { env: 'cozy fireplace setting, warm glow', mood: 'warm, mesmerizing' },
  ice: { env: 'winter wonderland, frost crystals', mood: 'crisp, beautiful' },
  magnet: { env: 'science laboratory with magnets', mood: 'curious, magnetic' },
  electric: { env: 'energy lab with safe electrical displays', mood: 'energizing, bright' },
  gravity: { env: 'space station with floating objects', mood: 'weightless, wonder' },
  wave: { env: 'ocean shore with gentle waves', mood: 'rhythmic, peaceful' },
  bubble: { env: 'sunny garden with soap bubbles floating', mood: 'playful, magical' },
  crystal: { env: 'crystal cave with colorful formations', mood: 'sparkling, wonder' },
  stone: { env: 'natural history museum with fossils', mood: 'ancient, discovery' },
  fossil: { env: 'natural history museum with fossils', mood: 'ancient, discovery' },
  color: { env: 'art studio with colorful paints', mood: 'vibrant, creative' },
  pattern: { env: 'nature setting with visible patterns', mood: 'ordered, beautiful' },
  story: { env: 'cozy reading nook with books', mood: 'imaginative, warm' },
  imagination: { env: 'dreamy cloud-like setting', mood: 'creative, limitless' },
  memory: { env: 'photo gallery with warm memories', mood: 'nostalgic, precious' },
  time: { env: 'clocktower at sunset', mood: 'contemplative, flowing' },
  change: { env: 'seasons transitioning, nature in flux', mood: 'accepting, dynamic' },
  question: { env: 'classroom with curiosity elements', mood: 'inquiring, engaged' },
  mirror: { env: 'elegant room with mirrors', mood: 'reflective, curious' },
  shadow: { env: 'sunny day with interesting shadows', mood: 'playful, mysterious' },
  season: { env: 'four seasons represented together', mood: 'cyclical, natural' },
  day: { env: 'sunrise to sunset landscape', mood: 'bright, hopeful' },
  night: { env: 'starry night sky, peaceful', mood: 'calm, wonder' },
  default: { env: 'bright modern learning studio', mood: 'curious, educational' }
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

const PHASES = [
  { name: 'hook', pose: 'welcoming open stance with arms slightly open, warm genuine smile' },
  { name: 'q1', pose: 'holding educational object with fascination, curious expression' },
  { name: 'q2', pose: 'thoughtful expression, chin resting on hand, pondering' },
  { name: 'q3', pose: 'encouraging smile, gesturing with open hand' },
  { name: 'wisdom', pose: 'hand on heart, proud accomplished smile' },
];

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

async function generateImage(prompt, outputPath) {
  try {
    const prediction = await replicate.predictions.create({
      model: 'lucataco/flux-dev-lora',
      version: 'a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
      input: {
        prompt, hf_lora: KELLY_LORA, lora_scale: 0.85,
        num_outputs: 1, aspect_ratio: '16:9', output_format: 'png',
        guidance_scale: 3.5, num_inference_steps: 28
      }
    });
    
    let result = prediction;
    while (result.status !== 'succeeded' && result.status !== 'failed') {
      await new Promise(r => setTimeout(r, 2000));
      result = await replicate.predictions.get(prediction.id);
    }
    
    if (result.status === 'failed') return { success: false, error: result.error };
    
    const imageUrl = result.output?.[0];
    if (imageUrl) {
      await downloadImage(imageUrl, outputPath);
      return { success: true, cost: 0.04 };
    }
    return { success: false, error: 'No URL' };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

async function generateLessonPhases(dayNumber, topic) {
  const paddedDay = String(dayNumber).padStart(3, '0');
  const dayDir = path.join(OUTPUT_BASE, paddedDay);
  const context = getContextForTopic(topic);
  
  console.log(`\n📚 Day ${dayNumber}: ${topic}`);
  console.log(`   🎨 ${context.env.substring(0, 40)}...`);
  
  let success = 0, cost = 0;
  
  for (const phase of PHASES) {
    const outputPath = path.join(dayDir, `${phase.name}.png`);
    
    if (fs.existsSync(outputPath)) {
      console.log(`   ⏭️ ${phase.name}: exists`);
      success++;
      continue;
    }
    
    const prompt = `kelly, photorealistic woman, late 20s, brown wavy hair with caramel highlights, hazel-brown eyes, powder blue cashmere sweater, in ${context.env}, ${phase.pose}, mood: ${context.mood}, cinematic, 8K`;
    
    process.stdout.write(`   🎨 ${phase.name}...`);
    const result = await generateImage(prompt, outputPath);
    
    if (result.success) {
      console.log(' ✅');
      success++;
      cost += result.cost;
    } else {
      console.log(` ❌ ${result.error}`);
    }
    
    await new Promise(r => setTimeout(r, 500));
  }
  
  return { success, cost };
}

async function main() {
  const startDay = parseInt(process.argv[2]) || 8;
  const endDay = parseInt(process.argv[3]) || 365;
  
  console.log('🚀 FULL 365 PHASE VISUAL GENERATOR');
  console.log('===================================\n');
  
  // Fetch all lessons from database
  console.log('📥 Fetching lessons from database...');
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
  
  console.log(`✅ Found ${lessons.length} lessons (Days ${startDay}-${endDay})`);
  console.log(`📊 Estimated: ${lessons.length * 5} images, ~$${(lessons.length * 5 * 0.04).toFixed(2)}\n`);
  
  let totalSuccess = 0, totalCost = 0;
  const startTime = Date.now();
  
  for (const lesson of lessons) {
    const result = await generateLessonPhases(lesson.day_number, lesson.topic);
    totalSuccess += result.success;
    totalCost += result.cost;
    
    const progress = ((lesson.day_number - startDay + 1) / lessons.length * 100).toFixed(1);
    const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
    const rate = totalSuccess / (elapsed || 1);
    const remaining = ((lessons.length * 5 - totalSuccess) / rate).toFixed(0);
    
    console.log(`   📊 ${progress}% | $${totalCost.toFixed(2)} | ${elapsed}min | ETA: ${remaining}min`);
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('🎉 GENERATION COMPLETE!');
  console.log(`✅ Images: ${totalSuccess}`);
  console.log(`💰 Cost: $${totalCost.toFixed(2)}`);
  console.log(`⏱️ Time: ${((Date.now() - startTime) / 1000 / 60).toFixed(1)} minutes`);
}

main().catch(console.error);


