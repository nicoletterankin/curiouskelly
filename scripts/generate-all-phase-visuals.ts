/**
 * COMPREHENSIVE PHASE VISUAL GENERATOR
 * Chief Academic Officer Edition
 * 
 * Generates pedagogically-intentional visuals for ALL 365 lessons
 * using the Curious Kelly LoRA model on Replicate/Flux.
 * 
 * Usage:
 *   npx ts-node scripts/generate-all-phase-visuals.ts --range=1-10
 *   npx ts-node scripts/generate-all-phase-visuals.ts --day=57
 *   npx ts-node scripts/generate-all-phase-visuals.ts --missing
 *   npx ts-node scripts/generate-all-phase-visuals.ts --all
 * 
 * Environment:
 *   REPLICATE_API_TOKEN - Required
 *   SUPABASE_URL - Required
 *   SUPABASE_SERVICE_KEY - Required
 */

import * as dotenv from 'dotenv';
dotenv.config();

import Replicate from 'replicate';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const CONFIG = {
  // Kelly LoRA - Our trained model
  LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_MODEL: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  LORA_SCALE: 0.85,
  
  // Kelly's consistent appearance (LOCKED - do not modify)
  KELLY_BASE: `kelly, photorealistic woman named Kelly, late 20s to early 30s, 
    brown wavy shoulder-length hair with caramel and honey highlights center-parted, 
    hazel-brown almond-shaped eyes, soft symmetrical features with natural makeup, 
    light-medium warm skin tone with healthy glow, 
    wearing soft powder blue cashmere crewneck sweater, 
    warm but professional expression, intelligent curious eyes`,
  
  // Image settings
  ASPECT_RATIO: '16:9',
  OUTPUT_FORMAT: 'png',
  GUIDANCE_SCALE: 3.5,
  INFERENCE_STEPS: 28,
  
  // Output directories
  OUTPUT_DIR_PHASES: path.join(process.cwd(), 'public', 'kelly', 'phases'),
  OUTPUT_DIR_LESSONS: path.join(process.cwd(), 'public', 'kelly', 'lessons'),
  
  // Rate limiting (ms between requests)
  RATE_LIMIT: 2000,
  BATCH_SIZE: 10,
};

// ═══════════════════════════════════════════════════════════════════
// VISUAL CONTEXT LIBRARY
// Maps topic keywords to visual environments and props
// ═══════════════════════════════════════════════════════════════════

const VISUAL_CONTEXTS: Record<string, {
  environment: string;
  props: string[];
  mood: string;
  palette: string;
}> = {
  // NATURE & EARTH SCIENCE
  water: {
    environment: 'lakeside with mountains, misty morning, ice crystals',
    props: ['glass of water', 'ice cubes', 'steam rising', 'rain clouds'],
    mood: 'fluid, transformative, wonder',
    palette: 'blues, silver, translucent'
  },
  clouds: {
    environment: 'open sky with dramatic clouds, sun rays, hilltop vista',
    props: ['fluffy cumulus clouds', 'rain drops', 'sun beams', 'weather vane'],
    mood: 'airy, expansive, magical',
    palette: 'sky blues, white, golden sunlight'
  },
  light: {
    environment: 'bright laboratory with prisms, rainbow spectrum, clean surfaces',
    props: ['glass prism with rainbow', 'beam of light', 'mirror', 'flashlight'],
    mood: 'bright, scientific, illuminating',
    palette: 'spectrum colors, white, crystal'
  },
  sound: {
    environment: 'music room with instruments, visible sound waves, acoustic panels',
    props: ['tuning fork', 'guitar strings', 'speaker cone', 'ripples in water'],
    mood: 'dynamic, rhythmic, vibrant',
    palette: 'warm oranges, vibrating patterns'
  },
  seed: {
    environment: 'garden setting, rich soil, warm spring sunshine',
    props: ['cross-section of seed', 'sprouting seedling', 'gardening tools', 'roots'],
    mood: 'nurturing, hopeful, growth',
    palette: 'earth browns, fresh greens, golden light'
  },
  star: {
    environment: 'night sky observatory, constellation backdrop, cosmic wonder',
    props: ['bright star close-up', 'telescope', 'constellation map', 'spiral galaxy'],
    mood: 'cosmic, awe-inspiring, vast',
    palette: 'deep blues, silver stars, cosmic purple'
  },
  friend: {
    environment: 'cozy living room, warm lighting, comfortable atmosphere',
    props: ['two hands clasping', 'photo album', 'board game', 'shared meal'],
    mood: 'warm, connected, trusting',
    palette: 'warm yellows, soft oranges, comfortable neutrals'
  },
  kindness: {
    environment: 'sunny park, community garden, people helping each other',
    props: ['heart symbol', 'helping hands', 'gift being given', 'smile'],
    mood: 'generous, rippling outward, hopeful',
    palette: 'soft pinks, warm reds, gentle greens'
  },
  listen: {
    environment: 'quiet library, comfortable chairs, attentive setting',
    props: ['ear close-up', 'person leaning in', 'closed book', 'cup of tea'],
    mood: 'attentive, present, understanding',
    palette: 'soft browns, quiet blues, warm lighting'
  },
  patience: {
    environment: 'zen garden, hourglass, meditative space',
    props: ['hourglass', 'growing plant timelapse', 'calm water', 'sunrise'],
    mood: 'calm, enduring, rewarding',
    palette: 'earth tones, soft greens, amber'
  },
  gratitude: {
    environment: 'golden hour field, abundance, harvest setting',
    props: ['journal and pen', 'hands in prayer', 'abundance of fruits', 'sunset'],
    mood: 'thankful, abundant, glowing',
    palette: 'golden yellows, warm oranges, soft light'
  },
  courage: {
    environment: 'mountain peak, challenging path, triumphant setting',
    props: ['lion silhouette', 'climbing rope', 'path through forest', 'shield'],
    mood: 'brave, determined, victorious',
    palette: 'bold reds, deep blues, mountain grays'
  },
  curious: {
    environment: 'explorer\'s study, maps and discoveries, wonder cabinet',
    props: ['magnifying glass', 'question marks', 'open book', 'compass'],
    mood: 'inquisitive, excited, discovering',
    palette: 'warm browns, exploration greens, discovery gold'
  },
  body: {
    environment: 'clean wellness space, anatomical models, healthy lifestyle',
    props: ['anatomical heart', 'skeleton model', 'healthy food', 'exercise equipment'],
    mood: 'appreciative, aware, balanced',
    palette: 'warm skin tones, medical whites, healthy greens'
  },
  breath: {
    environment: 'mountain air, fresh morning, open lungs visualization',
    props: ['lungs illustration', 'fresh air', 'yoga pose', 'dandelion seeds'],
    mood: 'refreshing, vital, centering',
    palette: 'sky blues, pure whites, soft greens'
  },
  move: {
    environment: 'active outdoors, playground, sports field',
    props: ['running shoes', 'bicycle', 'dancing figure', 'stretched muscles'],
    mood: 'energetic, joyful, alive',
    palette: 'active oranges, energetic yellows, movement blues'
  },
  rest: {
    environment: 'cozy bedroom, peaceful night, recovery space',
    props: ['comfortable bed', 'moon and stars', 'sleeping face', 'dream clouds'],
    mood: 'peaceful, restorative, calm',
    palette: 'deep purples, soft grays, starlight silver'
  },
  energy: {
    environment: 'power plant, lightning, energy transformation lab',
    props: ['battery', 'lightning bolt', 'solar panel', 'wind turbine'],
    mood: 'powerful, transforming, electric',
    palette: 'electric blues, energy yellows, power reds'
  },
  sense: {
    environment: 'sensory garden, diverse textures and colors, exploration',
    props: ['eye close-up', 'nose', 'ear', 'hand touching texture', 'tongue'],
    mood: 'aware, experiencing, discovering',
    palette: 'rainbow sensory colors, organic textures'
  },
  // DEFAULT fallback
  default: {
    environment: 'bright modern learning studio, clean background, professional',
    props: ['open book', 'light bulb', 'question mark', 'thumbs up'],
    mood: 'curious, educational, engaging',
    palette: 'clean whites, learning blues, warm accents'
  }
};

// ═══════════════════════════════════════════════════════════════════
// PHASE PROMPT TEMPLATES
// Each phase has a specific pedagogical purpose
// ═══════════════════════════════════════════════════════════════════

function getPhasePrompts(topic: string, context: typeof VISUAL_CONTEXTS['default']) {
  const { environment, props, mood } = context;
  
  return {
    hook: {
      name: 'hook',
      prompt: `${CONFIG.KELLY_BASE}, 
        standing in ${environment}, 
        welcoming open stance with arms slightly open in invitation,
        warm genuine smile showing excitement about today's topic: "${topic}",
        looking directly at viewer with curiosity and enthusiasm,
        full body visible, natural confident posture,
        mood: ${mood},
        cinematic photography, natural lighting, 8K, shallow depth of field`,
      purpose: 'Capture attention, establish topic presence'
    },
    
    q1: {
      name: 'q1',
      prompt: `${CONFIG.KELLY_BASE},
        in ${environment},
        holding and examining ${props[0]} with genuine fascination,
        curious engaged expression, eyebrows slightly raised in wonder,
        pointing at or gesturing toward the object to highlight detail,
        teaching moment - sharing first discovery about "${topic}",
        upper body focus with prop clearly visible,
        cinematic photography, soft directional lighting, 8K`,
      purpose: 'First question/fact - introduce core concept'
    },
    
    q2: {
      name: 'q2',
      prompt: `${CONFIG.KELLY_BASE},
        in ${environment},
        thoughtful contemplative expression, chin resting gently on hand,
        looking at ${props[1]} with deep consideration,
        pondering a deeper question about "${topic}",
        seated in director's chair or comfortable position,
        inviting the learner to think more deeply,
        soft lighting creating intimate learning moment, 8K`,
      purpose: 'Second question/fact - deeper exploration'
    },
    
    q3: {
      name: 'q3',
      prompt: `${CONFIG.KELLY_BASE},
        in ${environment},
        encouraging supportive expression with warm smile,
        gesturing toward ${props[2]} with open hand,
        leaning forward slightly with engagement and enthusiasm,
        explaining an important concept about "${topic}",
        body language says "you're getting this!",
        warm educational lighting, confident teaching pose, 8K`,
      purpose: 'Third question/fact - building understanding'
    },
    
    wisdom: {
      name: 'wisdom',
      prompt: `${CONFIG.KELLY_BASE},
        in ${environment} at golden hour,
        standing proudly with hand placed gently on heart,
        satisfied accomplished smile showing pride in learner's journey,
        ${props[3]} visible in background as symbol of mastery,
        sense of completion, growth, and wisdom achieved,
        looking at camera with warmth and encouragement,
        cinematic wide shot, inspirational golden light, 8K`,
      purpose: 'Final wisdom - celebration of learning'
    }
  };
}

// ═══════════════════════════════════════════════════════════════════
// CONTEXT MATCHER
// Finds the best visual context for a given topic
// ═══════════════════════════════════════════════════════════════════

function getContextForTopic(topic: string): typeof VISUAL_CONTEXTS['default'] {
  const topicLower = topic.toLowerCase();
  
  // Match topic to context
  for (const [keyword, context] of Object.entries(VISUAL_CONTEXTS)) {
    if (keyword === 'default') continue;
    if (topicLower.includes(keyword)) {
      return context;
    }
  }
  
  // Try partial matches
  const partialMatches: Record<string, string[]> = {
    water: ['rain', 'ocean', 'lake', 'river', 'liquid', 'ice', 'steam'],
    clouds: ['sky', 'weather', 'atmosphere', 'air'],
    light: ['sun', 'bright', 'ray', 'optic', 'color', 'rainbow'],
    sound: ['music', 'hear', 'audio', 'wave', 'noise', 'voice'],
    seed: ['plant', 'grow', 'tree', 'leaf', 'flower', 'garden', 'nature'],
    star: ['space', 'planet', 'moon', 'galaxy', 'universe', 'cosmos', 'astro'],
    friend: ['relationship', 'trust', 'connection', 'together'],
    kindness: ['help', 'care', 'generous', 'share', 'give'],
    listen: ['hear', 'attention', 'focus', 'understand'],
    patience: ['wait', 'time', 'slow', 'calm', 'persistent'],
    gratitude: ['thank', 'appreciate', 'grateful', 'blessing'],
    courage: ['brave', 'fear', 'strong', 'hero', 'challenge'],
    curious: ['wonder', 'question', 'discover', 'explore', 'learn'],
    body: ['health', 'muscle', 'bone', 'organ', 'cell', 'blood', 'brain'],
    breath: ['lung', 'oxygen', 'inhale', 'exhale', 'respir'],
    move: ['exercise', 'run', 'walk', 'dance', 'sport', 'physical'],
    rest: ['sleep', 'dream', 'relax', 'recover', 'night'],
    energy: ['power', 'electric', 'fuel', 'force', 'solar', 'battery'],
    sense: ['see', 'smell', 'taste', 'touch', 'feel', 'perception']
  };
  
  for (const [context, keywords] of Object.entries(partialMatches)) {
    if (keywords.some(kw => topicLower.includes(kw))) {
      return VISUAL_CONTEXTS[context];
    }
  }
  
  return VISUAL_CONTEXTS['default'];
}

// ═══════════════════════════════════════════════════════════════════
// DATABASE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

const supabase = createClient(
  process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_KEY!
);

interface Lesson {
  id: string;
  day_number: number;
  topic: string;
  icon_emoji: string | null;
  universal_truth: string | null;
}

async function getAllLessons(): Promise<Lesson[]> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, icon_emoji, universal_truth')
    .order('day_number');
  
  if (error) throw error;
  return data || [];
}

async function getLessonsInRange(start: number, end: number): Promise<Lesson[]> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, icon_emoji, universal_truth')
    .gte('day_number', start)
    .lte('day_number', end)
    .order('day_number');
  
  if (error) throw error;
  return data || [];
}

async function getMissingPhaseLessons(): Promise<Lesson[]> {
  // Get all lessons
  const all = await getAllLessons();
  
  // Check which don't have phase visuals
  const missing: Lesson[] = [];
  for (const lesson of all) {
    const dayDir = path.join(CONFIG.OUTPUT_DIR_PHASES, String(lesson.day_number).padStart(3, '0'));
    const hookPath = path.join(dayDir, 'hook.png');
    
    if (!fs.existsSync(hookPath)) {
      missing.push(lesson);
    }
  }
  
  return missing;
}

// ═══════════════════════════════════════════════════════════════════
// IMAGE GENERATION
// ═══════════════════════════════════════════════════════════════════

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

async function downloadImage(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : require('http');
    protocol.get(url, (response: any) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        downloadImage(response.headers.location).then(resolve).catch(reject);
        return;
      }
      const chunks: Buffer[] = [];
      response.on('data', (chunk: Buffer) => chunks.push(chunk));
      response.on('end', () => resolve(Buffer.concat(chunks)));
      response.on('error', reject);
    }).on('error', reject);
  });
}

async function generateImage(prompt: string, outputPath: string): Promise<{ success: boolean; error?: string; size?: number }> {
  try {
    console.log(`     Generating image...`);
    
    const output = await replicate.run(CONFIG.LORA_MODEL, {
      input: {
        prompt: prompt,
        hf_lora: CONFIG.LORA_URL,
        lora_scale: CONFIG.LORA_SCALE,
        num_outputs: 1,
        aspect_ratio: CONFIG.ASPECT_RATIO,
        output_format: CONFIG.OUTPUT_FORMAT,
        guidance_scale: CONFIG.GUIDANCE_SCALE,
        num_inference_steps: CONFIG.INFERENCE_STEPS
      }
    }) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    if (!imageUrl) {
      return { success: false, error: 'No image URL returned' };
    }
    
    const buffer = await downloadImage(imageUrl);
    
    // Ensure directory exists
    const dir = path.dirname(outputPath);
    fs.mkdirSync(dir, { recursive: true });
    
    fs.writeFileSync(outputPath, buffer);
    
    return { success: true, size: buffer.length };
    
  } catch (error: any) {
    return { success: false, error: error.message };
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN GENERATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

async function generateLessonPhaseVisuals(lesson: Lesson, phases = ['hook', 'q1', 'q2', 'q3', 'wisdom']): Promise<{ success: number; failed: number }> {
  const { day_number, topic, icon_emoji } = lesson;
  const paddedDay = String(day_number).padStart(3, '0');
  const dayDir = path.join(CONFIG.OUTPUT_DIR_PHASES, paddedDay);
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📚 Day ${day_number}: ${icon_emoji || '📖'} ${topic}`);
  console.log(`${'═'.repeat(60)}`);
  
  // Get visual context for this topic
  const context = getContextForTopic(topic);
  console.log(`📍 Environment: ${context.environment.substring(0, 50)}...`);
  console.log(`🎨 Mood: ${context.mood}`);
  
  // Get phase prompts
  const phasePrompts = getPhasePrompts(topic, context);
  
  let success = 0;
  let failed = 0;
  
  for (const phaseName of phases) {
    const phase = phasePrompts[phaseName as keyof typeof phasePrompts];
    if (!phase) {
      console.log(`  ⚠️ Unknown phase: ${phaseName}`);
      continue;
    }
    
    const outputPath = path.join(dayDir, `${phaseName}.png`);
    
    // Skip if already exists
    if (fs.existsSync(outputPath)) {
      console.log(`  ⏭️ ${phaseName}: Already exists`);
      success++;
      continue;
    }
    
    console.log(`\n  🎬 ${phaseName.toUpperCase()}`);
    console.log(`     Purpose: ${phase.purpose}`);
    
    const result = await generateImage(phase.prompt, outputPath);
    
    if (result.success) {
      console.log(`     ✅ Saved: ${phaseName}.png (${((result.size || 0) / 1024).toFixed(1)} KB)`);
      success++;
    } else {
      console.log(`     ❌ Failed: ${result.error}`);
      failed++;
    }
    
    // Rate limiting
    await new Promise(r => setTimeout(r, CONFIG.RATE_LIMIT));
  }
  
  console.log(`\n📊 Day ${day_number} Result: ${success}/${phases.length} phases generated`);
  return { success, failed };
}

async function generateRange(start: number, end: number) {
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  PHASE VISUAL GENERATOR - Days ${start} to ${end}`);
  console.log(`${'█'.repeat(60)}`);
  
  const lessons = await getLessonsInRange(start, end);
  console.log(`Found ${lessons.length} lessons to process\n`);
  
  const results = { totalSuccess: 0, totalFailed: 0, days: [] as any[] };
  
  for (const lesson of lessons) {
    const { success, failed } = await generateLessonPhaseVisuals(lesson);
    results.totalSuccess += success;
    results.totalFailed += failed;
    results.days.push({
      day: lesson.day_number,
      topic: lesson.topic,
      success,
      failed
    });
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📊 FINAL RESULTS`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`✅ Success: ${results.totalSuccess} images`);
  console.log(`❌ Failed: ${results.totalFailed} images`);
  console.log(`📅 Days processed: ${lessons.length}`);
  
  return results;
}

async function generateMissing() {
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  PHASE VISUAL GENERATOR - Missing Days Only`);
  console.log(`${'█'.repeat(60)}`);
  
  const missing = await getMissingPhaseLessons();
  console.log(`Found ${missing.length} days without phase visuals\n`);
  
  if (missing.length === 0) {
    console.log('✅ All days have phase visuals!');
    return { totalSuccess: 0, totalFailed: 0, days: [] };
  }
  
  const results = { totalSuccess: 0, totalFailed: 0, days: [] as any[] };
  
  for (const lesson of missing) {
    const { success, failed } = await generateLessonPhaseVisuals(lesson);
    results.totalSuccess += success;
    results.totalFailed += failed;
    results.days.push({
      day: lesson.day_number,
      topic: lesson.topic,
      success,
      failed
    });
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📊 FINAL RESULTS`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`✅ Success: ${results.totalSuccess} images`);
  console.log(`❌ Failed: ${results.totalFailed} images`);
  console.log(`📅 Days processed: ${missing.length}`);
  
  return results;
}

// ═══════════════════════════════════════════════════════════════════
// CLI INTERFACE
// ═══════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  
  console.log('\n🎨 CURIOUS KELLY PHASE VISUAL GENERATOR');
  console.log('   Chief Academic Officer Edition\n');
  
  // Check environment
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not set');
    process.exit(1);
  }
  if (!process.env.SUPABASE_URL && !process.env.NEXT_PUBLIC_SUPABASE_URL) {
    console.error('❌ SUPABASE_URL not set');
    process.exit(1);
  }
  if (!process.env.SUPABASE_SERVICE_KEY) {
    console.error('❌ SUPABASE_SERVICE_KEY not set');
    process.exit(1);
  }
  
  // Parse arguments
  const dayArg = args.find(a => a.startsWith('--day='));
  const rangeArg = args.find(a => a.startsWith('--range='));
  const missingArg = args.includes('--missing');
  const allArg = args.includes('--all');
  const dryRun = args.includes('--dry-run');
  
  if (dryRun) {
    console.log('🔍 DRY RUN MODE - No images will be generated\n');
    const missing = await getMissingPhaseLessons();
    console.log(`Days needing phase visuals: ${missing.length}`);
    console.log(`First 10: ${missing.slice(0, 10).map(l => l.day_number).join(', ')}`);
    console.log(`Estimated images: ${missing.length * 5}`);
    console.log(`Estimated cost: ~$${(missing.length * 5 * 0.04).toFixed(2)} (at $0.04/image)`);
    return;
  }
  
  if (dayArg) {
    const day = parseInt(dayArg.split('=')[1]);
    const lessons = await getLessonsInRange(day, day);
    if (lessons.length > 0) {
      await generateLessonPhaseVisuals(lessons[0]);
    } else {
      console.error(`❌ Lesson ${day} not found`);
    }
  } else if (rangeArg) {
    const [start, end] = rangeArg.split('=')[1].split('-').map(Number);
    await generateRange(start, end);
  } else if (missingArg) {
    await generateMissing();
  } else if (allArg) {
    await generateRange(1, 365);
  } else {
    console.log(`
Usage:
  npx ts-node scripts/generate-all-phase-visuals.ts --day=1        # Single day
  npx ts-node scripts/generate-all-phase-visuals.ts --range=1-30   # Day range
  npx ts-node scripts/generate-all-phase-visuals.ts --missing      # Only missing days
  npx ts-node scripts/generate-all-phase-visuals.ts --all          # All 365 days
  npx ts-node scripts/generate-all-phase-visuals.ts --dry-run      # Preview what would be generated

Environment Variables Required:
  REPLICATE_API_TOKEN
  SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)
  SUPABASE_SERVICE_KEY
    `);
  }
}

main().catch(console.error);

