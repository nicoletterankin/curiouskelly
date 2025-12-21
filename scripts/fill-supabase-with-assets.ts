#!/usr/bin/env npx tsx
/**
 * 🚀 FILL SUPABASE WITH ASSETS
 * 
 * Master orchestrator that:
 * 1. Generates phase visuals for each lesson
 * 2. Uploads to Supabase storage
 * 3. Registers URLs in kelly_video_assets table
 * 4. Updates core_lessons with hero/thumbnail URLs
 * 
 * Usage:
 *   npx tsx scripts/fill-supabase-with-assets.ts --range=1-10
 *   npx tsx scripts/fill-supabase-with-assets.ts --day=57
 *   npx tsx scripts/fill-supabase-with-assets.ts --all
 *   npx tsx scripts/fill-supabase-with-assets.ts --dry-run
 * 
 * Environment:
 *   REPLICATE_API_TOKEN - For image generation
 *   SUPABASE_URL
 *   SUPABASE_SERVICE_ROLE_KEY
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import Replicate from 'replicate';
import * as https from 'https';
import * as http from 'http';

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const CONFIG = {
  // Kelly LoRA Model
  LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_MODEL: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  LORA_SCALE: 0.85,
  
  // Storage buckets
  VISUALS_BUCKET: 'lesson-visuals',
  THUMBNAILS_BUCKET: 'lesson-thumbnails',
  
  // Rate limits
  RATE_LIMIT_MS: 2000,
  
  // Phases
  PHASES: ['hook', 'q1', 'q2', 'q3', 'wisdom'] as const,
};

// ═══════════════════════════════════════════════════════════════════════════
// KELLY BASE PROMPT (LOCKED)
// ═══════════════════════════════════════════════════════════════════════════

const KELLY_BASE = `kelly, photorealistic woman named Kelly, late 20s to early 30s, 
brown wavy shoulder-length hair with caramel and honey highlights center-parted, 
hazel-brown almond-shaped eyes, soft symmetrical features with natural makeup, 
light-medium warm skin tone with healthy glow, 
wearing soft powder blue cashmere crewneck sweater, 
warm but professional expression, intelligent curious eyes`;

// ═══════════════════════════════════════════════════════════════════════════
// PHASE PROMPT GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

function getPhasePrompt(topic: string, phase: string): string {
  const prompts: Record<string, string> = {
    hook: `${KELLY_BASE}, 
      standing in modern bright learning studio,
      welcoming open stance with arms slightly open in invitation,
      warm genuine smile showing excitement about today's topic: "${topic}",
      looking directly at viewer with curiosity and enthusiasm,
      full body visible, natural confident posture,
      cinematic photography, natural lighting, 8K, shallow depth of field`,
    
    q1: `${KELLY_BASE},
      in modern learning studio with relevant props,
      curious engaged expression, eyebrows slightly raised in wonder,
      pointing at or gesturing toward something interesting,
      teaching moment - sharing first discovery about "${topic}",
      upper body focus, natural lighting,
      cinematic photography, 8K`,
    
    q2: `${KELLY_BASE},
      thoughtful contemplative expression, chin resting gently on hand,
      pondering a deeper question about "${topic}",
      seated comfortably, inviting the learner to think more deeply,
      soft lighting creating intimate learning moment,
      cinematic photography, 8K`,
    
    q3: `${KELLY_BASE},
      encouraging supportive expression with warm smile,
      gesturing with open hand while explaining,
      leaning forward slightly with engagement and enthusiasm,
      body language says "you're getting this!",
      warm educational lighting, confident teaching pose,
      cinematic photography, 8K`,
    
    wisdom: `${KELLY_BASE},
      standing proudly at golden hour lighting,
      satisfied accomplished smile showing pride in learner's journey,
      hand placed gently on heart,
      sense of completion, growth, and wisdom achieved,
      looking at camera with warmth and encouragement,
      cinematic wide shot, inspirational golden light, 8K`,
  };
  
  return prompts[phase] || prompts.hook;
}

// ═══════════════════════════════════════════════════════════════════════════
// DATABASE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

interface Lesson {
  id: string;
  day_number: number;
  topic: string;
  icon_emoji: string | null;
}

async function getLessonsInRange(start: number, end: number): Promise<Lesson[]> {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, icon_emoji')
    .gte('day_number', start)
    .lte('day_number', end)
    .order('day_number');
  
  if (error) throw error;
  return data || [];
}

async function checkExistingAssets(dayNumber: number): Promise<Set<string>> {
  const { data } = await supabase
    .from('kelly_video_assets')
    .select('phase')
    .eq('day_number', dayNumber)
    .eq('asset_type', 'image');
  
  return new Set((data || []).map(a => a.phase));
}

async function registerAsset(asset: {
  day_number: number;
  phase: string;
  public_url: string;
  storage_path: string;
}) {
  const { error } = await supabase
    .from('kelly_video_assets')
    .upsert({
      day_number: asset.day_number,
      phase: asset.phase,
      template: 'visual',
      asset_type: 'image',
      age_bucket: null,
      language: 'en',
      storage_bucket: CONFIG.VISUALS_BUCKET,
      storage_path: asset.storage_path,
      public_url: asset.public_url,
      resolution: '1344x768',
      status: 'generated',
      quality_tier: 'production',
    }, {
      onConflict: 'day_number,phase,template,asset_type'
    });
  
  if (error) {
    console.error(`  ❌ DB Error: ${error.message}`);
    return false;
  }
  return true;
}

async function updateLessonUrls(dayNumber: number, heroUrl: string, thumbnailUrl: string) {
  const { error } = await supabase
    .from('core_lessons')
    .update({
      hero_image_url: heroUrl,
      thumbnail_url: thumbnailUrl,
    })
    .eq('day_number', dayNumber);
  
  if (error) {
    console.error(`  ❌ Failed to update lesson URLs: ${error.message}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE GENERATION
// ═══════════════════════════════════════════════════════════════════════════

const replicate = REPLICATE_TOKEN ? new Replicate({ auth: REPLICATE_TOKEN }) : null;

async function downloadImage(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
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

async function generateImage(prompt: string): Promise<Buffer | null> {
  if (!replicate) {
    console.log('  ⚠️ No REPLICATE_API_TOKEN - skipping generation');
    return null;
  }
  
  try {
    const output = await replicate.run(CONFIG.LORA_MODEL, {
      input: {
        prompt,
        hf_lora: CONFIG.LORA_URL,
        lora_scale: CONFIG.LORA_SCALE,
        num_outputs: 1,
        aspect_ratio: '16:9',
        output_format: 'png',
        guidance_scale: 3.5,
        num_inference_steps: 28,
      }
    }) as any;
    
    // Handle Replicate SDK FileOutput objects - use String() to get URL
    let imageUrl: string | null = null;
    
    if (Array.isArray(output) && output.length > 0) {
      const first = output[0];
      // FileOutput objects have toString that returns the URL
      const urlStr = String(first);
      if (urlStr && urlStr.startsWith('http')) {
        imageUrl = urlStr;
      }
    } else if (typeof output === 'string') {
      imageUrl = output;
    }
    
    if (!imageUrl) {
      console.log('  ⚠️ No image URL in output');
      return null;
    }
    
    console.log(`  🔗 Image URL: ${imageUrl.substring(0, 60)}...`);
    
    return await downloadImage(imageUrl);
  } catch (error: any) {
    console.error(`  ❌ Generation error: ${error.message}`);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// STORAGE UPLOAD
// ═══════════════════════════════════════════════════════════════════════════

async function uploadToStorage(buffer: Buffer, path: string): Promise<string | null> {
  const { error } = await supabase.storage
    .from(CONFIG.VISUALS_BUCKET)
    .upload(path, buffer, {
      contentType: 'image/png',
      upsert: true,
    });
  
  if (error) {
    // Try creating bucket if it doesn't exist
    if (error.message.includes('Bucket not found')) {
      await supabase.storage.createBucket(CONFIG.VISUALS_BUCKET, { public: true });
      return uploadToStorage(buffer, path);
    }
    console.error(`  ❌ Upload error: ${error.message}`);
    return null;
  }
  
  const { data: { publicUrl } } = supabase.storage
    .from(CONFIG.VISUALS_BUCKET)
    .getPublicUrl(path);
  
  return publicUrl;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN GENERATION PIPELINE
// ═══════════════════════════════════════════════════════════════════════════

async function processLesson(lesson: Lesson, dryRun: boolean): Promise<{ success: number; skipped: number; failed: number }> {
  const { day_number, topic, icon_emoji } = lesson;
  const paddedDay = String(day_number).padStart(3, '0');
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📚 Day ${day_number}: ${icon_emoji || '📖'} ${topic}`);
  console.log(`${'═'.repeat(60)}`);
  
  const existing = await checkExistingAssets(day_number);
  console.log(`  Existing assets: ${existing.size > 0 ? [...existing].join(', ') : 'none'}`);
  
  const results = { success: 0, skipped: 0, failed: 0 };
  let heroUrl = '';
  
  for (const phase of CONFIG.PHASES) {
    if (existing.has(phase)) {
      console.log(`  ⏭️ ${phase}: Already exists`);
      results.skipped++;
      continue;
    }
    
    console.log(`  🎬 ${phase.toUpperCase()}: Generating...`);
    
    if (dryRun) {
      console.log(`     Would generate: day_${paddedDay}/${phase}.png`);
      results.success++;
      continue;
    }
    
    // Generate image
    const prompt = getPhasePrompt(topic, phase);
    const imageBuffer = await generateImage(prompt);
    
    if (!imageBuffer) {
      console.log(`     ❌ Failed to generate`);
      results.failed++;
      continue;
    }
    
    // Upload to storage
    const storagePath = `day_${paddedDay}/${phase}.png`;
    const publicUrl = await uploadToStorage(imageBuffer, storagePath);
    
    if (!publicUrl) {
      results.failed++;
      continue;
    }
    
    // Register in database
    await registerAsset({
      day_number,
      phase,
      public_url: publicUrl,
      storage_path: storagePath,
    });
    
    console.log(`     ✅ Uploaded: ${storagePath}`);
    results.success++;
    
    // Save hook as hero image
    if (phase === 'hook') {
      heroUrl = publicUrl;
    }
    
    // Rate limiting
    await new Promise(r => setTimeout(r, CONFIG.RATE_LIMIT_MS));
  }
  
  // Update lesson with hero URL
  if (heroUrl && !dryRun) {
    await updateLessonUrls(day_number, heroUrl, heroUrl);
    console.log(`  📝 Updated core_lessons with hero_image_url`);
  }
  
  return results;
}

async function processRange(start: number, end: number, dryRun: boolean) {
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  🚀 FILL SUPABASE WITH ASSETS`);
  console.log(`  Days ${start} to ${end} ${dryRun ? '(DRY RUN)' : ''}`);
  console.log(`${'█'.repeat(60)}`);
  
  const lessons = await getLessonsInRange(start, end);
  console.log(`Found ${lessons.length} lessons\n`);
  
  if (lessons.length === 0) {
    console.log('❌ No lessons found in range');
    return;
  }
  
  const totals = { success: 0, skipped: 0, failed: 0, days: 0 };
  
  for (const lesson of lessons) {
    const result = await processLesson(lesson, dryRun);
    totals.success += result.success;
    totals.skipped += result.skipped;
    totals.failed += result.failed;
    totals.days++;
  }
  
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📊 FINAL RESULTS`);
  console.log(`${'═'.repeat(60)}`);
  console.log(`📅 Days processed: ${totals.days}`);
  console.log(`✅ Generated: ${totals.success} images`);
  console.log(`⏭️ Skipped: ${totals.skipped} images`);
  console.log(`❌ Failed: ${totals.failed} images`);
  
  if (dryRun) {
    console.log(`\n💡 Run without --dry-run to generate and upload images`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  
  // Check environment
  if (!SUPABASE_URL || !SUPABASE_KEY) {
    console.error('❌ Missing Supabase credentials');
    process.exit(1);
  }
  
  // Parse arguments
  const dayArg = args.find(a => a.startsWith('--day='));
  const rangeArg = args.find(a => a.startsWith('--range='));
  const allArg = args.includes('--all');
  
  if (dayArg) {
    const day = parseInt(dayArg.split('=')[1]);
    await processRange(day, day, dryRun);
  } else if (rangeArg) {
    const [start, end] = rangeArg.split('=')[1].split('-').map(Number);
    await processRange(start, end, dryRun);
  } else if (allArg) {
    await processRange(1, 365, dryRun);
  } else {
    console.log(`
🚀 FILL SUPABASE WITH ASSETS

Usage:
  npx tsx scripts/fill-supabase-with-assets.ts --day=1        # Single day
  npx tsx scripts/fill-supabase-with-assets.ts --range=1-30   # Day range
  npx tsx scripts/fill-supabase-with-assets.ts --all          # All 365 days
  npx tsx scripts/fill-supabase-with-assets.ts --dry-run      # Preview only

Environment Required:
  REPLICATE_API_TOKEN
  SUPABASE_URL (or PUBLIC_SUPABASE_URL)
  SUPABASE_SERVICE_ROLE_KEY
    `);
  }
}

main().catch(console.error);
