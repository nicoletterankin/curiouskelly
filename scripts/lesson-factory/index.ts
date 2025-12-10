/**
 * 🏭 CURIOUS KELLY LESSON FACTORY (LEGACY)
 * 
 * ⚠️ THIS FILE IS DEPRECATED
 * 
 * Use the new UNIFIED FACTORY instead:
 *   npx tsx scripts/lesson-factory/unified-factory.ts --day 1
 * 
 * Or run preflight check first:
 *   npx tsx scripts/lesson-factory/preflight-check.ts --day 1
 * 
 * The new unified factory includes:
 * - Full HD video generation (ElevenLabs → Flux+LoRA → MiniMax → Sync Labs lipsync-2-pro)
 * - Infographic generation (Flux Pro)
 * - Option card images (512×512)
 * - Response videos with different Kelly expressions
 * - Supabase upload with database updates
 * - Cloudflare R2 backup
 * - Language/age/tone expansion support
 * 
 * See: vom/UNIFIED_LESSON_FACTORY_FINAL.md for complete documentation
 */

import * as dotenv from 'dotenv';
dotenv.config();

import { createClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

// ═══════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════

const CONFIG = {
  // API Clients
  SUPABASE_URL: process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_KEY!,
  
  // Kelly LoRA
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  KELLY_LORA_MODEL: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  KELLY_LORA_SCALE: 0.85,
  
  // Output
  OUTPUT_BASE: path.join(process.cwd(), 'public', 'kelly'),
  
  // Rate limiting
  RATE_LIMIT_MS: 2000,
  
  // Quality thresholds
  MIN_QUALITY_SCORE: 0.7,
};

// ═══════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════

type GenerationTier = 'free' | 'premium' | 'masterwork';

interface GenerationJob {
  id: string;
  lessonId: string;
  tier: GenerationTier;
  status: string;
  progress: number;
  costs: CostBreakdown;
}

interface CostBreakdown {
  text: number;
  visuals: number;
  audio: number;
  video: number;
  total: number;
}

interface VisualContext {
  environment: string;
  props: string[];
  mood: string;
  color_palette: string;
  lighting: string;
}

interface GeneratedAsset {
  type: 'image' | 'audio' | 'video' | 'text';
  subtype: string;
  url: string;
  localPath: string;
  cost: number;
  generator: string;
  prompt?: string;
}

// ═══════════════════════════════════════════════════════════════
// CLIENTS
// ═══════════════════════════════════════════════════════════════

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);
const anthropic = new Anthropic();
const replicate = new Replicate();

// ═══════════════════════════════════════════════════════════════
// VISUAL CONTEXT MATCHING
// ═══════════════════════════════════════════════════════════════

async function getVisualContextForTopic(topic: string): Promise<VisualContext> {
  // Try to match from database
  const { data: contexts } = await supabase
    .from('visual_contexts')
    .select('*');
  
  if (contexts && contexts.length > 0) {
    const topicLower = topic.toLowerCase();
    
    // Find best matching context
    for (const ctx of contexts) {
      if (ctx.keywords.some((kw: string) => topicLower.includes(kw.toLowerCase()))) {
        // Update usage count
        await supabase
          .from('visual_contexts')
          .update({ usage_count: (ctx.usage_count || 0) + 1 })
          .eq('id', ctx.id);
        
        return {
          environment: ctx.environment,
          props: ctx.props,
          mood: ctx.mood,
          color_palette: ctx.color_palette,
          lighting: ctx.lighting,
        };
      }
    }
  }
  
  // Default context
  return {
    environment: 'bright modern learning studio with clean white background',
    props: ['open book', 'light bulb', 'globe', 'plant'],
    mood: 'curious, educational, engaging',
    color_palette: 'clean whites, learning blues, warm accents',
    lighting: 'soft natural studio lighting',
  };
}

// ═══════════════════════════════════════════════════════════════
// PROMPT BUILDING
// ═══════════════════════════════════════════════════════════════

async function getPromptTemplate(name: string): Promise<string | null> {
  const { data } = await supabase
    .from('prompt_templates')
    .select('template')
    .eq('name', name)
    .eq('is_active', true)
    .single();
  
  return data?.template || null;
}

function fillPromptTemplate(template: string, variables: Record<string, string>): string {
  let filled = template;
  for (const [key, value] of Object.entries(variables)) {
    filled = filled.replace(new RegExp(`{{${key}}}`, 'g'), value);
  }
  return filled;
}

// ═══════════════════════════════════════════════════════════════
// IMAGE GENERATION
// ═══════════════════════════════════════════════════════════════

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

async function generateKellyImage(
  prompt: string,
  outputPath: string,
  aspectRatio: string = '16:9'
): Promise<{ success: boolean; cost: number; error?: string }> {
  try {
    const output = await replicate.run(CONFIG.KELLY_LORA_MODEL, {
      input: {
        prompt: prompt,
        hf_lora: CONFIG.KELLY_LORA_URL,
        lora_scale: CONFIG.KELLY_LORA_SCALE,
        num_outputs: 1,
        aspect_ratio: aspectRatio,
        output_format: 'png',
        guidance_scale: 3.5,
        num_inference_steps: 28,
      },
    }) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    if (!imageUrl) {
      return { success: false, cost: 0.04, error: 'No image URL returned' };
    }
    
    const buffer = await downloadImage(imageUrl);
    
    // Ensure directory exists
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, buffer);
    
    return { success: true, cost: 0.04 }; // Approximate Flux cost
    
  } catch (error: any) {
    return { success: false, cost: 0, error: error.message };
  }
}

// ═══════════════════════════════════════════════════════════════
// PHASE VISUAL GENERATION
// ═══════════════════════════════════════════════════════════════

async function generatePhaseVisuals(
  lessonId: string,
  dayNumber: number,
  topic: string,
  context: VisualContext,
  jobId: string
): Promise<GeneratedAsset[]> {
  const assets: GeneratedAsset[] = [];
  const paddedDay = String(dayNumber).padStart(3, '0');
  const outputDir = path.join(CONFIG.OUTPUT_BASE, 'phases', paddedDay);
  
  const phases = [
    {
      name: 'hook',
      templateName: 'kelly_hook',
      variables: { environment: context.environment, mood: context.mood },
    },
    {
      name: 'q1',
      templateName: 'kelly_teaching',
      variables: { environment: context.environment, prop: context.props[0] || 'educational object' },
    },
    {
      name: 'q2',
      templateName: 'kelly_thinking',
      variables: { environment: context.environment, prop: context.props[1] || 'thoughtful object' },
    },
    {
      name: 'q3',
      templateName: 'kelly_teaching',
      variables: { environment: context.environment, prop: context.props[2] || 'educational prop' },
    },
    {
      name: 'wisdom',
      templateName: 'kelly_wisdom',
      variables: { environment: context.environment, prop: context.props[3] || 'symbolic achievement' },
    },
  ];
  
  for (const phase of phases) {
    const outputPath = path.join(outputDir, `${phase.name}.png`);
    
    // Skip if exists
    if (fs.existsSync(outputPath)) {
      console.log(`  ⏭️ ${phase.name}: Already exists`);
      assets.push({
        type: 'image',
        subtype: `phase_${phase.name}`,
        url: `/kelly/phases/${paddedDay}/${phase.name}.png`,
        localPath: outputPath,
        cost: 0,
        generator: 'existing',
      });
      continue;
    }
    
    // Get template and fill
    const template = await getPromptTemplate(phase.templateName);
    if (!template) {
      console.log(`  ⚠️ No template found for ${phase.templateName}`);
      continue;
    }
    
    const prompt = fillPromptTemplate(template, phase.variables);
    
    console.log(`  🎨 Generating ${phase.name}...`);
    const result = await generateKellyImage(prompt, outputPath);
    
    if (result.success) {
      console.log(`  ✅ ${phase.name} generated`);
      
      // Record asset in database
      await supabase.from('lesson_assets').insert({
        lesson_id: lessonId,
        job_id: jobId,
        asset_type: 'image',
        asset_subtype: `phase_${phase.name}`,
        storage_path: outputPath,
        public_url: `/kelly/phases/${paddedDay}/${phase.name}.png`,
        generator: 'flux-dev-lora',
        prompt: prompt,
        generation_cost_usd: result.cost,
        is_approved: true,
        approval_method: 'auto',
      });
      
      // Record cost
      await supabase.from('generation_costs').insert({
        job_id: jobId,
        provider: 'replicate',
        service: 'flux-dev-lora',
        cost_usd: result.cost,
      });
      
      assets.push({
        type: 'image',
        subtype: `phase_${phase.name}`,
        url: `/kelly/phases/${paddedDay}/${phase.name}.png`,
        localPath: outputPath,
        cost: result.cost,
        generator: 'flux-dev-lora',
        prompt: prompt,
      });
    } else {
      console.log(`  ❌ ${phase.name} failed: ${result.error}`);
    }
    
    // Rate limiting
    await new Promise(r => setTimeout(r, CONFIG.RATE_LIMIT_MS));
  }
  
  return assets;
}

// ═══════════════════════════════════════════════════════════════
// MAIN ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════

async function generateLessonAssets(
  dayNumber: number,
  tier: GenerationTier = 'free'
): Promise<{ success: boolean; assets: GeneratedAsset[]; totalCost: number }> {
  console.log('\n' + '█'.repeat(60));
  console.log(`  LESSON FACTORY - Day ${dayNumber} (${tier.toUpperCase()} tier)`);
  console.log('█'.repeat(60));
  
  // Get lesson data
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();
  
  if (lessonError || !lesson) {
    console.error(`❌ Lesson ${dayNumber} not found`);
    return { success: false, assets: [], totalCost: 0 };
  }
  
  console.log(`\n📚 Topic: ${lesson.topic}`);
  console.log(`📝 Truth: ${lesson.universal_truth}`);
  
  // Create generation job
  const { data: job } = await supabase
    .from('lesson_generation_jobs')
    .insert({
      lesson_id: lesson.id,
      tier: tier,
      requested_assets: ['phase_visuals'],
      status: 'visuals',
      progress: 0,
    })
    .select()
    .single();
  
  if (!job) {
    console.error('❌ Failed to create generation job');
    return { success: false, assets: [], totalCost: 0 };
  }
  
  console.log(`\n🔧 Job ID: ${job.id}`);
  
  // Get visual context
  console.log('\n📍 Finding visual context...');
  const context = await getVisualContextForTopic(lesson.topic);
  console.log(`   Environment: ${context.environment.substring(0, 50)}...`);
  console.log(`   Props: ${context.props.slice(0, 3).join(', ')}`);
  console.log(`   Mood: ${context.mood}`);
  
  // Generate phase visuals
  console.log('\n🎨 Generating phase visuals...');
  const assets = await generatePhaseVisuals(
    lesson.id,
    dayNumber,
    lesson.topic,
    context,
    job.id
  );
  
  // Calculate total cost
  const totalCost = assets.reduce((sum, a) => sum + a.cost, 0);
  
  // Update job status
  await supabase
    .from('lesson_generation_jobs')
    .update({
      status: 'completed',
      progress: 100,
      actual_cost_usd: totalCost,
      completed_at: new Date().toISOString(),
    })
    .eq('id', job.id);
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 GENERATION COMPLETE');
  console.log('═'.repeat(60));
  console.log(`✅ Assets generated: ${assets.length}`);
  console.log(`💰 Total cost: $${totalCost.toFixed(4)}`);
  
  return {
    success: true,
    assets,
    totalCost,
  };
}

// ═══════════════════════════════════════════════════════════════
// BATCH GENERATION
// ═══════════════════════════════════════════════════════════════

async function generateBatch(
  startDay: number,
  endDay: number,
  tier: GenerationTier = 'free'
): Promise<{ totalAssets: number; totalCost: number }> {
  console.log('\n' + '▓'.repeat(60));
  console.log(`  BATCH GENERATION - Days ${startDay} to ${endDay}`);
  console.log('▓'.repeat(60));
  
  let totalAssets = 0;
  let totalCost = 0;
  
  for (let day = startDay; day <= endDay; day++) {
    const result = await generateLessonAssets(day, tier);
    totalAssets += result.assets.length;
    totalCost += result.totalCost;
    
    // Progress update
    const progress = ((day - startDay + 1) / (endDay - startDay + 1) * 100).toFixed(1);
    console.log(`\n📈 Batch progress: ${progress}% (${day - startDay + 1}/${endDay - startDay + 1} lessons)`);
  }
  
  console.log('\n' + '▓'.repeat(60));
  console.log('📊 BATCH COMPLETE');
  console.log('▓'.repeat(60));
  console.log(`✅ Total assets: ${totalAssets}`);
  console.log(`💰 Total cost: $${totalCost.toFixed(2)}`);
  
  return { totalAssets, totalCost };
}

// ═══════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  
  // Validate environment
  if (!CONFIG.SUPABASE_URL) {
    console.error('❌ SUPABASE_URL not configured');
    process.exit(1);
  }
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not configured');
    process.exit(1);
  }
  
  // Parse arguments
  const dayArg = args.find(a => a.startsWith('--day='));
  const rangeArg = args.find(a => a.startsWith('--range='));
  const tierArg = args.find(a => a.startsWith('--tier='));
  
  const tier = (tierArg?.split('=')[1] || 'free') as GenerationTier;
  
  if (dayArg) {
    const day = parseInt(dayArg.split('=')[1]);
    await generateLessonAssets(day, tier);
  } else if (rangeArg) {
    const [start, end] = rangeArg.split('=')[1].split('-').map(Number);
    await generateBatch(start, end, tier);
  } else {
    console.log(`
🏭 CURIOUS KELLY LESSON FACTORY

Usage:
  npx ts-node scripts/lesson-factory --day=1 --tier=free
  npx ts-node scripts/lesson-factory --range=1-10 --tier=free
  npx ts-node scripts/lesson-factory --range=8-365 --tier=free

Tiers:
  free       - Standard quality (~$5/lesson)
  premium    - High quality with variants (~$100/lesson)
  masterwork - Documentary quality (~$5000/lesson)

Environment Variables Required:
  SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)
  SUPABASE_SERVICE_KEY
  REPLICATE_API_TOKEN
    `);
  }
}

main().catch(console.error);



