#!/usr/bin/env npx tsx
/**
 * 🎨 BYOK VISUAL GENERATOR
 * 
 * Generates Kelly visuals using community BYOK keys
 * with the Kelly LoRA from Hugging Face.
 * 
 * Supports: Replicate, Fal.ai, Together AI
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import Replicate from 'replicate';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// ═══════════════════════════════════════════════════════════════════════════
// KELLY LORA CONFIGURATION (LOCKED)
// ═══════════════════════════════════════════════════════════════════════════

const KELLY_LORA = {
  url: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  replicateModel: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  scale: 0.85,
  trigger: 'kelly',
};

const KELLY_BASE_PROMPT = `kelly, photorealistic woman named Kelly, late 20s to early 30s, 
brown wavy shoulder-length hair with caramel and honey highlights center-parted, 
hazel-brown almond-shaped eyes, soft symmetrical features with natural makeup, 
light-medium warm skin tone with healthy glow, 
wearing soft powder blue cashmere crewneck sweater, 
warm but professional expression, intelligent curious eyes`;

// ═══════════════════════════════════════════════════════════════════════════
// PHASE TEMPLATES (Maps to Kelly expressions)
// ═══════════════════════════════════════════════════════════════════════════

const PHASE_PROMPTS: Record<string, string> = {
  hook: 'excited expression, slight head tilt, welcoming smile, arms visible, studio lighting',
  cliff: 'curious expression, one eyebrow slightly raised, questioning look, thoughtful pose',
  fact1: 'engaged explaining gesture, direct eye contact, teaching moment, professional',
  fact2: 'animated discussion, hands gesturing, passionate about topic, warm smile',
  fact3: 'enlightening revelation, eyes bright, sharing discovery, genuine enthusiasm',
  wisdom: 'sincere expression, hand near heart, wise knowing look, gentle smile, warm',
  outro: 'proud smile, welcoming arms, celebration of learning, joyful eyes',
};

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

interface GenerationOptions {
  dayNumber: number;
  phase: string;
  track: 'learn' | 'grow';
  topic: string;
  provider: 'replicate' | 'fal' | 'together';
  apiKey: string;
}

/**
 * Build the full Kelly prompt for a specific phase and topic
 */
function buildPrompt(topic: string, phase: string): string {
  const phasePrompt = PHASE_PROMPTS[phase] || PHASE_PROMPTS.fact1;
  
  return `${KELLY_BASE_PROMPT}, ${phasePrompt}, 
teaching about "${topic}", 
white cyclorama studio background, 
director's chair with warm wood frame, 
natural window lighting from upper right, 
professional photography, 8K quality, photorealistic`;
}

/**
 * Generate with Replicate (FLUX + LoRA)
 */
async function generateWithReplicate(prompt: string, apiKey: string): Promise<string> {
  const replicate = new Replicate({ auth: apiKey });
  
  const output = await replicate.run(KELLY_LORA.replicateModel as `${string}/${string}:${string}`, {
    input: {
      prompt,
      hf_lora: KELLY_LORA.url,
      lora_scale: KELLY_LORA.scale,
      num_outputs: 1,
      aspect_ratio: '16:9',
      output_format: 'webp',
      guidance_scale: 3.5,
      output_quality: 90,
      num_inference_steps: 28,
    },
  }) as any;

  // Handle Replicate SDK FileOutput objects - use String() to get URL
  if (Array.isArray(output) && output.length > 0) {
    const urlStr = String(output[0]);
    if (urlStr && urlStr.startsWith('http')) {
      return urlStr;
    }
  }
  throw new Error('No valid image URL in Replicate output');
}

/**
 * Generate with Fal.ai
 */
async function generateWithFal(prompt: string, apiKey: string): Promise<string> {
  const response = await fetch('https://fal.run/fal-ai/flux-lora', {
    method: 'POST',
    headers: {
      'Authorization': `Key ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt,
      loras: [{
        path: KELLY_LORA.url,
        scale: KELLY_LORA.scale,
      }],
      image_size: 'landscape_16_9',
      num_inference_steps: 28,
      guidance_scale: 3.5,
      num_images: 1,
      enable_safety_checker: true,
    }),
  });

  const data = await response.json();
  return data.images?.[0]?.url || '';
}

/**
 * Main generation function
 */
async function generateVisual(options: GenerationOptions): Promise<string | null> {
  const { dayNumber, phase, track, topic, provider, apiKey } = options;
  
  console.log(`🎨 Generating Day ${dayNumber} ${phase} (${track} track)`);
  console.log(`   Topic: ${topic}`);
  console.log(`   Provider: ${provider}`);
  
  const prompt = buildPrompt(topic, phase);
  console.log(`   Prompt: ${prompt.substring(0, 100)}...`);
  
  let imageUrl: string;
  
  try {
    switch (provider) {
      case 'replicate':
        imageUrl = await generateWithReplicate(prompt, apiKey);
        break;
      case 'fal':
        imageUrl = await generateWithFal(prompt, apiKey);
        break;
      default:
        throw new Error(`Unsupported provider: ${provider}`);
    }
    
    console.log(`   ✅ Generated: ${imageUrl.substring(0, 60)}...`);
    
    // Download and upload to Supabase
    const response = await fetch(imageUrl);
    const buffer = await response.arrayBuffer();
    
    const fileName = `${track}/day-${String(dayNumber).padStart(3, '0')}/${phase}.webp`;
    
    const { error: uploadError } = await supabase.storage
      .from('lesson-visuals')
      .upload(fileName, buffer, {
        contentType: 'image/webp',
        upsert: true,
      });
    
    if (uploadError) {
      console.log(`   ⚠️ Upload failed: ${uploadError.message}`);
      return imageUrl; // Still return the generated URL
    }
    
    const { data: urlData } = supabase.storage
      .from('lesson-visuals')
      .getPublicUrl(fileName);
    
    // Log contribution
    await logContribution(provider, dayNumber, phase);
    
    console.log(`   ☁️ Uploaded: ${urlData.publicUrl}`);
    return urlData.publicUrl;
    
  } catch (err: any) {
    console.log(`   ❌ Error: ${err.message}`);
    return null;
  }
}

/**
 * Log contribution to community stats
 */
async function logContribution(provider: string, dayNumber: number, phase: string) {
  try {
    await supabase.from('community_contributions').insert({
      provider,
      resource_type: 'image',
      day_number: dayNumber,
      phase,
      estimated_cost_cents: 5, // ~$0.05 per image
    });
  } catch (err) {
    // Silent fail - logging is best-effort
  }
}

/**
 * Generate visuals for a range of days
 */
async function generateRange(
  startDay: number,
  endDay: number,
  track: 'learn' | 'grow',
  provider: 'replicate' | 'fal',
  apiKey: string
) {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎨 BYOK VISUAL GENERATOR                                  ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
  
  const phases = ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro'];
  let generated = 0;
  let failed = 0;
  
  for (let day = startDay; day <= endDay; day++) {
    // Get lesson topic
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('topic')
      .eq('day_number', day)
      .eq('track', track)
      .single();
    
    if (!lesson?.topic) {
      console.log(`⚠️ Day ${day}: No topic found, skipping`);
      continue;
    }
    
    for (const phase of phases) {
      const result = await generateVisual({
        dayNumber: day,
        phase,
        track,
        topic: lesson.topic,
        provider,
        apiKey,
      });
      
      if (result) {
        generated++;
      } else {
        failed++;
      }
      
      // Rate limit
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Generated: ${generated} visuals`);
  console.log(`❌ Failed: ${failed}`);
}

// CLI
const args = process.argv.slice(2);
const startDay = parseInt(args[0]) || 354;
const endDay = parseInt(args[1]) || startDay;
const track = (args[2] as 'learn' | 'grow') || 'learn';
const provider = (args[3] as 'replicate' | 'fal') || 'replicate';
const apiKey = args[4] || process.env.REPLICATE_API_TOKEN || '';

if (!apiKey) {
  console.log('Usage: npx tsx scripts/byok-visual-generator.ts <startDay> <endDay> <track> <provider> <apiKey>');
  console.log('Example: npx tsx scripts/byok-visual-generator.ts 354 360 learn replicate r8_xxx');
  process.exit(1);
}

generateRange(startDay, endDay, track, provider, apiKey).catch(console.error);

// Export for use in other scripts
export { generateVisual, buildPrompt, KELLY_LORA, PHASE_PROMPTS };


