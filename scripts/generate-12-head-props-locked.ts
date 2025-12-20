#!/usr/bin/env npx tsx
/**
 * 🎭 GENERATE 12 HEAD-PROP ARCHETYPES (LOCKED POSE)
 * 
 * Set 2: Differentiated by Head Accessories.
 * Strategy: Locked Pose (Center/Direct) + Simple, Realistic Head Props.
 * This ensures consistency with the "Clean" batch while adding variety.
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

const CONFIG = {
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.95,
  MASTER_SEED: 123456, // Different seed
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-head-only'),
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN! });
const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

const KELLY_BASE = `kelly, young woman late 20s, brown wavy shoulder-length hair with caramel highlights, hazel-brown eyes, flawless skin, light natural makeup, wearing soft powder blue cashmere crewneck sweater`;
const SCENE = `pure white seamless cyclorama photography studio background, professional soft studio lighting, 8K UHD, photorealistic`;
const FRAMING = `head and shoulders portrait, chest-up framing, subject centered, no hands visible, clean composition`;
const NEGATIVE = `hands, holding items, messy hair, harsh shadows, open mouth, teeth showing too much, distorted face, complex background, text, watermark`;

// LOCKED POSE + HEAD PROP PROMPTS
const ARCHETYPES_PROPS: Record<string, string> = {
  "scientist": "wearing clear safety lab goggles resting on forehead above eyebrows. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "explorer": "wearing a weathered fabric bandana headband tied around forehead. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "rebel": "wearing black sunglasses pushed up into hair on top of head. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "architect": "wearing a yellow pencil tucked behind right ear. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "diplomat": "wearing a thin elegant velvet headband. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "empath": "wearing a small delicate flower clip in hair on left side. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "macgyver": "wearing a simple baseball cap worn backwards. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "mystic": "wearing a delicate gold chain headpiece across forehead. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "provider": "wearing a cozy cream-colored knit ear-warmer headband. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "storyteller": "wearing vintage reading glasses perched on top of head. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "strategist": "wearing sleek modern wireless headphones around neck (not on ears). pose: facing camera directly, head perfectly centered, shoulders squared.",
  "survivor": "wearing a simple olive green beanie cap. pose: facing camera directly, head perfectly centered, shoulders squared."
};

async function generate(archetype: string, description: string) {
  console.log(`\n🎭 Generating Prop: ${archetype.toUpperCase()}`);
  const prompt = `${KELLY_BASE}, ${description}, ${FRAMING}, ${SCENE}`;
  
  try {
    const output = await replicate.run("lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d", {
      input: {
        prompt,
        hf_lora: CONFIG.KELLY_LORA_URL,
        lora_scale: CONFIG.LORA_SCALE,
        num_outputs: 1,
        aspect_ratio: "1:1",
        output_format: "png",
        guidance_scale: 3.5,
        num_inference_steps: 28,
        seed: CONFIG.MASTER_SEED,
        disable_safety_checker: true,
        negative_prompt: NEGATIVE
      }
    }) as any;

    const imageUrl = Array.isArray(output) ? output[0] : output;
    const res = await fetch(imageUrl);
    const buffer = Buffer.from(await res.arrayBuffer());
    
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
    const localPath = path.join(CONFIG.OUTPUT_DIR, `kelly_${archetype}_prop_head.png`);
    fs.writeFileSync(localPath, buffer);
    console.log(`   💾 Saved: ${localPath}`);

    const remotePath = `heygen/archetypes-head-only/kelly_${archetype}_prop_head.png`;
    await supabase.storage.from('kelly-templates').upload(remotePath, buffer, { upsert: true, contentType: 'image/png' });
    const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
    console.log(`   ☁️ Uploaded: ${data.publicUrl}`);
    return data.publicUrl;

  } catch (e: any) {
    console.error(`   ❌ Failed: ${e.message}`);
    return null;
  }
}

async function main() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  👒 GENERATING 12 HEAD-PROP ARCHETYPES (LOCKED POSE)         ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  
  const results: Record<string, string> = {};
  const keys = Object.keys(ARCHETYPES_PROPS);
  
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const url = await generate(key, ARCHETYPES_PROPS[key]);
    if (url) results[key] = url;
    if (i < keys.length - 1) await new Promise(r => setTimeout(r, 2000));
  }
  
  fs.writeFileSync(path.join(CONFIG.OUTPUT_DIR, 'prop_urls.json'), JSON.stringify(results, null, 2));
  console.log(`\n✅ DONE! All 12 prop archetypes generated in: ${CONFIG.OUTPUT_DIR}`);
}

main().catch(console.error);

















