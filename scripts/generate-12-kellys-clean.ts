#!/usr/bin/env npx tsx
/**
 * 🎭 GENERATE 12 CLEAN KELLY ARCHETYPES (NO PROPS)
 * 
 * Strategy: Pure Expression & Head Pose differentiation.
 * "Classy", clean, and safe for HeyGen moderation/animation.
 * 
 * Uses: Curious Kelly LoRA → Replicate flux-dev-lora
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.95, // High fidelity to Kelly
  MASTER_SEED: 888888, // New seed for fresh start
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-clean'),
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN! });
const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// LOCKED ELEMENTS
// =============================================================================

const KELLY_BASE = `kelly, young woman late 20s, brown wavy shoulder-length hair with caramel highlights, hazel-brown eyes, flawless skin, light natural makeup, wearing soft powder blue cashmere crewneck sweater`;
const SCENE = `pure white seamless cyclorama photography studio background, professional soft studio lighting, 8K UHD, photorealistic, shot on Hasselblad`;
const FRAMING = `head and shoulders portrait, chest-up framing, subject centered, no hands visible, clean composition`;
const NEGATIVE = `props, accessories, glasses, goggles, hats, jewelry, earrings, hands, holding items, messy hair, harsh shadows, open mouth, teeth showing too much, distorted face`;

// =============================================================================
// 12 ARCHETYPES - PURE EXPRESSION & POSE
// =============================================================================

const ARCHETYPES: Record<string, string> = {
  "scientist": "expression: focused analytical gaze, one eyebrow slightly raised, knowing intellectual smile. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "explorer": "expression: wide eyes sparkling with wonder, bright enthusiastic smile, delighted curiosity. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "rebel": "expression: confident asymmetric smirk, intense direct eye contact, slight challenge in eyes. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "architect": "expression: thoughtful concentrated look, lips pressed together in focus, deep analysis. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "diplomat": "expression: warm welcoming smile, soft approachable eyes, trustworthy and open. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "empath": "expression: gentle compassionate softness, eyes radiating deep understanding, slight empathetic smile. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "macgyver": "expression: bright ingenious spark in eyes, excited creative grin, ready to solve problems. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "mystic": "expression: serene knowing smile, deep wise eyes, peaceful transcendence. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "provider": "expression: warm protective smile, reassuring steady gaze, maternal safety. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "storyteller": "expression: animated captivating face, eyes sparkling with a secret, engaging charm. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "strategist": "expression: sharp calculating gaze, confident winner's smile, intelligence. pose: facing camera directly, head perfectly centered, shoulders squared.",
  "survivor": "expression: serious determined resilience, steady unshakeable gaze, brave and strong. pose: facing camera directly, head perfectly centered, shoulders squared."
};

// =============================================================================
// GENERATION
// =============================================================================

async function generate(archetype: string, description: string) {
  console.log(`\n🎭 Generating: ${archetype.toUpperCase()}`);
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
    
    // Download & Save
    const res = await fetch(imageUrl);
    const buffer = Buffer.from(await res.arrayBuffer());
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
    const localPath = path.join(CONFIG.OUTPUT_DIR, `kelly_${archetype}_clean.png`);
    fs.writeFileSync(localPath, buffer);
    console.log(`   💾 Saved: ${localPath}`);

    // Upload to Supabase
    const remotePath = `heygen/archetypes-clean/kelly_${archetype}_clean.png`;
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
  console.log('║  ✨ GENERATING 12 CLEAN KELLYS (NO PROPS)                    ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  
  const results: Record<string, string> = {};
  const keys = Object.keys(ARCHETYPES);
  
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const url = await generate(key, ARCHETYPES[key]);
    if (url) results[key] = url;
    if (i < keys.length - 1) await new Promise(r => setTimeout(r, 2000));
  }
  
  fs.writeFileSync(path.join(CONFIG.OUTPUT_DIR, 'urls.json'), JSON.stringify(results, null, 2));
  console.log('\n✅ DONE! All clean archetypes generated.');
}

main().catch(console.error);

