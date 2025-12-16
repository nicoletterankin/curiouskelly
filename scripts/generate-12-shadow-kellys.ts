#!/usr/bin/env npx tsx
/**
 * 🎭 GENERATE 12 SHADOW ARCHETYPES (CLEAN / NO PROPS)
 * 
 * Set 2: Psychological Shadows (The "Anti-Kellys")
 * Useful for roleplay, "what not to do", and shadow work lessons.
 * 
 * Strategy: Locked Pose + Distinct Psychological Expressions
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
  LORA_SCALE: 0.95,
  MASTER_SEED: 999999, // Distinct seed from Set 1
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-clean'), // Same folder as requested
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
// Ensure pose is identical to Set 1
const FRAMING = `head and shoulders portrait, chest-up framing, subject centered, no hands visible, clean composition`;
const NEGATIVE = `props, accessories, glasses, goggles, hats, jewelry, earrings, hands, holding items, messy hair, harsh shadows, open mouth, teeth showing too much, distorted face, face paint, makeup, clown nose`;

// =============================================================================
// 12 SHADOW ARCHETYPES - PSYCHOLOGICAL EXPRESSIONS
// =============================================================================

const SHADOW_ARCHETYPES: Record<string, string> = {
  // 1. The Clown (Distraction/Silly)
  "clown": "expression: exaggerated playful goofy grin, wide eyes, eyebrows raised high, silly entertaining look. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 2. The Victim (Helplessness)
  "victim": "expression: sad pleading puppy-dog eyes, eyebrows furrowed in worry, slight frown, vulnerable and helpless. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 3. The Bully (Aggression)
  "bully": "expression: sneering arrogant smile, narrowed cruel eyes, intense intimidating gaze, confident superiority. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 4. The Judge (Criticism)
  "judge": "expression: stern disapproval, pursed lips, critical narrowing of eyes, looking down nose slightly (but head level), harsh judgment. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 5. The Perfectionist (Anxiety/Control)
  "perfectionist": "expression: tense tight smile, eyes wide with anxious alertness, strained perfection, worried about making a mistake. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 6. The People Pleaser (Fawning)
  "pleaser": "expression: overly eager fake smile, desperate to be liked, wide eyes seeking approval, ingratiating expression. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 7. The Skeptic (Doubt)
  "skeptic": "expression: one eyebrow raised high in doubt, mouth pulled to side in disbelief, cynical gaze, questioning everything. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 8. The Pessimist (Gloom)
  "pessimist": "expression: gloomy Eeyore expression, heavy eyelids, downturned mouth, resigned to failure, low energy. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 9. The Diva (Entitlement)
  "diva": "expression: haughty superior look, chin slightly lifted (but centered), bored with you, expectant and demanding. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 10. The Robot (Numbness/Dissociation)
  "robot": "expression: completely blank neutral face, dead emotionless eyes, slack features, zero affect, uncanny valley stare. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 11. The Ghost (Fear/Invisible)
  "ghost": "expression: wide fearful eyes, pale shock, looking like she's seen a ghost, frozen in fear, holding breath. pose: facing camera directly, head perfectly centered, shoulders squared.",
  
  // 12. The Saboteur (Mischief)
  "saboteur": "expression: sly cunning smirk, eyes looking sideways (but head center), planning mischief, secretive and tricky. pose: facing camera directly, head perfectly centered, shoulders squared."
};

// =============================================================================
// GENERATION
// =============================================================================

async function generate(archetype: string, description: string) {
  console.log(`\n🎭 Generating Shadow: ${archetype.toUpperCase()}`);
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
    const localPath = path.join(CONFIG.OUTPUT_DIR, `kelly_${archetype}_clean.png`); // Same naming convention
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
  console.log('║  🌚 GENERATING 12 SHADOW ARCHETYPES (CLEAN)                  ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  
  const results: Record<string, string> = {};
  const keys = Object.keys(SHADOW_ARCHETYPES);
  
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const url = await generate(key, SHADOW_ARCHETYPES[key]);
    if (url) results[key] = url;
    if (i < keys.length - 1) await new Promise(r => setTimeout(r, 2000));
  }
  
  // Append to existing JSON if possible, or new one
  const jsonPath = path.join(CONFIG.OUTPUT_DIR, 'shadow_urls.json');
  fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  console.log(`\n✅ DONE! All 12 shadow archetypes generated in: ${CONFIG.OUTPUT_DIR}`);
}

main().catch(console.error);








