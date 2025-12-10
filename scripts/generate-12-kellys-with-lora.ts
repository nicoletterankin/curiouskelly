#!/usr/bin/env npx tsx
/**
 * 🎨 GENERATE 12 KELLY ARCHETYPE PHOTOS - WITH TRAINED LORA
 * 
 * Uses YOUR Curious Kelly LoRA for CONSISTENT character across all 12 archetypes.
 * Ready for HeyGen upload.
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

// =============================================================================
// CONFIGURATION - LOCKED
// =============================================================================

const CONFIG = {
  // Kelly LoRA - THE REAL ONE
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.90, // Strong for consistency
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-lora'),
  
  // Supabase
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// KELLY CHARACTER - LOCKED (matches LoRA training)
// =============================================================================

const KELLY_BASE = `kelly, woman with brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater`;

const SCENE_BASE = `pure white cyclorama photography studio background, professional studio lighting, clean minimal background, shot on Hasselblad H6D-100c, 85mm f/2.8, shallow depth of field, professional fashion photography, 8K UHD, photorealistic`;

// =============================================================================
// 12 ARCHETYPE EXPRESSIONS
// =============================================================================

const ARCHETYPE_PROMPTS: Record<string, { expression: string; energy: string }> = {
  "scientist": {
    expression: "focused analytical gaze, one eyebrow slightly raised, knowing slight smile, direct confident eye contact, chin slightly up",
    energy: "intellectual, evidence-based, curious researcher, \"I have the data\""
  },
  "explorer": {
    expression: "wide eyes sparkling with wonder, excited genuine smile showing teeth, eyebrows raised in delighted curiosity, head tilted slightly right, looking slightly upward",
    energy: "adventurous, discovering something amazing, childlike wonder, \"let's explore this together!\""
  },
  "rebel": {
    expression: "confident asymmetric smirk, one corner of mouth raised, intense direct eye contact, eyebrows slightly furrowed with challenge, chin down slightly, looking up through eyebrows",
    energy: "edgy, questioning authority, bold challenger, \"question everything\""
  },
  "architect": {
    expression: "thoughtful concentrated look, lips pressed together slightly, eyes showing deep focus and analysis, calm inner confidence, head perfectly centered and balanced",
    energy: "systematic, structured, building understanding, \"let me show you the blueprint\""
  },
  "diplomat": {
    expression: "warm welcoming smile, soft approachable eyes, gentle head nod feeling, open trustworthy expression, head tilted slightly with warmth",
    energy: "balanced, understanding, bridging perspectives, \"I see all sides\""
  },
  "empath": {
    expression: "gentle compassionate smile, eyes full of understanding, soft caring gaze, slightly parted lips as if listening deeply, head tilted with warmth and care, leaning in",
    energy: "nurturing, emotionally connected, deeply feeling, \"I feel what you feel\""
  },
  "macgyver": {
    expression: "practical creative grin, eyes bright with an idea, asymmetrical knowing smile, engaged and ready to act, head tilted forward slightly",
    energy: "resourceful, hands-on problem solver, inventive, \"here's how we can use this\""
  },
  "mystic": {
    expression: "serene knowing smile, eyes with depth and ancient wisdom, peaceful profound gaze, subtle mysterious quality, slight upward contemplative tilt",
    energy: "philosophical, seeing deeper meaning, spiritual insight, \"there's something deeper here\""
  },
  "provider": {
    expression: "warm protective smile, reassuring steady eyes, confident yet gentle, maternal strength and care, grounded centered position, stable and reliable",
    energy: "nurturing protector, safety and security, \"I'll keep you safe\""
  },
  "storyteller": {
    expression: "animated expressive face, eyes sparkling with a secret to share, dramatic engaging smile, theatrical captivating presence, dynamic angle, about to speak",
    energy: "narrative magic, captivating audience, dramatic flair, \"let me tell you a story\""
  },
  "strategist": {
    expression: "sharp focused gaze, confident knowing look, slight smile of someone who has figured out the winning move, chin slightly up, commanding authoritative angle",
    energy: "tactical genius, chess master, calculated confidence, \"here's the smart move\""
  },
  "survivor": {
    expression: "serious determined look, no-nonsense direct gaze, eyes showing resilience and hard-won wisdom, set jaw, straight on, solid grounded position",
    energy: "practical grit, tough resilience, real-world tested, \"when things get hard, you'll need this\""
  }
};

// =============================================================================
// GENERATION
// =============================================================================

async function generateKellyArchetype(archetype: string, config: { expression: string; energy: string }): Promise<string | null> {
  console.log(`\n🎨 Generating: Kelly - ${archetype.toUpperCase()}`);
  console.log(`   Expression: ${config.expression.substring(0, 50)}...`);
  
  const fullPrompt = `${KELLY_BASE}, ${config.expression}, ${SCENE_BASE}`;
  
  try {
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: fullPrompt,
          hf_lora: CONFIG.KELLY_LORA_URL,
          lora_scale: CONFIG.LORA_SCALE,
          num_outputs: 1,
          aspect_ratio: "1:1", // Square for HeyGen Photo Avatar
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          prompt_strength: 0.8,
          num_inference_steps: 28,
          disable_safety_checker: true
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`   📥 Downloading from Replicate...`);
    
    // Download image
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    const imageBuffer = Buffer.from(await response.arrayBuffer());
    
    // Save locally
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
    const localPath = path.join(CONFIG.OUTPUT_DIR, `kelly_archetype_${archetype}.png`);
    fs.writeFileSync(localPath, imageBuffer);
    console.log(`   💾 Saved: ${localPath}`);
    
    // Upload to Supabase
    const remotePath = `heygen/archetypes-lora/kelly_${archetype}.png`;
    await supabase.storage.from('kelly-templates').upload(remotePath, imageBuffer, {
      upsert: true,
      contentType: 'image/png',
    });
    const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
    console.log(`   ☁️ Uploaded: ${data.publicUrl}`);
    
    return data.publicUrl;
    
  } catch (error: any) {
    console.error(`   ❌ Failed: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════════════════╗');
  console.log('║  🎨 GENERATING 12 KELLY ARCHETYPE PHOTOS WITH TRAINED LORA             ║');
  console.log('║  For HeyGen Photo Avatars                                              ║');
  console.log('╚════════════════════════════════════════════════════════════════════════╝');
  console.log(`\n⚡ Using Kelly LoRA: ${CONFIG.KELLY_LORA_URL}`);
  console.log(`⚡ LoRA Scale: ${CONFIG.LORA_SCALE}`);

  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not found!');
    process.exit(1);
  }

  const results: Record<string, string> = {};
  const archetypes = Object.keys(ARCHETYPE_PROMPTS);
  
  for (const archetype of archetypes) {
    const url = await generateKellyArchetype(archetype, ARCHETYPE_PROMPTS[archetype]);
    
    if (url) {
      results[archetype] = url;
    } else {
      results[archetype] = 'FAILED';
    }
    
    // Rate limit - wait between requests
    console.log('   ⏳ Waiting 5 seconds...');
    await new Promise(r => setTimeout(r, 5000));
  }

  // Summary
  console.log('\n\n' + '═'.repeat(70));
  console.log('📋 RESULTS');
  console.log('═'.repeat(70));
  
  let successCount = 0;
  for (const [archetype, url] of Object.entries(results)) {
    const status = url.startsWith('http') ? '✅' : '❌';
    if (url.startsWith('http')) successCount++;
    console.log(`${status} ${archetype.padEnd(12)} ${url.startsWith('http') ? url : 'FAILED'}`);
  }
  
  console.log(`\n📊 Success: ${successCount}/12`);
  
  // Save mapping file
  const mappingPath = path.join(CONFIG.OUTPUT_DIR, 'archetype_urls.json');
  fs.writeFileSync(mappingPath, JSON.stringify(results, null, 2));
  console.log(`💾 Mapping saved: ${mappingPath}`);
  
  if (successCount === 12) {
    console.log('\n🎉 ALL 12 KELLYS GENERATED WITH LORA!');
    console.log('\n🎯 NEXT STEPS:');
    console.log('1. Review images in: generated-images/kelly-archetypes-lora/');
    console.log('2. Upload each to HeyGen: app.heygen.com → Avatars → Create');
    console.log('3. Copy Avatar IDs and update the pipeline');
  }
}

main().catch(console.error);

