#!/usr/bin/env npx tsx
/**
 * 🎨 GENERATE 12 KELLY ARCHETYPE PHOTOS
 * 
 * Uses Flux to create 12 consistent Kelly variations,
 * one for each archetype, ready for HeyGen upload.
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

const CONFIG = {
  FAL_API_KEY: process.env.FAL_KEY || process.env.FAL_API_KEY,
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-archetypes'),
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// Base prompt elements for consistency
const BASE_PROMPT = `photorealistic portrait of Kelly, a young woman in her late 20s, 
brown wavy hair with caramel highlights, warm brown eyes, 
wearing a light blue ribbed sweater, clean white background, 
professional headshot lighting, shoulders visible, 
high quality, 8k, detailed skin texture`;

// Archetype-specific expression and energy
const ARCHETYPE_PROMPTS: Record<string, string> = {
  "scientist": `${BASE_PROMPT}, 
    EXPRESSION: focused analytical gaze, one eyebrow slightly raised, 
    knowing slight smile, direct confident eye contact,
    HEAD: straight on, chin slightly up,
    ENERGY: intellectual, evidence-based, curious researcher`,

  "explorer": `${BASE_PROMPT},
    EXPRESSION: wide eyes sparkling with wonder, excited genuine smile showing teeth,
    eyebrows raised in delighted curiosity,
    HEAD: tilted slightly to the right, looking slightly upward,
    ENERGY: adventurous, discovering something amazing, childlike wonder`,

  "rebel": `${BASE_PROMPT},
    EXPRESSION: confident asymmetric smirk, one corner of mouth raised,
    intense direct eye contact, eyebrows slightly furrowed with challenge,
    HEAD: chin down slightly, looking up through eyebrows, defiant angle,
    ENERGY: edgy, questioning authority, bold challenger`,

  "architect": `${BASE_PROMPT},
    EXPRESSION: thoughtful concentrated look, lips pressed together slightly,
    eyes showing deep focus and analysis, calm inner confidence,
    HEAD: perfectly centered and balanced, composed posture,
    ENERGY: systematic, structured, building understanding`,

  "diplomat": `${BASE_PROMPT},
    EXPRESSION: warm welcoming smile, soft approachable eyes,
    gentle head nod feeling, open trustworthy expression,
    HEAD: tilted slightly with warmth, inviting angle,
    ENERGY: balanced, understanding, bridging perspectives`,

  "empath": `${BASE_PROMPT},
    EXPRESSION: gentle compassionate smile, eyes full of understanding,
    soft caring gaze, slightly parted lips as if listening deeply,
    HEAD: tilted with warmth and care, leaning in feeling,
    ENERGY: nurturing, emotionally connected, deeply feeling`,

  "macgyver": `${BASE_PROMPT},
    EXPRESSION: practical creative grin, eyes bright with an idea,
    asymmetrical knowing smile, engaged and ready to act,
    HEAD: tilted forward slightly, action-ready posture,
    ENERGY: resourceful, hands-on problem solver, inventive`,

  "mystic": `${BASE_PROMPT},
    EXPRESSION: serene knowing smile, eyes with depth and ancient wisdom,
    peaceful profound gaze, subtle mysterious quality,
    HEAD: slight upward contemplative tilt, transcendent angle,
    ENERGY: philosophical, seeing deeper meaning, spiritual insight`,

  "provider": `${BASE_PROMPT},
    EXPRESSION: warm protective smile, reassuring steady eyes,
    confident yet gentle, maternal strength and care,
    HEAD: grounded centered position, stable and reliable,
    ENERGY: nurturing protector, safety and security`,

  "storyteller": `${BASE_PROMPT},
    EXPRESSION: animated expressive face, eyes sparkling with a secret to share,
    dramatic engaging smile, theatrical captivating presence,
    HEAD: dynamic angle, mid-gesture feeling, about to speak,
    ENERGY: narrative magic, captivating audience, dramatic flair`,

  "strategist": `${BASE_PROMPT},
    EXPRESSION: sharp focused gaze, confident knowing look,
    slight smile of someone who has figured out the winning move,
    HEAD: chin slightly up, commanding authoritative angle,
    ENERGY: tactical genius, chess master, calculated confidence`,

  "survivor": `${BASE_PROMPT},
    EXPRESSION: serious determined look, no-nonsense direct gaze,
    eyes showing resilience and hard-won wisdom, set jaw,
    HEAD: straight on, solid grounded position, unshakeable,
    ENERGY: practical grit, tough resilience, real-world tested`,
};

async function generateImage(archetype: string, prompt: string): Promise<string> {
  console.log(`\n🎨 Generating: Kelly - ${archetype}`);
  
  const response = await fetch('https://fal.run/fal-ai/flux-pro/v1.1', {
    method: 'POST',
    headers: {
      'Authorization': `Key ${CONFIG.FAL_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: prompt,
      image_size: 'square_hd', // 1024x1024
      num_images: 1,
      enable_safety_checker: false,
      seed: 42, // Consistent seed for similarity
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Fal.ai error: ${error}`);
  }

  const result = await response.json();
  const imageUrl = result.images?.[0]?.url;
  
  if (!imageUrl) {
    throw new Error('No image URL in response');
  }

  // Download the image
  const imageResponse = await fetch(imageUrl);
  const imageBuffer = Buffer.from(await imageResponse.arrayBuffer());
  
  // Save locally
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
  const localPath = path.join(CONFIG.OUTPUT_DIR, `kelly_archetype_${archetype}.png`);
  fs.writeFileSync(localPath, imageBuffer);
  console.log(`   💾 Saved: ${localPath}`);
  
  // Upload to Supabase
  const remotePath = `heygen/archetypes/kelly_${archetype}.png`;
  await supabase.storage.from('kelly-templates').upload(remotePath, imageBuffer, {
    upsert: true,
    contentType: 'image/png',
  });
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
  console.log(`   ☁️ Uploaded: ${data.publicUrl}`);
  
  return data.publicUrl;
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎨 GENERATING 12 KELLY ARCHETYPE PHOTOS                   ║');
  console.log('║  For HeyGen Photo Avatars                                  ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  if (!CONFIG.FAL_API_KEY) {
    console.error('❌ FAL_KEY or FAL_API_KEY not found in environment');
    process.exit(1);
  }

  const results: Record<string, string> = {};
  const archetypes = Object.keys(ARCHETYPE_PROMPTS);
  
  for (const archetype of archetypes) {
    try {
      const url = await generateImage(archetype, ARCHETYPE_PROMPTS[archetype]);
      results[archetype] = url;
    } catch (error: any) {
      console.error(`   ❌ Failed: ${error.message}`);
      results[archetype] = 'FAILED';
    }
    
    // Rate limit - wait between requests
    await new Promise(r => setTimeout(r, 2000));
  }

  console.log('\n\n' + '═'.repeat(60));
  console.log('📋 RESULTS');
  console.log('═'.repeat(60));
  
  for (const [archetype, url] of Object.entries(results)) {
    console.log(`${archetype}: ${url.startsWith('http') ? '✅' : '❌'}`);
  }
  
  // Save mapping file
  const mappingPath = path.join(CONFIG.OUTPUT_DIR, 'archetype_urls.json');
  fs.writeFileSync(mappingPath, JSON.stringify(results, null, 2));
  console.log(`\n💾 Mapping saved: ${mappingPath}`);
  
  console.log('\n\n🎯 NEXT STEPS:');
  console.log('1. Review images in: generated-images/kelly-archetypes/');
  console.log('2. Upload each to HeyGen: app.heygen.com → Avatars → Create');
  console.log('3. Copy Avatar IDs and add to the pipeline');
}

main().catch(console.error);

