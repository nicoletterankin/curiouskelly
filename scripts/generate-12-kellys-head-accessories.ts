#!/usr/bin/env npx tsx
/**
 * 🎭 GENERATE 12 KELLY ARCHETYPES - HEAD ACCESSORIES ONLY
 * 
 * Each archetype is distinguished ONLY by head/face accessories.
 * Body, hands, sweater, and pose stay IDENTICAL for lesson compatibility.
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
  // Kelly LoRA
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.90,
  
  // Fixed seed for pose consistency
  MASTER_SEED: 777777,
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-head-only'),
  
  // Supabase
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// LOCKED ELEMENTS (IDENTICAL across all 12)
// =============================================================================

// Kelly's consistent appearance
const KELLY_FACE_KID = `kelly, child girl around 9 years old (kid), warm hazel-brown expressive eyes, youthful face proportions with slightly rounder cheeks, small natural nose, warm genuine smile, medium brown hair with subtle caramel highlights, soft waves at shoulder length, center-parted, no makeup, natural child appearance`;
const KELLY_FACE_TEEN = `kelly, teenage girl around 15 years old (teen), warm hazel-brown expressive eyes, youthful teen face proportions with emerging adult bone structure, clear natural youthful skin, warm genuine smile, medium brown hair with subtle caramel highlights, soft waves at shoulder length, center-parted, no heavy makeup, natural teen appearance`;
const KELLY_FACE_ADULT = `kelly, adult woman in late twenties, brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown expressive eyes, soft natural features, flawless skin, light natural makeup`;
const KELLY_FACE_ELDER = `kelly, elder woman in early eighties, silver-white hair with soft waves and faint warm undertones, hazel-brown expressive eyes, soft natural features, gentle laugh lines, natural weathering, dignified graceful aging, very light natural makeup`;
const KELLY_FACE_SUPER_ELDER = `kelly, extremely elderly centenarian woman around 100 years old (super elder), very old grandmother, warm hazel-brown expressive eyes with deep kindness, extremely deep facial wrinkles and creases, deep forehead lines, deep crow's feet, deep under-eye creases with sagging eyelids, deep nasolabial folds, deep marionette lines, pronounced jowls, crepey thin translucent skin texture, visible age spots (liver spots) and sun spots, thin lips, slightly sunken cheeks and temples, gentle sagging skin consistent with extreme old age, dignified fragile grace, very sparse wispy silver-white hair (thin fine strands, receding hairline, visible scalp, thinning crown), minimal natural eyebrows, no heavy makeup (only very light natural makeup), warm genuine smile`;

// LOCKED body and pose - IDENTICAL for all archetypes
const LOCKED_BODY = `wearing soft powder blue cashmere crewneck sweater, shoulders relaxed, arms naturally at sides with hands NOT visible in frame, body facing camera with slight natural angle`;

// LOCKED framing - chest up, clean
const LOCKED_FRAMING = `head and shoulders portrait, chest-up framing, subject perfectly centered, face in sharp focus, clean composition`;

// LOCKED background and lighting
const LOCKED_SCENE = `pure white seamless cyclorama photography studio backdrop, professional three-point studio lighting, soft even illumination, no shadows on background`;

// Camera specs
const CAMERA = `shot on Hasselblad H6D-100c, 85mm f/2.8, shallow depth of field, 8K UHD, photorealistic, professional headshot`;

// Negative prompt (keeps us in-bounds for consistent lesson assets)
const NEGATIVE_BASE = `hands, fingers, holding items, jewelry on hands, messy hair, harsh shadows, open mouth, teeth showing, distorted face, extra limbs, complex background, text, watermark, logo, cartoon, anime, illustration, painting, 3d render, cgi, plastic skin, uncanny valley`;

// =============================================================================
// 12 ARCHETYPE HEAD ACCESSORIES
// Each changes ONLY what's on/around the head
// =============================================================================

const ARCHETYPE_HEADS: Record<string, {
  headAccessory: string;
  expression: string;
  description: string;
}> = {

  // ═══════════════════════════════════════════════════════════════════════════
  // 1. SCIENTIST - Safety goggles on forehead
  // ═══════════════════════════════════════════════════════════════════════════
  "scientist": {
    headAccessory: "clear laboratory safety goggles with elastic strap pushed up onto forehead resting above eyebrows like a headband, goggles catching studio light with subtle gleam",
    expression: "focused analytical gaze with one eyebrow slightly raised, knowing intellectual smile, direct confident eye contact, chin slightly lifted with scholarly authority",
    description: "Lab goggles on forehead"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 2. EXPLORER - Aviator goggles + bandana
  // ═══════════════════════════════════════════════════════════════════════════
  "explorer": {
    headAccessory: "vintage brass and leather aviator flight goggles pushed up on top of head, weathered tan leather headband bandana tied around forehead peeking under goggles",
    expression: "wide eyes sparkling with wonder and excitement, bright genuine smile showing enthusiasm, eyebrows raised in delighted curiosity, gaze directed slightly upward dreamily",
    description: "Aviator goggles + bandana"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 3. REBEL - Dark sunglasses pushed up in hair
  // ═══════════════════════════════════════════════════════════════════════════
  "rebel": {
    headAccessory: "classic black wayfarer sunglasses pushed up and resting on top of head in hair, slightly tousled effortlessly cool hairstyle, small silver hoop earring in one ear",
    expression: "confident asymmetric smirk with one corner of mouth raised defiantly, intense direct eye contact with slight narrowing, eyebrows slightly furrowed in playful challenge",
    description: "Sunglasses in hair + earring"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 4. ARCHITECT - Pencil behind ear + glasses on head
  // ═══════════════════════════════════════════════════════════════════════════
  "architect": {
    headAccessory: "classic yellow drafting pencil tucked behind right ear visible through hair, tortoiseshell reading glasses pushed up on top of head",
    expression: "thoughtful concentrated look with lips pressed together slightly, eyes showing deep analytical focus, calm composed confidence, perfectly balanced centered head position",
    description: "Pencil behind ear + glasses"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 5. DIPLOMAT - Elegant pearl studs + thin headband
  // ═══════════════════════════════════════════════════════════════════════════
  "diplomat": {
    headAccessory: "elegant classic pearl stud earrings, thin navy blue velvet headband pushing hair back slightly, refined polished appearance",
    expression: "warm welcoming diplomatic smile, soft approachable eyes radiating understanding and openness, gentle head tilt conveying genuine interest and receptiveness",
    description: "Pearl studs + velvet headband"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 6. EMPATH - Soft fabric headband + small flower
  // ═══════════════════════════════════════════════════════════════════════════
  "empath": {
    headAccessory: "soft dusty rose pink fabric headband gently holding hair back, tiny dried lavender sprig tucked behind left ear, gentle feminine styling",
    expression: "gentle compassionate smile full of warmth, eyes radiating deep understanding and emotional connection, soft caring gaze, head tilted with genuine empathy and care",
    description: "Pink headband + lavender"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 7. MACGYVER - Shop safety glasses + utility bandana
  // ═══════════════════════════════════════════════════════════════════════════
  "macgyver": {
    headAccessory: "clear protective shop safety glasses with side shields pushed up on forehead, red paisley utility bandana tied around head keeping hair back",
    expression: "practical creative grin of someone with a clever solution, eyes bright and sparkling with ingenious idea, asymmetrical knowing smile, engaged ready-to-act energy",
    description: "Shop glasses + red bandana"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 8. MYSTIC - Amethyst gem + delicate chain headpiece
  // ═══════════════════════════════════════════════════════════════════════════
  "mystic": {
    headAccessory: "small teardrop amethyst crystal gem adhered to center of forehead at third eye position, delicate thin gold chain headpiece draped across hairline",
    expression: "serene knowing smile with ancient wisdom in eyes, peaceful profound gaze seeing beyond the visible, subtle ethereal mysterious quality, slight upward contemplative tilt",
    description: "Third eye amethyst + gold chain"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 9. PROVIDER - Cozy knit headband
  // ═══════════════════════════════════════════════════════════════════════════
  "provider": {
    headAccessory: "wide cozy cream-colored cable knit headband ear warmer wrapped around head, warm nurturing domestic aesthetic, soft comforting appearance",
    expression: "warm protective nurturing smile, reassuring steady eyes that promise safety and care, confident yet gentle maternal energy, grounded stable reliable gaze",
    description: "Cream knit headband"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 10. STORYTELLER - Vintage glasses on chain + feather
  // ═══════════════════════════════════════════════════════════════════════════
  "storyteller": {
    headAccessory: "vintage round gold-rimmed reading glasses pushed up on top of head, thin gold glasses chain hanging down, small peacock feather tucked decoratively in hair",
    expression: "animated expressive face mid-story, eyes sparkling with secrets to share, dramatic engaging smile, theatrical captivating presence suggesting wonder",
    description: "Gold glasses + peacock feather"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 11. STRATEGIST - Sharp angular glasses + chess clip
  // ═══════════════════════════════════════════════════════════════════════════
  "strategist": {
    headAccessory: "sharp modern angular black-framed glasses pushed up on top of head, small gold chess queen hair clip pinning back one side of hair",
    expression: "sharp focused calculating gaze, confident knowing look of someone thinking several moves ahead, slight strategic smile, chin slightly raised with commanding authority",
    description: "Angular glasses + chess clip"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 12. SURVIVOR - Tactical bandana + dog tags
  // ═══════════════════════════════════════════════════════════════════════════
  "survivor": {
    headAccessory: "olive green tactical military bandana tied around forehead, silver military dog tags on ball chain visible at neckline, rugged prepared appearance",
    expression: "serious determined look with no-nonsense direct gaze, eyes showing resilience and hard-won wisdom, set jaw of someone who has been tested, unshakeable composure",
    description: "Military bandana + dog tags"
  }
};

// =============================================================================
// BUILD PROMPT
// =============================================================================

type AgeVariantKey = 'kid' | 'teen' | 'adult' | 'elder' | 'super_elder';
type AgeVariant = {
  key: AgeVariantKey;
  label: string;
  kellyFace: string;
  seed: number;
  loraScale?: number;
  negativeExtra?: string;
};

const AGE_VARIANTS: Record<AgeVariantKey, AgeVariant> = {
  kid: {
    key: 'kid',
    label: '6-12',
    kellyFace: KELLY_FACE_KID,
    seed: 777776,
    loraScale: CONFIG.LORA_SCALE,
    negativeExtra: `adult, elder, super elder, wrinkles, crow's feet, gray hair, white hair, heavy makeup, professional business attire`
  },
  teen: {
    key: 'teen',
    label: '13-17',
    kellyFace: KELLY_FACE_TEEN,
    seed: 777775,
    loraScale: CONFIG.LORA_SCALE,
    negativeExtra: `child, kid, toddler, baby features, adult, elder, super elder, wrinkles, crow's feet, gray hair, white hair, heavy makeup`
  },
  adult: {
    key: 'adult',
    label: '18-35',
    kellyFace: KELLY_FACE_ADULT,
    seed: 777777,
    loraScale: CONFIG.LORA_SCALE,
    negativeExtra: `child, teenager, elderly, gray hair, white hair, deep wrinkles`
  },
  elder: {
    key: 'elder',
    label: '61-102',
    kellyFace: KELLY_FACE_ELDER,
    // Slightly different seed to avoid “same face, just recolored” artifacts
    seed: 777779,
    loraScale: CONFIG.LORA_SCALE,
    negativeExtra: `child, teenager, baby face, overly young, smooth unaged skin`
  },
  super_elder: {
    key: 'super_elder',
    label: '90-110',
    kellyFace: KELLY_FACE_SUPER_ELDER,
    seed: 888880,
    // Reduce LoRA further so the prompt can push truly centenarian features.
    loraScale: 0.55,
    negativeExtra: `child, teenager, adult, middle-aged, too young, smooth skin, airbrushed skin, no wrinkles, heavy makeup, glam makeup, youthful skin, thick hair, full youthful hair`
  }
};

function buildPrompt(archetype: string, age: AgeVariant): string {
  const config = ARCHETYPE_HEADS[archetype];
  
  // Combine all elements with head accessory being prominent
  return `${age.kellyFace}, ${config.headAccessory}, ${config.expression}, ${LOCKED_BODY}, ${LOCKED_FRAMING}, ${LOCKED_SCENE}, ${CAMERA}`;
}

// =============================================================================
// GENERATION
// =============================================================================

function getAgeOutputDir(age: AgeVariant): string {
  // Avoid overwriting the existing “adult” set in the root directory.
  // We write age-variants to: generated-images/kelly-archetypes-head-only/age/<key>/
  return path.join(CONFIG.OUTPUT_DIR, 'age', age.key);
}

function getAgeRemoteBase(age: AgeVariant): string {
  return `heygen/archetypes-head-only/age/${age.key}`;
}

async function generateArchetype(archetype: string, age: AgeVariant): Promise<string | null> {
  const config = ARCHETYPE_HEADS[archetype];
  const fullPrompt = buildPrompt(archetype, age);
  const negativePrompt = `${NEGATIVE_BASE}${age.negativeExtra ? `, ${age.negativeExtra}` : ''}`;
  
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`🎭 GENERATING: Kelly as ${archetype.toUpperCase()} (age ${age.label})`);
  console.log(`${'─'.repeat(70)}`);
  console.log(`👒 HEAD: ${config.description}`);
  console.log(`😊 EXPRESSION: ${config.expression.substring(0, 60)}...`);
  console.log(`${'─'.repeat(70)}`);
  
  try {
    console.log('🔄 Calling Replicate flux-dev-lora...');
    
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: fullPrompt,
          hf_lora: CONFIG.KELLY_LORA_URL,
          lora_scale: age.loraScale ?? CONFIG.LORA_SCALE,
          num_outputs: 1,
          aspect_ratio: "1:1",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          prompt_strength: 0.8,
          num_inference_steps: 28,
          seed: age.seed,
          disable_safety_checker: true,
          negative_prompt: negativePrompt
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`✅ Generated!`);
    console.log(`📥 Downloading...`);
    
    // Download
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    const imageBuffer = Buffer.from(await response.arrayBuffer());
    
    // Save locally
    const outDir = getAgeOutputDir(age);
    fs.mkdirSync(outDir, { recursive: true });
    const localPath = path.join(outDir, `kelly_${archetype}_head.png`);
    fs.writeFileSync(localPath, imageBuffer);
    console.log(`💾 Saved: ${localPath}`);
    
    // Upload to Supabase
    const remotePath = `${getAgeRemoteBase(age)}/kelly_${archetype}_head.png`;
    await supabase.storage.from('kelly-templates').upload(remotePath, imageBuffer, {
      upsert: true,
      contentType: 'image/png',
    });
    const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
    console.log(`☁️ Uploaded: ${data.publicUrl}`);
    
    return data.publicUrl;
    
  } catch (error: any) {
    console.error(`❌ FAILED: ${error.message}`);
    return null;
  }
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('╔══════════════════════════════════════════════════════════════════════════╗');
  console.log('║  🎭 GENERATING 12 KELLY ARCHETYPES - HEAD ACCESSORIES ONLY               ║');
  console.log('║  Body identical • Hands free • Only head changes • Optional age variants ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════╝');
  console.log(`\n⚡ Kelly LoRA: ${CONFIG.KELLY_LORA_URL}`);
  console.log(`⚡ LoRA Scale: ${CONFIG.LORA_SCALE}`);
  console.log(`📂 Output: ${CONFIG.OUTPUT_DIR}\n`);

  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not found!');
    process.exit(1);
  }

  // CLI:
  //   tsx scripts/generate-12-kellys-head-accessories.ts --ages kid,teen,adult,elder,super_elder
  // Defaults to generating the Core 60 (kid, teen, adult, elder, super_elder).
  const args = new Set(process.argv.slice(2));
  const agesArg = process.argv.find(a => a.startsWith('--ages='))?.split('=')[1];
  const agesList = (agesArg ? agesArg.split(',') : ['kid', 'teen', 'adult', 'elder', 'super_elder']).map(s => s.trim()).filter(Boolean);
  const selectedAges: AgeVariant[] = agesList
    .map((k) => AGE_VARIANTS[k as AgeVariantKey])
    .filter(Boolean);

  if (selectedAges.length === 0) {
    console.error('❌ No valid ages selected. Use --ages=kid,teen,adult,elder,super_elder');
    process.exit(1);
  }

  const results: Record<string, Record<string, { url: string; description: string }>> = {};
  const archetypes = Object.keys(ARCHETYPE_HEADS);
  
  console.log(`📋 ARCHETYPES (Head Accessories Only):`);
  archetypes.forEach((a, i) => {
    console.log(`   ${(i+1).toString().padStart(2)}. ${a.padEnd(12)} → ${ARCHETYPE_HEADS[a].description}`);
  });
  console.log(`\n📋 AGE VARIANTS:`);
  selectedAges.forEach((a, i) => {
    console.log(`   ${(i+1).toString().padStart(2)}. ${a.key.padEnd(8)} → ${a.label}`);
  });
  
  console.log(`\n🚀 Starting generation...\n`);
  
  for (const age of selectedAges) {
    results[age.key] = {};

    for (let i = 0; i < archetypes.length; i++) {
      const archetype = archetypes[i];
      const url = await generateArchetype(archetype, age);

      results[age.key][archetype] = {
        url: url || 'FAILED',
        description: ARCHETYPE_HEADS[archetype].description
      };

      console.log(`\n📊 Progress (${age.key}): ${i + 1}/${archetypes.length}`);

      if (i < archetypes.length - 1) {
        console.log('⏳ Waiting 5 seconds...\n');
        await new Promise(r => setTimeout(r, 5000));
      }
    }
  }

  // Summary
  console.log('\n\n' + '═'.repeat(80));
  console.log('📋 GENERATION COMPLETE');
  console.log('═'.repeat(80) + '\n');
  
  for (const age of selectedAges) {
    let success = 0;
    console.log(`AGE: ${age.key} (${age.label})`);
    console.log('ARCHETYPE      HEAD ACCESSORY                              STATUS');
    console.log('─'.repeat(80));

    for (const [arch, result] of Object.entries(results[age.key])) {
      const ok = result.url.startsWith('http');
      if (ok) success++;
      console.log(`${arch.padEnd(14)} ${result.description.padEnd(40)} ${ok ? '✅ SUCCESS' : '❌ FAILED'}`);
    }

    console.log('─'.repeat(80));
    console.log(`📊 Score (${age.key}): ${success}/12\n`);

    // Save per-age JSON mapping into the age output dir
    const ageDir = getAgeOutputDir(age);
    const jsonPath = path.join(ageDir, 'archetype_head_urls.json');
    fs.writeFileSync(jsonPath, JSON.stringify(results[age.key], null, 2));
    console.log(`💾 Results (${age.key}): ${jsonPath}`);
  }
  
  // Save manifest
  const manifest = {
    generated: new Date().toISOString(),
    concept: "HEAD ACCESSORIES ONLY - body and hands stay identical for lesson compatibility",
    loraUrl: CONFIG.KELLY_LORA_URL,
    ages: selectedAges.map(a => ({ key: a.key, label: a.label, seed: a.seed, loraScale: a.loraScale ?? CONFIG.LORA_SCALE })),
    archetypes: Object.entries(ARCHETYPE_HEADS).map(([name, config]) => ({
      name,
      headAccessory: config.headAccessory,
      expression: config.expression,
      description: config.description,
      promptsByAge: Object.fromEntries(selectedAges.map(a => [a.key, buildPrompt(name, a)])),
      urlsByAge: Object.fromEntries(selectedAges.map(a => [a.key, results[a.key]?.[name]?.url || 'NOT GENERATED']))
    })),
  };
  
  const manifestPath = path.join(CONFIG.OUTPUT_DIR, 'manifest.age-variants.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`📜 Manifest: ${manifestPath}`);
  
  console.log('\n✅ Done.');
  console.log('ℹ️ Tip: default run generates the Core 60 (kid, teen, adult, elder, super_elder) into /age/<bucket>/ with upserts to Supabase.');
}

main().catch(console.error);










