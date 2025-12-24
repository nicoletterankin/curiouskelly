#!/usr/bin/env npx tsx
/**
 * 🎭 GENERATE 12 KELLY ARCHETYPES WITH DISTINCTIVE PROPS
 * 
 * Each archetype gets a VISUAL PROP that makes it instantly recognizable.
 * ALL 12 use IDENTICAL pose, framing, and positioning so Kelly stays 
 * perfectly stationary when flipping between images.
 * 
 * Uses: Curious Kelly LoRA → Replicate flux-dev-lora
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
  LORA_SCALE: 0.90,
  
  // Fixed seed for pose consistency across all 12
  MASTER_SEED: 424242,
  
  // Output
  OUTPUT_DIR: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-with-props'),
  
  // Supabase
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// LOCKED POSE & FRAMING (shared across ALL 12 archetypes)
// =============================================================================

const LOCKED_POSE = `standing in exact center of frame, shoulders squared to camera, slight three-quarter turn to left, arms relaxed at sides with hands visible at waist level, weight evenly distributed, chin level with horizon, eyes looking directly at camera`;

const LOCKED_FRAMING = `waist-up portrait, subject perfectly centered, symmetrical composition, head positioned in upper third of frame`;

const LOCKED_LIGHTING = `three-point studio lighting, soft key light from camera-left at 45 degrees, fill light from camera-right, hair light from above-behind, even illumination on face, no harsh shadows`;

const LOCKED_BACKGROUND = `pure white seamless cyclorama photography studio backdrop, infinite white background, clean and minimal`;

const CAMERA_SPECS = `shot on Hasselblad H6D-100c, 85mm f/2.8 lens, shallow depth of field on background, tack sharp focus on eyes, 8K UHD resolution, photorealistic`;

// =============================================================================
// KELLY CHARACTER - LOCKED (matches LoRA training)
// =============================================================================

const KELLY_BASE = `kelly, young woman late 20s, brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown expressive eyes, soft natural features, flawless skin, light natural makeup, wearing soft powder blue cashmere crewneck sweater`;

// =============================================================================
// 12 ARCHETYPE PROMPTS WITH DISTINCTIVE PROPS
// Each prop is carefully chosen to be instantly recognizable at a glance
// =============================================================================

const ARCHETYPE_WITH_PROPS: Record<string, { 
  prop: string; 
  propPlacement: string;
  expression: string; 
  energy: string;
  visualDetails: string;
}> = {
  
  // ═══════════════════════════════════════════════════════════════════════════
  // 1. SCIENTIST - Lab goggles + clipboard
  // ═══════════════════════════════════════════════════════════════════════════
  "scientist": {
    prop: "clear safety lab goggles pushed up on forehead like a headband, holding a clipboard with scientific charts in left hand at waist level",
    propPlacement: "goggles resting on top of head just above hairline, clipboard held casually at hip",
    expression: "focused analytical gaze, one eyebrow slightly raised in curiosity, knowing slight smile suggesting she's found something interesting, direct confident eye contact, chin slightly raised with intellectual authority",
    energy: "evidence-based confidence, curious researcher who trusts data, 'I have the proof'",
    visualDetails: "goggles have slight gleam from studio lights, clipboard shows visible graph lines and data points"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 2. EXPLORER - Vintage brass compass + adventure hat
  // ═══════════════════════════════════════════════════════════════════════════
  "explorer": {
    prop: "vintage brass compass with ornate engravings held delicately in right palm at chest level, leather explorer's hat with a wide brim hanging behind shoulders on cord",
    propPlacement: "compass positioned at center chest clearly visible, hat suspended behind back on brown leather cord",
    expression: "wide eyes sparkling with wonder and anticipation, excited genuine smile showing teeth, eyebrows raised high in delighted curiosity, gaze directed slightly upward as if seeing new horizons",
    energy: "adventurous discovery, childlike wonder at the unknown, 'there's so much to discover!'",
    visualDetails: "compass needle visible pointing north, brass patina showing age and use, hat shows subtle weathering from expeditions"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 3. REBEL - Aviator sunglasses + leather jacket draped
  // ═══════════════════════════════════════════════════════════════════════════
  "rebel": {
    prop: "classic black aviator sunglasses hooked into sweater neckline at center chest, vintage black leather motorcycle jacket draped over left shoulder like a cape",
    propPlacement: "sunglasses hanging at collar V-point, jacket slung casually over one shoulder",
    expression: "confident asymmetric smirk with one corner of mouth raised defiantly, intense direct eye contact with slight narrowing, eyebrows slightly furrowed in challenge, chin down looking up through eyebrows with rebellious attitude",
    energy: "bold challenger energy, questioning authority, 'rules are made to be questioned'",
    visualDetails: "sunglasses have chrome arms catching light, leather jacket shows authentic creases and silver zippers"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 4. ARCHITECT - Blueprint tube + drafting pencil
  // ═══════════════════════════════════════════════════════════════════════════
  "architect": {
    prop: "architectural blueprint tube in blue plastic held vertically in left hand at hip, classic yellow drafting pencil tucked behind right ear",
    propPlacement: "blueprint tube held upright alongside body, pencil visible behind ear through hair",
    expression: "thoughtful concentrated look with lips pressed together slightly, eyes showing deep analytical focus and calculated planning, calm inner confidence of someone who sees the whole picture, head perfectly centered and balanced, composed posture",
    energy: "systematic builder, structured thinker, 'let me show you how this comes together'",
    visualDetails: "blueprint tube has visible cap and label, pencil is sharpened to a fine point, both items look professional and well-used"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 5. DIPLOMAT - Olive branch + two small flags
  // ═══════════════════════════════════════════════════════════════════════════
  "diplomat": {
    prop: "small olive branch with silver-green leaves held gracefully in right hand at heart level, two miniature crossed flags in subtle neutral colors tucked into left hand",
    propPlacement: "olive branch positioned at center chest like offering peace, flags held low at waist discretely",
    expression: "warm welcoming diplomatic smile, soft approachable eyes radiating understanding, gentle head nod feeling, open and trustworthy expression, head tilted slightly to the side with genuine interest",
    energy: "bridge builder, sees all perspectives, 'let's find common ground together'",
    visualDetails: "olive leaves are fresh and vibrant, flags are small silk on wooden sticks, both props suggest peace and unity"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 6. EMPATH - Hands forming heart shape + soft glow effect
  // ═══════════════════════════════════════════════════════════════════════════
  "empath": {
    prop: "both hands positioned at heart level forming a gentle heart shape with fingers and thumbs touching, small dried lavender sprig tucked behind left ear",
    propPlacement: "hands creating heart shape at center chest, lavender visible in hair",
    expression: "gentle compassionate smile full of warmth, eyes full of deep understanding and emotional connection, soft caring gaze that sees into your soul, slightly parted lips as if listening with whole being, head tilted with genuine care and empathy",
    energy: "emotional connector, feels what you feel, 'your feelings are valid and I understand'",
    visualDetails: "hands show delicate pose with natural light between fingers, lavender adds soft purple accent"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 7. MACGYVER - Swiss army knife + roll of duct tape
  // ═══════════════════════════════════════════════════════════════════════════
  "macgyver": {
    prop: "red Swiss army knife with multiple tools partially extended held in right hand at waist level, silver duct tape roll hanging from left wrist like a bracelet",
    propPlacement: "knife displayed at hip showing tools, tape roll worn on wrist like oversized bangle",
    expression: "practical creative grin of someone with a clever solution, eyes bright and sparkling with a new idea, asymmetrical knowing smile, engaged and ready to improvise, head tilted forward slightly with action-ready energy",
    energy: "resourceful problem solver, can fix anything with anything, 'I've got just the thing'",
    visualDetails: "knife shows screwdriver blade and scissors extended, duct tape is classic silver with visible texture"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 8. MYSTIC - Crystal sphere + third eye gem
  // ═══════════════════════════════════════════════════════════════════════════
  "mystic": {
    prop: "clear quartz crystal sphere the size of a tangerine cradled gently in both palms at solar plexus level, small amethyst gem adhered to forehead at third eye position",
    propPlacement: "crystal ball held centered at chest in cupped hands, gem centered on forehead between eyebrows",
    expression: "serene knowing smile with ancient wisdom in eyes, eyes with remarkable depth and mysterious knowledge, peaceful profound gaze seeing beyond the visible, subtle ethereal quality, slight upward contemplative tilt of head",
    energy: "spiritual seer, perceives deeper meaning, 'there is more to this than meets the eye'",
    visualDetails: "crystal sphere has internal rainbow refractions from studio lights, amethyst is deep purple and catches light"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 9. PROVIDER - Woven basket + fresh bread
  // ═══════════════════════════════════════════════════════════════════════════
  "provider": {
    prop: "small woven wicker basket with handle held in crook of left arm at waist, fresh artisan bread loaf with golden crust visible inside basket",
    propPlacement: "basket cradled protectively against hip, bread visible and warm-looking",
    expression: "warm protective nurturing smile, reassuring steady eyes that promise safety, confident yet gentle maternal energy, grounded centered position radiating stability, reliable and trustworthy gaze",
    energy: "nurturing protector, source of comfort and security, 'I will take care of you'",
    visualDetails: "basket shows natural wicker weave pattern, bread has rustic flour dusting and golden brown crust"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 10. STORYTELLER - Open storybook + theatrical mask
  // ═══════════════════════════════════════════════════════════════════════════
  "storyteller": {
    prop: "antique leather-bound storybook held open in both hands at chest level showing illustrated pages, small golden comedy theater mask attached to a ribbon hanging from right wrist",
    propPlacement: "book held open facing outward at heart level, mask dangling from wrist on satin ribbon",
    expression: "animated expressive face mid-story, eyes sparkling with a secret to share, dramatic engaging smile with theatrical presence, captivating energy of someone about to reveal something amazing, dynamic slight lean forward",
    energy: "narrative enchanter, weaves magic with words, 'once upon a time...'",
    visualDetails: "book pages show vintage illustrations and ornate text, mask is classic Greek comedy style in gold finish"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 11. STRATEGIST - Chess queen piece + tactical map
  // ═══════════════════════════════════════════════════════════════════════════
  "strategist": {
    prop: "elegant white marble chess queen piece held between thumb and forefinger of right hand at shoulder level, rolled tactical map with visible grid lines in left hand at hip",
    propPlacement: "chess piece raised and displayed prominently near face, map held low at side",
    expression: "sharp focused calculating gaze, confident knowing look of someone three moves ahead, slight victorious smile of having figured out the winning strategy, chin slightly raised with commanding authoritative angle",
    energy: "tactical genius, masterful planner, 'I see the path to victory'",
    visualDetails: "chess queen is carved white marble with fine details, map shows battle lines and strategic markers in red and blue"
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // 12. SURVIVOR - First aid kit + paracord bracelet
  // ═══════════════════════════════════════════════════════════════════════════
  "survivor": {
    prop: "compact red first aid kit with white cross held firmly in right hand at waist, thick braided paracord survival bracelet in olive green on left wrist",
    propPlacement: "first aid kit gripped confidently at hip, paracord bracelet visible on opposite wrist",
    expression: "serious determined look with no-nonsense direct gaze, eyes showing hard-won resilience and tested wisdom, set jaw of someone who has been through challenges, steady unshakeable composure, grounded solid stance",
    energy: "tested survivor, practical and prepared, 'when things get tough, you'll want to know this'",
    visualDetails: "first aid kit is compact field type with clear white cross, paracord shows intricate weave pattern in military green"
  }
};

// =============================================================================
// BUILD COMPLETE PROMPT
// =============================================================================

function buildPrompt(archetype: string): string {
  const config = ARCHETYPE_WITH_PROPS[archetype];
  
  return `${KELLY_BASE}, ${LOCKED_POSE}, ${config.prop}, ${config.propPlacement}, ${config.expression}, ${config.visualDetails}, ${LOCKED_FRAMING}, ${LOCKED_LIGHTING}, ${LOCKED_BACKGROUND}, ${CAMERA_SPECS}`;
}

// =============================================================================
// GENERATION FUNCTION
// =============================================================================

async function generateKellyArchetypeWithProp(archetype: string): Promise<string | null> {
  const config = ARCHETYPE_WITH_PROPS[archetype];
  const fullPrompt = buildPrompt(archetype);
  
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`🎭 GENERATING: Kelly as ${archetype.toUpperCase()}`);
  console.log(`${'─'.repeat(70)}`);
  console.log(`📦 PROP: ${config.prop.substring(0, 60)}...`);
  console.log(`😊 EXPRESSION: ${config.expression.substring(0, 50)}...`);
  console.log(`⚡ ENERGY: ${config.energy}`);
  console.log(`${'─'.repeat(70)}`);
  
  try {
    console.log('🔄 Calling Replicate flux-dev-lora...');
    
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: fullPrompt,
          hf_lora: CONFIG.KELLY_LORA_URL,
          lora_scale: CONFIG.LORA_SCALE,
          num_outputs: 1,
          aspect_ratio: "1:1",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          prompt_strength: 0.8,
          num_inference_steps: 28,
          seed: CONFIG.MASTER_SEED, // SAME SEED FOR CONSISTENT POSE
          disable_safety_checker: true
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`✅ Generated successfully!`);
    console.log(`📥 Downloading...`);
    
    // Download image
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    const imageBuffer = Buffer.from(await response.arrayBuffer());
    
    // Save locally
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
    const localPath = path.join(CONFIG.OUTPUT_DIR, `kelly_${archetype}_prop.png`);
    fs.writeFileSync(localPath, imageBuffer);
    console.log(`💾 Saved locally: ${localPath}`);
    
    // Upload to Supabase
    const remotePath = `heygen/archetypes-with-props/kelly_${archetype}_prop.png`;
    await supabase.storage.from('kelly-templates').upload(remotePath, imageBuffer, {
      upsert: true,
      contentType: 'image/png',
    });
    const { data } = supabase.storage.from('kelly-templates').getPublicUrl(remotePath);
    console.log(`☁️ Uploaded to Supabase: ${data.publicUrl}`);
    
    return data.publicUrl;
    
  } catch (error: any) {
    console.error(`❌ FAILED: ${error.message}`);
    return null;
  }
}

// =============================================================================
// MAIN EXECUTION
// =============================================================================

async function main() {
  console.log('╔══════════════════════════════════════════════════════════════════════════╗');
  console.log('║  🎭 GENERATING 12 KELLY ARCHETYPES WITH DISTINCTIVE PROPS                ║');
  console.log('║  Same Pose • Different Expressions • Unique Visual Props                 ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════╝');
  console.log(`\n⚡ Kelly LoRA: ${CONFIG.KELLY_LORA_URL}`);
  console.log(`⚡ LoRA Scale: ${CONFIG.LORA_SCALE}`);
  console.log(`⚡ Master Seed: ${CONFIG.MASTER_SEED} (ensures consistent pose)`);
  console.log(`📂 Output: ${CONFIG.OUTPUT_DIR}\n`);

  // Validate environment
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not found in environment!');
    process.exit(1);
  }
  
  if (!CONFIG.SUPABASE_URL || !CONFIG.SUPABASE_KEY) {
    console.error('❌ Supabase credentials not found!');
    process.exit(1);
  }

  const results: Record<string, { url: string; prop: string }> = {};
  const archetypes = Object.keys(ARCHETYPE_WITH_PROPS);
  
  console.log(`\n📋 ARCHETYPES TO GENERATE:`);
  archetypes.forEach((a, i) => {
    const prop = ARCHETYPE_WITH_PROPS[a].prop.split(',')[0];
    console.log(`   ${(i+1).toString().padStart(2)}. ${a.padEnd(12)} → ${prop}`);
  });
  
  console.log(`\n🚀 Starting generation...\n`);
  
  for (let i = 0; i < archetypes.length; i++) {
    const archetype = archetypes[i];
    const url = await generateKellyArchetypeWithProp(archetype);
    
    if (url) {
      results[archetype] = {
        url: url,
        prop: ARCHETYPE_WITH_PROPS[archetype].prop.split(',')[0]
      };
    } else {
      results[archetype] = {
        url: 'FAILED',
        prop: ARCHETYPE_WITH_PROPS[archetype].prop.split(',')[0]
      };
    }
    
    // Progress update
    console.log(`\n📊 Progress: ${i + 1}/${archetypes.length} complete`);
    
    // Rate limit between requests
    if (i < archetypes.length - 1) {
      console.log('⏳ Waiting 5 seconds before next generation...\n');
      await new Promise(r => setTimeout(r, 5000));
    }
  }

  // ==========================================================================
  // SUMMARY
  // ==========================================================================
  
  console.log('\n\n' + '═'.repeat(80));
  console.log('📋 GENERATION COMPLETE - RESULTS');
  console.log('═'.repeat(80) + '\n');
  
  let successCount = 0;
  let failedArchetypes: string[] = [];
  
  console.log('ARCHETYPE      PROP                                          STATUS');
  console.log('─'.repeat(80));
  
  for (const [archetype, result] of Object.entries(results)) {
    const isSuccess = result.url.startsWith('http');
    const status = isSuccess ? '✅ SUCCESS' : '❌ FAILED';
    if (isSuccess) successCount++;
    else failedArchetypes.push(archetype);
    
    console.log(`${archetype.padEnd(14)} ${result.prop.substring(0, 45).padEnd(45)} ${status}`);
  }
  
  console.log('─'.repeat(80));
  console.log(`\n📊 Final Score: ${successCount}/12 archetypes generated successfully\n`);
  
  // Save results JSON
  const mappingPath = path.join(CONFIG.OUTPUT_DIR, 'archetype_props_urls.json');
  fs.writeFileSync(mappingPath, JSON.stringify(results, null, 2));
  console.log(`💾 Results saved: ${mappingPath}`);
  
  // Save detailed manifest with prompts
  const manifest = {
    generated: new Date().toISOString(),
    masterSeed: CONFIG.MASTER_SEED,
    loraUrl: CONFIG.KELLY_LORA_URL,
    loraScale: CONFIG.LORA_SCALE,
    archetypes: Object.entries(ARCHETYPE_WITH_PROPS).map(([name, config]) => ({
      name,
      url: results[name]?.url || 'NOT GENERATED',
      prop: config.prop,
      propPlacement: config.propPlacement,
      expression: config.expression,
      energy: config.energy,
      visualDetails: config.visualDetails,
      fullPrompt: buildPrompt(name)
    }))
  };
  
  const manifestPath = path.join(CONFIG.OUTPUT_DIR, 'archetype_manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`📜 Full manifest saved: ${manifestPath}`);
  
  if (successCount === 12) {
    console.log('\n' + '═'.repeat(80));
    console.log('🎉 PERFECT! ALL 12 KELLYS GENERATED WITH PROPS!');
    console.log('═'.repeat(80));
    console.log('\n🎯 NEXT STEPS:');
    console.log(`   1. Review images: ${CONFIG.OUTPUT_DIR}`);
    console.log('   2. Verify Kelly is in SAME POSE across all 12');
    console.log('   3. Verify each PROP is clearly visible and matches archetype');
    console.log('   4. Upload to HeyGen for Photo Avatars if quality approved');
  } else {
    console.log('\n⚠️ Some archetypes failed. Re-run script to retry failed ones:');
    failedArchetypes.forEach(a => console.log(`   - ${a}`));
  }
}

main().catch(console.error);




















