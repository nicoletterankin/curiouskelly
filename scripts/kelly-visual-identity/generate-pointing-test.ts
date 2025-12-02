/**
 * Kelly Pointing Direction Test
 * 
 * Testing precise control over pointing direction
 * Key: Elbows stay on armrests to keep shot tight and consistent
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import Replicate from "replicate";
import * as fs from "fs";
import * as path from "path";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

// LOCKED SCENE
const SCENE = `pure white cyclorama photography studio, director's chair with black canvas and warm wood frame center frame, soft window light from upper right, camera straight on at eye level, medium shot waist up, 8K UHD, Hasselblad 100c 85mm f/2.8`;

// LOCKED APPEARANCE  
const KELLY = `kelly, brown wavy shoulder-length hair with caramel highlights, hazel-brown eyes, soft natural features, light natural makeup, soft powder blue cashmere crewneck sweater, calm confident expression, subtle warm smile`;

// KEY CONSTRAINT: Elbows stay on armrests
const ANCHOR = `elbows resting on chair armrests`;

// Test variations for pointing UP and DOWN
const POINTING_TESTS: Record<string, string> = {
  
  // === POINTING UP VARIATIONS ===
  
  point_up_v1: `${KELLY}, seated in director's chair, ${ANCHOR}, right forearm raised with index finger pointing straight up toward ceiling, eyes looking up following finger, subtle smile, ${SCENE}`,
  
  point_up_v2: `${KELLY}, seated in director's chair, ${ANCHOR}, right hand raised with finger pointing upward at 12 o'clock position, head tilted up slightly, gaze directed upward, inviting expression, ${SCENE}`,
  
  point_up_v3: `${KELLY}, seated in director's chair, ${ANCHOR}, right arm bent at elbow on armrest with hand raised and index finger extended pointing directly up, looking upward with encouraging expression, ${SCENE}`,
  
  point_up_v4: `${KELLY}, seated in director's chair, ${ANCHOR}, forearm vertical with index finger pointing to the sky, eyes gazing up at where finger points, warm knowing smile, ${SCENE}`,
  
  // === POINTING DOWN VARIATIONS ===
  
  point_down_v1: `${KELLY}, seated in director's chair, ${ANCHOR}, right forearm lowered with index finger pointing straight down toward floor, eyes looking down following finger, subtle smile, ${SCENE}`,
  
  point_down_v2: `${KELLY}, seated in director's chair, ${ANCHOR}, right hand lowered with finger pointing downward at 6 o'clock position, head tilted down slightly, gaze directed downward, inviting expression, ${SCENE}`,
  
  point_down_v3: `${KELLY}, seated in director's chair, ${ANCHOR}, right arm bent at elbow on armrest with hand lowered and index finger extended pointing directly down, looking downward with encouraging expression, ${SCENE}`,
  
  point_down_v4: `${KELLY}, seated in director's chair, ${ANCHOR}, forearm angled down with index finger pointing to the ground, eyes gazing down at where finger points, warm knowing smile, ${SCENE}`,
  
  // === OPEN PALM PRESENTING UP/DOWN (Vanna style) ===
  
  present_up_v1: `${KELLY}, seated in director's chair, ${ANCHOR}, right forearm raised with open palm facing up gesturing toward top of frame, eyes looking upward, elegant presenting gesture, ${SCENE}`,
  
  present_down_v1: `${KELLY}, seated in director's chair, ${ANCHOR}, right forearm lowered with open palm facing down gesturing toward bottom of frame, eyes looking downward, elegant presenting gesture, ${SCENE}`,
  
  // === TWO-HAND VARIATIONS ===
  
  both_up: `${KELLY}, seated in director's chair, both elbows on armrests, both forearms raised with index fingers pointing upward, looking up with bright encouraging expression, ${SCENE}`,
  
  both_down: `${KELLY}, seated in director's chair, both elbows on armrests, both forearms lowered with hands gesturing downward, looking down with inviting expression, ${SCENE}`,
};

const OUTPUT_DIR = path.join(process.cwd(), "pointing-test-batch");

async function generate(name: string, prompt: string): Promise<boolean> {
  console.log(`\n🎯 ${name}`);
  
  try {
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: prompt,
          hf_lora: LORA_URL,
          lora_scale: 0.9,
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed`);
    
    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(path.join(OUTPUT_DIR, `${name}.png`), buffer);
    
    console.log(`   ✅ Saved`);
    return true;
    
  } catch (error: any) {
    console.log(`   ❌ ${error.message}`);
    return false;
  }
}

async function main() {
  console.log("🎯 KELLY POINTING DIRECTION TEST");
  console.log("=".repeat(50));
  console.log("Testing precise up/down pointing with anchored elbows");
  console.log("");
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  let success = 0;
  const total = Object.keys(POINTING_TESTS).length;
  
  for (const [name, prompt] of Object.entries(POINTING_TESTS)) {
    if (await generate(name, prompt)) success++;
    await new Promise(r => setTimeout(r, 2000));
  }
  
  console.log("\n" + "=".repeat(50));
  console.log(`✅ Generated: ${success}/${total}`);
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  
  // Open folder
  const { execSync } = require("child_process");
  execSync(`explorer "${OUTPUT_DIR}"`);
}

main().catch(console.error);

