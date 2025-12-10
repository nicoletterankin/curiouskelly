/**
 * Kelly Precision Pointing - "Dissociated Limb" Strategy
 * 
 * GOAL: 20 perfect images with strict single-hand usage.
 * STRATEGY: Explicitly assign a "job" to the inactive hand (e.g., gripping armrest) 
 * to prevent clasping. Lock framing to "Medium Shot Waist Up".
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

// LOCKED SCENE - STRICT FRAMING
// "Medium shot from waist up" is crucial for splicing consistency
const SCENE = `pure white cyclorama photography studio, director's chair centered, soft window light from upper right, medium shot from waist up, camera fixed at eye level 1.5 meters away, 85mm lens, sharp focus on face and hand, 8K UHD, high fidelity`;

// LOCKED APPEARANCE
const KELLY_BASE = `kelly, brown wavy shoulder-length hair with caramel highlights, hazel-brown eyes, soft powder blue cashmere crewneck sweater, calm confident professional expression`;

// THE "ANCHOR" - This keeps the inactive hand busy so it doesn't clasp
const LEFT_ANCHOR = `left hand firmly gripping the chair armrest at waist level`;
const RIGHT_ANCHOR = `right hand firmly gripping the chair armrest at waist level`;

// 20 PRECISION PROMPTS
const PROMPTS: Record<string, string> = {
  
  // === GROUP 1: SELF (Me/My Heart) ===
  // Active: Right | Anchor: Left
  self_right_thumb: `${KELLY_BASE}, ${LEFT_ANCHOR}, right hand raised to chest level with thumb pointing at herself, confident smile, "it's me" gesture, ${SCENE}`,
  self_right_palm: `${KELLY_BASE}, ${LEFT_ANCHOR}, right hand placed gently flat against her heart, sincere expression, genuine connection, ${SCENE}`,
  
  // Active: Left | Anchor: Right
  self_left_thumb: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left hand raised to chest level with thumb pointing at herself, confident smile, "it's me" gesture, ${SCENE}`,
  self_left_palm: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left hand placed gently flat against her heart, sincere expression, genuine connection, ${SCENE}`,

  // === GROUP 2: CAMERA (You/The Learner) ===
  // Active: Right | Anchor: Left
  cam_right_index: `${KELLY_BASE}, ${LEFT_ANCHOR}, right arm extended forward, index finger pointing directly at the camera lens, "I choose you" gesture, engaging eye contact, ${SCENE}`,
  cam_right_open: `${KELLY_BASE}, ${LEFT_ANCHOR}, right arm extended forward with open palm facing up, offering gesture to the viewer, welcoming expression, ${SCENE}`,
  
  // Active: Left | Anchor: Right
  cam_left_index: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left arm extended forward, index finger pointing directly at the camera lens, "I choose you" gesture, engaging eye contact, ${SCENE}`,
  cam_left_open: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left arm extended forward with open palm facing up, offering gesture to the viewer, welcoming expression, ${SCENE}`,

  // === GROUP 3: BOTTOM RAIL (The Script/Context) ===
  // Active: Right | Anchor: Left
  bot_right_index: `${KELLY_BASE}, ${LEFT_ANCHOR}, right hand lowered, index finger pointing straight down towards the bottom frame edge, looking down at the point, "look at this text" gesture, ${SCENE}`,
  bot_right_palm: `${KELLY_BASE}, ${LEFT_ANCHOR}, right hand lowered with open palm facing down, gesturing to the bottom of the screen, looking down, indicating content below, ${SCENE}`,
  
  // Active: Left | Anchor: Right
  bot_left_index: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left hand lowered, index finger pointing straight down towards the bottom frame edge, looking down at the point, "look at this text" gesture, ${SCENE}`,
  bot_left_palm: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left hand lowered with open palm facing down, gesturing to the bottom of the screen, looking down, indicating content below, ${SCENE}`,

  // === GROUP 4: SIDE RAILS (The Choices) - RIGHT SIDE ===
  // Active: Right | Anchor: Left
  rail_right_index: `${KELLY_BASE}, ${LEFT_ANCHOR}, right arm extended to the right side, index finger pointing rigidly to the right frame edge, eyes looking right, "choice A is here" gesture, ${SCENE}`,
  rail_right_thumb: `${KELLY_BASE}, ${LEFT_ANCHOR}, right elbow tucked at side, right thumb jerking to the right, eyes glancing right, casual "check this out" gesture, ${SCENE}`,
  rail_right_open: `${KELLY_BASE}, ${LEFT_ANCHOR}, right arm sweeping to the right with open palm, presenting the right side of the screen, Vanna White style, ${SCENE}`,
  rail_right_lean: `${KELLY_BASE}, ${LEFT_ANCHOR}, leaning slightly right, right index finger pointing specifically at the right middle rail, intense focus on the right side, ${SCENE}`,

  // === GROUP 5: SIDE RAILS (The Choices) - LEFT SIDE ===
  // Active: Left | Anchor: Right
  rail_left_index: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left arm extended to the left side, index finger pointing rigidly to the left frame edge, eyes looking left, "choice B is here" gesture, ${SCENE}`,
  rail_left_thumb: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left elbow tucked at side, left thumb jerking to the left, eyes glancing left, casual "check this out" gesture, ${SCENE}`,
  rail_left_open: `${KELLY_BASE}, ${RIGHT_ANCHOR}, left arm sweeping to the left with open palm, presenting the left side of the screen, Vanna White style, ${SCENE}`,
  rail_left_lean: `${KELLY_BASE}, ${RIGHT_ANCHOR}, leaning slightly left, left index finger pointing specifically at the left middle rail, intense focus on the left side, ${SCENE}`,
};

const OUTPUT_DIR = path.join(process.cwd(), "public", "kelly", "precision-batch");

async function generate(name: string, prompt: string): Promise<boolean> {
  console.log(`\n🎯 Generating: ${name}`);
  // console.log(`   📝 ${prompt}`);
  
  try {
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: prompt,
          hf_lora: LORA_URL,
          lora_scale: 0.95, // Increased scale slightly for strict adherence
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 30 // Slight bump for quality
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed`);
    
    const buffer = Buffer.from(await response.arrayBuffer());
    
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    fs.writeFileSync(path.join(OUTPUT_DIR, `${name}.png`), buffer);
    
    console.log(`   ✅ Saved`);
    return true;
    
  } catch (error: any) {
    console.log(`   ❌ ${error.message}`);
    return false;
  }
}

async function main() {
  console.log("🎯 KELLY PRECISION CONTROL BATCH (20 IMAGES)");
  console.log("=".repeat(50));
  console.log("Strategy: Dissociated Limb Control + Frame Locking");
  console.log("Output: " + OUTPUT_DIR);
  console.log("");
  
  let success = 0;
  const entries = Object.entries(PROMPTS);
  const total = entries.length;
  
  // Run sequentially to avoid rate limits and monitor progress
  for (const [name, prompt] of entries) {
    if (await generate(name, prompt)) success++;
    // Small delay to be safe
    await new Promise(r => setTimeout(r, 1000));
  }
  
  console.log("\n" + "=".repeat(50));
  console.log(`✅ Batch Complete: ${success}/${total}`);
  
  const { execSync } = require("child_process");
  try {
    execSync(`explorer "${OUTPUT_DIR}"`);
  } catch (e) {
    console.log("Could not open folder automatically.");
  }
}

main().catch(console.error);







