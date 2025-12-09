/**
 * Kelly Choice Pairs - "This One" or "That One"
 * 
 * Generating PAIRED poses that mirror each other
 * Thumbs indicating direction, elbows at sides for tight natural framing
 * White space preserved for choice overlay
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

// LOCKED SCENE - tight medium shot
const SCENE = `pure white cyclorama studio, director's chair with black canvas and wood frame centered, soft window light, camera straight on, medium close-up chest and face, clean white background with space for text overlay, 8K, Hasselblad 85mm`;

// LOCKED APPEARANCE  
const KELLY = `kelly, brown wavy shoulder-length hair with caramel highlights, hazel-brown eyes, soft powder blue cashmere crewneck sweater, calm confident subtle smile, relaxed natural expression`;

// KEY: Elbows at sides, thumbs indicating direction
const ANCHOR = `elbows relaxed at her sides, arms close to body`;

// PAIRED POSES - each set is designed to be used together
const CHOICE_PAIRS: Record<string, string> = {
  
  // === PAIR 1: Thumbs Up/Down ===
  
  "pair1_thumbs_up": `${KELLY}, seated in director's chair, ${ANCHOR}, right hand at chest level with thumb pointing upward in casual thumbs up gesture, eyes glancing upward toward top of frame, encouraging expression like saying check this one out, ${SCENE}`,
  
  "pair1_thumbs_down": `${KELLY}, seated in director's chair, ${ANCHOR}, right hand at chest level with thumb pointing downward in casual gesture, eyes glancing downward toward bottom of frame, encouraging expression like saying or maybe this one, ${SCENE}`,
  
  // === PAIR 2: Open Hand Gesturing ===
  
  "pair2_gesture_up": `${KELLY}, seated in director's chair, ${ANCHOR}, right hand raised slightly with open palm and fingers loosely gesturing upward, wrist relaxed, eyes looking up with inviting expression, natural presenter pose, ${SCENE}`,
  
  "pair2_gesture_down": `${KELLY}, seated in director's chair, ${ANCHOR}, right hand lowered slightly with open palm and fingers loosely gesturing downward, wrist relaxed, eyes looking down with inviting expression, natural presenter pose, ${SCENE}`,
  
  // === PAIR 3: Subtle Head Tilt + Eye Direction ===
  
  "pair3_look_up": `${KELLY}, seated in director's chair, ${ANCHOR}, hands resting naturally, head tilted slightly upward, eyes gazing up toward top of frame with curious interested expression, subtle knowing smile, ${SCENE}`,
  
  "pair3_look_down": `${KELLY}, seated in director's chair, ${ANCHOR}, hands resting naturally, head tilted slightly downward, eyes gazing down toward bottom of frame with curious interested expression, subtle knowing smile, ${SCENE}`,
  
  // === PAIR 4: Two-Thumb Choice ===
  
  "pair4_both_choice_up": `${KELLY}, seated in director's chair, both hands at chest with thumbs up pointing upward, elbows tucked at sides, looking up with bright encouraging this is a great choice expression, ${SCENE}`,
  
  "pair4_both_choice_down": `${KELLY}, seated in director's chair, both hands at chest with thumbs pointing downward, elbows tucked at sides, looking down with bright encouraging this is a great choice expression, ${SCENE}`,
  
  // === PAIR 5: Casual Point with Whole Hand ===
  
  "pair5_casual_up": `${KELLY}, seated in director's chair, ${ANCHOR}, right hand casually indicating upward with relaxed fingers together, elbow bent close to body, looking up like she is presenting option A above, ${SCENE}`,
  
  "pair5_casual_down": `${KELLY}, seated in director's chair, ${ANCHOR}, right hand casually indicating downward with relaxed fingers together, elbow bent close to body, looking down like she is presenting option B below, ${SCENE}`,
  
  // === PAIR 6: Chin Rest Direction ===
  
  "pair6_chin_up": `${KELLY}, seated in director's chair, ${ANCHOR}, chin resting on hand thoughtfully, eyes looking upward toward top of frame, considering expression like hmm that one looks interesting, ${SCENE}`,
  
  "pair6_chin_down": `${KELLY}, seated in director's chair, ${ANCHOR}, chin resting on hand thoughtfully, eyes looking downward toward bottom of frame, considering expression like hmm or maybe that one, ${SCENE}`,

};

const OUTPUT_DIR = path.join(process.cwd(), "choice-pairs-batch");

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
  console.log("🎭 KELLY CHOICE PAIRS - This One or That One");
  console.log("=".repeat(50));
  console.log("Generating mirrored pairs for choice UI");
  console.log("");
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  let success = 0;
  const total = Object.keys(CHOICE_PAIRS).length;
  
  for (const [name, prompt] of Object.entries(CHOICE_PAIRS)) {
    if (await generate(name, prompt)) success++;
    await new Promise(r => setTimeout(r, 2000));
  }
  
  console.log("\n" + "=".repeat(50));
  console.log(`✅ Generated: ${success}/${total}`);
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log("\n📋 Pairs generated:");
  console.log("   Pair 1: Thumbs up/down");
  console.log("   Pair 2: Open hand gesture up/down");
  console.log("   Pair 3: Head tilt + eye direction");
  console.log("   Pair 4: Two-thumb choice");
  console.log("   Pair 5: Casual hand indicate");
  console.log("   Pair 6: Chin rest looking up/down");
}

main().catch(console.error);






