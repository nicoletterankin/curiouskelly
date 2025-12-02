/**
 * Kelly Directional Pointing - Index Finger Guide
 * 
 * Based on the hint pose (finger touching chin)
 * Hand stays in same position, finger extends to point
 * Eyes follow finger direction
 * 5 directions: left, right, up, down, center
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

// LOCKED SCENE - Same as hint
const SCENE = `pure white cyclorama photography studio, director's chair with black canvas seat and warm natural wood frame positioned center frame, soft natural window light from upper right creating gentle diagonal shadows on light gray seamless floor, camera positioned straight on at eye level, full body shot, professional studio photography, clean minimal background, 8K UHD, shot on Hasselblad 100c 85mm f/2.8`;

// LOCKED APPEARANCE - Same as hint
const KELLY_APPEARANCE = `kelly, brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup, soft powder blue cashmere crewneck sweater, medium wash blue jeans cuffed at ankle, white leather sneakers`;

// LOCKED PERSONALITY - Same as hint
const KELLY_VIBE = `calm confident expression, subtle warm smile, relaxed professional demeanor, approachable expert energy, poised composed posture`;

// BASE POSE - Finger touching chin (like hint)
const BASE_POSE = `finger touching chin thoughtfully`;

// 5 DIRECTIONAL VARIATIONS
const DIRECTIONAL_POINTS: Record<string, string> = {
  
  // Pointing LEFT - eyes follow left
  point_left: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, ${BASE_POSE}, index finger extended pointing to the left side of frame, eyes looking left following finger direction, slight knowing smile, as if directing attention to option on the left, ${SCENE}`,
  
  // Pointing RIGHT - eyes follow right
  point_right: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, ${BASE_POSE}, index finger extended pointing to the right side of frame, eyes looking right following finger direction, slight knowing smile, as if directing attention to option on the right, ${SCENE}`,
  
  // Pointing UP - eyes follow up
  point_up: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, ${BASE_POSE}, index finger extended pointing upward toward top of frame, eyes looking up following finger direction, slight knowing smile, as if directing attention to option above, ${SCENE}`,
  
  // Pointing DOWN - eyes follow down
  point_down: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, ${BASE_POSE}, index finger extended pointing downward toward bottom of frame, eyes looking down following finger direction, slight knowing smile, as if directing attention to option below, ${SCENE}`,
  
  // Pointing CENTER/STRAIGHT - eyes at camera (default)
  point_center: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, ${BASE_POSE}, index finger extended pointing forward toward camera, eyes looking directly at camera, slight knowing smile, as if about to share a helpful secret or guide, ${SCENE}`,
};

const OUTPUT_DIR = path.join(process.cwd(), "public", "kelly", "directions");

async function generate(name: string, prompt: string): Promise<boolean> {
  console.log(`\n🎯 Generating: ${name}`);
  console.log(`   👉 Direction: ${name.replace('point_', '')}`);
  
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
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    fs.writeFileSync(path.join(OUTPUT_DIR, `kelly_${name}.png`), buffer);
    
    console.log(`   ✅ Saved: kelly_${name}.png`);
    return true;
    
  } catch (error: any) {
    console.log(`   ❌ ${error.message}`);
    return false;
  }
}

async function main() {
  console.log("👆 KELLY DIRECTIONAL POINTING");
  console.log("=".repeat(50));
  console.log("Base pose: Finger touching chin (like hint)");
  console.log("Variations: Finger extends to point, eyes follow");
  console.log("");
  
  let success = 0;
  const total = Object.keys(DIRECTIONAL_POINTS).length;
  
  for (const [name, prompt] of Object.entries(DIRECTIONAL_POINTS)) {
    if (await generate(name, prompt)) success++;
    await new Promise(r => setTimeout(r, 2000));
  }
  
  console.log("\n" + "=".repeat(50));
  console.log(`✅ Generated: ${success}/${total}`);
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log("\n📋 Generated directions:");
  console.log("   • point_left - Finger left, eyes left");
  console.log("   • point_right - Finger right, eyes right");
  console.log("   • point_up - Finger up, eyes up");
  console.log("   • point_down - Finger down, eyes down");
  console.log("   • point_center - Finger forward, eyes at camera");
}

main().catch(console.error);

