/**
 * Kelly Final Choice Pair - Left/Right Thumbs
 * 
 * RIGHT hand thumbs up pointing RIGHT (Option 1 on right)
 * LEFT hand thumbs up pointing LEFT (Option 2 on left)
 * 
 * Elbows stay at sides for tight framing
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

// LOCKED SCENE - tight medium shot, centered
const SCENE = `pure white cyclorama studio, director's chair with black canvas and warm wood frame centered, soft window light from upper right, camera straight on at eye level, medium close-up chest and face, clean white background with ample space on sides for text overlay, 8K UHD, Hasselblad 100c 85mm f/2.8`;

// LOCKED APPEARANCE  
const KELLY = `kelly, brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup, soft powder blue cashmere crewneck sweater, calm confident expression, subtle warm smile, relaxed natural expression`;

// KEY CONSTRAINT: Elbows at sides for tight natural framing
const ANCHOR = `elbows relaxed at her sides, arms close to body, tight natural framing`;

// FINAL PAIR - Perfect mirrors
const FINAL_PAIR: Record<string, string> = {
  
  // RIGHT hand thumbs up pointing RIGHT (Option 1 goes on right side)
  choice_right: `${KELLY}, seated in director's chair, ${ANCHOR}, right hand at chest level with thumb pointing to the right side of frame, eyes glancing right toward where thumb points, encouraging expression like presenting option 1 on the right, subtle inviting smile, ${SCENE}`,
  
  // LEFT hand thumbs up pointing LEFT (Option 2 goes on left side)  
  choice_left: `${KELLY}, seated in director's chair, ${ANCHOR}, left hand at chest level with thumb pointing to the left side of frame, eyes glancing left toward where thumb points, encouraging expression like presenting option 2 on the left, subtle inviting smile, ${SCENE}`,
};

const OUTPUT_DIR = path.join(process.cwd(), "public", "kelly", "choices");
const PRODUCTION_DIR = path.join(process.cwd(), "public", "kelly", "poses");

async function generate(name: string, prompt: string): Promise<boolean> {
  console.log(`\n🎯 Generating: ${name}`);
  
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
    
    // Save to both locations
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    fs.mkdirSync(PRODUCTION_DIR, { recursive: true });
    
    fs.writeFileSync(path.join(OUTPUT_DIR, `${name}.png`), buffer);
    fs.writeFileSync(path.join(PRODUCTION_DIR, `kelly_${name}.png`), buffer);
    
    console.log(`   ✅ Saved to both locations`);
    return true;
    
  } catch (error: any) {
    console.log(`   ❌ ${error.message}`);
    return false;
  }
}

async function main() {
  console.log("🎭 KELLY FINAL CHOICE PAIR");
  console.log("=".repeat(50));
  console.log("Right hand → Right (Option 1)");
  console.log("Left hand → Left (Option 2)");
  console.log("");
  
  let success = 0;
  const total = Object.keys(FINAL_PAIR).length;
  
  for (const [name, prompt] of Object.entries(FINAL_PAIR)) {
    if (await generate(name, prompt)) success++;
    await new Promise(r => setTimeout(r, 2000));
  }
  
  console.log("\n" + "=".repeat(50));
  console.log(`✅ Generated: ${success}/${total}`);
  console.log(`📁 Choices: ${OUTPUT_DIR}`);
  console.log(`📁 Production: ${PRODUCTION_DIR}`);
}

main().catch(console.error);




