/**
 * THE IMAGE FACTORY - Kelly Production Line
 * 
 * Input: visual-manifest-december.json
 * Output: Generated Images in public/kelly/lessons/[day]/
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

// Load Manifest
const MANIFEST_PATH = path.join(process.cwd(), "scripts", "kelly-visual-identity", "visual-manifest-december.json");
const MANIFEST = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf-8"));

async function generateAsset(lessonDay: number, asset: any) {
  const outputDir = path.join(process.cwd(), "public", "kelly", "lessons", String(lessonDay));
  const outputPath = path.join(outputDir, asset.filename);
  
  console.log(`   🎨 Generating: ${asset.filename}`);
  // console.log(`      Prompt: ${asset.prompt}`);

  try {
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: asset.prompt,
          hf_lora: LORA_URL,
          lora_scale: 0.95,
          num_outputs: 1,
          // Aspect ratio logic
          aspect_ratio: asset.type.includes("guide") ? "9:16" : (asset.type.includes("reaction") ? "1:1" : "16:9"),
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 30
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed`);
    
    const buffer = Buffer.from(await response.arrayBuffer());
    
    fs.mkdirSync(outputDir, { recursive: true });
    fs.writeFileSync(outputPath, buffer);
    
    console.log(`      ✅ Saved`);
    return true;
    
  } catch (error: any) {
    console.log(`      ❌ Error: ${error.message}`);
    return false;
  }
}

async function main() {
  console.log("🏭 KELLY IMAGE FACTORY - December Pilot");
  console.log("=".repeat(60));
  
  for (const lesson of MANIFEST) {
    console.log(`\n📚 Lesson ${lesson.lesson_id}: ${lesson.title}`);
    
    for (const asset of lesson.assets) {
      await generateAsset(lesson.lesson_id, asset);
      // Rate limit safety
      await new Promise(r => setTimeout(r, 1000));
    }
  }
  
  console.log("\n" + "=".repeat(60));
  console.log("✅ Factory Run Complete");
}

main().catch(console.error);



