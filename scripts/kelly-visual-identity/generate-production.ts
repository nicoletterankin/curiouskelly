/**
 * Kelly Visual Identity - PRODUCTION PIPELINE
 * 
 * Uses YOUR trained LoRA from Hugging Face for perfect character consistency
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import Replicate from "replicate";
import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

// YOUR trained LoRA on Hugging Face
const LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

// Consistent scene - LOCKED
const SCENE = `pure white cyclorama photography studio, classic director's chair with black canvas seat and natural warm wood frame with round finials, professional studio lighting with soft natural window light from upper right casting gentle diagonal shadows on light gray seamless floor, clean minimal background, professional fashion photography, 8K UHD, shot on Hasselblad`;

// Kelly trigger word from your LoRA training
const KELLY = `kelly`;

// 12 Core poses
const POSES: Record<string, string> = {
  idle: `${KELLY}, seated in director's chair, relaxed natural posture, genuine warm smile, looking directly at camera, hands resting on armrests, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  thinking: `${KELLY}, seated in director's chair, chin resting thoughtfully on right hand, looking upward, contemplative curious expression, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  pointing_left: `${KELLY}, seated in director's chair, left arm extended pointing to the left, looking left, encouraging expression, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  pointing_right: `${KELLY}, seated in director's chair, right arm extended pointing to the right, looking right, encouraging expression, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  pointing_up: `${KELLY}, seated in director's chair, right arm raised pointing upward, looking up, engaged expression, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  pointing_down: `${KELLY}, seated in director's chair, right arm lowered pointing downward, looking down, helpful expression, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  encouraging: `${KELLY}, seated in director's chair, leaning forward, warm supportive smile, welcoming body language, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  hint: `${KELLY}, seated in director's chair, playful knowing expression, finger touching lips, mischievous smirk, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  celebrating: `${KELLY}, seated in director's chair, both arms raised in celebration, big joyful smile, excited expression, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  supportive: `${KELLY}, seated in director's chair, warm empathetic expression, gentle head tilt, hand on heart, caring look, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  proud: `${KELLY}, seated in director's chair, hand on heart, satisfied accomplished smile, proud posture, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`,
  
  excited: `${KELLY}, seated in director's chair, forward lean, bright enthusiastic eyes, excited smile, hands clasped, soft blue cashmere sweater, blue jeans, white sneakers, ${SCENE}`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-production");
const PRODUCTION_DIR = path.join(process.cwd(), "public", "kelly", "poses");

async function generateWithLoRA(poseName: string, prompt: string): Promise<Buffer | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  
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
    console.log(`📥 Downloading...`);
    
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    return Buffer.from(await response.arrayBuffer());
    
  } catch (error: any) {
    console.error(`❌ Error: ${error.message}`);
    return null;
  }
}

async function optimizeImage(inputPath: string, outputPath: string): Promise<void> {
  try {
    // Refresh PATH to include ImageMagick
    const magickPath = "C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe";
    const cmd = `"${magickPath}" "${inputPath}" -strip -quality 92 -resize "1920x1080>" -colorspace sRGB "${outputPath}"`;
    execSync(cmd, { stdio: 'pipe' });
    console.log(`✨ Optimized: ${path.basename(outputPath)}`);
  } catch {
    fs.copyFileSync(inputPath, outputPath);
    console.log(`📋 Copied: ${path.basename(outputPath)}`);
  }
}

async function main() {
  console.log("🚀 KELLY PRODUCTION PIPELINE");
  console.log("=".repeat(60));
  console.log(`🔗 LoRA: ${LORA_URL}`);
  console.log("");
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found!");
    process.exit(1);
  }
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(PRODUCTION_DIR, { recursive: true });
  
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log(`📁 Production: ${PRODUCTION_DIR}\n`);
  
  const results: Array<{ pose: string; success: boolean }> = [];
  
  for (const [poseName, prompt] of Object.entries(POSES)) {
    const buffer = await generateWithLoRA(poseName, prompt);
    
    if (buffer) {
      const rawPath = path.join(OUTPUT_DIR, `kelly_${poseName}_raw.png`);
      fs.writeFileSync(rawPath, buffer);
      
      const prodPath = path.join(PRODUCTION_DIR, `kelly_${poseName}.png`);
      await optimizeImage(rawPath, prodPath);
      
      console.log(`✅ ${poseName}`);
      results.push({ pose: poseName, success: true });
    } else {
      results.push({ pose: poseName, success: false });
    }
    
    await new Promise(r => setTimeout(r, 3000));
  }
  
  console.log("\n" + "=".repeat(60));
  const successful = results.filter(r => r.success).length;
  console.log(`✅ Successful: ${successful}/12`);
  
  if (successful === 12) {
    console.log("\n🎉 ALL 12 KELLY POSES GENERATED!");
    console.log(`📁 Production ready: ${PRODUCTION_DIR}`);
  }
}

main().catch(console.error);







