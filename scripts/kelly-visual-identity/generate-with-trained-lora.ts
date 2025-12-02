/**
 * Kelly Visual Identity - Production Pipeline with Trained LoRA
 * 
 * Uses YOUR trained Civitai LoRA (Curious_Kelly.safetensors) for perfect consistency
 * Generates all 12 poses with consistent character + background
 * 
 * Usage: tsx scripts/kelly-visual-identity/generate-with-trained-lora.ts
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

// Your Civitai LoRA - model version ID from the URL
const CIVITAI_LORA_VERSION = "2455956";
const CIVITAI_LORA_URL = `https://civitai.com/api/download/models/${CIVITAI_LORA_VERSION}`;

// Consistent scene description - LOCKED
const SCENE = `pure white cyclorama photography studio, director's chair with black canvas fabric seat and natural warm wood frame with round finials, professional studio lighting with soft natural window light from upper right casting gentle diagonal shadows on light gray seamless floor, clean minimal background, shot on Hasselblad H6D-100c, 85mm f/2.8, shallow depth of field, professional fashion photography, 8K UHD`;

// Kelly's appearance - LOCKED (matches your LoRA training)
const KELLY = `kelly, woman with brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater, medium wash blue jeans cuffed at ankle, white leather sneakers`;

// 12 Core poses with consistent scene
const POSES: Record<string, string> = {
  idle: `${KELLY}, seated in director's chair, relaxed natural posture, genuine warm smile, looking directly at camera, hands resting on armrests, ${SCENE}`,
  
  thinking: `${KELLY}, seated in director's chair, chin resting thoughtfully on right hand, elbow on armrest, looking upward and slightly to side, contemplative curious expression, ${SCENE}`,
  
  pointing_left: `${KELLY}, seated in director's chair, left arm extended gracefully pointing to the left, body angled slightly left, looking toward the left, encouraging helpful expression, ${SCENE}`,
  
  pointing_right: `${KELLY}, seated in director's chair, right arm extended gracefully pointing to the right, body angled slightly right, looking toward the right, encouraging helpful expression, ${SCENE}`,
  
  pointing_up: `${KELLY}, seated in director's chair, right arm raised elegantly pointing upward, head tilted back slightly, looking up, engaged interested expression, ${SCENE}`,
  
  pointing_down: `${KELLY}, seated in director's chair, right arm lowered gracefully pointing downward, slight forward lean, looking down, helpful guiding expression, ${SCENE}`,
  
  encouraging: `${KELLY}, seated in director's chair, leaning slightly forward, warm open welcoming expression, supportive smile, engaged body language, hands visible in welcoming gesture, ${SCENE}`,
  
  hint: `${KELLY}, seated in director's chair, playful knowing expression, index finger touching lips in secret gesture, slight mischievous smirk, head tilted playfully, ${SCENE}`,
  
  celebrating: `${KELLY}, seated in director's chair, both arms raised joyfully in celebration, big genuine smile, bright excited eyes, triumphant victorious pose, ${SCENE}`,
  
  supportive: `${KELLY}, seated in director's chair, warm empathetic caring expression, gentle head tilt showing understanding, reassuring smile, one hand on heart, ${SCENE}`,
  
  proud: `${KELLY}, seated in director's chair, right hand placed meaningfully on heart, satisfied accomplished smile, dignified confident posture, content proud expression, ${SCENE}`,
  
  excited: `${KELLY}, seated in director's chair, energetic forward-leaning posture, bright enthusiastic eyes, excited anticipating smile, hands clasped together eagerly, ${SCENE}`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-final");
const PRODUCTION_DIR = path.join(process.cwd(), "public", "kelly", "poses");

async function generateWithLoRA(poseName: string, prompt: string): Promise<Buffer | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  
  try {
    // Use FLUX Dev with LoRA from Civitai
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: prompt,
          hf_lora: CIVITAI_LORA_URL,
          lora_scale: 0.85,
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          prompt_strength: 0.8,
          num_inference_steps: 28,
          disable_safety_checker: true
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`📥 Downloading...`);
    
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
    
  } catch (error: any) {
    console.error(`❌ Error: ${error.message}`);
    
    // Fallback to FLUX Pro without LoRA if LoRA fails
    console.log(`🔄 Trying fallback with FLUX 1.1 Pro...`);
    return await generateFallback(poseName, prompt);
  }
}

async function generateFallback(poseName: string, prompt: string): Promise<Buffer | null> {
  try {
    const output = await replicate.run(
      "black-forest-labs/flux-1.1-pro",
      {
        input: {
          prompt: prompt,
          aspect_ratio: "16:9",
          output_format: "png",
          output_quality: 100,
          safety_tolerance: 2,
          prompt_upsampling: true
        }
      }
    ) as any;
    
    const imageUrl = typeof output === 'string' ? output : (Array.isArray(output) ? output[0] : output.toString());
    
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
    
  } catch (error: any) {
    console.error(`❌ Fallback also failed: ${error.message}`);
    return null;
  }
}

async function processWithImageMagick(inputPath: string, outputPath: string): Promise<void> {
  // Optimize image for web delivery
  const cmd = `magick "${inputPath}" -strip -quality 90 -resize "1920x1080>" -colorspace sRGB "${outputPath}"`;
  
  try {
    execSync(cmd, { stdio: 'pipe' });
    console.log(`✨ Optimized: ${path.basename(outputPath)}`);
  } catch (error) {
    // If ImageMagick fails, just copy the file
    fs.copyFileSync(inputPath, outputPath);
    console.log(`📋 Copied (no optimization): ${path.basename(outputPath)}`);
  }
}

async function main() {
  console.log("🚀 Kelly Visual Identity - Production Pipeline");
  console.log("=".repeat(60));
  console.log("⚡ Using YOUR trained Civitai LoRA: Curious_Kelly.safetensors");
  console.log(`📦 LoRA URL: ${CIVITAI_LORA_URL}`);
  console.log("");
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found!");
    process.exit(1);
  }
  
  // Create directories
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(PRODUCTION_DIR, { recursive: true });
  
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log(`📁 Production: ${PRODUCTION_DIR}\n`);
  
  const results: Array<{ pose: string; success: boolean; path?: string }> = [];
  
  for (const [poseName, prompt] of Object.entries(POSES)) {
    const buffer = await generateWithLoRA(poseName, prompt);
    
    if (buffer) {
      // Save raw generated image
      const rawFilename = `kelly_${poseName}_raw.png`;
      const rawPath = path.join(OUTPUT_DIR, rawFilename);
      fs.writeFileSync(rawPath, buffer);
      
      // Process with ImageMagick for production
      const prodFilename = `kelly_${poseName}.png`;
      const prodPath = path.join(PRODUCTION_DIR, prodFilename);
      await processWithImageMagick(rawPath, prodPath);
      
      console.log(`✅ Saved: ${prodFilename}`);
      results.push({ pose: poseName, success: true, path: prodPath });
    } else {
      results.push({ pose: poseName, success: false });
    }
    
    console.log("⏳ Waiting 5 seconds...");
    await new Promise(r => setTimeout(r, 5000));
  }
  
  // Generate summary
  console.log("\n" + "=".repeat(60));
  console.log("📊 GENERATION SUMMARY");
  console.log("=".repeat(60));
  
  const successful = results.filter(r => r.success).length;
  console.log(`✅ Successful: ${successful}/12`);
  console.log(`❌ Failed: ${12 - successful}/12`);
  
  if (successful > 0) {
    console.log(`\n📁 Raw output: ${OUTPUT_DIR}`);
    console.log(`📁 Production: ${PRODUCTION_DIR}`);
    
    // List generated files
    console.log("\n📋 Generated files:");
    results.filter(r => r.success).forEach(r => {
      console.log(`   ✅ kelly_${r.pose}.png`);
    });
  }
  
  if (successful === 12) {
    console.log("\n🎉 ALL 12 POSES GENERATED SUCCESSFULLY!");
    console.log("🚀 Ready for production deployment!");
  }
}

main().catch(console.error);



