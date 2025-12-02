/**
 * Kelly Visual Identity - InstantID (Face Reference)
 * 
 * Uses InstantID to maintain face consistency from a reference image
 * This is the ONLY way to get consistent character without LoRA
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

// Reference image - Kelly's face
const REFERENCE_IMAGE = path.join(process.cwd(), "daily-lesson-marketing/public/lessons/images/4.jpeg");

const POSE_PROMPTS: Record<string, string> = {
  idle: `professional studio photo, woman sitting in director's chair, relaxed posture, warm smile, looking at camera, hands on armrests, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  thinking: `professional studio photo, woman sitting in director's chair, chin resting on hand, looking up thoughtfully, contemplative expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  pointing_left: `professional studio photo, woman sitting in director's chair, left arm extended pointing left, looking left, encouraging expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  pointing_right: `professional studio photo, woman sitting in director's chair, right arm extended pointing right, looking right, encouraging expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  pointing_up: `professional studio photo, woman sitting in director's chair, arm raised pointing up, looking up, engaged expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  pointing_down: `professional studio photo, woman sitting in director's chair, arm pointing down, looking down, helpful expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  encouraging: `professional studio photo, woman sitting in director's chair, leaning forward, warm supportive smile, open body language, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  hint: `professional studio photo, woman sitting in director's chair, playful expression, finger on lips, knowing smirk, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  celebrating: `professional studio photo, woman sitting in director's chair, arms raised celebrating, big joyful smile, excited, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  supportive: `professional studio photo, woman sitting in director's chair, empathetic expression, head tilt, hand on heart, caring look, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  proud: `professional studio photo, woman sitting in director's chair, hand on heart, satisfied smile, proud posture, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`,
  excited: `professional studio photo, woman sitting in director's chair, forward lean, bright eyes, excited smile, hands clasped, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, 8k`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-instantid");

async function generateWithInstantID(poseName: string, prompt: string, faceImageBase64: string): Promise<Buffer | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  
  try {
    const output = await replicate.run(
      "zsxkib/instant-id:6af8583c541261472e92155d87bba80d5ad98461665802c6d74a5eb5d8487aba",
      {
        input: {
          image: `data:image/jpeg;base64,${faceImageBase64}`,
          prompt: prompt,
          negative_prompt: "cartoon, anime, illustration, painting, drawing, art, sketch, low quality, blurry, deformed",
          num_steps: 30,
          guidance_scale: 5,
          ip_adapter_scale: 0.8,
          controlnet_conditioning_scale: 0.8,
          num_outputs: 1,
          output_format: "png"
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
    return null;
  }
}

async function main() {
  console.log("🚀 Kelly Visual Identity - InstantID (Face Reference)");
  console.log("=".repeat(60));
  console.log("⚡ Using reference image for FACE CONSISTENCY!");
  console.log("");
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found!");
    process.exit(1);
  }
  
  // Load reference image
  if (!fs.existsSync(REFERENCE_IMAGE)) {
    console.error(`❌ Reference image not found: ${REFERENCE_IMAGE}`);
    process.exit(1);
  }
  
  console.log(`📷 Reference: ${REFERENCE_IMAGE}`);
  const faceImageBase64 = fs.readFileSync(REFERENCE_IMAGE).toString('base64');
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`📁 Output: ${OUTPUT_DIR}\n`);
  
  const results: Array<{ pose: string; success: boolean }> = [];
  
  for (const [poseName, prompt] of Object.entries(POSE_PROMPTS)) {
    const buffer = await generateWithInstantID(poseName, prompt, faceImageBase64);
    
    if (buffer) {
      const filename = `kelly_${poseName}_instantid_v1.png`;
      const outputPath = path.join(OUTPUT_DIR, filename);
      fs.writeFileSync(outputPath, buffer);
      console.log(`✅ Saved: ${filename}`);
      results.push({ pose: poseName, success: true });
    } else {
      results.push({ pose: poseName, success: false });
    }
    
    console.log("⏳ Waiting 5 seconds...");
    await new Promise(r => setTimeout(r, 5000));
  }
  
  console.log("\n" + "=".repeat(60));
  console.log("📊 GENERATION SUMMARY");
  console.log("=".repeat(60));
  
  const successful = results.filter(r => r.success).length;
  console.log(`✅ Successful: ${successful}/12`);
  console.log(`❌ Failed: ${12 - successful}/12`);
  
  if (successful > 0) {
    console.log(`\n📁 Output: ${OUTPUT_DIR}`);
  }
}

main().catch(console.error);



