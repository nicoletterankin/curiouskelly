/**
 * Kelly Visual Identity - FLUX 1.1 Pro (Photorealistic)
 * 
 * Uses FLUX 1.1 Pro on Replicate - the BEST quality model
 * Produces photorealistic images, not cartoons
 * 
 * Usage: tsx scripts/kelly-visual-identity/generate-with-flux-pro.ts
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

// Photorealistic prompt optimized for FLUX Pro
const KELLY_BASE = `RAW photo, professional studio photograph, a real woman named Kelly, age 28, natural brown wavy shoulder-length hair with subtle caramel highlights, center parted, hazel-brown eyes, soft natural features, light natural makeup, healthy glowing skin, wearing a soft powder blue cashmere crewneck sweater, medium wash blue jeans cuffed at ankle, clean white leather sneakers, sitting in a classic director's chair with black canvas seat and natural wood frame, pure white cyclorama photography studio, soft natural window light from upper right creating gentle shadows, calm confident friendly expression, shot on Hasselblad H6D-100c, 85mm f/1.8, shallow depth of field, photorealistic, hyperrealistic, 8k uhd, high detail skin texture, professional fashion photography`;

const POSE_PROMPTS: Record<string, string> = {
  idle: `${KELLY_BASE}, relaxed seated posture, genuine warm smile, direct eye contact with camera, hands resting naturally on armrests, welcoming friendly demeanor`,
  thinking: `${KELLY_BASE}, chin resting on right hand, elbow on armrest, gazing upward and to the side, thoughtful contemplative expression, curious look`,
  pointing_left: `${KELLY_BASE}, left arm extended to the left pointing with index finger, body angled slightly left, looking toward the left, encouraging helpful expression`,
  pointing_right: `${KELLY_BASE}, right arm extended to the right pointing with index finger, body angled slightly right, looking toward the right, encouraging helpful expression`,
  pointing_up: `${KELLY_BASE}, right arm raised above head pointing upward with index finger, looking up, engaged interested expression`,
  pointing_down: `${KELLY_BASE}, right arm lowered pointing downward with index finger, slight forward lean, looking down, helpful guiding expression`,
  encouraging: `${KELLY_BASE}, leaning slightly forward, warm open expression, supportive smile, engaged body language, hands visible in welcoming gesture`,
  hint: `${KELLY_BASE}, playful knowing expression, index finger touching lips in secret gesture, slight mischievous smirk, head tilted`,
  celebrating: `${KELLY_BASE}, both arms raised in victory celebration, big genuine smile, bright excited eyes, joyful triumphant pose`,
  supportive: `${KELLY_BASE}, warm empathetic expression, gentle head tilt, caring smile, one hand on heart, understanding look`,
  proud: `${KELLY_BASE}, hand placed on heart, satisfied accomplished smile, confident upright posture, proud content expression`,
  excited: `${KELLY_BASE}, forward leaning eager posture, bright enthusiastic eyes, excited smile, hands clasped together in anticipation`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-pro");

async function generateWithFluxPro(poseName: string, prompt: string): Promise<Buffer | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  console.log(`📝 Using FLUX 1.1 Pro (photorealistic)`);
  
  try {
    // Use FLUX 1.1 Pro - the highest quality model
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
    console.log(`📥 Downloading: ${imageUrl.substring(0, 60)}...`);
    
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
  console.log("🚀 Kelly Visual Identity - FLUX 1.1 Pro (Photorealistic)");
  console.log("=".repeat(60));
  console.log("⚡ This model produces PHOTOREALISTIC images, not cartoons!");
  console.log("");
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found!");
    process.exit(1);
  }
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`📁 Output: ${OUTPUT_DIR}\n`);
  
  const results: Array<{ pose: string; success: boolean }> = [];
  
  for (const [poseName, prompt] of Object.entries(POSE_PROMPTS)) {
    const buffer = await generateWithFluxPro(poseName, prompt);
    
    if (buffer) {
      const filename = `kelly_${poseName}_pro_v1.png`;
      const outputPath = path.join(OUTPUT_DIR, filename);
      fs.writeFileSync(outputPath, buffer);
      console.log(`✅ Saved: ${filename} (${(buffer.length / 1024 / 1024).toFixed(2)} MB)`);
      results.push({ pose: poseName, success: true });
    } else {
      results.push({ pose: poseName, success: false });
    }
    
    console.log("⏳ Waiting 3 seconds...");
    await new Promise(r => setTimeout(r, 3000));
  }
  
  console.log("\n" + "=".repeat(60));
  console.log("📊 GENERATION SUMMARY");
  console.log("=".repeat(60));
  
  const successful = results.filter(r => r.success).length;
  console.log(`✅ Successful: ${successful}/12`);
  console.log(`❌ Failed: ${12 - successful}/12`);
  
  if (successful > 0) {
    console.log(`\n📁 Output: ${OUTPUT_DIR}`);
    console.log("\n🎯 NEXT: Review images and run upload script");
  }
}

main().catch(console.error);



