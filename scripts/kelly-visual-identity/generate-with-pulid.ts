/**
 * Kelly Visual Identity - PuLID FLUX (Face Reference)
 * 
 * Uses PuLID with FLUX for face-consistent generation
 * This maintains Kelly's face from reference image
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
  idle: `professional studio photograph, woman sitting in director's chair with black fabric and wood frame, relaxed natural posture, warm genuine smile, looking directly at camera, hands resting on armrests, wearing soft blue cashmere crewneck sweater, medium wash blue jeans, white leather sneakers, pure white cyclorama studio background, soft natural window lighting from upper right, photorealistic, 8k uhd`,
  thinking: `professional studio photograph, woman sitting in director's chair, chin resting thoughtfully on right hand, elbow on armrest, looking upward contemplatively, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  pointing_left: `professional studio photograph, woman sitting in director's chair, left arm extended pointing to the left, looking left with encouraging expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  pointing_right: `professional studio photograph, woman sitting in director's chair, right arm extended pointing to the right, looking right with encouraging expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  pointing_up: `professional studio photograph, woman sitting in director's chair, right arm raised pointing upward, looking up with engaged expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  pointing_down: `professional studio photograph, woman sitting in director's chair, right arm lowered pointing downward, looking down helpfully, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  encouraging: `professional studio photograph, woman sitting in director's chair, leaning slightly forward, warm supportive smile, open welcoming body language, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  hint: `professional studio photograph, woman sitting in director's chair, playful knowing expression, finger touching lips in secret gesture, slight smirk, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  celebrating: `professional studio photograph, woman sitting in director's chair, both arms raised in celebration, big joyful smile, excited happy expression, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  supportive: `professional studio photograph, woman sitting in director's chair, warm empathetic expression, gentle head tilt, hand on heart, caring look, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  proud: `professional studio photograph, woman sitting in director's chair, hand placed on heart, satisfied accomplished smile, dignified posture, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`,
  excited: `professional studio photograph, woman sitting in director's chair, forward leaning eager posture, bright enthusiastic eyes, excited smile, hands clasped together, soft blue cashmere sweater, blue jeans, white sneakers, white studio background, soft lighting, photorealistic, 8k`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-pulid");

async function generateWithPuLID(poseName: string, prompt: string, faceImagePath: string): Promise<Buffer | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  
  try {
    // Read and convert image to data URI
    const imageBuffer = fs.readFileSync(faceImagePath);
    const base64 = imageBuffer.toString('base64');
    const dataUri = `data:image/jpeg;base64,${base64}`;
    
    const output = await replicate.run(
      "zsxkib/pulid:43d309c37ab4e62361e5e29b8e9e867fb2dcbcec77ae91206a8d95ac5dd451a0",
      {
        input: {
          prompt: prompt,
          main_face_image: dataUri,
          negative_prompt: "cartoon, anime, illustration, painting, drawing, art, sketch, low quality, blurry, deformed, ugly, bad anatomy, bad hands, extra fingers",
          num_steps: 20,
          start_step: 4,
          guidance_scale: 4,
          num_outputs: 1,
          id_weight: 1.0,
          output_format: "png",
          output_quality: 100
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
  console.log("🚀 Kelly Visual Identity - PuLID FLUX (Face Reference)");
  console.log("=".repeat(60));
  console.log("⚡ Using Kelly's face from reference image!");
  console.log("");
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found!");
    process.exit(1);
  }
  
  // Check reference image exists
  if (!fs.existsSync(REFERENCE_IMAGE)) {
    console.error(`❌ Reference image not found: ${REFERENCE_IMAGE}`);
    process.exit(1);
  }
  
  console.log(`📷 Reference: ${REFERENCE_IMAGE}`);
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`📁 Output: ${OUTPUT_DIR}\n`);
  
  const results: Array<{ pose: string; success: boolean }> = [];
  
  for (const [poseName, prompt] of Object.entries(POSE_PROMPTS)) {
    const buffer = await generateWithPuLID(poseName, prompt, REFERENCE_IMAGE);
    
    if (buffer) {
      const filename = `kelly_${poseName}_pulid_v1.png`;
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





