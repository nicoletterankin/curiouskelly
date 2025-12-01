/**
 * Kelly Visual Identity - FLUX Schnell Generation (Free, Fast)
 * 
 * Uses Replicate's free FLUX Schnell model
 * No LoRA needed, works immediately
 * 
 * Usage: tsx scripts/kelly-visual-identity/generate-with-flux-schnell.ts
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config(); // Also check .env

import Replicate from "replicate";
import * as fs from "fs";
import * as path from "path";
import * as https from "https";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const KELLY_BASE = `photorealistic woman named Kelly, late 20s to early 30s, brown wavy shoulder-length hair with caramel and honey highlights center-parted, hazel-brown almond-shaped eyes, soft symmetrical features with natural makeup, light-medium warm skin tone with healthy glow, wearing soft blue cashmere crewneck sweater, medium-wash relaxed-fit jeans cuffed at ankle, white leather sneakers minimal and clean, seated in director's chair with black fabric seat and back with warm wood frame and round finials, white cyclorama studio background, natural window light from upper right casting soft diagonal shadows on light gray white seamless floor, calm cool confident expression, professional photography, 8k, high quality`;

const POSE_PROMPTS: Record<string, string> = {
  idle: `${KELLY_BASE}, relaxed posture, slight warm smile, looking directly at camera, hands resting naturally on armrests`,
  thinking: `${KELLY_BASE}, chin resting on right hand, elbow on armrest, looking up and slightly to the side, contemplative thoughtful expression`,
  pointing_left: `${KELLY_BASE}, left arm extended to the left side of frame, index finger pointing left, body turned slightly left, looking toward the left with encouraging expression`,
  pointing_right: `${KELLY_BASE}, right arm extended to the right side of frame, index finger pointing right, body turned slightly right, looking toward the right with encouraging expression`,
  pointing_up: `${KELLY_BASE}, right arm raised above head, index finger pointing upward, head tilted back slightly, eyes looking up, engaged expression`,
  pointing_down: `${KELLY_BASE}, right arm lowered, index finger pointing downward toward the floor, slight forward lean, looking down at indicated spot, helpful expression`,
  encouraging: `${KELLY_BASE}, leaning slightly forward in chair, open welcoming expression, slight nod, warm supportive smile, engaged body language`,
  hint: `${KELLY_BASE}, playful knowing expression, right index finger touching lips in thoughtful secret gesture, eyes slightly narrowed with gentle mischief, slight knowing smirk`,
  celebrating: `${KELLY_BASE}, both arms raised in celebration gesture, big joyful genuine smile, eyes bright and wide with excitement, victorious energetic pose`,
  supportive: `${KELLY_BASE}, warm empathetic expression, slight head tilt to the side, gentle encouraging smile, open body language, hand on heart or reaching forward`,
  proud: `${KELLY_BASE}, right hand placed on heart, satisfied accomplished smile, dignified upright posture, warm proud expression, eyes soft and content`,
  excited: `${KELLY_BASE}, energetic forward-leaning posture, bright wide eyes, enthusiastic smile, hands clasped together in front of chest, ready and eager expression`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses");

async function downloadImage(url: string): Promise<Buffer> {
  // Use fetch instead of https.get for better compatibility
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download: ${response.status}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

async function generatePose(poseName: string): Promise<void> {
  const prompt = POSE_PROMPTS[poseName];
  
  console.log(`\n🎨 Generating: ${poseName}`);
  console.log(`📝 Prompt: ${prompt.substring(0, 100)}...`);
  
  try {
    const output = await replicate.run(
      "black-forest-labs/flux-schnell",
      {
        input: {
          prompt: prompt,
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          output_quality: 100,
          disable_safety_checker: false
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`📥 Downloading from: ${imageUrl}`);
    
    const buffer = await downloadImage(imageUrl);
    
    const filename = `kelly_${poseName}_flux_v1.png`;
    const outputPath = path.join(OUTPUT_DIR, filename);
    fs.writeFileSync(outputPath, buffer);
    
    console.log(`✅ Saved: ${filename}`);
    
  } catch (error: any) {
    console.error(`❌ Failed: ${poseName} - ${error.message}`);
    throw error;
  }
}

async function generateAllPoses() {
  console.log("🚀 Kelly Visual Identity - FLUX Schnell Generation");
  console.log("=".repeat(60));
  console.log("");
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found in environment!");
    console.error("\nGet your token from: https://replicate.com/account/api-tokens");
    console.error("Then add to .env.local: REPLICATE_API_TOKEN=your-token-here");
    process.exit(1);
  }
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`📁 Output directory: ${OUTPUT_DIR}\n`);
  
  const results: Array<{ pose: string; success: boolean }> = [];
  
  for (const poseName of Object.keys(POSE_PROMPTS)) {
    try {
      await generatePose(poseName);
      results.push({ pose: poseName, success: true });
      
      console.log("⏳ Waiting 2 seconds (rate limiting)...");
      await new Promise(resolve => setTimeout(resolve, 2000));
      
    } catch (error) {
      results.push({ pose: poseName, success: false });
    }
  }
  
  console.log("\n" + "=".repeat(60));
  console.log("📊 GENERATION SUMMARY");
  console.log("=".repeat(60));
  
  const successful = results.filter(r => r.success).length;
  console.log(`✅ Successful: ${successful}/12`);
  console.log(`❌ Failed: ${12 - successful}/12`);
  
  if (successful > 0) {
    console.log(`\n📁 Output: ${OUTPUT_DIR}`);
    console.log("\n🎯 NEXT STEPS:");
    console.log("1. Review the generated images");
    console.log("2. Run: npx tsx scripts/kelly-visual-identity/upload-to-r2.ts");
  }
}

if (require.main === module) {
  generateAllPoses().catch(console.error);
}

export { generateAllPoses, generatePose };

