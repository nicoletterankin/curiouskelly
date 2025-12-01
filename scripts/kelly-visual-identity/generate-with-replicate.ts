/**
 * Kelly Visual Identity - Replicate FLUX Generation
 * 
 * Uses Replicate API with your trained LoRA for higher quality
 * 
 * Usage: tsx scripts/kelly-visual-identity/generate-with-replicate.ts
 */

import Replicate from "replicate";
import * as fs from "fs";
import * as path from "path";
import * as https from "https";

// Initialize Replicate
const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

// Kelly's base appearance (same as before)
const KELLY_BASE = `photorealistic woman named Kelly, late 20s to early 30s, brown wavy shoulder-length hair with caramel and honey highlights center-parted, hazel-brown almond-shaped eyes, soft symmetrical features with natural makeup, light-medium warm skin tone with healthy glow, wearing soft blue cashmere crewneck sweater (hex #A8C4D9), medium-wash relaxed-fit jeans cuffed at ankle, white leather sneakers minimal and clean, seated in director's chair with black fabric seat and back with warm wood frame and round finials, white cyclorama studio background, natural window light from upper right casting soft diagonal shadows on light gray white seamless floor, calm cool confident expression, Mac Genius energy, warm but professional`;

// Pose prompts (same 12 poses)
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
  supportive: `${KELLY_BASE}, warm empathetic expression, slight head tilt to the side, gentle encouraging smile, open body language, hand on heart or reaching forward, NOT sad`,
  proud: `${KELLY_BASE}, right hand placed on heart, satisfied accomplished smile, dignified upright posture, warm proud expression, eyes soft and content`,
  excited: `${KELLY_BASE}, energetic forward-leaning posture, bright wide eyes, enthusiastic smile, hands clasped together in front of chest, ready and eager expression`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-replicate");

/**
 * Generate a single pose using Replicate + FLUX + LoRA
 */
async function generateWithReplicate(
  poseName: string,
  loraUrl: string
): Promise<{ url: string; buffer: Buffer }> {
  const prompt = POSE_PROMPTS[poseName];
  if (!prompt) throw new Error(`Unknown pose: ${poseName}`);
  
  console.log(`\n🎨 Generating: ${poseName}`);
  
  try {
    const output = await replicate.run(
      "black-forest-labs/flux-dev-lora",
      {
        input: {
          prompt: `kelly ${prompt}`,
          hf_lora: loraUrl, // Your trained LoRA from Civitai
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          output_quality: 100,
          guidance_scale: 3.5,
          num_inference_steps: 50, // Higher = better quality
          lora_scale: 1.0, // How much to use your LoRA
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    // Download the image
    const buffer = await downloadImage(imageUrl);
    
    console.log(`✅ Generated: ${poseName}`);
    return { url: imageUrl, buffer };
    
  } catch (error: any) {
    console.error(`❌ Failed: ${poseName} - ${error.message}`);
    throw error;
  }
}

/**
 * Download image from URL
 */
async function downloadImage(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    https.get(url, (response) => {
      const chunks: Buffer[] = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('end', () => resolve(Buffer.concat(chunks)));
      response.on('error', reject);
    });
  });
}

/**
 * Generate all 12 poses
 */
async function generateAllPoses() {
  console.log("🚀 Kelly Visual Identity - Replicate FLUX Generation");
  console.log("=" .repeat(60));
  
  // Get LoRA URL from command line or environment
  const loraUrl = process.argv[2] || process.env.KELLY_LORA_URL;
  
  if (!loraUrl) {
    console.error("❌ Error: LoRA URL required!");
    console.error("\nUsage:");
    console.error("  tsx generate-with-replicate.ts <civitai-lora-url>");
    console.error("\nOr set environment variable:");
    console.error("  KELLY_LORA_URL=https://civitai.com/api/download/models/YOUR_MODEL_ID");
    process.exit(1);
  }
  
  console.log(`📦 Using LoRA: ${loraUrl}`);
  console.log("");
  
  // Create output directory
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  const results: Array<{ pose: string; success: boolean; path?: string }> = [];
  
  for (const poseName of Object.keys(POSE_PROMPTS)) {
    try {
      const { buffer } = await generateWithReplicate(poseName, loraUrl);
      
      const filename = `kelly_${poseName}_replicate_v1.png`;
      const outputPath = path.join(OUTPUT_DIR, filename);
      fs.writeFileSync(outputPath, buffer);
      
      console.log(`💾 Saved: ${filename}`);
      results.push({ pose: poseName, success: true, path: outputPath });
      
      // Rate limiting
      console.log("⏳ Waiting 3 seconds...");
      await new Promise(resolve => setTimeout(resolve, 3000));
      
    } catch (error: any) {
      console.error(`❌ Failed: ${poseName}`);
      results.push({ pose: poseName, success: false });
    }
  }
  
  // Summary
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

// Run
if (require.main === module) {
  generateAllPoses().catch(console.error);
}

export { generateWithReplicate, generateAllPoses };

