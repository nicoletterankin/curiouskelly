/**
 * Kelly Visual Identity - Gemini Imagen 3 Generation
 * 
 * Uses Google's Imagen 3 via Gemini API for photorealistic images
 * 
 * Usage: tsx scripts/kelly-visual-identity/generate-with-gemini.ts
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import * as fs from "fs";
import * as path from "path";

// Get API key from environment
const API_KEY = process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY;

const KELLY_BASE = `Ultra photorealistic professional photograph of a woman named Kelly, late 20s to early 30s, brown wavy shoulder-length hair with caramel and honey highlights center-parted, hazel-brown almond-shaped expressive eyes, soft symmetrical features with natural subtle makeup, light-medium warm skin tone with healthy natural glow, wearing a soft blue cashmere crewneck sweater, medium-wash relaxed-fit blue jeans cuffed at ankle, white leather minimal sneakers, seated in a director's chair with black fabric seat and warm wood frame with round finials, pure white cyclorama photography studio background, professional studio lighting with natural window light from upper right casting soft diagonal shadows on light gray seamless floor, calm cool confident warm approachable expression, shot on Canon EOS R5, 85mm lens, f/2.8, professional fashion photography, 8K resolution, highly detailed skin texture`;

const POSE_PROMPTS: Record<string, string> = {
  idle: `${KELLY_BASE}, relaxed natural posture, slight genuine warm smile, looking directly at camera with friendly eye contact, hands resting naturally on chair armrests, default welcoming state`,
  thinking: `${KELLY_BASE}, chin resting thoughtfully on right hand, elbow on armrest, looking up and slightly to the side, contemplative curious expression, eyebrows slightly raised as if considering an interesting question`,
  pointing_left: `${KELLY_BASE}, left arm extended gracefully to the left side of frame, index finger pointing left, body turned slightly left, head turned left, looking toward the left with warm encouraging expression`,
  pointing_right: `${KELLY_BASE}, right arm extended gracefully to the right side of frame, index finger pointing right, body turned slightly right, head turned right, looking toward the right with warm encouraging expression`,
  pointing_up: `${KELLY_BASE}, right arm raised elegantly above head, index finger pointing upward toward ceiling, head tilted back slightly, eyes looking up with engaged interested expression`,
  pointing_down: `${KELLY_BASE}, right arm lowered gracefully, index finger pointing downward toward the floor, slight forward lean, looking down at indicated spot with helpful guiding expression`,
  encouraging: `${KELLY_BASE}, leaning slightly forward in chair with engaged body language, open welcoming warm expression, slight encouraging nod, genuine supportive smile, hands visible with open inviting palms`,
  hint: `${KELLY_BASE}, playful knowing expression, right index finger touching lips in thoughtful secret gesture, eyes slightly narrowed with gentle playful mischief, slight knowing smirk, head tilted`,
  celebrating: `${KELLY_BASE}, both arms raised joyfully in celebration victory gesture, big genuine joyful smile showing teeth, eyes bright and wide with excitement and happiness, energetic triumphant pose`,
  supportive: `${KELLY_BASE}, warm empathetic caring expression, slight head tilt to the side showing understanding, gentle encouraging reassuring smile, open supportive body language, one hand on heart`,
  proud: `${KELLY_BASE}, right hand placed meaningfully on heart, satisfied accomplished genuine smile, dignified upright proud posture, warm proud expression, eyes soft and content with achievement`,
  excited: `${KELLY_BASE}, energetic forward-leaning eager posture, bright wide enthusiastic eyes, big excited smile, hands clasped together in front of chest with anticipation, ready and eager expression`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-gemini");

async function generateWithGemini(poseName: string, prompt: string): Promise<string | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  
  // Use Imagen 3 endpoint
  const url = `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${API_KEY}`;
  
  const requestBody = {
    instances: [
      {
        prompt: prompt
      }
    ],
    parameters: {
      sampleCount: 1,
      aspectRatio: "16:9",
      personGeneration: "allow_adult",
      safetySetting: "block_only_high"
    }
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ API Error: ${response.status} - ${errorText}`);
      return null;
    }

    const data = await response.json() as any;
    
    if (data.predictions && data.predictions[0] && data.predictions[0].bytesBase64Encoded) {
      return data.predictions[0].bytesBase64Encoded;
    }
    
    console.error(`❌ No image in response:`, JSON.stringify(data, null, 2));
    return null;

  } catch (error: any) {
    console.error(`❌ Error: ${error.message}`);
    return null;
  }
}

async function generateAllPoses() {
  console.log("🚀 Kelly Visual Identity - Gemini Imagen 3 Generation");
  console.log("=".repeat(60));
  
  if (!API_KEY) {
    console.error("❌ No API key found!");
    console.error("Set GOOGLE_AI_API_KEY or GEMINI_API_KEY in .env.local");
    process.exit(1);
  }
  
  console.log(`🔑 Using API key: ${API_KEY.substring(0, 10)}...`);
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`📁 Output: ${OUTPUT_DIR}\n`);
  
  const results: Array<{ pose: string; success: boolean }> = [];
  
  for (const [poseName, prompt] of Object.entries(POSE_PROMPTS)) {
    const base64Image = await generateWithGemini(poseName, prompt);
    
    if (base64Image) {
      const buffer = Buffer.from(base64Image, 'base64');
      const filename = `kelly_${poseName}_gemini_v1.png`;
      const outputPath = path.join(OUTPUT_DIR, filename);
      fs.writeFileSync(outputPath, buffer);
      console.log(`✅ Saved: ${filename}`);
      results.push({ pose: poseName, success: true });
    } else {
      results.push({ pose: poseName, success: false });
    }
    
    // Rate limiting
    console.log("⏳ Waiting 3 seconds...");
    await new Promise(resolve => setTimeout(resolve, 3000));
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

if (require.main === module) {
  generateAllPoses().catch(console.error);
}






