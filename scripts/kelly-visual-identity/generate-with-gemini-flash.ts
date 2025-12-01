/**
 * Kelly Visual Identity - Gemini 2.0 Flash Image Generation
 * 
 * Uses Gemini 2.0 Flash experimental which supports image generation
 * 
 * Usage: tsx scripts/kelly-visual-identity/generate-with-gemini-flash.ts
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import { GoogleGenerativeAI } from "@google/generative-ai";
import * as fs from "fs";
import * as path from "path";

const API_KEY = process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY;

const genAI = new GoogleGenerativeAI(API_KEY!);

const KELLY_BASE = `Create a ultra photorealistic professional photograph of a woman named Kelly. She is in her late 20s to early 30s with brown wavy shoulder-length hair with caramel and honey highlights, center-parted. She has hazel-brown almond-shaped expressive eyes, soft symmetrical features with natural subtle makeup, light-medium warm skin tone with healthy natural glow. She is wearing a soft blue cashmere crewneck sweater, medium-wash relaxed-fit blue jeans cuffed at ankle, white leather minimal sneakers. She is seated in a director's chair with black fabric seat and warm wood frame with round finials. The setting is a pure white cyclorama photography studio with professional studio lighting with natural window light from upper right casting soft diagonal shadows. She has a calm, cool, confident, warm, approachable expression. Shot on Canon EOS R5, 85mm lens, f/2.8, professional fashion photography, 8K resolution, highly detailed.`;

const POSE_PROMPTS: Record<string, string> = {
  idle: `${KELLY_BASE} She has a relaxed natural posture with a slight genuine warm smile, looking directly at camera with friendly eye contact, hands resting naturally on chair armrests.`,
  thinking: `${KELLY_BASE} Her chin is resting thoughtfully on her right hand, elbow on armrest, looking up and slightly to the side with a contemplative curious expression.`,
  pointing_left: `${KELLY_BASE} Her left arm is extended gracefully to the left side of frame, index finger pointing left, body turned slightly left, looking toward the left with a warm encouraging expression.`,
  pointing_right: `${KELLY_BASE} Her right arm is extended gracefully to the right side of frame, index finger pointing right, body turned slightly right, looking toward the right with a warm encouraging expression.`,
  pointing_up: `${KELLY_BASE} Her right arm is raised elegantly above her head, index finger pointing upward, head tilted back slightly, eyes looking up with an engaged interested expression.`,
  pointing_down: `${KELLY_BASE} Her right arm is lowered gracefully, index finger pointing downward toward the floor, slight forward lean, looking down with a helpful guiding expression.`,
  encouraging: `${KELLY_BASE} She is leaning slightly forward in chair with engaged body language, open welcoming warm expression, slight encouraging nod, genuine supportive smile.`,
  hint: `${KELLY_BASE} She has a playful knowing expression, right index finger touching her lips in a thoughtful secret gesture, eyes slightly narrowed with gentle playful mischief, slight knowing smirk.`,
  celebrating: `${KELLY_BASE} Both her arms are raised joyfully in a celebration victory gesture, big genuine joyful smile showing teeth, eyes bright and wide with excitement and happiness.`,
  supportive: `${KELLY_BASE} She has a warm empathetic caring expression, slight head tilt to the side showing understanding, gentle encouraging reassuring smile, one hand on heart.`,
  proud: `${KELLY_BASE} Her right hand is placed meaningfully on her heart, satisfied accomplished genuine smile, dignified upright proud posture, eyes soft and content.`,
  excited: `${KELLY_BASE} She has an energetic forward-leaning eager posture, bright wide enthusiastic eyes, big excited smile, hands clasped together in front of chest with anticipation.`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-gemini");

async function generateWithGeminiFlash(poseName: string, prompt: string): Promise<Buffer | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  
  try {
    // Use gemini-2.0-flash-exp which supports image generation
    const model = genAI.getGenerativeModel({ 
      model: "gemini-2.0-flash-exp",
      generationConfig: {
        temperature: 1,
        topP: 0.95,
        topK: 40,
        maxOutputTokens: 8192,
      }
    });

    const result = await model.generateContent({
      contents: [{
        role: "user",
        parts: [{
          text: `Generate a photorealistic image: ${prompt}`
        }]
      }],
      generationConfig: {
        responseMimeType: "image/png"
      } as any
    });

    const response = result.response;
    
    // Check for inline data (image)
    if (response.candidates && response.candidates[0]) {
      const candidate = response.candidates[0];
      if (candidate.content && candidate.content.parts) {
        for (const part of candidate.content.parts) {
          if ((part as any).inlineData) {
            const inlineData = (part as any).inlineData;
            if (inlineData.mimeType?.startsWith('image/')) {
              console.log(`✅ Got image data`);
              return Buffer.from(inlineData.data, 'base64');
            }
          }
        }
      }
    }
    
    // If no image, log the text response
    const text = response.text();
    console.log(`❌ No image generated. Response: ${text.substring(0, 200)}...`);
    return null;

  } catch (error: any) {
    console.error(`❌ Error: ${error.message}`);
    return null;
  }
}

async function generateAllPoses() {
  console.log("🚀 Kelly Visual Identity - Gemini 2.0 Flash Generation");
  console.log("=".repeat(60));
  
  if (!API_KEY) {
    console.error("❌ No API key found!");
    process.exit(1);
  }
  
  console.log(`🔑 API key: ${API_KEY.substring(0, 10)}...`);
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`📁 Output: ${OUTPUT_DIR}\n`);
  
  const results: Array<{ pose: string; success: boolean }> = [];
  
  for (const [poseName, prompt] of Object.entries(POSE_PROMPTS)) {
    const imageBuffer = await generateWithGeminiFlash(poseName, prompt);
    
    if (imageBuffer) {
      const filename = `kelly_${poseName}_gemini_v1.png`;
      const outputPath = path.join(OUTPUT_DIR, filename);
      fs.writeFileSync(outputPath, imageBuffer);
      console.log(`💾 Saved: ${filename}`);
      results.push({ pose: poseName, success: true });
    } else {
      results.push({ pose: poseName, success: false });
    }
    
    console.log("⏳ Waiting 5 seconds...");
    await new Promise(resolve => setTimeout(resolve, 5000));
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

