/**
 * Kelly Visual Identity - Gemini Imagen Generation
 * 
 * Uses the correct Gemini API endpoint for image generation
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import * as fs from "fs";
import * as path from "path";

const API_KEY = process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY;

const KELLY_BASE = `Ultra photorealistic professional photograph of a woman named Kelly, late 20s, brown wavy shoulder-length hair with caramel highlights, hazel-brown eyes, soft features, natural makeup, wearing soft blue cashmere sweater, blue jeans, white sneakers, seated in director's chair with black fabric and wood frame, white cyclorama studio, professional lighting from upper right, 8K, Canon EOS R5, 85mm f/2.8`;

const POSE_PROMPTS: Record<string, string> = {
  idle: `${KELLY_BASE}, relaxed posture, warm smile, looking at camera, hands on armrests`,
  thinking: `${KELLY_BASE}, chin on hand, looking up, contemplative expression`,
  pointing_left: `${KELLY_BASE}, left arm extended pointing left, looking left, encouraging`,
  pointing_right: `${KELLY_BASE}, right arm extended pointing right, looking right, encouraging`,
  pointing_up: `${KELLY_BASE}, arm raised pointing up, looking up, engaged`,
  pointing_down: `${KELLY_BASE}, arm lowered pointing down, looking down, helpful`,
  encouraging: `${KELLY_BASE}, leaning forward, warm supportive smile, open body language`,
  hint: `${KELLY_BASE}, playful expression, finger on lips, knowing smirk`,
  celebrating: `${KELLY_BASE}, arms raised in celebration, big joyful smile, excited`,
  supportive: `${KELLY_BASE}, empathetic expression, head tilt, hand on heart, caring`,
  proud: `${KELLY_BASE}, hand on heart, satisfied smile, dignified posture`,
  excited: `${KELLY_BASE}, forward lean, bright eyes, excited smile, hands clasped`
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-gemini");

async function generateImage(poseName: string, prompt: string): Promise<Buffer | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  
  // Try using the imagen model with generateImages endpoint
  const url = `https://generativelanguage.googleapis.com/v1beta/models/imagegeneration:generateImages?key=${API_KEY}`;
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: { text: prompt },
        numberOfImages: 1,
        aspectRatio: "16:9"
      })
    });

    if (!response.ok) {
      const text = await response.text();
      
      // Try alternative endpoint
      console.log(`Trying alternative endpoint...`);
      return await tryAlternativeEndpoint(poseName, prompt);
    }

    const data = await response.json() as any;
    if (data.images && data.images[0]) {
      return Buffer.from(data.images[0].bytesBase64Encoded, 'base64');
    }
    
    return null;
  } catch (error: any) {
    console.error(`❌ Error: ${error.message}`);
    return await tryAlternativeEndpoint(poseName, prompt);
  }
}

async function tryAlternativeEndpoint(poseName: string, prompt: string): Promise<Buffer | null> {
  // Try Vertex AI style endpoint
  const url = `https://us-central1-aiplatform.googleapis.com/v1/projects/207034676667/locations/us-central1/publishers/google/models/imagegeneration@006:predict`;
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        instances: [{ prompt: prompt }],
        parameters: {
          sampleCount: 1,
          aspectRatio: "16:9"
        }
      })
    });

    if (!response.ok) {
      console.error(`❌ Alternative endpoint failed: ${response.status}`);
      return null;
    }

    const data = await response.json() as any;
    if (data.predictions && data.predictions[0]) {
      return Buffer.from(data.predictions[0].bytesBase64Encoded, 'base64');
    }
    return null;
  } catch (error: any) {
    console.error(`❌ Alternative error: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log("🚀 Kelly - Gemini Imagen Generation");
  console.log("=".repeat(60));
  console.log(`🔑 API key: ${API_KEY?.substring(0, 10)}...`);
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  let success = 0;
  for (const [poseName, prompt] of Object.entries(POSE_PROMPTS)) {
    const buffer = await generateImage(poseName, prompt);
    if (buffer) {
      const filename = `kelly_${poseName}_v1.png`;
      fs.writeFileSync(path.join(OUTPUT_DIR, filename), buffer);
      console.log(`✅ Saved: ${filename}`);
      success++;
    }
    await new Promise(r => setTimeout(r, 2000));
  }
  
  console.log(`\n✅ Generated: ${success}/12`);
}

main().catch(console.error);





