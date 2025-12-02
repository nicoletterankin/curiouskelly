/**
 * Kelly Thumbnail Generator - January Pilot
 * 
 * Implements the "Curious Kelly Thumbnail Generation System"
 * Uses the "Walking Kelly" profile pose and category-specific environments.
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

const LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

// === STYLE BIBLE CONSTANTS ===

const KELLY_ANCHOR = `kelly, young woman walking in profile view, long brown wavy hair, light blue crewneck sweater, blue jeans cuffed at ankles, white sneakers, mid-stride walking pose, natural arm movement, looking ahead, soft natural lighting on figure, subtle ground shadow`;

const STYLE_LOCKS = `photorealistic editorial photography, clean composition, cinematic color grading, soft shadows, 8k, professional photography, 16:9 aspect ratio`;

const NEGATIVE_PROMPT = `cartoon, illustration, anime, painting, drawing, sketch, blurry, low quality, watermark, text, logo, extra limbs, extra fingers, deformed, distorted face, wrong outfit, different clothes, holding objects, sitting, standing still, bad anatomy, distorted body`;

// === LESSON PROMPTS (Just Lesson 001 for Test) ===

const LESSONS = [
  {
    id: "001",
    slug: "starting-fresh",
    prompt_middle: `walking toward golden sunrise horizon, vast open landscape, morning mist in distance, warm golden hour lighting, sense of new beginning and possibility, soft orange and gold gradient sky`
  }
];

// === GENERATION LOGIC ===

const OUTPUT_DIR = path.join(process.cwd(), "public", "kelly", "thumbnails", "raw");

async function generateThumbnail(lesson: { id: string; slug: string; prompt_middle: string }) {
  const filename = `lesson-${lesson.id}-${lesson.slug}.png`;
  const fullPrompt = `${KELLY_ANCHOR}, ${lesson.prompt_middle}, ${STYLE_LOCKS}`;
  
  console.log(`\n🎨 Generating: ${filename}`);
  // console.log(`   📝 Prompt: ${fullPrompt}`);
  
  try {
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: fullPrompt,
          hf_lora: LORA_URL,
          lora_scale: 0.95, // High adherence to Kelly's look
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 30,
          extra_lora_scale: 0.8 // Slight boost for style
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed`);
    
    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(path.join(OUTPUT_DIR, filename), buffer);
    
    console.log(`   ✅ Saved: ${filename}`);
    return true;
    
  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`);
    return false;
  }
}

async function main() {
  console.log("🖼️ KELLY THUMBNAIL PILOT - Lesson 001 Test");
  console.log("=".repeat(50));
  console.log("Testing 'Walking Kelly' pose and Style Bible adherence");
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log("");
  
  let success = 0;
  for (const lesson of LESSONS) {
    if (await generateThumbnail(lesson)) success++;
  }
  
  console.log("\n" + "=".repeat(50));
  if (success === LESSONS.length) {
    console.log("✅ Test Complete");
    try {
        const { execSync } = require("child_process");
        execSync(`explorer "${OUTPUT_DIR}"`);
    } catch (e) {}
  } else {
    console.log("❌ Test Failed");
  }
}

main().catch(console.error);



