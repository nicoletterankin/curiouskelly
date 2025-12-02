/**
 * Kelly Visual Identity - PRESENTER MODE
 * 
 * Kelly as a calm, confident presenter - like Vanna White
 * Subtle expressions, consistent positioning, professional presenter poses
 * 
 * Usage: tsx scripts/kelly-visual-identity/generate-kelly-presenter.ts
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

const LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors";

// LOCKED SCENE - Never changes
// Camera: straight on, centered, full body visible
// Chair: always center-frame, same angle
// Lighting: consistent window light from upper right
const SCENE = `
pure white cyclorama photography studio, 
director's chair with black canvas seat and warm natural wood frame positioned center frame,
soft natural window light from upper right creating gentle diagonal shadows on light gray seamless floor,
camera positioned straight on at eye level,
full body shot,
professional studio photography,
clean minimal background,
8K UHD,
shot on Hasselblad 100c 85mm f/2.8
`.replace(/\n/g, ' ').trim();

// LOCKED APPEARANCE
const KELLY_APPEARANCE = `
kelly,
brown wavy shoulder-length hair with caramel highlights center-parted,
hazel-brown eyes,
soft natural features,
light natural makeup,
soft powder blue cashmere crewneck sweater,
medium wash blue jeans cuffed at ankle,
white leather sneakers
`.replace(/\n/g, ' ').trim();

// LOCKED PERSONALITY - calm, confident, approachable expert
// NOT: overly expressive, clownish, theatrical
// YES: subtle warmth, quiet confidence, professional presenter
const KELLY_VIBE = `
calm confident expression,
subtle warm smile,
relaxed professional demeanor,
approachable expert energy,
poised composed posture
`.replace(/\n/g, ' ').trim();

// PRESENTER POSES - Vanna White style
// Each pose is designed for the interactive lesson UI
const PRESENTER_POSES: Record<string, { prompt: string; description: string }> = {
  
  // === NEUTRAL/DEFAULT STATES ===
  
  idle: {
    description: "Default state - attentive, ready to guide",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, hands resting naturally on armrests, looking directly at camera with warm attentive expression, slight closed-lip smile, ${SCENE}`
  },
  
  listening: {
    description: "Waiting for user input - engaged, patient",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, hands folded gently in lap, head tilted very slightly, attentive listening expression, patient encouraging look, ${SCENE}`
  },
  
  // === PRESENTING OPTIONS (Vanna White style) ===
  
  present_left: {
    description: "Presenting Option A (left side) - elegant gesture toward left",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, right hand resting on armrest, left arm extended gracefully to the left with open palm facing up in presenting gesture, head turned slightly left, eyes looking toward the left side of frame, subtle inviting smile as if presenting a prize, ${SCENE}`
  },
  
  present_right: {
    description: "Presenting Option B (right side) - elegant gesture toward right", 
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, left hand resting on armrest, right arm extended gracefully to the right with open palm facing up in presenting gesture, head turned slightly right, eyes looking toward the right side of frame, subtle inviting smile as if presenting a prize, ${SCENE}`
  },
  
  present_both: {
    description: "Presenting both options - balanced welcoming gesture",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, both arms extended outward with open palms facing up in welcoming presenting gesture, looking at camera, warm inviting expression as if saying choose either one, ${SCENE}`
  },
  
  // === MOBILE/VERTICAL LAYOUT ===
  
  present_top: {
    description: "Presenting top option (mobile) - gesture upward",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, one hand resting on lap, other hand raised with open palm gesturing upward toward top of frame, eyes glancing upward, subtle smile presenting the option above, ${SCENE}`
  },
  
  present_bottom: {
    description: "Presenting bottom option (mobile) - gesture downward",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, one hand resting on lap, other hand lowered with open palm gesturing downward toward bottom of frame, eyes glancing downward, subtle smile presenting the option below, ${SCENE}`
  },
  
  // === FEEDBACK STATES ===
  
  thinking: {
    description: "Processing/considering - thoughtful pause",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, chin resting lightly on hand, thoughtful contemplative expression, eyes looking slightly upward as if considering, subtle knowing smile, ${SCENE}`
  },
  
  correct: {
    description: "Correct answer - warm approval, NOT over the top",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, hands together in gentle appreciative gesture, warm genuine smile, proud approving expression, slight nod, eyes bright with quiet satisfaction, ${SCENE}`
  },
  
  encourage: {
    description: "Incorrect but supportive - gentle reassurance",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, one hand on heart in empathetic gesture, warm understanding expression, gentle encouraging smile, supportive caring look, ${SCENE}`
  },
  
  hint: {
    description: "Offering a hint - playful but subtle",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, finger touching chin thoughtfully, slight knowing smile, eyes with gentle mischief, as if about to share a helpful secret, ${SCENE}`
  },
  
  complete: {
    description: "Lesson/phase complete - satisfied accomplishment",
    prompt: `${KELLY_APPEARANCE}, seated in director's chair facing camera, ${KELLY_VIBE}, hands resting contentedly, satisfied accomplished expression, warm proud smile, calm confident posture, ${SCENE}`
  }
};

const OUTPUT_DIR = path.join(process.cwd(), "generated-poses-presenter");
const PRODUCTION_DIR = path.join(process.cwd(), "public", "kelly", "poses");

async function generatePose(poseName: string, poseData: { prompt: string; description: string }): Promise<Buffer | null> {
  console.log(`\n🎨 Generating: ${poseName}`);
  console.log(`   📝 ${poseData.description}`);
  
  try {
    const output = await replicate.run(
      "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
      {
        input: {
          prompt: poseData.prompt,
          hf_lora: LORA_URL,
          lora_scale: 0.9,
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    console.log(`   📥 Downloading...`);
    
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.status}`);
    
    return Buffer.from(await response.arrayBuffer());
    
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return null;
  }
}

async function optimizeImage(inputPath: string, outputPath: string): Promise<void> {
  try {
    const magickPath = "C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe";
    const cmd = `"${magickPath}" "${inputPath}" -strip -quality 92 -resize "1920x1080>" -colorspace sRGB "${outputPath}"`;
    execSync(cmd, { stdio: 'pipe' });
  } catch {
    fs.copyFileSync(inputPath, outputPath);
  }
}

async function main() {
  console.log("🎬 KELLY PRESENTER MODE - Vanna White Style");
  console.log("=".repeat(60));
  console.log("✨ Calm, confident, professional presenter poses");
  console.log(`🔗 LoRA: ${LORA_URL}`);
  console.log("");
  
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not found!");
    process.exit(1);
  }
  
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(PRODUCTION_DIR, { recursive: true });
  
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log(`📁 Production: ${PRODUCTION_DIR}`);
  
  const results: Array<{ pose: string; success: boolean }> = [];
  const poseCount = Object.keys(PRESENTER_POSES).length;
  
  for (const [poseName, poseData] of Object.entries(PRESENTER_POSES)) {
    const buffer = await generatePose(poseName, poseData);
    
    if (buffer) {
      const rawPath = path.join(OUTPUT_DIR, `kelly_${poseName}_raw.png`);
      fs.writeFileSync(rawPath, buffer);
      
      const prodPath = path.join(PRODUCTION_DIR, `kelly_${poseName}.png`);
      await optimizeImage(rawPath, prodPath);
      
      console.log(`   ✅ Saved: kelly_${poseName}.png`);
      results.push({ pose: poseName, success: true });
    } else {
      results.push({ pose: poseName, success: false });
    }
    
    await new Promise(r => setTimeout(r, 3000));
  }
  
  console.log("\n" + "=".repeat(60));
  console.log("📊 GENERATION SUMMARY");
  console.log("=".repeat(60));
  
  const successful = results.filter(r => r.success).length;
  console.log(`✅ Successful: ${successful}/${poseCount}`);
  
  if (successful === poseCount) {
    console.log("\n🎉 ALL PRESENTER POSES GENERATED!");
    console.log(`📁 Production: ${PRODUCTION_DIR}`);
    console.log("\n📋 Generated poses:");
    Object.entries(PRESENTER_POSES).forEach(([name, data]) => {
      console.log(`   • ${name}: ${data.description}`);
    });
  }
}

main().catch(console.error);

