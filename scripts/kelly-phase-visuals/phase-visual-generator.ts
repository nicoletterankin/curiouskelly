/**
 * PHASE VISUAL GENERATOR - CAO Visual Learning System 2.0
 * 
 * Generates layered educational visuals PER PHASE:
 * - Layer 1: Background (Gemini Imagen - educational context)
 * - Layer 2: Kelly (LoRA via Replicate - consistent character)
 * - Layer 3: Diagrams/Props (Gemini - educational content)
 * 
 * Then composites them together for the final visual.
 */

import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });
dotenv.config();

import Replicate from "replicate";
import * as fs from "fs";
import * as path from "path";

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const CONFIG = {
  // Kelly LoRA settings
  KELLY_LORA: {
    model: "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
    loraUrl: "https://civitai.com/api/download/models/2455956",
    loraScale: 0.85,
    triggerWord: "kelly"
  },
  
  // Kelly's consistent appearance (matches LoRA training)
  KELLY_APPEARANCE: `kelly, woman late 20s, brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater`,
  
  // Output settings
  OUTPUT_DIR: path.join(process.cwd(), "public", "kelly", "phase-visuals"),
  
  // Gemini API
  GEMINI_API_KEY: process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY,
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN
};

// ═══════════════════════════════════════════════════════════════════
// LESSON PHASE DEFINITIONS
// ═══════════════════════════════════════════════════════════════════

interface PhaseVisual {
  phase: string;
  kellyPose: string;
  background: string;
  diagram?: string;
  emotionalTone: string;
}

// Day 1: "Starting Fresh" - Leaves/Photosynthesis
const DAY_1_PHASES: PhaseVisual[] = [
  {
    phase: "hook",
    kellyPose: "welcoming stance, arms slightly open, warm inviting smile, body angled toward viewer",
    background: "Sunlit forest clearing at golden hour, morning mist, rays of light filtering through leaves, lush green canopy, peaceful nature scene, no text, photorealistic, 8K",
    emotionalTone: "wonder, curiosity, invitation to learn"
  },
  {
    phase: "q1",
    kellyPose: "pointing upward with right hand, looking at where she points, explaining expression, engaged teaching posture",
    background: "Extreme macro close-up of a green leaf surface showing cellular structure, veins visible, water droplets, sunlight hitting leaf, scientific but beautiful, no text, photorealistic",
    diagram: "Simple photosynthesis diagram: sun arrow to leaf, CO2 + H2O arrow through leaf, O2 + glucose out, clean minimal style with labels",
    emotionalTone: "discovery, scientific wonder"
  },
  {
    phase: "q2",
    kellyPose: "thoughtful pose, chin resting on hand, contemplative curious expression, slight head tilt",
    background: "Cross-section view inside a plant cell, chloroplasts visible as green oval structures, cell wall, membrane, artistic scientific illustration style, soft lighting",
    diagram: "Chloroplast structure diagram with thylakoids, stroma labeled, clean educational style",
    emotionalTone: "deep thinking, molecular wonder"
  },
  {
    phase: "q3",
    kellyPose: "leaning forward slightly, encouraging supportive expression, hands visible in open gesture",
    background: "Underground root system of a tree, soil cross-section showing roots absorbing water, water droplets on root hairs, earth tones, educational illustration",
    diagram: "Water absorption diagram showing root to stem to leaf pathway with arrows",
    emotionalTone: "connection, understanding how it all works"
  },
  {
    phase: "wisdom",
    kellyPose: "proud confident stance, hand on heart, satisfied warm smile, accomplished posture",
    background: "Majestic oak tree at sunset, full canopy, golden light, single tree against beautiful sky, representing growth and completion, cinematic",
    emotionalTone: "accomplishment, synthesis of learning, growth mindset"
  }
];

// ═══════════════════════════════════════════════════════════════════
// KELLY POSE GENERATOR (using LoRA)
// ═══════════════════════════════════════════════════════════════════

async function generateKellyPose(
  replicate: Replicate,
  poseDescription: string,
  outputPath: string
): Promise<boolean> {
  
  const prompt = `${CONFIG.KELLY_APPEARANCE}, ${poseDescription}, professional studio photography, solid green screen background for compositing, full body visible, studio lighting, 8K quality, clean edges for easy background removal`;
  
  console.log(`  🎭 Generating Kelly pose...`);
  
  try {
    const output = await replicate.run(
      CONFIG.KELLY_LORA.model as `${string}/${string}:${string}`,
      {
        input: {
          prompt: prompt,
          hf_lora: CONFIG.KELLY_LORA.loraUrl,
          lora_scale: CONFIG.KELLY_LORA.loraScale,
          num_outputs: 1,
          aspect_ratio: "9:16", // Portrait for Kelly
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28
        }
      }
    ) as any;
    
    const imageUrl = Array.isArray(output) ? output[0] : output;
    
    // Handle stream response
    let imageData: Buffer;
    if (imageUrl.getReader) {
      const reader = imageUrl.getReader();
      const chunks: Uint8Array[] = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }
      imageData = Buffer.concat(chunks);
    } else {
      const response = await fetch(imageUrl);
      imageData = Buffer.from(await response.arrayBuffer());
    }
    
    fs.writeFileSync(outputPath, imageData);
    console.log(`  ✅ Kelly pose saved: ${path.basename(outputPath)}`);
    return true;
    
  } catch (error: any) {
    console.error(`  ❌ Kelly pose error: ${error.message}`);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════
// BACKGROUND GENERATOR (using Gemini or Flux for now)
// ═══════════════════════════════════════════════════════════════════

async function generateBackground(
  replicate: Replicate,
  backgroundDescription: string,
  outputPath: string
): Promise<boolean> {
  
  console.log(`  🌄 Generating background...`);
  
  try {
    // Using Flux Pro for backgrounds (better quality)
    const output = await replicate.run(
      "black-forest-labs/flux-1.1-pro",
      {
        input: {
          prompt: backgroundDescription,
          aspect_ratio: "16:9",
          output_format: "png",
          output_quality: 100,
          safety_tolerance: 2
        }
      }
    ) as any;
    
    const imageUrl = typeof output === 'string' ? output : output.toString();
    
    const response = await fetch(imageUrl);
    const imageData = Buffer.from(await response.arrayBuffer());
    
    fs.writeFileSync(outputPath, imageData);
    console.log(`  ✅ Background saved: ${path.basename(outputPath)}`);
    return true;
    
  } catch (error: any) {
    console.error(`  ❌ Background error: ${error.message}`);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════
// DIAGRAM GENERATOR (educational content)
// ═══════════════════════════════════════════════════════════════════

async function generateDiagram(
  replicate: Replicate,
  diagramDescription: string,
  outputPath: string
): Promise<boolean> {
  
  if (!diagramDescription) return true; // No diagram needed
  
  console.log(`  📊 Generating diagram...`);
  
  const prompt = `Educational diagram: ${diagramDescription}, clean white background, professional scientific illustration, clear labels, simple lines, educational textbook style, high contrast, easy to read`;
  
  try {
    const output = await replicate.run(
      "black-forest-labs/flux-1.1-pro",
      {
        input: {
          prompt: prompt,
          aspect_ratio: "1:1",
          output_format: "png",
          output_quality: 100
        }
      }
    ) as any;
    
    const imageUrl = typeof output === 'string' ? output : output.toString();
    const response = await fetch(imageUrl);
    const imageData = Buffer.from(await response.arrayBuffer());
    
    fs.writeFileSync(outputPath, imageData);
    console.log(`  ✅ Diagram saved: ${path.basename(outputPath)}`);
    return true;
    
  } catch (error: any) {
    console.error(`  ❌ Diagram error: ${error.message}`);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN GENERATOR
// ═══════════════════════════════════════════════════════════════════

async function generatePhaseVisuals(dayNumber: number, phases: PhaseVisual[]) {
  console.log(`\n${"═".repeat(60)}`);
  console.log(`🎓 PHASE VISUAL GENERATOR - Day ${dayNumber}`);
  console.log(`${"═".repeat(60)}\n`);
  
  if (!CONFIG.REPLICATE_API_TOKEN) {
    console.error("❌ REPLICATE_API_TOKEN not set!");
    return;
  }
  
  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  
  const dayDir = path.join(CONFIG.OUTPUT_DIR, dayNumber.toString().padStart(3, '0'));
  fs.mkdirSync(dayDir, { recursive: true });
  
  for (const phase of phases) {
    console.log(`\n📍 Phase: ${phase.phase.toUpperCase()}`);
    console.log(`   Emotional tone: ${phase.emotionalTone}`);
    console.log(`   ${"─".repeat(50)}`);
    
    const phaseDir = path.join(dayDir, phase.phase);
    fs.mkdirSync(phaseDir, { recursive: true });
    
    // Generate all layers
    await generateBackground(
      replicate,
      phase.background,
      path.join(phaseDir, "background.png")
    );
    
    await generateKellyPose(
      replicate,
      phase.kellyPose,
      path.join(phaseDir, "kelly.png")
    );
    
    if (phase.diagram) {
      await generateDiagram(
        replicate,
        phase.diagram,
        path.join(phaseDir, "diagram.png")
      );
    }
    
    // Add delay between phases
    console.log(`  ⏳ Waiting before next phase...`);
    await new Promise(r => setTimeout(r, 3000));
  }
  
  console.log(`\n${"═".repeat(60)}`);
  console.log(`✅ Day ${dayNumber} phase visuals complete!`);
  console.log(`📁 Output: ${dayDir}`);
  console.log(`${"═".repeat(60)}\n`);
}

// ═══════════════════════════════════════════════════════════════════
// ENTRY POINT
// ═══════════════════════════════════════════════════════════════════

async function main() {
  // Generate Day 1 as proof of concept
  await generatePhaseVisuals(1, DAY_1_PHASES);
}

main().catch(console.error);

