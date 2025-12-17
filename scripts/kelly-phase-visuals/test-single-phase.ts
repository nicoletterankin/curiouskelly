/**
 * TEST SINGLE PHASE - Proof of Concept
 * 
 * Generates just the HOOK phase for Day 1 to verify:
 * 1. Kelly LoRA is working and character is consistent
 * 2. Background generation is high quality
 * 3. The visual approach makes sense
 */

import * as dotenv from "dotenv";
dotenv.config({ path: "../../.env.local" });
dotenv.config({ path: "../../.env" });

import Replicate from "replicate";
import * as fs from "fs";
import * as path from "path";

const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN;

if (!REPLICATE_TOKEN) {
  console.error("❌ REPLICATE_API_TOKEN not found!");
  console.error("Set it in .env or .env.local");
  process.exit(1);
}

const replicate = new Replicate({ auth: REPLICATE_TOKEN });

// Kelly LoRA configuration
const KELLY_LORA = {
  model: "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d" as const,
  loraUrl: "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors",
  loraScale: 0.90
};

// Kelly's trained appearance
const KELLY = `kelly, woman late 20s, brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown eyes, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater`;

async function downloadImage(urlOrStream: any): Promise<Buffer> {
  if (urlOrStream.getReader) {
    // It's a ReadableStream
    const reader = urlOrStream.getReader();
    const chunks: Uint8Array[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    return Buffer.concat(chunks);
  } else {
    // It's a URL string
    const response = await fetch(urlOrStream);
    return Buffer.from(await response.arrayBuffer());
  }
}

async function testKellyLoRA() {
  console.log("\n" + "═".repeat(60));
  console.log("🧪 TEST 1: Kelly LoRA Character Consistency");
  console.log("═".repeat(60) + "\n");
  
  const prompt = `${KELLY}, welcoming stance, arms slightly open, warm inviting smile, standing in a professional photography studio, clean white backdrop, full body visible, professional studio lighting, 8K quality`;
  
  console.log("📝 Prompt:", prompt.substring(0, 100) + "...");
  console.log("\n⏳ Generating with Kelly LoRA...\n");
  
  try {
    const output = await replicate.run(
      KELLY_LORA.model,
      {
        input: {
          prompt: prompt,
          hf_lora: KELLY_LORA.loraUrl,
          lora_scale: KELLY_LORA.loraScale,
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "png",
          guidance_scale: 3.5,
          output_quality: 100,
          num_inference_steps: 28
        }
      }
    ) as any;
    
    const imageData = await downloadImage(Array.isArray(output) ? output[0] : output);
    
    const outputDir = path.join(process.cwd(), "test-output");
    fs.mkdirSync(outputDir, { recursive: true });
    
    const filepath = path.join(outputDir, "kelly-lora-test.png");
    fs.writeFileSync(filepath, imageData);
    
    console.log("✅ SUCCESS! Kelly generated with LoRA");
    console.log(`📁 Saved to: ${filepath}`);
    console.log(`📐 Size: ${(imageData.length / 1024).toFixed(1)} KB`);
    
    return true;
  } catch (error: any) {
    console.error("❌ FAILED:", error.message);
    return false;
  }
}

async function testEducationalBackground() {
  console.log("\n" + "═".repeat(60));
  console.log("🧪 TEST 2: Educational Background (Photosynthesis)");
  console.log("═".repeat(60) + "\n");
  
  const prompt = `Sunlit forest clearing at golden hour, morning mist rising, rays of light filtering through green leaves, lush canopy above, peaceful nature scene, no people, no text, cinematic photography, 8K, National Geographic style`;
  
  console.log("📝 Prompt:", prompt.substring(0, 100) + "...");
  console.log("\n⏳ Generating background...\n");
  
  try {
    const output = await replicate.run(
      "black-forest-labs/flux-1.1-pro",
      {
        input: {
          prompt: prompt,
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
    
    const outputDir = path.join(process.cwd(), "test-output");
    fs.mkdirSync(outputDir, { recursive: true });
    
    const filepath = path.join(outputDir, "background-test.png");
    fs.writeFileSync(filepath, imageData);
    
    console.log("✅ SUCCESS! Background generated");
    console.log(`📁 Saved to: ${filepath}`);
    console.log(`📐 Size: ${(imageData.length / 1024).toFixed(1)} KB`);
    
    return true;
  } catch (error: any) {
    console.error("❌ FAILED:", error.message);
    return false;
  }
}

async function main() {
  console.log("\n" + "█".repeat(60));
  console.log("  PHASE VISUAL SYSTEM - PROOF OF CONCEPT TEST");
  console.log("█".repeat(60));
  
  const kellyOk = await testKellyLoRA();
  
  console.log("\n⏳ Waiting 5 seconds between tests...\n");
  await new Promise(r => setTimeout(r, 5000));
  
  const bgOk = await testEducationalBackground();
  
  console.log("\n" + "═".repeat(60));
  console.log("📊 TEST RESULTS");
  console.log("═".repeat(60));
  console.log(`  Kelly LoRA:  ${kellyOk ? "✅ PASS" : "❌ FAIL"}`);
  console.log(`  Background:  ${bgOk ? "✅ PASS" : "❌ FAIL"}`);
  console.log("═".repeat(60));
  
  if (kellyOk && bgOk) {
    console.log("\n🎉 All tests passed! Ready to generate phase visuals.");
    console.log("📁 Check test-output/ folder for results.\n");
  } else {
    console.log("\n⚠️ Some tests failed. Review errors above.\n");
  }
}

main().catch(console.error);




