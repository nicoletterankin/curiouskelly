#!/usr/bin/env npx tsx
/**
 * 🎨 DAY 351 BACKUP GENERATOR
 * 
 * Generates ALTERNATE versions of all Day 351 visuals for A/B testing
 * and backup purposes. Saves to /backups/ subdirectories.
 * 
 * Usage:
 *   npx tsx scripts/generate-day-351-backups.ts
 *   npx tsx scripts/generate-day-351-backups.ts --kelly-only
 *   npx tsx scripts/generate-day-351-backups.ts --infographics-only
 */

import * as dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });
dotenv.config();

import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const CONFIG = {
  DAY: 351,
  FLUX_PRO: "black-forest-labs/flux-1.1-pro",
  KELLY_LORA: {
    model: "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
    loraUrl: "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors",
    loraScale: 0.90,
  },
  KELLY_BASE: `kelly, woman late 20s, brown wavy shoulder-length hair with caramel highlights center-parted, hazel-brown almond-shaped expressive eyes, soft natural features, light natural makeup, wearing soft powder blue cashmere crewneck sweater, blue jeans`,
  OUTPUT_BACKUPS: path.join(process.cwd(), 'public', 'kelly', 'backups', '351'),
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN,
  DELAY_MS: 4000
};

// BACKUP KELLY PROMPTS - Different variations
const KELLY_BACKUP_PROMPTS = [
  {
    id: "hook-alt",
    filename: "hook-alt.png",
    prompt: `${CONFIG.KELLY_BASE}, sitting peacefully in director's chair, eyes closed with serene smile, hands resting on lap, meditation visualization pose, soft purple cosmic background with subtle stars, peaceful aura, professional photography, 8K`
  },
  {
    id: "cliff-alt", 
    filename: "cliff-alt.png",
    prompt: `${CONFIG.KELLY_BASE}, leaning forward with intense curiosity, both hands gesturing as if explaining a revelation, eyes sparkling with discovery, abstract neural network visualization in background, teaching moment energy, professional photography, 8K`
  },
  {
    id: "wisdom-alt",
    filename: "wisdom-alt.png",
    prompt: `${CONFIG.KELLY_BASE}, warm genuine smile, one hand over heart in giving gesture, other hand extended palm-up as if offering wisdom, golden sunset light, peaceful wise mentor energy, cosmic background with gentle stars, professional photography, 8K`
  },
  {
    id: "outro-alt",
    filename: "outro-alt.png",
    prompt: `${CONFIG.KELLY_BASE}, friendly wave goodbye, warm encouraging smile, sunset behind creating silhouette glow, hand on heart, supportive coach energy sending learner off, twilight gradient background, professional photography, 8K`
  }
];

// BACKUP INFOGRAPHIC PROMPTS - Different visual approaches
const INFOGRAPHIC_BACKUP_PROMPTS = [
  {
    id: "brain-scan-alt",
    filename: "brain-scan-alt.png",
    prompt: `Educational infographic showing visualization science: Top-down view of brain MRI scan with two hemispheres, left side labeled "Physical Action" glowing bright blue, right side labeled "Mental Visualization" glowing identical bright blue, center showing 90% overlap statistic. Clean medical visualization style, dark background, scientific accuracy, 8K resolution`,
    aspectRatio: "16:9"
  },
  {
    id: "piano-study-alt",
    filename: "piano-study-alt.png", 
    prompt: `Educational infographic showing piano experiment: Three brain icons in a row, first brain connected to piano keys with "Physical Practice" label showing strong neural growth, second brain with thought bubble of piano "Mental Practice Only" showing nearly equal neural growth, third brain "Control" showing no change. Bar chart below showing improvement percentages. Clean scientific study aesthetic, 8K`,
    aspectRatio: "16:9"
  },
  {
    id: "olympic-alt",
    filename: "olympic-alt.png",
    prompt: `Inspirational infographic about mental training: Central image of Olympic rings with a meditating athlete silhouette inside. Large "50%" statistic prominently displayed. Text area for "Mental Rehearsal Time". Icons around edge showing surgeon, pianist, athlete all in visualization poses. Gold and blue color scheme, premium sports documentary aesthetic, 8K`,
    aspectRatio: "16:9"
  },
  {
    id: "cosmic-mind-alt",
    filename: "cosmic-mind-alt.png",
    prompt: `Cosmic consciousness visualization: Human head profile facing right, eyes closed in meditation, entire brain area filled with swirling purple and gold galaxy, stars and nebulae forming neural pathway patterns. Golden light rays emanating outward. Deep space background with floating stars. Represents infinite potential of visualization. Awe-inspiring digital art, 8K`,
    aspectRatio: "16:9"
  }
];

async function generateWithLoRA(replicate: Replicate, prompt: string, outputPath: string): Promise<boolean> {
  console.log(`  🎭 Generating: ${path.basename(outputPath)}...`);
  
  try {
    const output = await replicate.run(
      CONFIG.KELLY_LORA.model as `${string}/${string}:${string}`,
      {
        input: {
          prompt: prompt,
          hf_lora: CONFIG.KELLY_LORA.loraUrl,
          lora_scale: CONFIG.KELLY_LORA.loraScale,
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
    
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, imageData);
    console.log(`  ✅ Saved: ${path.basename(outputPath)}`);
    return true;
  } catch (error: any) {
    console.error(`  ❌ Error: ${error.message}`);
    return false;
  }
}

async function generateWithFlux(replicate: Replicate, prompt: string, outputPath: string, aspectRatio: string): Promise<boolean> {
  console.log(`  📊 Generating: ${path.basename(outputPath)}...`);
  
  try {
    const output = await replicate.run(CONFIG.FLUX_PRO, {
      input: {
        prompt: prompt,
        aspect_ratio: aspectRatio,
        output_format: "png",
        output_quality: 100,
        safety_tolerance: 2
      }
    }) as any;
    
    const imageUrl = typeof output === 'string' ? output : output.toString();
    const response = await fetch(imageUrl);
    const imageData = Buffer.from(await response.arrayBuffer());
    
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, imageData);
    console.log(`  ✅ Saved: ${path.basename(outputPath)}`);
    return true;
  } catch (error: any) {
    console.error(`  ❌ Error: ${error.message}`);
    return false;
  }
}

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║  🎨 DAY 351 BACKUP GENERATOR                                     ║
║  Creating alternate versions for A/B testing                     ║
╚══════════════════════════════════════════════════════════════════╝
`);

  if (!CONFIG.REPLICATE_API_TOKEN) {
    console.error('❌ REPLICATE_API_TOKEN not set!');
    process.exit(1);
  }

  const args = process.argv.slice(2);
  const kellyOnly = args.includes('--kelly-only');
  const infographicsOnly = args.includes('--infographics-only');
  const all = !kellyOnly && !infographicsOnly;

  const replicate = new Replicate({ auth: CONFIG.REPLICATE_API_TOKEN });
  let generated = 0;
  let failed = 0;

  const kellyDir = path.join(CONFIG.OUTPUT_BACKUPS, 'kelly');
  const infographicsDir = path.join(CONFIG.OUTPUT_BACKUPS, 'infographics');

  // Generate Kelly backups
  if (all || kellyOnly) {
    console.log('\n📸 GENERATING KELLY BACKUP VARIATIONS...');
    console.log(`   Output: ${kellyDir}\n`);
    
    for (const item of KELLY_BACKUP_PROMPTS) {
      const outputPath = path.join(kellyDir, item.filename);
      const success = await generateWithLoRA(replicate, item.prompt, outputPath);
      success ? generated++ : failed++;
      await new Promise(r => setTimeout(r, CONFIG.DELAY_MS));
    }
  }

  // Generate Infographic backups
  if (all || infographicsOnly) {
    console.log('\n📊 GENERATING INFOGRAPHIC BACKUP VARIATIONS...');
    console.log(`   Output: ${infographicsDir}\n`);
    
    for (const item of INFOGRAPHIC_BACKUP_PROMPTS) {
      const outputPath = path.join(infographicsDir, item.filename);
      const success = await generateWithFlux(replicate, item.prompt, outputPath, item.aspectRatio);
      success ? generated++ : failed++;
      await new Promise(r => setTimeout(r, CONFIG.DELAY_MS));
    }
  }

  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║  ✅ BACKUP GENERATION COMPLETE                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Generated: ${String(generated).padEnd(3)} backups                                     ║
║  Failed:    ${String(failed).padEnd(3)} backups                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  📁 Backups:  ${CONFIG.OUTPUT_BACKUPS}
╚══════════════════════════════════════════════════════════════════╝

Use these backups to A/B test which visuals perform best!
`);
}

main().catch(console.error);
