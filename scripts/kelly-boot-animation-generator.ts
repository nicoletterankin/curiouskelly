#!/usr/bin/env npx tsx
/**
 * 🚀 KELLY BOOT ANIMATION FRAME GENERATOR
 * 
 * Generates 3 animation frames for a boot/loading screen:
 * Frame 2: Transition - lowering hand, turning toward camera
 * Frame 3: Greeting - facing camera, warm expression
 * Frame 4: Smile - genuine warm smile, welcoming
 * 
 * Frame 1 (Thinking) already exists as kelly-curious-v3-seed33333333.png
 * 
 * USAGE:
 *   npx tsx scripts/kelly-boot-animation-generator.ts
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import https from 'https';
import http from 'http';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  REPLICATE_API_TOKEN: process.env.REPLICATE_API_TOKEN!,
  
  // Kelly LoRA - high scale for perfect consistency
  KELLY_LORA_URL: 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors',
  LORA_SCALE: 0.92, // High for maximum character consistency
  
  // Model
  FLUX_LORA_MODEL: 'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
  
  // Output - 1024x1024 square as requested
  WIDTH: 1024,
  HEIGHT: 1024,
  
  // Output directory
  OUTPUT_DIR: path.join(process.cwd(), 'kelly-boot-frames'),
  
  // Reference image (Frame 1)
  REFERENCE_IMAGE: path.join(process.cwd(), 'lora-training-dataset-expanded', 'kelly-curious-v3-seed33333333.png'),
  
  // Seed for consistency - using the same base as the reference image
  BASE_SEED: 33333333,
};

// =============================================================================
// BOOT ANIMATION FRAMES
// =============================================================================

interface AnimationFrame {
  id: string;
  filename: string;
  name: string;
  prompt: string;
  negativePrompt: string;
  seed: number;
}

const BOOT_FRAMES: AnimationFrame[] = [
  {
    id: 'transition',
    filename: 'kelly-boot-transition.png',
    name: 'Frame 2: Transition',
    prompt: `kelly, photorealistic woman, light blue crewneck sweater, long wavy brown hair with golden blonde highlights, brown eyes, lowering her hand from her chin, head slightly turned toward camera, soft natural expression, pure white background, natural soft lighting, medium shot, upper body visible, same identity, professional portrait, photorealistic quality`,
    negativePrompt: 'circle crop, logo, text, watermark, frame, border, harsh shadows, dark background',
    seed: CONFIG.BASE_SEED + 1,
  },
  {
    id: 'greeting',
    filename: 'kelly-boot-greeting.png',
    name: 'Frame 3: Greeting',
    prompt: `kelly, photorealistic woman, light blue crewneck sweater, long wavy brown hair with golden blonde highlights, brown eyes, facing camera directly, hands relaxed, warm welcoming expression, beginning of a smile, eye contact with viewer, pure white background, natural soft lighting, medium shot, upper body visible, same identity, professional portrait, photorealistic quality`,
    negativePrompt: 'circle crop, logo, text, watermark, frame, border, harsh shadows, dark background, looking away',
    seed: CONFIG.BASE_SEED + 2,
  },
  {
    id: 'smile',
    filename: 'kelly-boot-smile.png',
    name: 'Frame 4: Smile',
    prompt: `kelly, photorealistic woman, light blue crewneck sweater, long wavy brown hair with golden blonde highlights, brown eyes, facing camera, genuine warm smile, friendly engaged eyes, welcoming expression, pure white background, natural soft lighting, medium shot, upper body visible, same identity, professional portrait, photorealistic quality`,
    negativePrompt: 'circle crop, logo, text, watermark, frame, border, harsh shadows, dark background, looking away, serious, neutral expression',
    seed: CONFIG.BASE_SEED + 3,
  },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

async function downloadImage(url: string, outputPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(outputPath);
    const protocol = url.startsWith('https') ? https : http;
    
    protocol.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          downloadImage(redirectUrl, outputPath).then(resolve).catch(reject);
          return;
        }
      }
      
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(outputPath, () => {});
      reject(err);
    });
  });
}

async function generateFrame(replicate: Replicate, frame: AnimationFrame): Promise<string> {
  console.log(`\n🎬 Generating: ${frame.name}`);
  console.log(`   Filename: ${frame.filename}`);
  console.log(`   Seed: ${frame.seed}`);
  console.log(`   Prompt: ${frame.prompt.substring(0, 80)}...`);
  
  const input = {
    prompt: frame.prompt,
    negative_prompt: frame.negativePrompt,
    hf_lora: CONFIG.KELLY_LORA_URL,
    lora_scale: CONFIG.LORA_SCALE,
    num_outputs: 1,
    aspect_ratio: '1:1',
    output_format: 'png',
    guidance_scale: 7.5, // As requested
    output_quality: 100,
    num_inference_steps: 40, // Higher for quality
    seed: frame.seed,
    disable_safety_checker: true,
  };
  
  console.log(`   🔄 Calling Replicate API...`);
  const startTime = Date.now();
  
  const output = await replicate.run(CONFIG.FLUX_LORA_MODEL as `${string}/${string}:${string}`, { input });
  
  if (!output) {
    throw new Error(`No output from Replicate for ${frame.id}`);
  }
  
  // Extract URL from output
  let imageUrl: string;
  if (Array.isArray(output)) {
    imageUrl = String(output[0]);
  } else if (typeof output === 'object' && output !== null) {
    imageUrl = String((output as any).url || (output as any).toString());
  } else {
    imageUrl = String(output);
  }
  
  if (!imageUrl || imageUrl === '[object Object]' || imageUrl === 'undefined') {
    throw new Error(`Invalid output URL from Replicate: ${JSON.stringify(output)}`);
  }
  
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`   ✅ Generated in ${elapsed}s`);
  console.log(`   📥 Downloading...`);
  
  // Download the image
  const outputPath = path.join(CONFIG.OUTPUT_DIR, frame.filename);
  await downloadImage(imageUrl, outputPath);
  
  const stats = fs.statSync(outputPath);
  console.log(`   ✅ Saved: ${frame.filename} (${(stats.size / 1024).toFixed(1)} KB)`);
  
  return outputPath;
}

function copyReferenceImage(): boolean {
  const destPath = path.join(CONFIG.OUTPUT_DIR, 'kelly-boot-thinking.png');
  
  if (fs.existsSync(CONFIG.REFERENCE_IMAGE)) {
    fs.copyFileSync(CONFIG.REFERENCE_IMAGE, destPath);
    const stats = fs.statSync(destPath);
    console.log(`   ✅ Copied: kelly-boot-thinking.png (${(stats.size / 1024).toFixed(1)} KB)`);
    return true;
  } else {
    console.log(`   ⚠️ Reference image not found at: ${CONFIG.REFERENCE_IMAGE}`);
    return false;
  }
}

function generateReadme(generatedFrames: string[], seed: number): void {
  const readme = `# Kelly Boot Animation Frames

Generated: ${new Date().toISOString()}

## Frames

| Frame | Filename | Description |
|-------|----------|-------------|
| 1 | kelly-boot-thinking.png | Looking up and right, chin on hand, curious/thinking |
| 2 | kelly-boot-transition.png | Lowering hand, turning toward camera |
| 3 | kelly-boot-greeting.png | Facing camera, warm expression, beginning of smile |
| 4 | kelly-boot-smile.png | Full warm smile, welcoming expression |

## Generation Settings

- **Model**: lucataco/flux-dev-lora
- **LoRA**: CuriousKellycom/curious-kelly-lora
- **LoRA Scale**: ${CONFIG.LORA_SCALE}
- **Base Seed**: ${seed}
- **Size**: 1024×1024
- **Guidance Scale**: 7.5
- **Steps**: 40

## Frame Seeds

- Frame 1 (Thinking): ${seed} (reference image)
- Frame 2 (Transition): ${seed + 1}
- Frame 3 (Greeting): ${seed + 2}
- Frame 4 (Smile): ${seed + 3}

## Usage

These frames are designed for a boot/loading animation where Kelly transitions
from "thinking" to "greeting" the user:

1. **Thinking** → She's contemplating, looking curious
2. **Transition** → She notices the user, turns toward camera
3. **Greeting** → Making eye contact, warm welcome beginning
4. **Smile** → Full warm smile, ready to teach

## CSS Animation Example

\`\`\`css
@keyframes kelly-boot {
  0%, 20% { background-image: url('kelly-boot-thinking.png'); }
  25%, 45% { background-image: url('kelly-boot-transition.png'); }
  50%, 70% { background-image: url('kelly-boot-greeting.png'); }
  75%, 100% { background-image: url('kelly-boot-smile.png'); }
}
\`\`\`

## Quality Checklist

- [ ] Same person identity across all 4 frames
- [ ] Same outfit (light blue sweater) in all frames
- [ ] White background, no artifacts
- [ ] Natural progression of movement/expression
- [ ] No weird hands or anatomy issues
`;

  const readmePath = path.join(CONFIG.OUTPUT_DIR, 'README.md');
  fs.writeFileSync(readmePath, readme);
  console.log(`   📄 Generated README.md`);
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('═'.repeat(70));
  console.log('🚀 KELLY BOOT ANIMATION FRAME GENERATOR');
  console.log('═'.repeat(70));
  console.log(`\nGenerating ${BOOT_FRAMES.length} frames + copying reference:`);
  console.log('  1. Thinking (copy from reference)');
  console.log('  2. Transition (generate)');
  console.log('  3. Greeting (generate)');
  console.log('  4. Smile (generate)');
  console.log(`\nOutput: ${CONFIG.OUTPUT_DIR}`);
  console.log(`Size: ${CONFIG.WIDTH}×${CONFIG.HEIGHT}`);
  console.log(`LoRA Scale: ${CONFIG.LORA_SCALE}`);
  console.log(`Base Seed: ${CONFIG.BASE_SEED}`);
  console.log('═'.repeat(70));
  
  // Check API key
  if (!CONFIG.REPLICATE_API_TOKEN) {
    console.error('\n❌ ERROR: REPLICATE_API_TOKEN not found!');
    console.error('   Set it in your .env or .env.local file.');
    process.exit(1);
  }
  
  // Create output directory
  if (!fs.existsSync(CONFIG.OUTPUT_DIR)) {
    fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
    console.log(`\n📁 Created: ${CONFIG.OUTPUT_DIR}`);
  }
  
  // Copy reference image (Frame 1)
  console.log('\n📋 FRAME 1: Copying reference image...');
  const refCopied = copyReferenceImage();
  
  // Initialize Replicate
  const replicate = new Replicate({
    auth: CONFIG.REPLICATE_API_TOKEN,
  });
  
  // Generate each frame
  const generatedPaths: string[] = [];
  const startTime = Date.now();
  
  for (let i = 0; i < BOOT_FRAMES.length; i++) {
    const frame = BOOT_FRAMES[i];
    console.log(`\n[${i + 2}/4] Processing ${frame.name}...`);
    
    try {
      const outputPath = await generateFrame(replicate, frame);
      generatedPaths.push(outputPath);
    } catch (error) {
      console.error(`\n❌ ERROR generating ${frame.id}:`, error);
    }
  }
  
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  
  // Generate README
  console.log('\n📝 Creating README...');
  generateReadme(generatedPaths, CONFIG.BASE_SEED);
  
  // Summary
  console.log('\n' + '═'.repeat(70));
  console.log('✅ GENERATION COMPLETE');
  console.log('═'.repeat(70));
  console.log(`\nGenerated ${generatedPaths.length}/${BOOT_FRAMES.length} frames in ${elapsed}s`);
  console.log(`Reference image ${refCopied ? 'copied' : 'NOT FOUND'}`);
  
  console.log(`\n📁 Output folder: ${CONFIG.OUTPUT_DIR}`);
  console.log('\n📄 Files:');
  
  const allFiles = fs.readdirSync(CONFIG.OUTPUT_DIR);
  allFiles.forEach((file, i) => {
    const filePath = path.join(CONFIG.OUTPUT_DIR, file);
    const stats = fs.statSync(filePath);
    console.log(`   ${i + 1}. ${file} (${(stats.size / 1024).toFixed(1)} KB)`);
  });
  
  console.log('\n✨ Boot animation frames ready!');
  console.log('\n🔍 Quality Check Reminder:');
  console.log('   □ Same person identity across all 4 frames');
  console.log('   □ Same outfit (light blue sweater) in all frames');
  console.log('   □ White background, no artifacts');
  console.log('   □ Natural progression of movement/expression');
  console.log('   □ No weird hands or anatomy issues');
}

main().catch(console.error);
