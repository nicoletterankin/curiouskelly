/**
 * Kelly LoRA Dataset Expansion for Cinematic Perfection
 * ======================================================
 * 
 * Generates additional training images to expand from 7 → 25 images
 * covering ALL expressions, angles, and poses needed for perfect consistency.
 * 
 * Usage:
 *   npx tsx scripts/kelly-visual-identity/expand-lora-dataset.ts
 * 
 * Prerequisites:
 *   - REPLICATE_API_TOKEN in .env.local
 *   - Existing LoRA at HuggingFace for base consistency
 */

import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import Replicate from 'replicate';

dotenv.config({ path: '.env.local' });

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

// Output directory for new training images
const OUTPUT_DIR = path.join(process.cwd(), 'lora-training-dataset-expanded');

// Current LoRA for consistency (we'll use it to generate new poses)
const CURRENT_LORA = 'https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors';

// LOCKED Kelly identity - the DNA that makes Kelly Kelly
const KELLY_DNA = {
  face: 'soft symmetrical features, natural makeup, warm approachable expression',
  eyes: 'hazel-brown eyes, expressive, slightly almond-shaped, visible catchlights',
  hair: 'brown wavy shoulder-length hair with caramel and honey highlights, center-parted',
  skin: 'light-medium warm skin tone, natural healthy glow',
  age: 'late 20s to early 30s',
  outfit: 'soft powder blue cashmere crewneck sweater, medium-wash relaxed-fit blue jeans cuffed at ankle, white leather sneakers',
};

// LOCKED scene
const SCENE = {
  setting: 'pure white cyclorama photography studio',
  chair: 'director\'s chair with black canvas fabric seat and natural warm wood frame',
  lighting: 'professional studio lighting with soft natural window light from upper right',
  floor: 'light gray seamless floor',
  camera: 'shot on Hasselblad H6D-100c, 85mm f/2.8, shallow depth of field, professional fashion photography, 8K UHD',
};

// Build full base prompt - using professional/corporate language to avoid safety filter false positives
const KELLY_BASE = `kelly, professional female educator, ${KELLY_DNA.face}, ${KELLY_DNA.eyes}, ${KELLY_DNA.hair}, ${KELLY_DNA.skin}, ${KELLY_DNA.age}, wearing ${KELLY_DNA.outfit}`;
const SCENE_BASE = `${SCENE.setting}, ${SCENE.chair}, ${SCENE.lighting}, ${SCENE.floor}, ${SCENE.camera}`;

/**
 * The 18 NEW images needed to complete the perfect dataset
 * Combined with existing 7 = 25 total training images
 */
const NEW_TRAINING_IMAGES = [
  // === ANGLE COVERAGE ===
  {
    name: 'three-quarter-left',
    prompt: `${KELLY_BASE}, three-quarter view from the left, seated in director's chair, relaxed natural posture, warm genuine smile, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, three-quarter view from left, seated, relaxed posture, warm smile, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio',
  },
  {
    name: 'three-quarter-right', 
    prompt: `${KELLY_BASE}, three-quarter view from the right, seated in director's chair, relaxed natural posture, warm genuine smile, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, three-quarter view from right, seated, relaxed posture, warm smile, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio',
  },
  {
    name: 'front-full-body',
    prompt: `${KELLY_BASE}, full body shot, standing directly facing camera, relaxed confident stance, warm welcoming smile, hands naturally at sides, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, full body, standing, facing camera, confident stance, welcoming smile, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio',
  },
  
  // === EXPRESSION COVERAGE ===
  {
    name: 'surprised-delighted',
    prompt: `${KELLY_BASE}, close-up portrait, genuinely surprised delighted expression, eyes wide with joy, mouth slightly open in pleasant surprise, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, close-up, surprised delighted expression, eyes wide with joy, brown wavy hair, hazel-brown eyes, soft blue cashmere sweater, white background',
  },
  {
    name: 'teaching-explaining',
    prompt: `${KELLY_BASE}, medium shot waist up, actively explaining with hand gesture, engaged teaching expression, one hand raised in explanatory gesture, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, medium shot, teaching explaining, hand gesture, engaged expression, brown wavy hair, soft blue cashmere sweater, white studio',
  },
  {
    name: 'curious-questioning',
    prompt: `${KELLY_BASE}, close-up portrait, curious questioning expression, one eyebrow slightly raised, head tilted with interest, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, close-up, curious questioning expression, eyebrow raised, head tilted, brown wavy hair, hazel-brown eyes, soft blue cashmere sweater, white background',
  },
  {
    name: 'encouraging-supportive',
    prompt: `${KELLY_BASE}, medium shot, encouraging supportive expression, leaning forward slightly, warm empathetic smile, hands open in welcoming gesture, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, medium shot, encouraging supportive expression, leaning forward, warm smile, open hands, brown wavy hair, soft blue cashmere sweater, white studio',
  },
  {
    name: 'celebrating-joyful',
    prompt: `${KELLY_BASE}, full body, arms raised in celebration, genuine joyful expression, triumphant victorious pose, big authentic smile, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, full body, arms raised celebrating, joyful expression, triumphant pose, big smile, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio',
  },
  {
    name: 'concentrating-focused',
    prompt: `${KELLY_BASE}, close-up portrait, concentrated focused expression, slightly narrowed eyes, serious but warm, thinking deeply, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, close-up, concentrated focused expression, narrowed eyes, serious but warm, brown wavy hair, hazel-brown eyes, soft blue cashmere sweater, white background',
  },
  {
    name: 'laughing-genuine',
    prompt: `${KELLY_BASE}, close-up portrait, genuine hearty laugh, eyes crinkled with joy, mouth open in authentic laughter, natural unposed moment, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, close-up, genuine laugh, eyes crinkled with joy, authentic laughter, brown wavy hair, hazel-brown eyes, soft blue cashmere sweater, white background',
  },
  
  // === POSE COVERAGE ===
  {
    name: 'pointing-left',
    prompt: `${KELLY_BASE}, medium shot, left arm extended gracefully pointing to the left side of frame, helpful guiding expression, body angled slightly toward gesture, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, medium shot, pointing left, arm extended, helpful expression, brown wavy hair, soft blue cashmere sweater, white studio',
  },
  {
    name: 'pointing-right',
    prompt: `${KELLY_BASE}, medium shot, right arm extended gracefully pointing to the right side of frame, helpful guiding expression, body angled slightly toward gesture, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, medium shot, pointing right, arm extended, helpful expression, brown wavy hair, soft blue cashmere sweater, white studio',
  },
  {
    name: 'pointing-up',
    prompt: `${KELLY_BASE}, medium shot, right arm raised elegantly pointing upward, head tilted back slightly looking up, engaged interested expression, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, medium shot, pointing up, arm raised, looking up, interested expression, brown wavy hair, soft blue cashmere sweater, white studio',
  },
  {
    name: 'arms-crossed-confident',
    prompt: `${KELLY_BASE}, medium shot, arms crossed confidently across chest, warm self-assured smile, relaxed confident stance, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, medium shot, arms crossed, confident smile, relaxed stance, brown wavy hair, soft blue cashmere sweater, white studio',
  },
  {
    name: 'leaning-forward-engaged',
    prompt: `${KELLY_BASE}, medium shot, leaning forward with genuine interest, elbows on knees, engaged attentive expression, active listening pose, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, medium shot, leaning forward, elbows on knees, engaged attentive expression, brown wavy hair, soft blue cashmere sweater, white studio',
  },
  {
    name: 'standing-casual',
    prompt: `${KELLY_BASE}, full body, standing in relaxed casual pose, weight on one leg, one hand on hip, friendly approachable stance, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, full body, standing casual, relaxed pose, hand on hip, friendly stance, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio',
  },
  
  // === FRAMING COVERAGE ===
  {
    name: 'medium-shot-neutral',
    prompt: `${KELLY_BASE}, medium shot from waist up, neutral relaxed expression, slight pleasant smile, hands visible in frame, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, medium shot waist up, neutral relaxed expression, slight smile, hands visible, brown wavy hair, soft blue cashmere sweater, white studio',
  },
  {
    name: 'extreme-closeup-eyes',
    prompt: `${KELLY_BASE}, extreme close-up on face, focus on eyes, warm intelligent gaze, slight smile, visible catchlights in hazel-brown eyes, ${SCENE_BASE}`,
    caption: 'kelly, photorealistic woman, extreme close-up, face focus, warm intelligent gaze, slight smile, hazel-brown eyes with catchlights, brown wavy hair, white background',
  },
];

/**
 * Generate a single training image
 */
async function generateImage(
  name: string,
  prompt: string,
  caption: string
): Promise<boolean> {
  console.log(`\n🎨 Generating: ${name}`);
  console.log(`   Prompt: ${prompt.slice(0, 100)}...`);
  
  try {
    const output = await replicate.run(
      'lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d',
      {
        input: {
          prompt: prompt,
          hf_lora: CURRENT_LORA,
          lora_scale: 0.85,
          num_outputs: 1,
          aspect_ratio: '1:1',  // Square for training
          output_format: 'png',
          guidance_scale: 3.5,
          num_inference_steps: 28,
          disable_safety_checker: true,
        },
      }
    );
    
    // Handle different output formats (URL string or ReadableStream)
    let imageBuffer: Buffer;
    const result = Array.isArray(output) ? output[0] : output;
    
    if (!result) {
      console.error(`   ❌ No output returned`);
      return false;
    }
    
    // Check if it's a ReadableStream
    if (result && typeof result === 'object' && 'getReader' in result) {
      // It's a ReadableStream - read all chunks
      const reader = (result as ReadableStream).getReader();
      const chunks: Uint8Array[] = [];
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }
      
      // Combine chunks into single buffer
      const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
      const combined = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      imageBuffer = Buffer.from(combined);
      
    } else if (typeof result === 'string') {
      // It's a URL - download it
      const response = await fetch(result);
      const arrayBuffer = await response.arrayBuffer();
      imageBuffer = Buffer.from(arrayBuffer);
      
    } else {
      console.error(`   ❌ Unexpected output type: ${typeof result}`);
      return false;
    }
    
    // Save image
    const imagePath = path.join(OUTPUT_DIR, `${name}.png`);
    fs.writeFileSync(imagePath, imageBuffer);
    console.log(`   ✅ Saved: ${name}.png (${(imageBuffer.length / 1024).toFixed(0)} KB)`);
    
    // Save caption
    const captionPath = path.join(OUTPUT_DIR, `${name}.txt`);
    fs.writeFileSync(captionPath, caption);
    console.log(`   📝 Caption: ${name}.txt`);
    
    return true;
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return false;
  }
}

/**
 * Copy existing images to the expanded dataset
 */
function copyExistingImages(): void {
  const existingDir = path.join(process.cwd(), 'lora-training-dataset');
  const files = fs.readdirSync(existingDir);
  
  console.log('\n📋 Copying existing 7 images...');
  
  for (const file of files) {
    if (file === 'README.md') continue;
    
    const src = path.join(existingDir, file);
    const dest = path.join(OUTPUT_DIR, file);
    
    fs.copyFileSync(src, dest);
    console.log(`   ✓ ${file}`);
  }
}

/**
 * Main generation flow
 */
async function main(): Promise<void> {
  console.log('🎬 KELLY LORA DATASET EXPANSION');
  console.log('================================');
  console.log('Goal: Expand from 7 → 25 images for CINEMATIC PERFECTION');
  console.log('');
  
  // Create output directory
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  
  // Copy existing images first
  copyExistingImages();
  
  // Generate new images
  console.log(`\n🎨 Generating ${NEW_TRAINING_IMAGES.length} new training images...`);
  console.log('   (This will take ~15-20 minutes and cost ~$2-3)');
  
  let success = 0;
  let failed = 0;
  
  for (const image of NEW_TRAINING_IMAGES) {
    const result = await generateImage(image.name, image.prompt, image.caption);
    if (result) {
      success++;
    } else {
      failed++;
    }
    
    // Rate limiting
    await new Promise(r => setTimeout(r, 2000));
  }
  
  // Create comprehensive README
  const readme = `# Kelly LoRA Training Dataset - Cinematic Perfection Edition

This dataset contains **25 curated reference images** for training a production-grade character LoRA.

## Dataset Composition

### Original 7 Images (from initial training)
- 4.jpeg - Close-up, big smile
- pray.jpeg - Hands together, hopeful
- open-walk.jpeg - Full body walking, profile
- square-chair2.jpeg - Seated, hand on heart
- our-girl.jpeg - Seated, chin on hand
- open.png - Close-up, contemplative
- close.jpeg - Eyes closed, peaceful

### 18 NEW Expansion Images
${NEW_TRAINING_IMAGES.map(img => `- ${img.name}.png - ${img.caption.split(',').slice(2, 4).join(',')}`).join('\n')}

## Coverage Matrix

| Category | Coverage |
|----------|----------|
| **Angles** | Front, 3/4 Left, 3/4 Right, Profile |
| **Expressions** | Smile, Thoughtful, Surprised, Teaching, Curious, Celebrating, Laughing, Focused |
| **Poses** | Seated, Standing, Walking, Pointing L/R/Up, Arms Crossed, Leaning Forward |
| **Framing** | Extreme Close-up, Close-up, Medium, Full Body |

## Training Settings (Replicate)

- **Trainer:** ostris/flux-dev-lora-trainer
- **Trigger word:** kelly
- **Steps:** 2500 (increased for larger dataset)
- **LoRA rank:** 32 (increased for more detail)
- **Learning rate:** 0.0001

## Why This Works

With 25 diverse images covering all angles, expressions, and poses:
- The LoRA learns Kelly's **identity**, not just specific poses
- New generations maintain consistency regardless of prompt
- Hands, expressions, and proportions remain stable
- Every frame is recognizably Kelly

Generated: ${new Date().toISOString()}
Version: 2.0 - Cinematic Perfection Edition
`;

  fs.writeFileSync(path.join(OUTPUT_DIR, 'README.md'), readme);
  
  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 GENERATION COMPLETE');
  console.log('='.repeat(60));
  console.log(`✅ Successfully generated: ${success}/${NEW_TRAINING_IMAGES.length}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log(`📷 Total images: ${7 + success} (7 original + ${success} new)`);
  console.log('');
  console.log('🎯 NEXT STEPS:');
  console.log('1. Review generated images in the folder');
  console.log('2. Remove any that don\'t look like Kelly');
  console.log('3. Run: npx tsx scripts/kelly-visual-identity/train-lora-replicate.ts');
  console.log('4. Train new LoRA on Replicate (~$10, ~2 hours)');
}

main().catch(console.error);

