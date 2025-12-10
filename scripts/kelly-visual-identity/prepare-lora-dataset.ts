/**
 * Kelly Visual Identity Pipeline - LoRA Training Dataset Preparation
 * 
 * Prepares the 8 reference images with captions for Civitai LoRA training.
 * Creates a structured dataset folder with images and corresponding .txt caption files.
 * 
 * Usage: tsx scripts/kelly-visual-identity/prepare-lora-dataset.ts
 */

import * as fs from "fs";
import * as path from "path";

// Reference image locations
const REFERENCE_IMAGES = [
  {
    source: "daily-lesson-marketing/public/lessons/images/4.jpeg",
    name: "4.jpeg",
    caption: "kelly, photorealistic woman, close-up portrait, big genuine smile, direct eye contact, brown wavy hair with caramel highlights, hazel-brown eyes, soft blue cashmere sweater, white background, studio lighting"
  },
  {
    source: "daily-lesson-marketing/public/lessons/images/yay-pray-huh/pray.jpeg",
    name: "pray.jpeg",
    caption: "kelly, photorealistic woman, upper body, hands together near face in prayer pose, looking up and to the right, playful hopeful expression, brown wavy hair, soft blue cashmere sweater, white background"
  },
  {
    source: "daily-lesson-marketing/public/lessons/images/walk/open-walk.jpeg",
    name: "open-walk.jpeg",
    caption: "kelly, photorealistic woman, full body, walking, profile view, casual confident stride, director's chair in background, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio, natural lighting"
  },
  {
    source: "daily-lesson-marketing/public/lessons/images/square-chair/square-chair2.jpeg",
    name: "square-chair2.jpeg",
    caption: "kelly, photorealistic woman, full body, seated in director's chair, right hand on heart, sincere grateful expression, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio background"
  },
  {
    source: "daily-lesson-marketing/public/lessons/images/our-girl-too-excited/our-girl.jpeg",
    name: "our-girl.jpeg",
    caption: "kelly, photorealistic woman, full body, seated in director's chair, chin resting on hand, thoughtful curious expression, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio background"
  },
  {
    source: "daily-lesson-marketing/public/lessons/images/open-close/open.png",
    name: "open.png",
    caption: "kelly, photorealistic woman, close-up portrait, chin on hand, looking up and to the left, contemplative expression, brown wavy hair with highlights, hazel-brown eyes, soft blue cashmere sweater, white background"
  },
  {
    source: "daily-lesson-marketing/public/lessons/images/open-close/close.jpeg",
    name: "close.jpeg",
    caption: "kelly, photorealistic woman, close-up portrait, eyes closed, peaceful satisfied smile, chin resting on hand, brown wavy hair, soft blue cashmere sweater, white background"
  }
];

// Note: The 8th reference image "Curious_Kelly_in_final_pose_in_Chair.png" was not found
// We'll use the available 7 images which should still be sufficient for LoRA training

const OUTPUT_DIR = path.join(process.cwd(), "lora-training-dataset");

/**
 * Prepare LoRA training dataset
 */
async function prepareLoRADataset() {
  console.log("🎨 Kelly Visual Identity Pipeline - LoRA Dataset Preparation");
  console.log("=".repeat(60));
  
  // Create output directory
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`📁 Output directory: ${OUTPUT_DIR}`);
  
  let successCount = 0;
  let errorCount = 0;
  
  for (const ref of REFERENCE_IMAGES) {
    try {
      const sourcePath = path.join(process.cwd(), ref.source);
      
      // Check if source exists
      if (!fs.existsSync(sourcePath)) {
        console.error(`❌ Source not found: ${ref.source}`);
        errorCount++;
        continue;
      }
      
      // Copy image
      const destImagePath = path.join(OUTPUT_DIR, ref.name);
      fs.copyFileSync(sourcePath, destImagePath);
      console.log(`✅ Copied: ${ref.name}`);
      
      // Create caption file
      const captionPath = path.join(OUTPUT_DIR, ref.name.replace(/\.(jpeg|jpg|png)$/, ".txt"));
      fs.writeFileSync(captionPath, ref.caption);
      console.log(`📝 Caption: ${path.basename(captionPath)}`);
      
      successCount++;
      
    } catch (error: any) {
      console.error(`❌ Error processing ${ref.name}:`, error.message);
      errorCount++;
    }
  }
  
  // Create README for the dataset
  const readmePath = path.join(OUTPUT_DIR, "README.md");
  const readmeContent = `# Kelly LoRA Training Dataset

This dataset contains ${successCount} reference images of Kelly for training a character LoRA.

## Training Settings (Civitai)

- **Base model:** FLUX.1 Dev or SDXL 1.0
- **Training type:** Character/Person
- **Instance prompt:** \`kelly\`
- **Class prompt:** \`woman\`
- **Training steps:** 1500-2000
- **Learning rate:** 1e-4
- **Network dimension:** 32
- **Network alpha:** 16

## Kelly's Visual Identity (LOCKED)

### Physical Appearance
- Face: Soft, symmetrical features; natural makeup; warm, approachable expression
- Eyes: Hazel-brown, expressive, slightly almond-shaped
- Hair: Brown with caramel/honey highlights, wavy, shoulder-length, center-parted
- Skin: Light-medium warm tone, natural, healthy glow
- Age: Late 20s to early 30s

### Outfit (ALWAYS THE SAME)
- Top: Soft blue cashmere crewneck sweater (hex #A8C4D9)
- Bottoms: Medium-wash relaxed-fit jeans, cuffed at ankle
- Shoes: White leather sneakers (minimal, clean)
- No jewelry, no accessories

### Scene (ALWAYS THE SAME)
- Chair: Director's chair with black fabric seat/back, warm wood frame
- Background: White cyclorama studio
- Lighting: Natural window light from upper right
- Floor: Light gray/white seamless

## Files in This Dataset

${REFERENCE_IMAGES.map((ref, i) => `${i + 1}. **${ref.name}** - ${ref.caption.split(',').slice(2, 4).join(',')}`).join('\n')}

## Next Steps

1. Upload this entire folder to Civitai: https://civitai.com/models/train
2. Configure training with the settings above
3. Start training (estimated cost: $15-25, time: 4-6 hours)
4. Download the trained LoRA model
5. Use in generation pipeline with \`kelly\` trigger word

Generated: ${new Date().toISOString()}
`;
  
  fs.writeFileSync(readmePath, readmeContent);
  console.log(`\n📄 Created: README.md`);
  
  // Summary
  console.log("\n" + "=".repeat(60));
  console.log("📊 DATASET PREPARATION SUMMARY");
  console.log("=".repeat(60));
  console.log(`✅ Successfully prepared: ${successCount} images`);
  console.log(`❌ Errors: ${errorCount}`);
  console.log(`📁 Dataset location: ${OUTPUT_DIR}`);
  console.log("\n🎯 NEXT STEPS:");
  console.log("1. Review the images in the dataset folder");
  console.log("2. Go to https://civitai.com/models/train");
  console.log("3. Upload the entire 'lora-training-dataset' folder");
  console.log("4. Configure training settings as specified in README.md");
  console.log("5. Start training (~4-6 hours, $15-25)");
}

// Run if called directly
if (require.main === module) {
  prepareLoRADataset().catch(console.error);
}

export { prepareLoRADataset, REFERENCE_IMAGES };









