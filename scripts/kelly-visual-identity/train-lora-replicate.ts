/**
 * Kelly LoRA Training via Replicate
 * ==================================
 * Trains a FLUX LoRA with NO image limit (unlike Civitai's 5-image cap)
 * 
 * Usage:
 *   npx tsx scripts/kelly-visual-identity/train-lora-replicate.ts
 * 
 * Prerequisites:
 *   - REPLICATE_API_TOKEN in .env.local
 *   - Training images in lora-training-dataset/ folder
 *   - Each image needs a matching .txt caption file
 */

import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import archiver from 'archiver';

dotenv.config({ path: '.env.local' });

const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN;
if (!REPLICATE_TOKEN) {
  console.error('❌ REPLICATE_API_TOKEN not found in .env.local');
  process.exit(1);
}

const CONFIG = {
  // Training dataset location - use expanded dataset if it exists
  DATASET_DIR: fs.existsSync(path.join(process.cwd(), 'lora-training-dataset-expanded'))
    ? path.join(process.cwd(), 'lora-training-dataset-expanded')
    : path.join(process.cwd(), 'lora-training-dataset'),
  
  // Output for the zip file
  ZIP_OUTPUT: path.join(process.cwd(), 'lora-training-dataset.zip'),
  
  // Where to upload the zip (you'll need to host this)
  // Options: Supabase Storage, Cloudflare R2, or any public URL
  
  // Training parameters - OPTIMIZED FOR 25 IMAGES
  TRIGGER_WORD: 'kelly',
  STEPS: 2500,           // Increased for larger dataset (100 steps per image)
  LORA_RANK: 32,         // Increased for more detail and consistency
  LEARNING_RATE: 0.0001, // 1e-4 standard
  
  // Your destination (where the trained model goes)
  DESTINATION: 'curiouskelly/curious-kelly-lora-v2',
};

/**
 * Create a zip file from the training dataset
 */
async function createDatasetZip(): Promise<string> {
  console.log('📦 Creating training dataset zip...');
  
  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream(CONFIG.ZIP_OUTPUT);
    const archive = archiver('zip', { zlib: { level: 9 } });
    
    output.on('close', () => {
      console.log(`✅ Created: ${CONFIG.ZIP_OUTPUT} (${(archive.pointer() / 1024 / 1024).toFixed(2)} MB)`);
      resolve(CONFIG.ZIP_OUTPUT);
    });
    
    archive.on('error', reject);
    archive.pipe(output);
    
    // Add all images and their caption files
    const files = fs.readdirSync(CONFIG.DATASET_DIR);
    const imageFiles = files.filter(f => /\.(jpeg|jpg|png|webp)$/i.test(f));
    
    console.log(`📷 Found ${imageFiles.length} training images`);
    
    for (const imageFile of imageFiles) {
      const imagePath = path.join(CONFIG.DATASET_DIR, imageFile);
      archive.file(imagePath, { name: imageFile });
      
      // Check for caption file
      const captionFile = imageFile.replace(/\.(jpeg|jpg|png|webp)$/i, '.txt');
      const captionPath = path.join(CONFIG.DATASET_DIR, captionFile);
      
      if (fs.existsSync(captionPath)) {
        archive.file(captionPath, { name: captionFile });
        console.log(`  ✓ ${imageFile} + ${captionFile}`);
      } else {
        console.log(`  ⚠ ${imageFile} (no caption file)`);
      }
    }
    
    archive.finalize();
  });
}

/**
 * Upload zip to a hosting service and get public URL
 * For now, this provides instructions - you can integrate with Supabase/R2
 */
async function getDatasetUrl(): Promise<string> {
  console.log('\n📤 Dataset Upload Required');
  console.log('='.repeat(50));
  console.log('The training zip needs to be publicly accessible.');
  console.log('');
  console.log('Options:');
  console.log('1. Upload to Supabase Storage (you have this)');
  console.log('2. Upload to Cloudflare R2');
  console.log('3. Use any file hosting with public URLs');
  console.log('');
  console.log(`Zip location: ${CONFIG.ZIP_OUTPUT}`);
  console.log('');
  
  // For automated flow, you could integrate with Supabase here
  // For now, return a placeholder
  return 'UPLOAD_ZIP_AND_PASTE_URL_HERE';
}

/**
 * Start training on Replicate
 */
async function startTraining(datasetUrl: string): Promise<void> {
  console.log('\n🚀 Starting Replicate Training');
  console.log('='.repeat(50));
  
  const payload = {
    // Using ostris/flux-dev-lora-trainer
    destination: CONFIG.DESTINATION,
    input: {
      input_images: datasetUrl,
      trigger_word: CONFIG.TRIGGER_WORD,
      steps: CONFIG.STEPS,
      lora_rank: CONFIG.LORA_RANK,
      learning_rate: CONFIG.LEARNING_RATE,
      autocaption: false,  // We have our own captions
      autocaption_prefix: `${CONFIG.TRIGGER_WORD}, `,
    }
  };
  
  console.log('Training config:');
  console.log(JSON.stringify(payload.input, null, 2));
  
  const response = await fetch(
    'https://api.replicate.com/v1/models/ostris/flux-dev-lora-trainer/versions/d995297071a44dcb72244e6c19462111649ec86a9646c32df56daa7f14801944/trainings',
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${REPLICATE_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    }
  );
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Training failed to start: ${error}`);
  }
  
  const training = await response.json();
  
  console.log('\n✅ Training Started!');
  console.log('='.repeat(50));
  console.log(`Training ID: ${training.id}`);
  console.log(`Status URL: https://replicate.com/p/${training.id}`);
  console.log('');
  console.log('⏱️  Estimated time: 1-2 hours');
  console.log('💰 Estimated cost: $5-15');
  console.log('');
  console.log('When complete, download your LoRA from:');
  console.log(`https://replicate.com/curiouskelly/curious-kelly-lora-v2`);
}

/**
 * Main flow
 */
async function main() {
  console.log('🎨 Kelly LoRA Training via Replicate');
  console.log('=====================================');
  console.log('Unlike Civitai (5 image limit), Replicate has NO limit!');
  console.log('');
  
  // Step 1: Check dataset
  if (!fs.existsSync(CONFIG.DATASET_DIR)) {
    console.error(`❌ Dataset directory not found: ${CONFIG.DATASET_DIR}`);
    process.exit(1);
  }
  
  const images = fs.readdirSync(CONFIG.DATASET_DIR)
    .filter(f => /\.(jpeg|jpg|png|webp)$/i.test(f));
  
  console.log(`📷 Found ${images.length} training images in dataset`);
  
  if (images.length < 5) {
    console.warn('⚠️  Recommended: 15-25 images for best character consistency');
  }
  
  // Step 2: Create zip
  await createDatasetZip();
  
  // Step 3: Get URL (manual for now)
  console.log('\n' + '='.repeat(50));
  console.log('NEXT STEPS:');
  console.log('='.repeat(50));
  console.log('');
  console.log('1. Upload the zip file to get a public URL:');
  console.log(`   ${CONFIG.ZIP_OUTPUT}`);
  console.log('');
  console.log('2. Use Replicate\'s web UI (easiest):');
  console.log('   https://replicate.com/ostris/flux-dev-lora-trainer/train');
  console.log('');
  console.log('3. Or use the API with this config:');
  console.log(`
curl -X POST https://api.replicate.com/v1/models/ostris/flux-dev-lora-trainer/versions/d995297071a44dcb72244e6c19462111649ec86a9646c32df56daa7f14801944/trainings \\
  -H "Authorization: Bearer ${REPLICATE_TOKEN?.slice(0, 8)}..." \\
  -H "Content-Type: application/json" \\
  -d '{
    "destination": "${CONFIG.DESTINATION}",
    "input": {
      "input_images": "YOUR_ZIP_URL_HERE",
      "trigger_word": "${CONFIG.TRIGGER_WORD}",
      "steps": ${CONFIG.STEPS},
      "lora_rank": ${CONFIG.LORA_RANK}
    }
  }'
`);
}

main().catch(console.error);

