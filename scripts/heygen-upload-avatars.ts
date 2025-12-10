#!/usr/bin/env npx tsx
/**
 * 🎨 HEYGEN AVATAR UPLOADER
 * 
 * Uploads the 12 Kelly archetype images to HeyGen as Photo Avatars.
 * Returns the talking_photo_id for each, ready for video generation.
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY!,
  IMAGES_DIR: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-lora'),
};

const ARCHETYPES = [
  'scientist',
  'explorer',
  'rebel',
  'architect',
  'diplomat',
  'empath',
  'macgyver',
  'mystic',
  'provider',
  'storyteller',
  'strategist',
  'survivor',
];

async function uploadToHeyGen(imagePath: string, name: string): Promise<string | null> {
  console.log(`\n📤 Uploading: ${name}`);
  
  // Read the image file
  const imageBuffer = fs.readFileSync(imagePath);
  const base64Image = imageBuffer.toString('base64');
  
  // Try the talking_photo endpoint
  try {
    const response = await fetch('https://api.heygen.com/v1/talking_photo', {
      method: 'POST',
      headers: {
        'X-Api-Key': CONFIG.HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: `data:image/png;base64,${base64Image}`,
        name: `Kelly ${name}`,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.log(`   ⚠️ v1/talking_photo failed: ${error}`);
      
      // Try v2 endpoint
      return await tryV2Endpoint(imagePath, name);
    }

    const result = await response.json();
    console.log(`   ✅ Uploaded! ID: ${result.data?.talking_photo_id || result.data?.id}`);
    return result.data?.talking_photo_id || result.data?.id;

  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return null;
  }
}

async function tryV2Endpoint(imagePath: string, name: string): Promise<string | null> {
  console.log(`   🔄 Trying v2 endpoint...`);
  
  const imageBuffer = fs.readFileSync(imagePath);
  const base64Image = imageBuffer.toString('base64');
  
  try {
    const response = await fetch('https://api.heygen.com/v2/photo_avatar', {
      method: 'POST',
      headers: {
        'X-Api-Key': CONFIG.HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: `data:image/png;base64,${base64Image}`,
        name: `Kelly ${name}`,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.log(`   ⚠️ v2/photo_avatar also failed: ${error}`);
      return null;
    }

    const result = await response.json();
    console.log(`   ✅ Uploaded via v2! ID: ${result.data?.id}`);
    return result.data?.id;

  } catch (error: any) {
    console.error(`   ❌ v2 Error: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎨 HEYGEN KELLY AVATAR UPLOADER                           ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  if (!CONFIG.HEYGEN_API_KEY) {
    console.error('❌ HEYGEN_API_KEY not found in environment');
    process.exit(1);
  }

  // Check for images
  if (!fs.existsSync(CONFIG.IMAGES_DIR)) {
    console.error(`❌ Images directory not found: ${CONFIG.IMAGES_DIR}`);
    console.error('Run generate-12-kellys-with-lora.ts first!');
    process.exit(1);
  }

  const results: Record<string, string | null> = {};

  for (const archetype of ARCHETYPES) {
    const imagePath = path.join(CONFIG.IMAGES_DIR, `kelly_archetype_${archetype}.png`);
    
    if (!fs.existsSync(imagePath)) {
      console.log(`⚠️ Image not found: ${imagePath}`);
      results[archetype] = null;
      continue;
    }

    const avatarId = await uploadToHeyGen(imagePath, archetype);
    results[archetype] = avatarId;

    // Rate limit
    await new Promise(r => setTimeout(r, 2000));
  }

  // Summary
  console.log('\n\n' + '═'.repeat(60));
  console.log('📋 AVATAR IDS - Copy these to heygen-day1-batch.ts');
  console.log('═'.repeat(60));
  console.log('\nconst KELLY_AVATAR_IDS: Record<string, string> = {');
  
  for (const [archetype, id] of Object.entries(results)) {
    const formattedName = `The ${archetype.charAt(0).toUpperCase() + archetype.slice(1)}`;
    if (id) {
      console.log(`  "${formattedName}": "${id}",`);
    } else {
      console.log(`  "${formattedName}": "UPLOAD_FAILED",`);
    }
  }
  
  console.log('};');

  // Save to file
  const outputPath = path.join(CONFIG.IMAGES_DIR, 'heygen_avatar_ids.json');
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`\n💾 Saved to: ${outputPath}`);
}

main().catch(console.error);

