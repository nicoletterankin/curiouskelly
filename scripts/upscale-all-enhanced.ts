#!/usr/bin/env npx tsx
/**
 * Upscale all remaining enhanced images to 2x
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN! });

const ENHANCED_DIR = path.join(process.cwd(), 'public', 'ziggurat-concepts', 'enhanced');

const TO_UPSCALE = [
  'dawn-awakening-enhanced.png',
  'night-silhouette-enhanced.png', 
  'morning-broadcast-enhanced.png',
  'twilight-transition-enhanced.png',
  'aerial-overview-enhanced.png',
  'observatory-interior-enhanced.png',
];

async function fileToDataUrl(filePath: string): Promise<string> {
  const buffer = fs.readFileSync(filePath);
  const base64 = buffer.toString('base64');
  return `data:image/png;base64,${base64}`;
}

async function upscale(filename: string): Promise<void> {
  const inputPath = path.join(ENHANCED_DIR, filename);
  const outputName = filename.replace('-enhanced.png', '-upscaled.png');
  const outputPath = path.join(ENHANCED_DIR, outputName);
  
  if (!fs.existsSync(inputPath)) {
    console.log(`⚠️ Not found: ${filename}`);
    return;
  }
  
  console.log(`📐 Upscaling: ${filename}`);
  
  try {
    const imageDataUrl = await fileToDataUrl(inputPath);
    
    const output = await replicate.run(
      "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa",
      {
        input: {
          image: imageDataUrl,
          scale: 2,
          face_enhance: false,
        }
      }
    ) as any;

    let imageUrl = typeof output === 'string' ? output : String(output);
    
    if (imageUrl) {
      const response = await fetch(imageUrl);
      const buffer = Buffer.from(await response.arrayBuffer());
      fs.writeFileSync(outputPath, buffer);
      console.log(`   ✅ Saved: ${outputPath}`);
    }
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
  }
}

async function main() {
  console.log('📐 Upscaling remaining enhanced images...\n');
  
  for (const file of TO_UPSCALE) {
    await upscale(file);
    await new Promise(r => setTimeout(r, 500));
  }
  
  console.log('\n🎉 Done!');
}

main().catch(console.error);
