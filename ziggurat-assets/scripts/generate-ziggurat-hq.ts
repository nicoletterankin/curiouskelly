#!/usr/bin/env npx tsx
/**
 * 🏛️ ZIGGURAT CONCEPT GENERATOR - HIGH QUALITY VERSION
 * 
 * Generates premium quality hero images using Flux Dev.
 */

import 'dotenv/config';
import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

const OUTPUT_DIR = path.join(process.cwd(), 'public', 'ziggurat-concepts');

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Hero images with enhanced prompts
const HERO_PROMPTS = {
  'hero-dawn-hq': {
    prompt: `Ultra high resolution architectural photography of a monumental stepped pyramid building at golden hour dawn, inverted ziggurat architectural form with 13 distinct horizontal terrace levels where each level projects outward more than the one above creating dramatic stepped silhouette, the building is constructed of weathered brutalist precast concrete panels in warm cream beige color, each terrace edge is illuminated with bright electric blue LED strip lighting color hex 3B82F6, the LED strips create horizontal glowing lines across the entire width of each terrace, stunning Pacific Ocean sunrise in background with dramatic pink orange purple gradient sky, lush Southern California coastal hillside landscape, the building stands as a beacon on the hilltop, professional architectural photography by Julius Shulman, Hasselblad medium format camera, ultra sharp detail, dramatic lighting, award winning architectural photograph`,
    aspectRatio: '16:9',
  },
  
  'hero-night-hq': {
    prompt: `Professional architectural night photography of massive stepped pyramid building under starry night sky, ziggurat form with 13 horizontal terrace levels clearly visible in silhouette, each terrace edge subtly outlined with dim blue LED accent lighting at 15 percent brightness creating gentle blue horizontal lines, dark indigo navy sky filled with stars, building appears as a dark geometric form with minimal light pollution, respectful ambient lighting much dimmer than surrounding area, Southern California coastal hills visible in moonlight, the building rests quietly in the night, architectural photography by Ezra Stoller, shot on large format camera, long exposure, peaceful contemplative atmosphere, professional real estate photography`,
    aspectRatio: '16:9',
  },
};

async function generateImage(name: string, prompt: string, aspectRatio: string): Promise<string | null> {
  console.log(`\n🎨 Generating HQ: ${name}`);
  
  try {
    // Use flux-dev for higher quality
    const output = await replicate.run(
      "black-forest-labs/flux-dev",
      {
        input: {
          prompt,
          aspect_ratio: aspectRatio,
          num_outputs: 1,
          output_format: "png",
          output_quality: 100,
          num_inference_steps: 50,
          guidance: 3.5,
        }
      }
    ) as any;

    let imageUrl: string | null = null;
    
    if (Array.isArray(output) && output.length > 0) {
      const firstOutput = output[0];
      if (typeof firstOutput === 'string') {
        imageUrl = firstOutput;
      } else if (firstOutput && typeof firstOutput === 'object') {
        imageUrl = String(firstOutput);
      }
    } else if (typeof output === 'string') {
      imageUrl = output;
    }
    
    if (imageUrl) {
      console.log(`   ✅ Generated: ${imageUrl.substring(0, 80)}...`);
      
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to download: ${response.status}`);
      }
      const buffer = Buffer.from(await response.arrayBuffer());
      const outputPath = path.join(OUTPUT_DIR, `${name}.png`);
      fs.writeFileSync(outputPath, buffer);
      console.log(`   💾 Saved: ${outputPath}`);
      
      return outputPath;
    }
    
    return null;
  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🏛️  THE ZIGGURAT - High Quality Hero Images');
  console.log('═══════════════════════════════════════════════════════════════');

  for (const [name, config] of Object.entries(HERO_PROMPTS)) {
    await generateImage(name, config.prompt, config.aspectRatio);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  console.log('\n🎉 Done!');
}

main().catch(console.error);
