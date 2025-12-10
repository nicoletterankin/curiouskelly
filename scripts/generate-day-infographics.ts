
import * as dotenv from 'dotenv';
dotenv.config();

import Replicate from 'replicate';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

const CONFIG = {
  // Flux Pro for high quality infographics
  MODEL: "black-forest-labs/flux-1.1-pro",
  
  // Output directory
  OUTPUT_DIR_PHASES: path.join(process.cwd(), 'public', 'kelly', 'phases'),
};

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

async function downloadImage(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : require('http');
    protocol.get(url, (response: any) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        downloadImage(response.headers.location).then(resolve).catch(reject);
        return;
      }
      const chunks: Buffer[] = [];
      response.on('data', (chunk: Buffer) => chunks.push(chunk));
      response.on('end', () => resolve(Buffer.concat(chunks)));
      response.on('error', reject);
    }).on('error', reject);
  });
}

async function generateImage(prompt: string, outputPath: string) {
  console.log(`     🎨 Generating: ${prompt.substring(0, 50)}...`);
  
  try {
    const output = await replicate.run(CONFIG.MODEL, {
      input: {
        prompt: prompt,
        aspect_ratio: "16:9",
        output_format: "png",
        output_quality: 100,
        safety_tolerance: 2
      }
    }) as any;
    
    const imageUrl = typeof output === 'string' ? output : output.toString();
    const buffer = await downloadImage(imageUrl);
    
    fs.writeFileSync(outputPath, buffer);
    console.log(`     ✅ Saved to ${path.basename(outputPath)}`);
    return true;
  } catch (error: any) {
    console.error(`     ❌ Error: ${error.message}`);
    return false;
  }
}

async function processDay(dayNumber: number) {
  console.log(`\nProcessing Day ${dayNumber}...`);
  const dayDir = path.join(CONFIG.OUTPUT_DIR_PHASES, String(dayNumber).padStart(3, '0') || String(dayNumber));
  const planPath = path.join(dayDir, 'visual-plan.json');
  
  if (!fs.existsSync(planPath)) {
    console.error(`  ❌ No visual-plan.json found at ${planPath}`);
    // Try without padding
    const dayDirNoPad = path.join(CONFIG.OUTPUT_DIR_PHASES, String(dayNumber));
    const planPathNoPad = path.join(dayDirNoPad, 'visual-plan.json');
    if (fs.existsSync(planPathNoPad)) {
        console.log(`  Found at ${planPathNoPad}`);
        // Proceed with no pad logic if needed, but for now let's just stick to standard
    }
    return;
  }

  const plan = JSON.parse(fs.readFileSync(planPath, 'utf-8'));
  
  for (const item of plan) {
    const phase = item.phase.toLowerCase();
    const outputPath = path.join(dayDir, `${phase}.png`);
    
    if (fs.existsSync(outputPath)) {
      console.log(`  ⏭️ ${phase}.png already exists`);
      continue;
    }
    
    // Construct prompt
    const prompt = `${item.visualDescription}, ${item.style}, educational infographic, high quality, 8K, ${item.textOverlay ? `text overlay "${item.textOverlay}"` : ''}`;
    
    await generateImage(prompt, outputPath);
  }
}

const day = process.argv[2] ? parseInt(process.argv[2]) : 344;
processDay(day);

